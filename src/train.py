import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import logging
from omegaconf import DictConfig, OmegaConf
import hydra
import json

import numpy as np

from tqdm import trange, tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import torch
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tb

from src.utils.experiment_utils import set_all_seeds, make_dirs, save_images
from src.sampling_patterns.learned3d import Learned3d
from src.recon_algorithms.diffusion_utils import load_net
from src.recon_algorithms.diffusion import DiffusionMRIReconstruction
from src.data.data_utils import split_dataset
from src.data.fastMRI import BrainMultiCoil, KneesMultiCoil
from src.data.dict_dataset import DictDataset
from src.utils.metric_utils import Metrics
from src.utils.training_utils import make_noisy_sample, single_step_posterior_estimate, calculate_loss
from src.utils.helpers import get_mvue_torch, get_min_max, normalize

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig) -> None:
    # (0) Set up
    print(OmegaConf.to_yaml(cfg))
    
    log.info("Setting all seeds")
    set_all_seeds(cfg.seed)
    
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    
    # (1) Setup Sampling pattern
    if cfg.pattern.sample_pattern == 'Learned3d':
        pattern_class = Learned3d
    else:
        raise NotImplementedError(f"Pattern class {cfg.pattern.sample_pattern} not implemented.") 
    
    log.info("Initialising Sampling Pattern")
    
    sampling_pattern = pattern_class(num_acs_lines=cfg.pattern.num_acs_lines,
                                     R=cfg.pattern.R,
                                     length=cfg.data.image_size,
                                     device=device,
                                     cut_corners=cfg.pattern.cut_corners,
                                     init_dist=cfg.pattern.init_dist,
                                     sampler=cfg.pattern.sampler,
                                     tau=cfg.pattern.tau)
    
    # (2) Setup datasets
    if cfg.data.dataset == "BrainMultiCoil":
        dataset_class = BrainMultiCoil
    elif cfg.data.dataset == "KneesMultiCoil":
        dataset_class = KneesMultiCoil
    elif cfg.data.dataset == "DictDataset":
        dataset_class = DictDataset
    else:
        raise NotImplementedError(f"Dataset class {cfg.data.dataset} not implemented.")
    
    log.info("Initialising datasets")
    
    if cfg.data.dataset == "DictDataset":
        train_split = DictDataset(data_fname=cfg.data.train_file, log=log)
        val_split = DictDataset(data_fname=cfg.data.val_file, log=log)
        test_split = DictDataset(data_fname=cfg.data.test_file, log=log)
    else:
        train_dataset = dataset_class(input_dir=cfg.data.train_input_dir,
                                    maps_dir=cfg.data.train_maps_dir,
                                    file_pattern=cfg.data.file_pattern,
                                    ignore_slice_list=cfg.data.ignore_slice_list,
                                    image_size=cfg.data.image_size,
                                    num_slices_path=cfg.data.train_num_slices_path,
                                    load_slice_info=cfg.data.load_slice_info,
                                    save_slice_info=cfg.data.save_slice_info,
                                    kspace_pad=cfg.data.kspace_pad,
                                    remove_start=cfg.data.remove_start,
                                    remove_end=cfg.data.remove_end,
                                    cache_data=cfg.data.cache_data,
                                    log=log)
        
        test_dataset = dataset_class(input_dir=cfg.data.test_input_dir,
                                    maps_dir=cfg.data.test_maps_dir,
                                    file_pattern=cfg.data.file_pattern,
                                    ignore_slice_list=cfg.data.ignore_slice_list,
                                    image_size=cfg.data.image_size,
                                    num_slices_path=cfg.data.test_num_slices_path,
                                    load_slice_info=cfg.data.load_slice_info,
                                    save_slice_info=cfg.data.save_slice_info,
                                    kspace_pad=cfg.data.kspace_pad,
                                    remove_start=cfg.data.remove_start,
                                    remove_end=cfg.data.remove_end,
                                    cache_data=cfg.data.cache_data,
                                    log=log)

        split_dict = split_dataset(train_set=train_dataset,
                                test_set=test_dataset,
                                num_train=cfg.data.num_train,
                                num_val=cfg.data.num_val,
                                num_test=cfg.data.num_test,
                                seed=cfg.seed,
                                log=log)

        train_split, val_split, test_split = split_dict['train'], split_dict['val'], split_dict['test']

    train_loader = DataLoader(train_split, 
                              batch_size=cfg.data.train_batch_size,
                              shuffle=True,
                              num_workers=1,
                              drop_last=True,
                              persistent_workers=True,
                              pin_memory=True)
    val_loader = DataLoader(val_split,
                            batch_size=cfg.data.val_batch_size,
                            shuffle=False,
                            num_workers=1,
                            drop_last=False,
                            persistent_workers=False,
                            pin_memory=True)
    test_loader = DataLoader(test_split,
                             batch_size=cfg.data.test_batch_size,
                             shuffle=False,
                             num_workers=1,
                             drop_last=False,
                             persistent_workers=False,
                             pin_memory=True)
    
    # (3) Check and set up num_iters if needed
    # NOTE right now this only works for 3D sampling patterns
    if cfg.training.num_iters == -1:
        updates_per_epoch = cfg.data.num_train / cfg.data.train_batch_size
        cfg.training.num_iters = int(np.ceil(((cfg.data.image_size**2) / 
                                    cfg.pattern.R - cfg.pattern.num_acs_lines**2) / updates_per_epoch))
        
        log.info(f"Setting epochs to {cfg.training.num_iters}")
        
    # (4) Check if we need to initialise an optimizer
    if cfg.training.optimizer == "adam":
        opt = torch.optim.Adam(sampling_pattern.parameters(), lr=cfg.training.lr)
    elif cfg.training.optimizer == "greedy_topk":
        opt = None
    else:
        raise NotImplementedError(f"Optimizer {cfg.training.optimizer} not implemented.")
    
    #TODO make a way to load a pattern from a .pt file
    
    # (5) Initialise the network and recon
    edm_path = os.path.join(os.path.dirname(__file__), "recon_algorithms", "edm")
    log.info(f"EDM path: {edm_path}")
    sys.path.append(edm_path)
    
    net = load_net(cfg.recon.net_path, device=device)
    
    recon_alg = DiffusionMRIReconstruction(net=net,
                                           hard_consistent_output=cfg.recon.hard_consistent_output,
                                           alg_type=cfg.recon.alg_type,
                                           steps=cfg.recon.steps,
                                           rho=cfg.recon.rho,
                                           S_churn=cfg.recon.S_churn,
                                           S_min=cfg.recon.S_min,
                                           S_max=cfg.recon.S_max,
                                           S_noise=cfg.recon.S_noise,
                                           sigma_min=cfg.recon.sigma_min,
                                           sigma_max=cfg.recon.sigma_max,
                                           **cfg.recon.kwargs)
    
    # (6) Logging and metrics
    metrics = Metrics()
    
    log_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    log.info(f"Log directory: {log_dir}")
    make_dirs(log_dir, ["images", "tensorboard"])
    
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
        
    tb_logger = tb.SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
    
    # (7) Save a snap of the initial sampling patterns before any training
    P = sampling_pattern.sample_mask(n=1).detach().cpu() #[1,1,H,W]
    P_prob = sampling_pattern.probabilistic_mask().detach().cpu() #[1,1,H,W]
    
    pattern_path = os.path.join(log_dir, "images", "learned_masks")
    make_dirs(os.path.join(log_dir, "images"), ["learned_masks"])
    
    save_images(P, ["Sample_00"], pattern_path)
    save_images(P_prob, ["Prob_00"], pattern_path)
    
    acceleration = (torch.numel(P) / torch.sum(P)).item()
    log.info(f"Initial acceleration: {acceleration}")
    
    ##########################################
    ############## TRAINING LOOP #############
    ##########################################
    finished_flag = False
    
    with logging_redirect_tqdm():
        for epoch in trange(cfg.training.num_iters, unit=" epoch"):
            # (0) Checkpoint
            if epoch % cfg.training.checkpoint_every == 0:
                #NOTE add checkpointing functionality
                log.info("Saving checkpoint...") 
            
            # (1) Train
            for i, (item, idx) in tqdm(enumerate(train_loader), desc="Training", unit=" batch"):
                # (a) grab variables and move to gpu
                FSx, S, x = item['ksp'].to(device), item['s_maps'].to(device), item['gt_image'].to(device)
                scan_idx, slice_idx = item['scan_idx'].tolist(), item['slice_idx'].tolist()
                
                # (b) grab pattern, make noisy sample, and estimate posterior mean
                P = sampling_pattern.sample_mask(n=x.shape[0])
                
                x_t, sigma_t = make_noisy_sample(x=x, sigma_t=None, normalize_input=True)
                
                x_hat = single_step_posterior_estimate(net=net, x_t=x_t, sigma_t=sigma_t, FSx=FSx, P=P, S=S, 
                                                       likelihood_step_size=cfg.training.likelihood_step_size)
                
                train_loss = calculate_loss(x_hat=x_hat, x=x, loss_type=cfg.training.loss_type)
                train_loss.backward()
                
                # (c) update pattern
                if cfg.training.optimizer == "adam":
                    opt.step()
                    opt.zero_grad()
                elif cfg.training.optimizer == "greedy_topk":
                    finished_flag = sampling_pattern.greedy_topk_step(k=cfg.training.k, 
                                                                      include_conjugates=cfg.training.include_conjugates)
                
                # (d) logging metrics and saving images for the current batch
                with torch.no_grad():
                    # (i) metrics
                    resid = x_hat - x
                    gt_mse = torch.mean(torch.square(resid), dim=[1,2,3]) 
                    gt_mae = torch.mean(torch.abs(resid), dim=[1,2,3]) 
                    
                    R_sample = (P.shape[2] * P.shape[3]) / torch.sum(P, dim=[1, 2, 3])
                    
                    metrics_dict = {"train_loss": np.array([train_loss.item()] * x.shape[0]),
                                    "sigma_t": sigma_t.squeeze().detach().cpu().numpy(),
                                    "gt_mse": gt_mse.squeeze().detach().cpu().numpy(),
                                    "gt_mae": gt_mae.squeeze().detach().cpu().numpy(),
                                    "R_sample": R_sample.squeeze().detach().cpu().numpy()}
                    metrics.add_external_metrics(metrics_dict, iter_num=epoch, iter_type="train")
                    metrics.calc_iter_metrics(x_hat=x_hat, x=x, iter_num=epoch, iter_type="train")
                    
                    # (ii) save images
                    if (i == 0) and ((epoch % cfg.training.checkpoint_every == 0) or (epoch == cfg.training.num_iters - 1)):
                        #Save sampling patterns
                        P = sampling_pattern.sample_mask(n=1).detach().cpu() #[1,1,H,W]
                        P_prob = sampling_pattern.probabilistic_mask().detach().cpu() #[1,1,H,W]
                        
                        pattern_path = os.path.join(log_dir, "images", "learned_masks")
                        
                        save_images(P, [f"Sample_{epoch}"], pattern_path)
                        save_images(P_prob, [f"Prob_{epoch}"], pattern_path)
                        
                        #Save reconstructions at every iteration
                        x_idx = [f"{scan_id}_{slice_id}" for scan_id, slice_id in zip(scan_idx, slice_idx)]
                        x_resid_idx = [f"{idx}_resid" for idx in x_idx]
                        x_resid_stretched_idx = [f"{idx}_resid_stretched" for idx in x_idx]
                        
                        # norm_factor = np.percentile(torch.norm(x, dim=1).detach().cpu().numpy(), q=99, axis=(1, 2)) #[N]
                        # norm_factor = torch.from_numpy(norm_factor).to(device).view(-1, 1, 1, 1)
                        norm_factor = 1
                        
                        x_hat_vis = x_hat / norm_factor
                        x_vis = x / norm_factor
                        x_resid = x_hat_vis - x_vis
                        x_resid_stretched = 5 * x_resid
                        
                        recovered_path = os.path.join(log_dir, "images",  "train_recon", f"epoch_{epoch}")
                        save_images(x_hat_vis, x_idx, recovered_path)
                        save_images(x_resid, x_resid_idx, recovered_path)
                        save_images(x_resid_stretched, x_resid_stretched_idx, recovered_path)
                        
                        #Save the ground truth images only once
                        if epoch == 0:
                            true_path = os.path.join(log_dir, "images",  "train")
                            save_images(x_vis, x_idx, true_path)

                # (e) check if we are done
                if finished_flag:
                    log.info("Desired acceleration reached. Exiting training loop.")
                    break
            
            # (f) log metrics for the entire epoch
            metrics.aggregate_iter_metrics(iter_num=epoch, iter_type="train")
            metrics.add_metrics_to_tb(tb_logger=tb_logger, step=epoch, iter_type="train")
            if (epoch % cfg.training.checkpoint_every == 0) or (epoch == cfg.training.num_iters - 1):
                log.info(metrics.get_all_metrics(iter_num=epoch, iter_type="train"))
            
            # (2) Validate
            if (epoch + 1) % cfg.training.val_every == 0:
                for i, (item, idx) in tqdm(enumerate(val_loader), desc="Validation", unit=" batch"):
                    pass #NOTE add validation functionality
            
            #Check if complete
            if finished_flag:
                log.info("Finishing logging and starting testing (if desired)")
                
                if cfg.data.dataset != "DictDataset":
                    train_loader.dataset.dataset.teardown()
                    val_loader.dataset.dataset.teardown()
                else:
                    train_loader.dataset.teardown()
                    val_loader.dataset.teardown()
                    
                break
    
        #(3) Test
        for i, (item, idx) in tqdm(enumerate(test_loader), desc="Testing", unit=" batch"):
            # (a) grab variables and move to gpu
            FSx, S, x = item['ksp'].to(device), item['s_maps'].to(device), item['gt_image'].to(device)
            scan_idx, slice_idx = item['scan_idx'].tolist(), item['slice_idx'].tolist()
            
            # (b) grab pattern, make initial noisy sample, and sample from posterior
            P = sampling_pattern.sample_mask(n=x.shape[0]).detach()
            
            y = P * FSx
            x_hat_mvue = get_mvue_torch(y, S)
            norm_mins, norm_maxes = get_min_max(x_hat_mvue)
            x_init = normalize(x_hat_mvue, norm_mins, norm_maxes)
            x_init = x_init + cfg.recon.sigma_max * torch.randn_like(x_init)
            
            x_hat = recon_alg(x_init=x_init, FSx=FSx, P=P, S=S)
            
            # (d) logging metrics and saving images for the current batch
            with torch.no_grad():
                # (i) metrics
                resid = x_hat - x
                gt_mse = torch.mean(torch.square(resid), dim=[1,2,3]) 
                gt_mae = torch.mean(torch.abs(resid), dim=[1,2,3]) 
                
                R_sample = (P.shape[2] * P.shape[3]) / torch.sum(P, dim=[1, 2, 3])
                
                metrics_dict = {"gt_mse": gt_mse.squeeze().detach().cpu().numpy(),
                                "gt_mae": gt_mae.squeeze().detach().cpu().numpy(),
                                "R_sample": R_sample.squeeze().detach().cpu().numpy()}
                metrics.add_external_metrics(metrics_dict, iter_num=epoch, iter_type="test")
                metrics.calc_iter_metrics(x_hat=x_hat, x=x, iter_num=epoch, iter_type="test")
                
                # (ii) save images
                if i == 0:
                    #Save sampling patterns on first batch
                    P = sampling_pattern.sample_mask(n=1).detach().cpu() #[1,1,H,W]
                    P_prob = sampling_pattern.probabilistic_mask().detach().cpu() #[1,1,H,W]
                    
                    pattern_path = os.path.join(log_dir, "images", "learned_masks")
                    
                    save_images(P, [f"Sample_{epoch}"], pattern_path)
                    save_images(P_prob, [f"Prob_{epoch}"], pattern_path)
                
                #Save reconstructions at every iteration
                x_idx = [f"{scan_id}_{slice_id}" for scan_id, slice_id in zip(scan_idx, slice_idx)]
                x_resid_idx = [f"{idx}_resid" for idx in x_idx]
                x_resid_stretched_idx = [f"{idx}_resid_stretched" for idx in x_idx]
                
                # norm_factor = np.percentile(torch.norm(x, dim=1).detach().cpu().numpy(), q=99, axis=(1, 2)) #[N]
                # norm_factor = torch.from_numpy(norm_factor).to(device).view(-1, 1, 1, 1)
                norm_factor = 1
                
                x_hat_vis = x_hat / norm_factor
                x_vis = x / norm_factor
                x_resid = x_hat_vis - x_vis
                x_resid_stretched = 5 * x_resid
                
                recovered_path = os.path.join(log_dir, "images",  "test_recon", f"epoch_{epoch}")
                save_images(x_hat_vis, x_idx, recovered_path)
                save_images(x_resid, x_resid_idx, recovered_path)
                save_images(x_resid_stretched, x_resid_stretched_idx, recovered_path)
                
                #save ground truth images at every test iteration
                true_path = os.path.join(log_dir, "images",  "test")
                save_images(x_vis, x_idx, true_path)
                
                # (iii) grab the stats and save to a file
                metric_dict = metrics.get_dict("test")[f'iter_{epoch}']
                psnr_array = metric_dict['psnr'][-len(x_idx):]
                ssim_array = metric_dict['ssim'][-len(x_idx):]
                sample_metric_dicts = [{"Slice": idx, "PSNR": psnr_array[i], "SSIM": ssim_array[i]} for i, idx in enumerate(x_idx)]
                metric_path = os.path.join(recovered_path, "sample_metrics.json")
                with open(metric_path, 'a') as f:
                    json.dump(sample_metric_dicts, f, indent=4)
                    
                avg_metric_dict = [{"MEAN PSNR": np.mean(metric_dict['psnr']), "MEAN SSIM": np.mean(metric_dict['ssim']),
                                    "STD PSNR": np.std(metric_dict['psnr']), "STD SSIM": np.std(metric_dict['ssim'])}]
                avg_metric_path = os.path.join(recovered_path, "avg_sample_metrics.json")
                with open(avg_metric_path, 'w') as f:
                    json.dump(avg_metric_dict, f, indent=4)
                    
        # (f) log metrics for the entire epoch
        metrics.aggregate_iter_metrics(iter_num=epoch, iter_type="test")
        metrics.add_metrics_to_tb(tb_logger=tb_logger, step=epoch, iter_type="test")
        log.info(metrics.get_all_metrics(iter_num=epoch, iter_type="test"))
    
        #NOTE add checkpointing functionality
        log.info("Saving final checkpoint...") 

        if cfg.data.dataset != "DictDataset":
            test_loader.dataset.dataset.teardown()
        else:
            test_loader.dataset.teardown()
            
if __name__ == "__main__":
    sys.exit(train())
