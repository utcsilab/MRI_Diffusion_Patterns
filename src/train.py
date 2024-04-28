import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import logging
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np

import torch
from torch.utils.data import DataLoader

from src.utils.experiment_utils import set_all_seeds
from src.sampling_patterns.learned3d import Learned3d

from src.data.data_utils import split_dataset
from src.data.fastMRI import BrainMultiCoil, KneesMultiCoil

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    log.info("Setting all seeds")
    set_all_seeds(cfg.seed)
    
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Setup Sampling pattern
    if cfg.pattern.sample_pattern == 'Learned3d':
        pattern_class = Learned3d
    else:
        raise NotImplementedError(f"Pattern class {cfg.pattern.sample_pattern} not implemented.") 
    
    log.info("Initialising Sampling Pattern")
    
    sampling_pattern = Learned3d(num_acs_lines=cfg.pattern.num_acs_lines,
                                 R=cfg.pattern.R,
                                 length=cfg.data.image_size,
                                 device=device,
                                 cut_corners=cfg.pattern.cut_corners,
                                 init_dist=cfg.pattern.init_dist,
                                 sampler=cfg.pattern.sampler,
                                 tau=cfg.pattern.tau)
    
    # Setup datasets
    if cfg.data.dataset == "BrainMultiCoil":
        dataset_class = BrainMultiCoil
    elif cfg.data.dataset == "KneesMultiCoil":
        dataset_class = KneesMultiCoil
    else:
        raise NotImplementedError(f"Dataset class {cfg.data.dataset} not implemented.")
    
    log.info("Initialising datasets")
    
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

    train_loader = DataLoader(split_dict['train'], 
                              batch_size=cfg.data.train_batch_size,
                              shuffle=True,
                              num_workers=16,
                              drop_last=True)
    val_loader = DataLoader(split_dict['val'],
                            batch_size=cfg.data.val_batch_size,
                            shuffle=False,
                            num_workers=16,
                            drop_last=False)
    test_loader = DataLoader(split_dict['test'],
                             batch_size=cfg.data.test_batch_size,
                             shuffle=False,
                             num_workers=16,
                             drop_last=False)
    
    # Check and set up num_iters if needed
    # NOTE right now this only works for 3D sampling patterns
    if cfg.training.num_iters == -1:
        updates_per_epoch = cfg.data.num_train / cfg.data.train_batch_size
        cfg.training.num_iters = int(np.ceil(((cfg.data.image_size**2) / 
                                    cfg.pattern.R - cfg.pattern.num_acs_lines**2) / updates_per_epoch))
        
        log.info(f"Setting epochs to {cfg.training.num_iters}")
        
    # Check if we need to initialise an optimizer
    if cfg.training.optimizer == "adam":
        opt = torch.optim.Adam(sampling_pattern.parameters(), lr=cfg.training.lr)
    elif cfg.training.optimizer == "greedy_topk":
        pass
    else:
        raise NotImplementedError(f"Optimizer {cfg.training.optimizer} not implemented.")
    
    
    
if __name__ == "__main__":
    train()
