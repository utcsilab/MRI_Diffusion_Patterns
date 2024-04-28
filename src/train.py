import logging
from omegaconf import DictConfig, OmegaConf
import hydra
import torch

from src.sampling_patterns.learned3d import Learned3d
from src.data.fastMRI import BrainMultiCoil, KneesMultiCoil

log = logging.getLogger(__name__)


@hydra.main(version_base=None)
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    log.info("Info level message")
    
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    
    sampling_pattern = Learned3d(num_acs_lines=cfg.pattern.num_acs_lines,
                                 R=cfg.pattern.R,
                                 length=cfg.pattern.length,
                                 device=device,
                                 cut_corners=cfg.pattern.cut_corners,
                                 init_dist=cfg.pattern.init_dist,
                                 sampler=cfg.pattern.sampler,
                                 tau=cfg.pattern.tau)
    
    dataset_class = BrainMultiCoil if cfg.data.dataset == "brain" else KneesMultiCoil
    train_dataset = dataset_class()

if __name__ == "__main__":
    train()
