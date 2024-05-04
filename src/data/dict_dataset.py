from src.utils.experiment_utils import load_if_pickled
from torch.utils.data import Dataset
import torch

class DictDataset(Dataset):
    def __init__(self, data_fname, log):
        """
        Dataset class for loading a dictionary from a pickle file.
        
        Args:
            data_fname (str): The path to the pickle file containing the data.
            log (Logger): The logger to use for logging.
        """
        log.info(f"Loading data from {data_fname}")
        self.data = load_if_pickled(data_fname)
        self.keys = list(self.data.keys())
        log.info(f"Loaded {len(self.keys)} samples.")
    
    def __len__(self):
        return len(self.keys)
    
    def teardown(self):
        """
        Deletes the cached tensors
        """
        for k in self.keys:
            for k2 in list(self.data[k].keys()):
                del self.data[k][k2]
            del self.data[k]
        del self.data
    
    def transfer_to_device(self, device):
        """
        Transfers the data to the device.
        
        Args:
            device (torch.device): The device to transfer the data to.
        """
        for k in self.keys:
            for k2 in list(self.data[k].keys()):
                if torch.is_tensor(self.data[k][k2]):
                    self.data[k][k2] = self.data[k][k2].to(device)
    
    def __getitem__(self, idx):
        return self.data[self.keys[idx]], idx
    