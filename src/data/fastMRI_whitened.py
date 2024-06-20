import numpy as np
import h5py
import os
from tqdm import tqdm
import sigpy as sp
import copy
from multiprocessing import Manager

import torch
from torch.utils.data import Dataset

from src.data.data_utils import get_all_files
from src.utils.helpers import get_mvue_numpy

class BrainMultiCoilWhitened(Dataset):
    def __init__(self, 
                 data_dir,
                 file_list,
                 image_size=320,
                 acs_size = 20,
                 pad_coils=True,
                 remove_start=0,
                 remove_end=5,
                 cache_data=False,
                 log=None):
        """
        Whitened fastMRI multicoil brain dataset.
        
        Assumes subdirectories "coil_sens", "ground_truths", and "ksp" exist within data_dir.
        
        Data are stored as .npy volumes with shape [H, W, N_coils, N_slices] for coils and ksp, 
            and [H, W, N_slices] for ground truths.
        
        Triplets of (coil_sens, ksp, ground_truth) should be stored with the same filename in their 
            respective subdirectories.
        """
        super.__init__()
        
        self.data_dir = data_dir
        self.file_list = file_list
        self.image_size = image_size
        self.acs_size = acs_size
        self.pad_coils = pad_coils
        self.remove_start = remove_start
        self.remove_end = remove_end
        self.cache_data = cache_data
        self.log = log
        
        self.ksp_dir = os.path.join(data_dir, 'ksp')
        self.coil_sens_dir = os.path.join(data_dir, 'coil_sens')
        self.gt_dir = os.path.join(data_dir, 'ground_truths')
        
        #Access volume metadata to get number of slices
        if log is not None:
            log.info('Accessing volume metadata...')
            
        self.num_slices = np.zeros((len(file_list)), dtype=int)
        self.max_N_coils = 0
        
        for i, fname in tqdm(enumerate(file_list)):
            all_exist = os.path.isfile(os.path.join(self.ksp_dir, fname)) and \
                        os.path.isfile(os.path.join(self.coil_sens_dir, fname)) and \
                        os.path.isfile(os.path.join(self.gt_dir, fname))
            assert all_exist, 'Missing data for volume: {}'.format(fname)
            
            H, W, N_coils, N_slices = np.load(os.path.join(self.ksp_dir, fname), mmap_mode='r+').shape
            
            assert H == W == image_size, "Incorrect shape for volume: {}".format(fname)
            assert N_slices > (remove_end + remove_start), "Not enough slices for volume: {}".format(fname)
            
            if N_coils > self.max_N_coils:
                self.max_N_coils = N_coils
            
            self.num_slices[i] = N_slices - (remove_end + remove_start)
        
        self.slice_mapper = np.cumsum(self.num_slices) - 1 #counts from 0
        
        #Make the acs mask
        center_line_idx = np.arange((image_size - acs_size) // 2, (image_size + acs_size) // 2)
        mask = np.zeros((image_size, image_size), dtype=bool)
        mask[center_line_idx] = True
        mask = mask * mask.transpose(1, 0)
        self.acs_mask = mask #[H, W]
        
        if self.cache_data:
            manager = Manager()
            self.dataset_cache = manager.dict()
    
    def __len__(self):
        return self.slice_mapper[-1] + 1
    
    def teardown(self):
        """
        Deletes the cached tensors if they exist
        """
        if (not self.cache_data) or (not hasattr(self, "dataset_cache")):
            return
        
        for k in list(self.dataset_cache.keys()):
            for k2 in list(self.dataset_cache[k].keys()):
                del self.dataset_cache[k][k2]
            del self.dataset_cache[k]
        del self.dataset_cache
    
    def __getitem__(self, idx):
        #see if the sample is already cached
        if self.cache_data and (idx in self.dataset_cache):
            return self.dataset_cache[idx], idx
        
        # Get scan and slice index
        # First scan for which index is in the valid cumulative range
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # Offset from cumulative range
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)
        slice_idx += self.remove_start #skip some at the start if desired
        
        # Grab the data
        s_maps = np.load(os.path.join(self.coil_sens_dir, self.file_list[scan_idx]), mmap_mode='r+')[..., slice_idx]
        s_maps = s_maps.transpose(2, 0, 1) #[N_coils, H, W]
        
        ksp = np.load(os.path.join(self.ksp_dir, self.file_list[scan_idx]), mmap_mode='r+')[..., slice_idx]
        ksp = ksp.transpose(2, 0, 1) #[N_coils, H, W]
        
        gt_mvue = np.load(os.path.join(self.gt_dir, self.file_list[scan_idx]), mmap_mode='r+')[..., slice_idx] #[H, W]
        gt_mvue = np.stack((np.real(gt_mvue),np.imag(gt_mvue)), axis=0) #[2, H, W]
        
        # Get the normalisation factor from the acs region
        masked_ksp = ksp * self.acs_mask # [N_coils, H, W]
        mvue_masked = get_mvue_numpy(masked_ksp, s_maps) # [H, W]
        acs_norm_factor = np.percentile(np.abs(mvue_masked), 99)
        
        # Apply optional padding
        if self.pad_coils and s_maps.shape[0] < self.max_N_coils and ksp.shape[0] < self.max_N_coils:
            s_maps = np.pad(s_maps, ((0, self.max_N_coils - s_maps.shape[0]), (0, 0), (0, 0)), mode='constant')
            ksp = np.pad(ksp, ((0, self.max_N_coils - ksp.shape[0]), (0, 0), (0, 0)), mode='constant')
            
        # Output
        sample = {'ksp': torch.from_numpy(ksp.astype(np.complex64)),
                  's_maps': torch.from_numpy(s_maps.astype(np.complex64)),
                  'gt_image': torch.from_numpy(gt_mvue.astype(np.float32)),
                  'acs_norm_factor': acs_norm_factor,
                  'scan_idx': scan_idx,
                  'slice_idx': slice_idx}
        
        if self.cache_data:
            self.dataset_cache[idx] = copy.deepcopy(sample)
        
        return sample, idx
    