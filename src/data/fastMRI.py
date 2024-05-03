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


class BrainMultiCoil(Dataset):
    def __init__(self, 
                 input_dir, 
                 maps_dir,
                 file_pattern, 
                 ignore_slice_list=None,
                 image_size=384, 
                 num_slices_path=None, 
                 load_slice_info=False, 
                 save_slice_info=False, 
                 kspace_pad=28,
                 remove_start=0,
                 remove_end=5,
                 cache_data=False,
                 log=None):
        """
        FastMRI multicoil brain dataset

        Args:
            input_dir (str or path): Path to multi-coil k-space data as .h5 files.
            maps_dir (str or path): Path to coil sensitivity maps.
            file_pattern (str): Pattern to match files in input_dir. E.G. '*T2*.h5'
            ignore_slice_list (list, optional): List of file names to ignore. Defaults to None.
            image_size (int, optional): Crop size. Defaults to 384.
            num_slices_path (str or path, optional): Numpy file containing the number of slices in each volume. 
                                                     Will be read from if 'load_slice_info' is set to 'True'.
                                                     Will be saved to if 'save_slice_info' is set to 'True'.
                                                     Defaults to None.
            load_slice_info (bool, optional): Whether to load slice info instead of calculating from volumes. 
                                              Defaults to False.
            save_slice_info (bool, optional): Whether to save slice info after calculating. 
                                              Defaults to False.
            kspace_pad (int, optional): Number of channels to pad each sample to enable batching in dataloaders. 
                                        Defaults to 28.
            remove_start (int, optional): Number of slices to ignore from the start of each volume. 
                                          Defaults to 0.
            remove_end (int, optional): Number of slices to ignore from the end of each volume. 
                                        Defaults to 5.
            cache_data (bool, optional): Whether to cache samples in RAM as they are seen. 
                                         Defaults to False.
            log (logging.Logger, optional): Logger for logging. Defaults to None.
        """
        # Attributes
        self.input_dir = input_dir
        self.maps_dir = maps_dir
        self.image_size = image_size
        self.kspace_pad = kspace_pad
        self.remove_start = remove_start
        self.remove_end = remove_end
        self.cache_data = cache_data
        
        self.file_list = get_all_files(input_dir, pattern=file_pattern)
        if ignore_slice_list is not None:
            self.file_list = [f for f in self.file_list if os.path.basename(f) not in ignore_slice_list]
        
        # Access meta-data of each scan to get number of slices
        if not load_slice_info:
            log.info("Reading " + str(len(self.file_list)) + " Scans")
            self.num_slices = np.zeros((len(self.file_list,)), dtype=int)
            for idx, file in tqdm(enumerate(self.file_list)):
                input_file = os.path.join(self.input_dir, os.path.basename(file))
                with h5py.File(input_file, 'r') as data:
                    self.num_slices[idx] = int(np.array(data['kspace']).shape[0])
            
            #Check if we want to save the slice info 
            # want to save the raw info before removing slices from start or end!            
            if save_slice_info:
                log.info("Saving compiled scan information!")
                np.save(num_slices_path, self.num_slices)
        else:
            log.info("Loading available scan information")
            self.num_slices = np.load(num_slices_path)
        
        self.num_slices = self.num_slices - (self.remove_start + self.remove_end)    
        self.slice_mapper = np.cumsum(self.num_slices) - 1 # Counts from '0'
        
        if self.cache_data:
            manager = Manager()
            self.dataset_cache = manager.dict()
            # self.dataset_cache = {}
    
    def __len__(self):
        total_slices = int(np.sum(self.num_slices))
        
        return total_slices
    
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
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
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

        # Load maps for specific scan and slice
        maps_file = os.path.join(self.maps_dir,
                                 os.path.basename(self.file_list[scan_idx]))
        with h5py.File(maps_file, 'r') as data:
            # Get maps
            s_maps = np.asarray(data['s_maps'][slice_idx])

        # Load raw data for specific scan and slice
        raw_file = os.path.join(self.input_dir,
                                os.path.basename(self.file_list[scan_idx]))
        with h5py.File(raw_file, 'r') as data:
            # Get k-space
            gt_ksp = np.asarray(data['kspace'][slice_idx])

        # Crop extra lines and reduce FoV in phase-encode
        gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], gt_ksp.shape[1], self.image_size))

        # Reduce FoV by half in the readout direction
        gt_ksp = sp.ifft(gt_ksp, axes=(-2,))
        gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], self.image_size,
                                    gt_ksp.shape[2]))
        gt_ksp = sp.fft(gt_ksp, axes=(-2,)) # Back to k-space

        # Crop extra lines and reduce FoV in phase-encode
        s_maps = sp.fft(s_maps, axes=(-2, -1)) # These are now maps in k-space
        s_maps = sp.resize(s_maps, (
            s_maps.shape[0], s_maps.shape[1], self.image_size))

        # Reduce FoV by half in the readout direction
        s_maps = sp.ifft(s_maps, axes=(-2,))
        s_maps = sp.resize(s_maps, (s_maps.shape[0], self.image_size,
                                    s_maps.shape[2]))
        s_maps = sp.fft(s_maps, axes=(-2,)) # Back to k-space
        s_maps = sp.ifft(s_maps, axes=(-2, -1)) # Finally convert back to image domain

        # find mvue image
        gt_mvue = get_mvue_numpy(gt_ksp, s_maps)

        ksp = gt_ksp

        #NOTE this is legacy pre-scaling - EDM is trained with different scaling
        # commenting this for now, but old results were using both scalings!!!
        gt_mvue_scale_factor = np.percentile(np.abs(gt_mvue),99)
        ksp /= gt_mvue_scale_factor
        gt_mvue /= gt_mvue_scale_factor

        gt_mvue_two_channel = np.zeros((2,) + gt_mvue.shape, dtype=np.float32)
        gt_mvue_two_channel[0] = np.real(gt_mvue).astype(np.float32)
        gt_mvue_two_channel[1] = np.imag(gt_mvue).astype(np.float32)

        #Apply optional K-space padding
        #This allows us to make the non-batch dimensions of all the samples homogenous,
        #   and allows for batch size > 1
        if self.kspace_pad:
            if (ksp.shape[0] < self.kspace_pad) and (s_maps.shape[0] < self.kspace_pad):
                ksp = np.pad(ksp, ((0,self.kspace_pad - ksp.shape[0]), (0,0), (0,0)))
                s_maps = np.pad(s_maps, ((0,self.kspace_pad - s_maps.shape[0]), (0,0), (0,0)))

        # Output
        sample = {'ksp': ksp.astype(np.complex64), #[C, H, W] complex64 numpy array
                  's_maps': s_maps.astype(np.complex64), #[C, H, W] complex64 numpy array
                  'gt_image': gt_mvue_two_channel.astype(np.float32),
                  'scan_idx': scan_idx,
                  'slice_idx': slice_idx}
        
        if self.cache_data:
            self.dataset_cache[idx] = copy.deepcopy(sample)

        return sample, idx

class KneesMultiCoil(BrainMultiCoil):
    def __init__(self, 
                 input_dir, 
                 maps_dir,
                 file_pattern, 
                 ignore_slice_list=None,
                 image_size=320, 
                 num_slices_path=None, 
                 load_slice_info=False, 
                 save_slice_info=False, 
                 kspace_pad=False,
                 remove_start=10,
                 remove_end=0,
                 cache_data=False,
                 log=None):
        
        super(KneesMultiCoil, self).__init__(input_dir=input_dir,
                                             maps_dir=maps_dir,
                                             file_pattern=file_pattern,
                                             ignore_slice_list=ignore_slice_list,
                                             image_size=image_size,
                                             num_slices_path=num_slices_path,
                                             load_slice_info=load_slice_info,
                                             save_slice_info=save_slice_info,
                                             kspace_pad=kspace_pad,
                                             remove_start=remove_start,
                                             remove_end=remove_end,
                                             cache_data=cache_data,
                                             log=log)
        