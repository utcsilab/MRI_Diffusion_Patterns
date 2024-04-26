from torch.utils.data import Dataset

class BrainMultiCoil(Dataset):
    def __init__(self, 
                 input_dir, 
                 maps_dir, 
                 image_size=384, 
                 num_slices_path=None, 
                 slice_mapper_path=None, 
                 load_slice_info=False, 
                 save_slice_info=False, 
                 kspace_pad=28,
                 remove_start=0,
                 remove_end=5,
                 cache_data=False):
        """
        FastMRI multicoil brain dataset

        Args:
            input_dir (str or path): Path to multi-coil k-space data as .h5 files.
            maps_dir (str or path): Path to coil sensitivity maps.
            image_size (int, optional): Crop size. Defaults to 384.
            num_slices_path (str or path, optional): Numpy file containing the number of slices in each volume. 
                                                     Will be read from if 'load_slice_info' is set to 'True'.
                                                     Will be saved to if 'save_slice_info' is set to 'True'.
                                                     Defaults to None.
            slice_mapper_path (str or path, optional): Numpy file containing the cumsum of slices for each volume. 
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
        """
        