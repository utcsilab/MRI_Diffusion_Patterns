import torchvision
import torch
import os
import pickle
import random
import numpy as np
import sys


def save_image(image, path):
    """
    Saves the given image to the specified path.

    Args:
        image (torch.Tensor): The image to be saved. Dimension should be (1, H, W) or (2, H, W).
                                If (2, H, W), the image is converted to grayscale before saving.
        path (str): The path where the image will be saved. Should include the filename and extension.
    """
    x_png = image.detach().cpu()

    if x_png.shape[0] == 2:
        x_png = torch.linalg.norm(x_png, dim=0, keepdim=True)
    
    torchvision.utils.save_image(x_png, path)
    
def save_images(images, labels, save_prefix):
    """
    Save a batch of images to png files and .pt files.
    
    Args:
        images (torch.Tensor): The images to be saved. Dimension should be (N, 1, H, W) or (N, 2, H, W).
        labels (list): The labels of the images. Can be integers, strings, or torch.Tensors.
        save_prefix (str): The prefix of the path where the images will be saved. Should not include the filename.
    """
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)
    
    for image_num, image in zip(labels, images):
        if isinstance(image_num, torch.Tensor):
            save_image(image, os.path.join(save_prefix, str(image_num.item())+'.png'))
            torch.save(image, os.path.join(save_prefix, str(image_num.item())+'.pt'))
        elif isinstance(image_num, int):
            save_image(image, os.path.join(save_prefix, str(image_num)+'.png'))
            torch.save(image, os.path.join(save_prefix, str(image_num)+'.pt'))
        elif isinstance(image_num, str):
            save_image(image, os.path.join(save_prefix, image_num +'.png'))
            torch.save(image, os.path.join(save_prefix, image_num +'.pt'))
        else:
            raise NotImplementedError("Bad type given to save_images for labels.")
        
def save_to_pickle(data, pkl_filepath):
    """
    Save the data to a pickle file.
    
    Args:
        data (object): The data to be saved.
        pkl_filepath (str): The path where the data will be saved. Should include the filename and extension.
    """
    with open(pkl_filepath, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)
        
def load_if_pickled(pkl_filepath):
    """
    Load the data from a pickle file if it exists.
    
    Args:
        pkl_filepath (str): The path of the pickle file to be loaded.
        
    Returns:
        object: The data loaded from the pickle file.
    """
    if os.path.isfile(pkl_filepath):
        with open(pkl_filepath, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
    else:
        data = {}
    return data

def set_all_seeds(random_seed: int):
    """
    Sets random seeds in numpy, torch, and random.
    
    Args:
        random_seed (int): The seed to set.
    """
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

def make_dirs(root_dir, sub_dirs, overwrite=False):
    """
    Create directories in the given root directory.
    
    Args:
        root_dir (str): The root directory where the subdirectories will be created.
        sub_dirs (list): The list of subdirectories to be created.
        overwrite (bool): If True, the directories will be created even if they already exist.
    """
    for sub_dir in sub_dirs:
        path = os.path.join(root_dir, sub_dir)
        if not os.path.exists(path) or overwrite:
            os.makedirs(path, exist_ok=True)
        else:
            sys.exit("Folder exists. Program halted.")

