import torch
import numpy as np
import os
import glob


def get_all_files(folder, pattern='*'):
    """
    Return a list of all files in a given folder that match a given pattern.
    
    Args:
        folder (str): Folder path
        pattern (str): File pattern to match
    
    Returns:
        files (list): List of file paths
    """
    files = [x for x in glob.iglob(os.path.join(folder, pattern))]
    return sorted(files)

def split_dataset(train_set, test_set, num_train, num_val, num_test, seed, log):
    """
    Split a given dataset into train, val, and test sets.
    
    Args:
        train_set (torch.utils.data.Dataset): Training dataset
        test_set: (torch.utils.data.Dataset): Testing dataset
        num_train: (int) Number of training samples
        num_val: (int) Number of validation samples (from training set)
        num_test: (int) Number of testing samples.
        seed: (int) Random seed for reproducibility.
        log (logging.Logger): Logger for logging messages.
    
    Returns:
        out_dict (dict): Dictionary containing the train, val, and test datasets.
    """
    tr_indices = list(range(len(train_set))) if train_set is not None else []
    te_indices = list(range(len(test_set)))

    log.info(f"Train Dataset Size: {len(tr_indices)}")
    log.info(f"Test Dataset Size: {len(te_indices)}")

    random_state = np.random.get_state()
    np.random.seed(seed)
    np.random.shuffle(tr_indices)
    np.random.seed(seed)
    np.random.shuffle(te_indices)
    np.random.set_state(random_state)

    train_indices = tr_indices[:num_train] if train_set is not None else []
    val_indices = tr_indices[num_train:num_train+num_val] if train_set is not None else []
    test_indices = te_indices[:num_test]

    train_dataset = torch.utils.data.Subset(train_set, train_indices) if train_set is not None else None
    val_dataset = torch.utils.data.Subset(train_set, val_indices) if train_set is not None else None
    test_dataset = torch.utils.data.Subset(test_set, test_indices)

    out_dict = {'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset}
    
    return out_dict
