"""
@file dataloader.py
@author Ryan Missel

Holds the WebDataset classes for the available datasets
"""
import torch
import numpy as np

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, file_path, config):
        # Load data
        npzfile = np.load(file_path)
        self.images = npzfile['images'].astype(np.float32)
        self.images = (self.images > 0).astype('float32')

        # Only load the position, not velocity
        self.state = npzfile['states'].astype(np.float32)[:, :, :2]
        self.state_dim = self.state.shape[-1]

        # Modify based on dataset percent
        rand_idx = np.random.choice(range(self.images.shape[0]), size=int(self.images.shape[0] * config.dataset_percent), replace=False)
        self.images = self.images[rand_idx]
        self.state = self.state[rand_idx]
        print(f"Images: {self.images.shape} | States: {self.state.shape}")

        # Get data dimensions
        self.sequences, self.timesteps = self.images.shape[0], self.images.shape[1]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        """ Couple images and controls together for compatibility with other datasets """
        image = torch.from_numpy(self.images[idx, :])
        state = torch.from_numpy(self.state[idx, :])
        return torch.Tensor([idx]), image, state


if __name__ == '__main__':
    # Test loading dataset and visualizing it
    dataset = BaseDataset("../data/bouncing_ball/bouncingball_8000samples/train.npz", {'dataset_percent': 1.0})

    import matplotlib.pyplot as plt
    plt.imshow(dataset.images[0, 0])
    plt.show()
