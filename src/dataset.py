"""
PyTorch Dataset for the 2D slices made by prepare_slices.py.
Same as in the U-Net project (../C1), but it takes a list of cases instead of a split name,
because here the split changes with every fold.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BrainSliceDataset(Dataset):
    def __init__(self, slices_dir, cases, augment=False):
        self.dir = Path(slices_dir)
        self.augment = augment
        # mmap_mode="r": only the slice we ask for is read from disk
        self.images = {c: np.load(self.dir / f"{c}_img.npy", mmap_mode="r") for c in cases}
        self.labels = {c: np.load(self.dir / f"{c}_lbl.npy", mmap_mode="r") for c in cases}
        self.index = [(c, i) for c in cases for i in range(len(self.labels[c]))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        case, i = self.index[idx]
        image = torch.from_numpy(np.array(self.images[case][i], dtype=np.float32))  # (4, H, W)
        label = torch.from_numpy(np.array(self.labels[case][i], dtype=np.int64))  # (H, W)

        if self.augment:
            # left-right flip (same flip for image and label) + small intensity change
            if torch.rand(1) < 0.5:
                image, label = image.flip(-1), label.flip(-1)
            image = image * (1 + 0.1 * (torch.rand(1) - 0.5)) + 0.1 * (torch.rand(1) - 0.5)

        return image, label
