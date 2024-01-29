import os
import torch
import pandas as pd
import cv2 as cv
import numpy as np
import albumentations as A
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader

class TestDataset(Dataset):
    """Dataset."""

    def __init__(self, csv, attr, transform) -> None:
        """
        Initialize an `ATLASDataset`.

        """
        self.csv = pd.read_csv(csv, low_memory=False).reset_index(drop=True)
        self.transform = transform
        self.attr = attr

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.csv.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a sample from the dataset.

        :param idx: The index of the sample to return.
        :return: A sample from the dataset.
        """
        row = self.csv.iloc[idx]
        # print(os.getcwd())
        image_path = os.path.join('./data/skin/', row.filepath)
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        if self.transform is not None:
          res = self.transform(image=image)
          image = res['image'].astype(np.float32)
        else:
          image = image.astype(np.float32)

        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)
        data = torch.tensor(image).float()
        label = torch.tensor(self.csv.iloc[idx].target).float()
        attr = torch.tensor(self.csv.loc[idx, self.attr])

        return data, label, attr


def get_dataset(csv, attr, transform=None, batch_size=64, shuffle=False):
    if transform is None:
      transform = A.Compose([
          A.Resize(256, 256),
          A.HorizontalFlip(p=0.5),
          A.Normalize()
      ])

    data_loader = DataLoader(TestDataset(csv, attr, transform), batch_size=batch_size, shuffle=shuffle)

    return data_loader