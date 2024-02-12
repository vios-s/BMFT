import os
import torch
import pandas as pd
import cv2 as cv
import numpy as np
import albumentations as A
from typing import Dict, List, Tuple, Any
from torch.utils.data import Dataset, DataLoader
from arguments import *

args = parse_args()


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
        if 'data/skin/' not in row.filepath:
            image_path = os.path.join('./data/skin/', row.filepath)
        else:
            image_path = os.path.join(row.filepath)

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


def criterion_func(df):
    lst = df['target'].value_counts().sort_index().tolist()
    sum_lst = sum(lst)
    class_freq = []
    for i in lst:
        class_freq.append(i / sum_lst * 100)
    weights = torch.tensor(class_freq, dtype=torch.float32)

    weights = weights / weights.sum()
    weights = 1.0 / weights
    weights = weights / weights.sum()
    weights = weights.to(device)

    return weights


def load_dataset(dataset, batch_size=64, shuffle=False) -> DataLoader[Any]:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )


def get_dataset(csv, attr, transform=None):
    if transform is None:
        transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.Normalize()
        ])

    dataset = TestDataset(csv, attr, transform)

    return dataset
