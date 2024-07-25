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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset(Dataset):
    """Dataset."""

    def __init__(self, csv, attr, transform) -> None:
        """
        Initialize an `ATLASDataset`.

        """
        self.csv = csv
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
        if args.task == 'skin':
            if 'data/skin/' not in row.filepath:
                if 'data/Skin/' in row.filepath:
                    image_path = row.filepath
                    image_path = image_path.replace('data/Skin/', 'data/skin/')
                else:
                    image_path = os.path.join('data/skin/', row.filepath)
            else:
                image_path = row.filepath
        if args.task == 'xray':
            if 'data/chestXray/' not in row.filepath:
                image_path = os.path.join('data/chestXray/', row.filepath)
            else:
                image_path = row.filepath


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


def target_dataset_balance(df, target):
    target0 = df[df[target] == 0]
    target1 = df[df[target] == 1]
    if len(target0) > len(target1):
        # index = np.random.randint(len(target1), size=len(target0))
        index = np.random.randint(len(target1), size=len(target1))
        target1 = target1.iloc[list(index)]
    else:
        index = np.random.randint(len(target0), size=len(target1))
        target0 = target0.iloc[list(index)]
    csv = pd.concat([target0, target1], ignore_index=True)
    return csv


def dataset_balance(df, attribute, target, downsample=0):
    group0 = df[df[attribute] == 0]
    group1 = df[df[attribute] == 1]
    if len(group0) > len(group1):
        df0, df1 = split_dataset(group0, group1, target, downsample)
    else:
        df0, df1 = split_dataset(group1, group0, target, downsample)

    csv = pd.concat([df0, df1], ignore_index=True)
    return csv


def split_dataset(df0, df1, target, downsample):
    group0_target0 = df0[df0[target] == 0]
    group0_target1 = df0[df0[target] == 1]
    group1_target0 = df1[df1[target] == 0]
    group1_target1 = df1[df1[target] == 1]
    if downsample:
        index_target0 = np.random.randint(len(group0_target0), size=len(group1_target0))
        down_target0 = group0_target0.iloc[list(index_target0)]
        index_target1 = np.random.randint(len(group0_target1), size=len(group1_target1))
        down_target1 = group0_target1.iloc[list(index_target1)]
        df0 = pd.concat([down_target0, down_target1], ignore_index=True)
    else:
        index_target0 = np.random.randint(len(group1_target0), size=len(group0_target0))
        up_target0 = group1_target0.iloc[list(index_target0)]
        index_target1 = np.random.randint(len(group1_target1), size=len(group0_target1))
        up_target1 = group0_target1.iloc[list(index_target1)]
        df1 = pd.concat([up_target0, up_target1], ignore_index=True)

    return df0, df1


def cal_pos_weight(df):
    pos_count = df['target'].sum() * 1.0 + 1e-10
    neg_count = (len(df) - pos_count)*1.0
    ratio = neg_count/pos_count
    pos_weight = torch.tensor(ratio, device=device)

    return pos_weight


def load_dataset(dataset, batch_size=64, shuffle=False) -> DataLoader[Any]:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )


def get_dataset(csv, attr, transform=None, mode='train'):
    if mode == 'train':
        if transform is None:
            transform = A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.Normalize()
            ])
    else:
        transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize()
            ])

    dataset = TestDataset(csv, attr, transform)

    return dataset
