import cv2 as cv
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import numpy as np
import pandas as pd


from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

import albumentations as A

class ISICDataset(Dataset):
    """ISIC Dataset."""

    def __init__(self, csv, split, mode, transform: transforms.Compose, transform2: transforms.Compose) -> None:
        """
        Initialize an `ISICDataset`.
        
        """
        self.csv = csv.reset_index(drop=True)
        self.split = split
        self.mode = mode
        self.transform = transform
        self.transform2 = transform2
        

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.csv.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a sample from the dataset.
        
        :param idx: The index of the sample to return.
        :return: A sample from the dataset.
        """
        row = self.csv.iloc[idx]
        
        image = cv.imread(row.filepath)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        
        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)
            
        # Augmenting duplicated images to treat as new data points
        if self.transform2 is not None:
            if self.csv.marked[idx] == 1:
                res = self.transform2(image=image)
                image = res['image'].astype(np.float32)
        
        image = image.transpose(2, 0, 1)
        
        data = torch.tensor(image).float()
        
        if self.mode == 'test':
            return data, torch.tensor(self.csv.iloc[idx].target).long()
        
        else:
            if self.args.instrument:
                return data, torch.tensor(self.csv.iloc[idx].target).long(), \
                            torch.tensor(self.csv.iloc[idx].instrument).long(), \
                            torch.tensor(self.csv.iloc[idx].marked).long
                            
            elif self.args.instrument and self.
            
            

class ISICDataModule(LightningDataModule):
    """ISIC DataModule.
    
    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```
    """
    
    def __init__(
        self,
        csv: str = "data/",
        split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        transform = None,
        transform2= None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `ISICDataModule`.
        
        :param csv: The data directory. Defaults to `"data/"`.
        :param split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        # data transformations
        if self.transforms.skew:
            self.transforms_marked = A.Compose([
                A.Transpose(p=0.5),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
            ])
        else:
            self.transforms_marked = None

        # Augmentations for all training data
        if self.model.name == 'inception':  # Special augmentation for inception to provide 299x299 images
            self.transforms_train = A.Compose([
                A.Resize(299, 299),
                A.Normalize()
            ])
        else:
            self.transforms_train = A.Compose([
                A.Resize(self.data.image_size, self.data.image_size),
                A.Normalize()
            ])

        # Augmentations for validation data
        self.transforms_val = A.Compose([
            A.Resize(self.data.image_size, self.data.image_size),
            A.Normalize()
        ])
        

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
    
    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. 
        Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = ISICDataset(self.hparams.csv, split=self.hparams.split, mode='train', transform=self.hparams.transforms1, transform2=self.hparams.transforms2)
            valset = ISICDataset(self.hparams.csv, split=self.hparams.split, mode='val', transform=self.hparams.transforms1, transform2=self.hparams.transforms2)
            testset = ISICDataset(self.hparams.csv, split=self.hparams.split, mode='test', transform=self.hparams.transforms1, transform2=self.hparams.transforms2)
            
            
    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train, 
            batch_size=self.batch_size_per_device,
            sampler=RandomSampler(self.data_train),
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            drop_last=True)
        
    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True
        )