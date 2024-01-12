import os
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

    def __init__(self, csv, transform) -> None:
        """
        Initialize an `ISICDataset`.
        
        """
        self.csv = pd.read_csv(csv, low_memory=False).reset_index(drop=True)
        self.transform = transform
        

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.csv.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a sample from the dataset.
        
        :param idx: The index of the sample to return.
        :return: A sample from the dataset.
        """
        row = self.csv.iloc[idx]
        # print(os.getcwd())
        image = cv.imread(os.path.join('./data/skin/', row.filepath))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)
        
        image = image.transpose(2, 0, 1)
        
        data = torch.tensor(image).float()
        
        return data, torch.tensor(self.csv.iloc[idx].target).float()
            

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
        data_dir: str = "data/skin/csv/",
        dataset: str = "isic_balanced",
        testset: str = "",
        transform = None,
        test_transform = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `ISICDataModule`.
        
        :param csv: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        if transform is None:
            self.transform= A.Compose([
                A.Resize(256, 256),
                A.Normalize()
            ])
            
        if test_transform is None:
            self.test_transform= A.Compose([
                A.Resize(256, 256),
                A.Normalize()
            ])
    
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None



        self.batch_size_per_device = batch_size
    
    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of classes (2).
        """
        return 2
    
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
            self.data_train = ISICDataset(self.hparams.data_dir + self.hparams.dataset + '_train.csv', transform=self.transform)
            self.data_val = ISICDataset(self.hparams.data_dir + self.hparams.dataset + '_val.csv', transform=self.transform)
            # self.test = ISICDataset(self.hparams.data_dir + self.hparams.dataset + self.hparams.testset, transform=self.test_transforms)
            self.test = None
            
            
    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train, 
            batch_size=self.batch_size_per_device,
            shuffle=True,
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

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        
    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass