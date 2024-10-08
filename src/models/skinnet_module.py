from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.auroc import AUROC

class SkinNetLitModule(LightningModule):
    """A `LightningModule` for SkinNet classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```
    """
    
    def __init__(
        self,
        net: torch.nn.Module,
        head: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None: 
        """Initialize a `SkinNetLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net', 'head'])

        # model
        self.net = net
        self.head = head

        # class_counts = torch.tensor([951.0, 33647.0])
        # class_weights = 1. / class_counts
        #
        # # Normalizing the weights (optional)
        # class_weights = class_weights / class_weights.sum()

        self.criterion = torch.nn.BCEWithLogitsLoss()
        # loss function
        # self.criterion = torch.nn.BCELoss()
        # torch.nn.CrossEntropyLoss(reduction='none')

        # metrics
        self.train_acc = Accuracy(task="binary", num_classes=2)
        self.val_acc = Accuracy(task="binary", num_classes=2)
        self.test_acc = Accuracy(task="binary", num_classes=2)

        self.train_auc = AUROC(task="binary", num_classes=2)
        self.val_auc = AUROC(task="binary", num_classes=2)
        self.test_auc = AUROC(task="binary", num_classes=2)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # for tracking best so far validation accuracy
        # self.val_acc_best = MaxMetric()
        self.val_auc_best = MaxMetric()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: The input tensor.
        :return: The output tensor.
        """
        return self.head(self.net(x))
    
    def on_train_start(self) -> None:
        """Called when the train begins."""
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_auc.reset()
        # self.val_acc_best.reset()
        self.val_auc_best.reset()
        
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]):
        """Perform a step through the model `self.net`.

        :param batch: The batch of data.
        """
        x, y = batch
        logits = self.forward(x).squeeze(1)
        count_pos = torch.sum(y) * 1.0 + 1e-10
        count_neg = torch.sum(1. - y) * 1.0
        beta = count_neg / count_pos
        beta_back = count_pos / (count_pos + count_neg)

        bce1 = torch.nn.BCEWithLogitsLoss(pos_weight=beta)
        loss = beta_back * bce1(logits, y)
        # loss = self.criterion(logits, y)
        preds = ((logits>0.5).float())
        return loss, preds, y
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data.
        
        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        :param batch_idx: The batch index.
        :return: A tensor of losses.
        """
        
        loss, preds, targets = self.model_step(batch)
        
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_auc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        pass
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_auc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc) # update best so far val acc
        auc = self.val_auc.compute()  # get current val auc
        self.val_auc_best(auc)
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/auc_best", self.val_auc_best.compute(), sync_dist=True, prog_bar=True)
        
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_auc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}