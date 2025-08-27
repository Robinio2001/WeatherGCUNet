import torch
import torch.nn.functional as F
import lightning as L
import argparse

class LightningBase(L.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--es_patience', type=int, default=5)
        return parser
    
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)


    def loss_func(self, y_pred, y_true):
        # Average over batch
        return F.l1_loss(y_pred, y_true, reduction="mean")

    def training_step(self, batch, batch_idx):
        X, y = batch

        pred, A = self(X)
        task_loss = self.loss_func(pred, y)
        
        train_loss = task_loss

        # Logging
        self.log("task_loss", task_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss", train_loss, on_step=True, on_epoch=True)

        if self.global_step % 100 == 0:
            self.logger.experiment.add_image(f"adj/A", A[:, :].detach().unsqueeze(0), self.global_step)

        return train_loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return {
            "optimizer": optim,
            "monitor": "val_loss"
        }
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        pred, _, = self(X)
        val_loss = self.loss_func(pred, y)

        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

        return val_loss