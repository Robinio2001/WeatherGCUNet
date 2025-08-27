import torch
import lightning as L
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from argparse import ArgumentParser
from scipy.io import loadmat

class DataModule(L.LightningDataModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_dir', type=str, default='data/step1.mat')
        parser.add_argument('--val_ratio', type=float, default=0.1)

        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # Should be adjusted based on the hardware on which the code is run
        self.num_workers=0
        self.persistent_workers=True
        self.pin_memory=True
    
    def setup(self, stage):
        mat = loadmat(self.hparams.data_dir)
        if stage=="fit":
            # [Samples, V, T, C] where V = cities, T=timesteps, C=weather variables
            X_mat = mat["Xtr"]
            # [Samples, V] where V are the target cities
            Y_mat = mat["Ytr"]

            X = torch.tensor(X_mat).permute(0, 3, 2, 1).float()
            Y = torch.tensor(Y_mat).float()

            self.train_val_dataset = TensorDataset(X, Y)

            num_samples = len(self.train_val_dataset)
            indices = list(range(num_samples))
            split = int(np.floor(self.hparams.val_ratio*num_samples))

            np.random.seed(42)
            np.random.shuffle(indices)

            train_ids, val_ids = indices[split:], indices[:split]
            self.train_sampler = SubsetRandomSampler(train_ids)
            self.val_sampler = SubsetRandomSampler(val_ids)
        
        if stage=="test":
            X_mat = mat["Xtest"]
            Y_mat = mat["Ytest"]

            X = torch.tensor(X_mat).permute(0, 3, 2, 1).float()
            Y = torch.tensor(Y_mat).float()

            self.test_dataset = TensorDataset(X, Y)

    def train_dataloader(self):
        return DataLoader(self.train_val_dataset,
                          batch_size=self.hparams.batch_size,
                          sampler=self.train_sampler,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.train_val_dataset,
                          batch_size=self.hparams.batch_size,
                          sampler=self.val_sampler,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)