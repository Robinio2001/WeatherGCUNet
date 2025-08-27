import lightning as L
import numpy as np
from utils.data_module import DataModule
from model.model import WeatherGC_UNet
from argparse import ArgumentParser

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

def train_regression(hparams):
    dm = DataModule(hparams)

    model = WeatherGC_UNet(hparams)

    tb_logger = TensorBoardLogger(save_dir=hparams.default_dir + hparams.delimiter + 'logs')
    lr_monitor = LearningRateMonitor()

    checkpoint_callback = ModelCheckpoint(dirpath=hparams.default_dir + hparams.delimiter + 'saved_models',
                                          filename=hparams.filename,
                                          monitor='val_loss',
                                          save_top_k=1,
                                          mode='min')
    
    es_callback = EarlyStopping(monitor='val_loss',
                                patience=hparams.es_patience,
                                mode='min')
    
    trainer = L.Trainer(accelerator="auto",
                        default_root_dir=hparams.default_dir,
                        max_epochs=hparams.epochs,
                        logger=tb_logger,
                        callbacks=[checkpoint_callback, es_callback, lr_monitor],
                        fast_dev_run=hparams.fast_dev_run,
                        deterministic=True)

    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = DataModule.add_model_specific_args(parser)
    parser = WeatherGC_UNet.add_model_specific_args(parser)

    parser.add_argument('--model', type=str, default='WeatherGC_UNet')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--default_dir', type=str, default='lightning/')
    parser.add_argument('--filename', type=str, default='{model}-{epoch}-{val_loss:.5f}')

    args = parser.parse_args()

    # Data loading
    args.fast_dev_run = True
    args.valid_size = 0.1
    args.delimiter = '/'

    # Data specific
    args.num_nodes = 5
    args.num_timesteps = 30

    # Model specific
    args.pool_ratio = 0.8 # Ratio of nodes that should be selected during pooling
    args.sigma = 0.35
    
    train_regression(args)