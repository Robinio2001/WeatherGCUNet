# WeatherGC-UNet
Code implementation and saved model checkpoints for the WeatherGC-UNet model.

## Installation
Clone the repository with git:

```
git clone https://github.com/Robinio2001/WeatherGCUNet.git
```

We use Poetry for dependency management. Installing the required packages:

```
poetry install --no-root
```

Activate the virtual environment created by Poetry:

```
eval $(poetry env activate)
```

## Data
The required dataset can be found [here](https://drive.google.com/drive/folders/1dqtmmfyKg5X2aCa2W8pBvPCrxE7Tc974?usp=sharing). The input timestep (time lag) is 30, it was derived from the dataset that can be found [here](https://sites.google.com/view/siamak-mehrkanoon/code-data).

Ensure the data is located in the '/data' folder, which should be the files ```step1.mat``` until ```step4.mat```. These files represent the different prediction horizons, from 6H to 24H respectively.

### Training
The training script can be run by running ```python train_model_lightning.py``` with the desired hyperparameters. E.g.

```
python train_model_lightning.py --batch_size 32 --data_dir data/step1.mat --epochs 50 --es_patience 5
```

During training, the model checkpoint with the lowest validation loss will be saved in ```lightning/saved_models```. 

It is possible to save all checkpoints, or the top-k, by changing this value inside ```train_model_lightning.py``` in the ```CheckpointCallback```.

### Evaluation
The model can be evaluated using ```test_model_lightning.py```, providing the path to the desired checkpoint as hyperparameter. 

It is important to provide the correct dataset to ensure that the model is evaluated on the right prediction horizon. For this, the right data path needs to be provided. Example of evaluating a model on the 18H data:

```
python test_model_lightning.py --model_ckpt checkpoints/18H-epoch=21-val_loss=0.04894.ckpt --data_dir data/step3.mat
```

At the end of the test script, the average MAE and MSE values over the whole test set are provided, as well as the average values per city.


