import torch
import torch.nn.functional as F
from utils.data_module import DataModule
from model.model import WeatherGC_UNet
from argparse import ArgumentParser, Namespace

torch.manual_seed(123)
torch.set_printoptions(precision=20)

# Get specific arguments
ap = ArgumentParser()
ap = DataModule.add_model_specific_args(ap)
ap = WeatherGC_UNet.add_model_specific_args(ap)
ap.add_argument('--model_ckpt', type=str, default="checkpoints/6H-epoch=18-val_loss=0.03635.ckpt")
args = vars(ap.parse_args())
args["batch_size"] = 6

dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(dev)

dm = DataModule(args)
dm.setup("test")
test_loader = dm.test_dataloader()

# From scale1.mat
scaler = torch.tensor([28.3, 62.52179487, 25.]).to(dev)

model_path = args["model_ckpt"]

# Load model weights from checkpoint
checkpoint = torch.load(args["model_ckpt"], map_location=torch.device('cpu'))
hparams = Namespace(**checkpoint["hyper_parameters"])

best_model = WeatherGC_UNet(hparams)
best_model.load_state_dict(checkpoint["state_dict"])
best_model.to(dev)
best_model.eval()

print(f"Model Path = {model_path}")

loss_func = F.l1_loss
loss_func_2 = F.mse_loss

with torch.no_grad():
    test_loss = 0.0
    test_loss_2 = 0.0
    test_num = 0

    for xb, yb in test_loader:
        pred, _ = best_model(xb.to(dev))
        batch_test_loss = loss_func(pred * scaler, yb.to(dev) * scaler, reduction="none")
        batch_test_loss_2 = loss_func_2(pred * scaler, yb.to(dev) * scaler, reduction='none')

        test_loss += torch.sum(batch_test_loss, dim=0)
        test_loss_2 += torch.sum(batch_test_loss_2, dim=0)
        test_num += len(xb)

test_loss /= test_num #average of all samples
print("\nTest MAE loss of the model:", test_loss)
print("Average:", torch.mean(test_loss), "\n")

test_loss_2 /= test_num # average of all samples
print("Test MSE loss of the model:", test_loss_2)
print("Average:", torch.mean(test_loss_2), "\n")