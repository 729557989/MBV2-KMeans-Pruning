import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from test import test_model
from train import train_model

from MobileNetV2 import MobileNetV2


if __name__ == "__main__":
    # Get CUDA device
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = "cpu"

    # Data Preprocessing
    data_dir = "tiny-224/"
    num_workers = {"train": 0, "val": 0, "test": 0}
    batch_size = 100

    data_transforms = {
        "train": transforms.Compose(
            [
                # transforms.RandomRotation(20),
                # transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]
        ),
    }
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val", "test"]
    }
    dataloaders = {
        x: data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers[x])
        for x in ["train", "val", "test"]
    }
    
    # Load MobileNetV2
    torch.manual_seed(42)
    
    # model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model_ft = MobileNetV2()
    
    # print(model_ft)
    model_ft = model_ft.to(device) # only support image of size 224x224

    # Loss Function
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.00166, momentum=0.9)
    
    # Train
    epochs = 10
    
    best_epoch = train_model(
        output_path="MobileNetV2_224",
        model=model_ft,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer_ft,
        device=device,
        num_epochs=epochs,
    )
    
    # Test
    model_ft.load_state_dict(torch.load(f"models/MobileNetV2_224/model_{best_epoch}_epoch.pt"))
    test_model(model=model_ft, dataloaders=dataloaders, criterion=criterion, device=device)