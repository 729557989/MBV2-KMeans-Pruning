import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from helpers import train_model, test_model
from MobileNetV2 import MobileNetV2


if __name__ == "__main__":
        # Get CUDA device
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = "cpu"
        
    torch.manual_seed(42)

    # Data Preprocessing
    data_dir = "100-sports/"
    num_workers = {"train": 0, "val": 0, "test": 0}
    batch_size = 32

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
    
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    
    model_ft = MobileNetV2()
    model_ft = model_ft.to(device) # only support image of size 224x224

    # Optimizer
    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.008, weight_decay=0.0001)
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=16, gamma=0.3)

    # Train
    epochs = 80

    best_epoch = train_model(
        output_path="MobileNetV2_224",
        model=model_ft,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer_ft,
        device=device,
        num_epochs=epochs,
        scheduler=scheduler
    )

    # Test
    model_ft.load_state_dict(torch.load(f"models/MobileNetV2_224/model_{best_epoch}_epoch.pt"))
    test_model(model=model_ft, dataloaders=dataloaders, criterion=criterion, device=device)