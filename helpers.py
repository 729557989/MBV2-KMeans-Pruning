import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from livelossplot import PlotLosses


def train_model(output_path, model, dataloaders, criterion, optimizer, device, num_epochs=5, scheduler=None) -> int:
    (Path("models") / output_path).mkdir(parents=True, exist_ok=True)
    since = time.time()
    liveloss = PlotLosses()

    best_acc = 0.0
    best = 0

    for epoch in range(num_epochs):
        logs = {}
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=f"Epoch {epoch+1}/{num_epochs} [{phase}]"):  # tqdm wrapper added
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # statistics
                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            if phase == "train":
                prefix = ""
                if scheduler != None:
                    scheduler.step()
            else:
                prefix = "val_"

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best = epoch + 1

            logs[prefix + "log loss"] = epoch_loss.item()
            logs[prefix + "accuracy"] = epoch_acc.item()

        liveloss.update(logs)
        liveloss.send()

        torch.save(model.state_dict(), f"./models/{output_path}/model_{epoch + 1}_epoch.pt")
        
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best Validation Accuracy: {}, Epoch: {}".format(best_acc, best))
    return best

def test_model(model, dataloaders, criterion, device, phase="test"):
    since = time.time()
    # Each epoch has a training and validation phase
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloaders[phase].dataset)
    epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

    time_elapsed = time.time() - since
    print("Test Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))
    print("Test complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))



def split_list_evenly(original_list, num_splits=6):
    split_size = len(original_list) // num_splits
    remainder = len(original_list) % num_splits

    splits = []
    start = 0
    for i in range(num_splits):
        end = start + split_size + (1 if i < remainder else 0)
        splits.append(original_list[start:end])
        start = end

    return splits

def visualize_weights(model):
    layers = []
    for i, (name, param) in enumerate(model.named_parameters()):
        if param.dim() > 1:
            layers.append((name, param))
    layers_to_visualize = split_list_evenly(layers)
    for i, l in enumerate(layers_to_visualize):
        plot_weight_distribution(l, i+1)

def plot_weight_distribution(layers_to_visualize, idx, bins=256, count_nonzero_only=False):
    fig, axes = plt.subplots(3, 3, figsize=(10, 6))
    axes = axes.ravel()
    plot_index = 0
    layers = []
    for name, param in layers_to_visualize:
        if param.dim() > 1:
            layers.append(param)
            ax = axes[plot_index]
            if count_nonzero_only:
                param_cpu = param.detach().view(-1).cpu()
                param_cpu = param_cpu[param_cpu != 0].view(-1)
                ax.hist(param_cpu, bins=bins, density=True,
                        color = 'blue', alpha = 0.5)
            else:
                ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True,
                        color = 'blue', alpha = 0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
    fig.suptitle(f'Histogram of Weights, Layer Group {idx}')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()