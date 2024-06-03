# %% パターン4でインストールしたPyTorchの動作確認
import torch
import numpy as np
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
x_ones = torch.ones_like(x_data) # x_dataの特性（プロパティ）を維持
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_dataのdatatypeを上書き更新
print(f"Random Tensor: \n {x_rand} \n")

# %% CPUとGPU（device="mpu"）の速度比較
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import CIFAR10
import time
import matplotlib.pyplot as plt

SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 5  # number of epochs
NUM_LOAD_WORKERS = 0  # Number of workers for DataLoader

def train_cifar10_resnet18(device):
    if not torch.backends.mps.is_available():
        raise Exception('MPS is not available')
    # Set random seed
    torch.manual_seed(SEED)
    ###### 1. Create dataset & Preprocessing ######
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Augmentation by flipping
        transforms.ColorJitter(),  # Augmentation by randomly changing brightness, contrast, saturation, and hue.
        transforms.RandomRotation(degrees=10),  # Augmentation by rotation
        transforms.ToTensor(),  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
        transforms.Normalize((0.5,), (0.5,))  # Normalization
    ])
    # Load train dataset
    train_dataset = CIFAR10(root = '.', train = True, transform = transform, download=True)
    # Define class names
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    # Define mini-batch DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS)
    # Load validation dataset
    val_dataset = CIFAR10(root = '.', train = False, transform = transform, download=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOAD_WORKERS)
    ###### 2. Define Model ######
    # Load a pretrained model
    model = models.resnet18(pretrained=True)
    print(model)
    # Freeze all the parameters (https://pytorch.org/examples/beginner/transfer_learning_tutorial.html#convnet-as-fixed-feature-extractor)
    for param in model.parameters():
        param.requires_grad = False
    # Modify the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(idx_to_class.keys()))
    # Send the model to GPU
    model.to(device)
    ###### 3. Define Criterion & Optimizer ######
    criterion = nn.CrossEntropyLoss()  # Criterion (Cross entropy loss)
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Optimizer (Adam). Only parameters in the final layer are set.
    ###### 4. Training ######
    losses = []  # Array for storing loss (criterion)
    accs = []  # Array for storing accuracy
    val_losses = []  # Array for validation loss
    val_accs = []  # Array for validation accuracy
    start = time.time()  # For elapsed time
    # Epoch loop
    for epoch in range(NUM_EPOCHS):
        # Initialize training metrics
        running_loss = 0.0  # Initialize running loss
        running_acc = 0.0  # Initialize running accuracy
        # Mini-batch loop
        for imgs, labels in train_loader:
            # Send images and labels to GPU
            imgs = imgs.to(device)
            labels = labels.to(device)
            # Update parameters
            optimizer.zero_grad()  # Initialize gradient
            output = model(imgs)  # Forward (Prediction)
            loss = criterion(output, labels)  # Calculate criterion
            loss.backward()  # Backpropagation (Calculate gradient)
            optimizer.step()  # Update parameters (Based on optimizer algorithm)
            # Store running losses and accs
            running_loss += loss.item()  # Update running loss
            pred = torch.argmax(output, dim=1)  # Predicted labels
            running_acc += torch.mean(pred.eq(labels).float())  # Update running accuracy
        # Calculate average of running losses and accs
        running_loss /= len(train_loader)
        losses.append(running_loss)
        running_acc /= len(train_loader)
        accs.append(running_acc.cpu())

        # Calculate validation metrics
        val_running_loss = 0.0  # Initialize validation running loss
        val_running_acc = 0.0  # Initialize validation running accuracy
        for val_imgs, val_labels in val_loader:
            val_imgs = val_imgs.to(device)
            val_labels = val_labels.to(device)
            val_output = model(val_imgs)  # Forward (Prediction)
            val_loss = criterion(val_output, val_labels)  # Calculate criterion
            val_running_loss += val_loss.item()   # Update running loss
            val_pred = torch.argmax(val_output, dim=1)  # Predicted labels
            val_running_acc += torch.mean(val_pred.eq(val_labels).float())  # Update running accuracy
        val_running_loss /= len(val_loader)
        val_losses.append(val_running_loss)
        val_running_acc /= len(val_loader)
        val_accs.append(val_running_acc.cpu())

        print(f'epoch: {epoch}, loss: {running_loss}, acc: {running_acc},  val_loss: {val_running_loss}, val_acc: {val_running_acc}, elapsed_time: {time.time() - start}')

    ###### 5. Model evaluation and visualization ######
    # Plot loss history
    plt.plot(losses, label='Train loss')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Loss history')
    plt.legend()
    plt.show()
    # Plot accuracy history
    plt.plot(accs, label='Train accuracy')
    plt.plot(val_accs, label='Validation accuracy')
    plt.title('Accuracy history')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train_cifar10_resnet18(device='mps')
    train_cifar10_resnet18(device='cpu')

# %%
