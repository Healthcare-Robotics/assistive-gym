from datetime import time, datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

from deepnn.model.net import MyNet
from deepnn.preprocess.custom_dataset import CustomDataset

DATA_PATH = os.path.join(os.getcwd(), os.path.join('data'))
CHECKPOINT_PATH = os.path.join(os.getcwd(), os.path.join('checkpoints'))

# TODO: move this one to yaml
# Hyperparameters
input_size = 82  # 72 + 10
hidden_size1 = 200
hidden_size2 = 100
output_size = 32  # MNIST has 10 classes
num_epochs = 1000
batch_size = 16
learning_rate = 0.1


def get_data_split():  # 60% train, 20% val, 20% test
    datasets = CustomDataset(DATA_PATH, transform=None)

    train_size, val_size = int(len(datasets) * 0.6), int(len(datasets) * 0.2)
    test_size = len(datasets) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(datasets, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_data_split()

    # init model
    model = MyNet(input_size, hidden_size1, hidden_size2, output_size).to(device)

    criterion = nn.MSELoss()  # For regression, we use Mean Squared Error loss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Initialize the SummaryWriter
    writer = SummaryWriter('runs/experiment_1')  # Specify the directory for logging

    # Train the model
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(train_loader):
            # print (features.shape, labels.shape)
            # Move data to the defined device
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                # Log the loss value to TensorBoard
                writer.add_scalar('training loss', loss, epoch * len(train_loader) + i)
            if (i + 1) % 20 == 0:
                with torch.no_grad():
                    # Calculate the validation loss
                    val_loss = 0.0
                    for features, labels in val_loader:
                        # Move data to the correct device
                        features = features.to(device)
                        labels = labels.to(device)
                        # Forward pass
                        outputs = model(features)
                        loss = criterion(outputs, labels)
                        # Update running loss value
                        val_loss += loss.item() * features.size(0)
                    # Calculate the average loss over the entire validation dataset
                    average_val_loss = val_loss / len(val_loader.dataset)


                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Validation Loss: {average_val_loss:.4f}')
                # Log the validation loss to TensorBoard
                writer.add_scalar('validation loss', average_val_loss, epoch * len(train_loader) + i)
                model.train()
    writer.close()
    test_model(model, test_loader, criterion)
    # Save the model checkpoint with time stamp
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, f'model_epoch_{num_epochs}_{datetime.now()}.ckpt'))


def test_model(model, test_loader, criterion):
    """
    Evaluate the performance of a neural network model on a test dataset.

    Parameters:
    model (nn.Module): The neural network model.
    test_loader (DataLoader): DataLoader object for the test dataset.
    criterion (nn.Module): Loss function used for the model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Ensure the model is in evaluation mode (as it affects specific layers like dropout)
    model.eval()

    # To accumulate the losses and the number of examples seen
    running_loss = 0.0
    total_examples = 0

    # No need to track gradients for evaluation, saving memory and computations
    with torch.no_grad():
        for features, labels in test_loader:
            # Move data to the correct device\
            features = features.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Update running loss value
            running_loss += loss.item() * features.size(0)  # loss.item() gives the mean loss per batch
            total_examples += features.size(0)

    # Calculate the average loss over the entire test dataset
    average_loss = running_loss / total_examples

    # Additional metrics can be calculated such as R2 score, MAE, etc.
    print(f'Average Loss on the Test Set: {average_loss:.4f}')

    # Put the model back to training mode
    model.train()

    return average_loss  # Depending on your needs, you might want to return other metrics.


if __name__ == '__main__':
    train()
