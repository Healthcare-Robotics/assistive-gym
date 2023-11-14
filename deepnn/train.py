import json
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

from ray.tune.schedulers import ASHAScheduler
from deepnn.model.net import MyNet
from deepnn.preprocess.custom_dataset import CustomDataset
from deepnn.utils.data_parser import ModelOutput
from deepnn.utils.loss_utils import cal_loss

INPUT_PATH = os.path.join(os.getcwd(), os.path.join('data', 'input'))
OUTPUT_PATH = os.path.join(os.getcwd(), os.path.join('data', 'output'))
CHECKPOINT_PATH = os.path.join(os.getcwd(), os.path.join('checkpoints'))

# TODO: move this one to yaml
# Hyperparameters
input_size = 82  # 72 + 10
hidden_size1 = 128
hidden_size2 = 64
hidden_size3 = 32
output_size = 32  # 32
num_epochs = 200

def get_data_split(batch_size, object):  # 60% train, 20% val, 20% test
    datasets = CustomDataset(INPUT_PATH, object, transform=None)

    train_size, val_size = int(len(datasets) * 0.7), int(len(datasets) * 0.1)
    test_size = len(datasets) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(datasets, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def train(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_data_split(config['batch_size'], config['object'])

    # init model
    model = MyNet(input_size, config['h1_size'], config['h2_size'], config['h3_size'], output_size).to(device)

    criterion = nn.MSELoss()  # For regression, we use Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # Initialize the SummaryWriter
    writer = SummaryWriter('runs/experiment_1')  # Specify the directory for logging

    # Train the model
    for epoch in range(num_epochs):
        for i, train_data in enumerate(train_loader):
            # print (features.shape, labels.shape)
            # Move data to the defined device
            features, labels = train_data['feature'], train_data['label']
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Report the metric to optimize
            tune.report(loss=loss.item())

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                # Log the loss value to TensorBoard
                writer.add_scalar('training loss', loss, epoch * len(train_loader) + i)
            if (i + 1) % 20 == 0:
                with torch.no_grad():
                    # Calculate the validation loss
                    val_loss = 0.0
                    for val_data in val_loader:
                        features, labels = val_data['feature'], val_data['label']
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
    torch.save(
            {'config': config,
             'model': model.state_dict()
            }, os.path.join(CHECKPOINT_PATH, f'model_{config["object"]}_epoch_{num_epochs}_{datetime.now()}.ckpt'))


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
        for test_data in test_loader:
            features, labels = test_data['feature'], test_data['label']
            # Move data to the correct device\
            features = features.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Update running loss value
            running_loss += loss.item() * features.size(0)  # loss.item() gives the mean loss per batch
            total_examples += features.size(0)

            # calculate custom loss
            # TODO: see if we need to run it outside
            for i in range(len(outputs)):
                label, output = labels[i], outputs[i]
                label = label.cpu().numpy()
                output = output.cpu().numpy()
                label_obj = ModelOutput.from_tensor(label)
                output_obj = ModelOutput.from_tensor(output)
                human_joint_angle_loss, robot_joint_angle_loss, robot_base_loss, robot_base_rot_loss = cal_loss(
                    label_obj, output_obj)
                print(
                    f'Human joint angle err (deg): {human_joint_angle_loss}, Robot joint angle err (deg): {robot_joint_angle_loss}, '
                    f'robot base pos err (m): {robot_base_loss}, robot base orient err: {robot_base_rot_loss}')

    # Calculate the average loss over the entire test dataset
    average_loss = running_loss / total_examples

    # Additional metrics can be calculated such as R2 score, MAE, etc.
    print(f'Average Loss on the Test Set: {average_loss:.4f}')

    # Put the model back to training mode
    model.train()

    return average_loss  # Depending on your needs, you might want to return other metrics.

def train_with_ray(config):

    # Use the ASHA scheduler
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=100,
        grace_period=1,
        reduction_factor=2
    )

    # Run the experiment
    result = tune.run(
        train,
        resources_per_trial={"cpu": 1},
        config=config,
        num_samples=1,
        scheduler=scheduler,
    )

    # Print the best result
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))

    train(best_trial.config)

def eval_model(model_checkpoint):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    saved_data = torch.load(os.path.join(CHECKPOINT_PATH, model_checkpoint))
    config = saved_data['config']
    model = MyNet(input_size, config['h1_size'], config['h2_size'], config['h3_size'], output_size).to(device)

    model.load_state_dict(saved_data['model'])
    model.eval()

    datasets = CustomDataset(INPUT_PATH, config['object'], transform=None)
    eval_loader = DataLoader(datasets, batch_size=1, shuffle=False)
    criterion = nn.MSELoss()  # For regression, we use Mean Squared Error loss

    with torch.no_grad():
        for i, data in enumerate(eval_loader):
            features, labels, input_files, output_files = data['feature'], data['label'], data['feature_path'], data[
                'label_path']
            # Move data to the correct device\
            features = features.to(device)
            # Forward pass
            outputs = model(features)
            # save to file
            for j in range(len(outputs)):
                output = outputs[j]
                output = output.cpu().numpy()
                output_obj = ModelOutput.from_tensor(output)
                data = output_obj.convert_to_dict()
                output_file = output_files[j]
                output_file = output_file.replace('input/searchoutput', 'output')
                # write to output file and create folder if not exist
                print(f'Writing to {output_file}')
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w') as f:
                    f.write(json.dumps(data, indent=4))


if __name__ == '__main__':
    # Define the hyperparameter search space
    config = {
        "lr": tune.loguniform(1e-5, 1e-1),
        "weight_decay": tune.loguniform(1e-5, 1e-1),
        "h1_size": tune.grid_search(list(range(256, 1024, 128))),
        "h2_size": tune.grid_search(list(range(128, 256, 32))),
        "h3_size": tune.grid_search(list(range(32, 128, 16))),
        "batch_size": tune.choice([16, 32, 64]),
        "object": "cane"
    }

    train_with_ray(config)

    # model_checkpoint= 'model_cane_epoch_200_2023-11-05 23:04:22.032313.ckpt'
    # eval_model(model_checkpoint)
