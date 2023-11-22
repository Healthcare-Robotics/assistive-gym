import json
import torch
import os
from torch.utils.data import Dataset, DataLoader

from deepnn.utils.data_parser import ModelInput, ModelOutput

SEARCH_INPUT_DIR = 'searchinput'
SEARCH_OUTPUT_DIR = 'searchoutput'

# TODO: split such that you have unseen person for testing
class CustomDataset(Dataset):
    def __init__(self, directory, object=None, transform=None):
        """
        Initialize the object with the directory where the JSON files are located.
        Each JSON file contains a single data sample.

        Parameters:
        directory (str): Directory containing the JSON files.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.transform = transform
        self.inputfile_list =[]
        self.outputfile_list = []
        # Expect the directory to contain two subdirectories: 'input' and 'output'
        input_dir, output_dir = os.path.join(directory, SEARCH_INPUT_DIR), os.path.join(directory, SEARCH_OUTPUT_DIR)
        # print (input_dir, output_dir)
        # list all subdirectories in 'output'
        person_ids = sorted([f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))])
        # print(person_ids)
        for p in person_ids:
            subinput_dir, sub_output_dir = os.path.join(input_dir, p), os.path.join(output_dir, p)
            pose_ids = [f for f in os.listdir(sub_output_dir) if os.path.isdir(os.path.join(sub_output_dir, f))]
            inputfile_list = []
            outputfile_list = []
            for pose_id in pose_ids:
                subsub_output_dir = os.path.join(sub_output_dir, pose_id)

                outputfile_list.extend([os.path.join(subsub_output_dir, f) for f in os.listdir(subsub_output_dir) if f.endswith(object + '.json')])
                inputfile_list.append(os.path.join(subinput_dir, pose_id+'.json'))

            self.inputfile_list.extend(inputfile_list)
            self.outputfile_list.extend(outputfile_list)
        assert len(self.inputfile_list) == len(self.outputfile_list)


    def __len__(self):
        """Return the total number of data points (JSON files)."""
        return len(self.inputfile_list)

    def __getitem__(self, idx):
        """
        Load the JSON file corresponding to 'idx' and return the processed data.
        """
        # Build file path
        inputfile_path = os.path.join(self.directory, self.inputfile_list[idx])
        outputfile_path = os.path.join(self.directory, self.outputfile_list[idx])

        # Load JSON file
        with open(inputfile_path, 'r') as infile:
            input = json.load(infile)
        with open(outputfile_path, 'r') as outfile:
            output = json.load(outfile)

        # TODO: cut this parsing to a class
        model_input = ModelInput(input['pose'], input['betas'])
        model_output = ModelOutput(output['joint_angles'], output['robot']['original'][0], output['robot']['original'][1], output['robot_joint_angles'])

        feature = model_input.to_tensor()
        label = model_output.to_tensor()

        if self.transform:
            feature = self.transform(feature)

        return {
            'feature': feature,
            'label': label,
            'feature_path': inputfile_path,
            'label_path': outputfile_path
        }

if __name__ == '__main__':
    # Path to the directory containing your JSON files
    directory_path = os.path.join(os.getcwd(), os.path.join('data'))

    # Define transformations here (if any)
    transformations = None

    # Create an instance of your dataset
    json_dataset = CustomDataset(directory_path, transform=transformations)
    print ("Number of samples:", len(json_dataset))

    # Create a data loader
    data_loader = DataLoader(json_dataset, batch_size=16, shuffle=True)

    # Iterate over data to see how data is loaded
    for batch_idx, (features, labels) in enumerate(data_loader):
        print("Batch:", batch_idx)
        # print("Features:", features)
        # print("Labels:", labels)