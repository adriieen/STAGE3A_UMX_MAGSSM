import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import warnings
import importlib.util
import json

from tqdm import tqdm


from src.utils.train_test import evaluate
from src.utils.GoogleSpeechCommands import SubsetSC
from src.utils.experimentManager import ExperimentManagerLoadFunction, ExperimentManagerReadExistingEntry
from src.utils.LogFile import Echo_STDIO_to_File
from src.utils.export_model import export_layer_parameters

import matplotlib.pyplot as plt


results_path = './results_journal/'
export_path = './export_model/'
run = 0
step_scale_list = [[1, 1, 1], [1, 2, 22], [1, 4, 64]]


full_export_path = os.path.join(export_path, f'export_model_run_{run}')

if not os.path.exists(full_export_path):
    os.makedirs(full_export_path, exist_ok=True)


# Arg parser
parser = argparse.ArgumentParser(description='Keyword spotting')
parser.add_argument('--device', default='cuda:0', type=str,
                    help='Device', choices=['cuda:0', 'cuda:1', 'cpu'])
parser.add_argument('--seed', default=1234, type=int, help='Seed')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size')


# Parse arguments
args = parser.parse_args()
device = args.device
pin_memory = True if (device == 'cuda:0') or (device == 'cuda:1') else False
seed = args.seed
num_workers = args.num_workers
batch_size = args.batch_size
device = torch.device(device if torch.cuda.is_available() else "cpu")
transform = None  # transform.to(device)
criterion = torch.nn.CrossEntropyLoss()


# Create a directory to save the model
model_save_path = ExperimentManagerLoadFunction(results_path, run=run)
model_config = ExperimentManagerReadExistingEntry(model_save_path)

# Log to file
log_file = os.path.join(full_export_path, 'Export.log')
file_stream = Echo_STDIO_to_File(log_file)
sys.stdout = file_stream
print("Logging Started")
print("model config:")
print(model_config)

# Dynamic import of the model, from backup folder in the model_save_path
try:
    spec = importlib.util.spec_from_file_location("src.model.classifier", os.path.join(
        model_save_path, 'backup', 'src', 'model', 'classifier.py'))
    print(spec)
    ssm_model = importlib.util.module_from_spec(spec)
    print(ssm_model)
    sys.modules["module.name"] = ssm_model
    spec.loader.exec_module(ssm_model)

    SC_Model_classifier = ssm_model.SC_Model_classifier
except Exception as e:
    print("Error importing model from backup folder")
    print(e)
    print("Switing to default model")
    from src.model.classifier import SC_Model_classifier


# set seed for pytorch and numpy
torch.manual_seed(seed)
np.random.seed(seed)

# Load data
print("Start loading data")
dataset_path = "./data/SpeechCommands/"

test_set = SubsetSC(dataset_path, "testing")
print("End loading data")

print("Generate Datalaoder")
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False,
                                          drop_last=False, num_workers=num_workers, pin_memory=pin_memory)

for inputs, labels in test_loader:
    print(inputs.shape)
    break

# test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
print("Generate Datalaoder end")

# Create model
model = SC_Model_classifier(input_size=model_config['input_size'],
                            classes=model_config['classes'],
                            hidden_sizes=model_config['hidden_sizes'],
                            output_sizes=model_config['output_sizes'],
                            ZeroOrderHoldRegularization=model_config['zeroOrderHoldRegularization'],
                            input_bias=model_config['input_bias'],
                            bias_init=model_config['bias_init'],
                            output_bias=model_config['output_bias'],
                            norm=model_config['norm'],
                            complex_output=model_config['complex_output'],
                            norm_type=model_config['norm_type'],
                            B_C_init=model_config['B_C_init'],
                            stability=model_config['stability'],
                            trainable_SkipLayer=model_config['trainable_skip_connections'],
                            act=model_config['act'],
                            )
# Load the model
if os.path.exists(os.path.join(model_save_path, 'best_valid_loss_model.pt')):
    model.load_state_dict(torch.load(os.path.join(
        model_save_path, f'best_valid_acc_model.pt')), strict=False)
else:
    warnings.warn('No model found, using random initialization')


def infere_model(inputs, weights_dict):
    step_scale = 1
    batch_size = inputs.shape[0]
    assert inputs.shape == (batch_size, 16000, 1)
    length = len(weights_dict)
    for index in range(length-1):
        if 'A' in weights_dict[index].keys():
            step_scale_new = weights_dict[index]['step_scale']
            if step_scale_new != step_scale:
                # print(f"Step scale changed from {step_scale} to {step_scale_new}")
                inputs = inputs[:, ::step_scale_new//step_scale, :]
                step_scale = step_scale_new

            A = weights_dict[index]['A']
            B = weights_dict[index]['B']
            B_bias = weights_dict[index]['B_bias']
            C = weights_dict[index]['C']
            C_bias = weights_dict[index]['C_bias']

            BU = inputs@B.T + B_bias.reshape(1, 1, -1)
            state = np.zeros_like(BU)
            state[:, 0, :] = BU[:, 0, :]
            for i in range(1, inputs.shape[1]):
                state[:, i, :] = state[:, i-1, :]*A + BU[:, i, :]
            output = state@C.T + C_bias

            output = np.real(output)

            if 'SkipLayer' in weights_dict[index].keys():
                SkipLayer = weights_dict[index]['SkipLayer']
                output = np.where(output > 0, output, 0.01*output) + inputs@SkipLayer.T
            else:
                output = np.where(output > 0, output, 0.01*output) + inputs

            inputs = np.real(output)
    W = weights_dict[length-1]['W']
    b = weights_dict[length-1]['b']
    output = inputs.mean(axis=1)@W.T + b.reshape(1, -1)
    return output

# Model Wrapper
class ModelWrapper(torch.nn.Module):
    def __init__(self, weights_dict):
        super(ModelWrapper, self).__init__()
        self.weights_dict = weights_dict

    def forward(self, x):
        inputs = x.cpu().detach().numpy()
        output = infere_model(inputs, self.weights_dict)
        return torch.from_numpy(output).to(x.device)


# calculate number of parameters and MACs
default_macs = model.get_number_of_MACs()
number_of_parameters = model.get_number_of_parameters()

df_test = pd.DataFrame(columns=['step_scale', 'test_loss_model', 'test_acc_model',
                       'test_loss_numpy', 'test_acc_numpy', 'flops percent', 'flops'])

# Export and test the model
with tqdm(enumerate(step_scale_list), total=len(step_scale_list), ncols=150) as pbar:
    for i, step_scale in pbar:
        step = {}
        step['step_scale'] = [step_scale]
        model.set_step_scale(step_scale, [1, *step_scale[:-1]])
        model.to(device)
        macs = model.get_number_of_MACs()
        macs_percent = model.get_number_of_MACs()/default_macs*100
        step['flops'] = macs
        step['flops percent'] = macs_percent

        # Test the model
        test_loss_model, test_acc_model = evaluate(model, criterion, test_loader)
        pbar.write(
            f'Torch Step Scale: {step_scale} macs:{macs_percent:5.2f}%  Test loss: {test_loss_model}, Test accuracy: {test_acc_model}')
        step['test_loss_model'] = test_loss_model
        step['test_acc_model'] = test_acc_model

        # export the model
        weights_dict, _ = export_layer_parameters(model, step_scale)
        np.save(os.path.join(full_export_path, f'model_dict_{str(step_scale)}.npy'), weights_dict)

        # test numpy model
        model_numpy = ModelWrapper(weights_dict)
        test_loss_numpy, test_acc_numpy = evaluate(model_numpy, criterion, test_loader)
        pbar.write(f'Numpy Step Scale: {step_scale} macs:{macs_percent:5.2f}%  Test loss: {test_loss_numpy}, Test accuracy: {test_acc_numpy}')
        step['test_loss_numpy'] = test_loss_numpy
        step['test_acc_numpy'] = test_acc_numpy

        # print(step)
       
        # save test results to csv
        df_test = pd.concat([df_test, pd.DataFrame(step,index=[0])], ignore_index=True)
        df_test.reset_index(drop=True, inplace=True)
        df_test.to_csv(os.path.join(full_export_path, f'export_accuracy_results.csv'))
