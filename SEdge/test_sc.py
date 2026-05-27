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


from src.utils.train_test import  evaluate
from src.utils.GoogleSpeechCommands import SubsetSC
from src.utils.experimentManager import ExperimentManagerLoadFunction, ExperimentManagerReadExistingEntry
from src.utils.LogFile import Echo_STDIO_to_File

import matplotlib.pyplot as plt


results_path = './results_journal/'
run = 0

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
log_file = os.path.join(model_save_path, 'Test.log')
file_stream = Echo_STDIO_to_File(log_file)
sys.stdout = file_stream
print("Logging Started")
print("model config:")
print(model_config)

# Dynamic import of the model, from backup folder in the model_save_path
try:
    spec = importlib.util.spec_from_file_location("src.model.classifier", os.path.join(model_save_path, 'backup', 'src', 'model', 'classifier.py'))
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
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)

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

model.eval()
model.to(device)
model.eval()


# calculate number of parameters and MACs
default_macs = model.get_number_of_MACs()
macs_per_layer = []
for layer in model.seq:
    macs_per_layer.append(layer.get_number_of_MACs())
    # print(layer.get_number_of_MACs())
macs_per_layer.append(model.decoder.in_features*model.decoder.out_features)
macs_per_layer = np.array(macs_per_layer)
print('MACs per Layer', macs_per_layer)

plt.bar(x=np.arange(len(model.seq)+1),
        height=macs_per_layer, label='macs per layer')
plt.savefig(os.path.join(model_save_path, 'macs_per_layer.png'))
plt.title('MACs per layer')
plt.xlabel('layer')
plt.ylabel('MACs')
plt.close()

# calculate number of parameters
number_of_parameters = model.get_number_of_parameters()
params_per_layer = []
for layer in model.seq:
    params_per_layer.append(layer.get_number_of_parameters())
    # print(layer.get_number_of_parameters())
params_per_layer.append(model.decoder.in_features *
                        model.decoder.out_features + model.decoder.out_features)
params_per_layer = np.array(params_per_layer)
print('Params per Layer', params_per_layer)

plt.bar(x=np.arange(len(model.seq)+1),
        height=params_per_layer, label='params per layer')
plt.savefig(os.path.join(model_save_path, 'params_per_layer.png'))
plt.title('Params per layer')
plt.xlabel('layer')
plt.ylabel('params')
plt.close()


# sweeping function for step scale
def sweep(current_scale, remaining_depth, upper_bound=16):
    if remaining_depth == 0:
        return [[]]
    results = []
    for i in range(1, upper_bound+1):
        if current_scale*i>upper_bound+1:
            break
        results += [[current_scale*i,*ans] for ans in  sweep(current_scale*i, remaining_depth - 1, upper_bound=upper_bound)]
    return results


length = len(model_config['hidden_sizes'])

step_scale_list = sweep(1, length, upper_bound=32)



# Get the test results or create a new dataframe
if os.path.exists(os.path.join(model_save_path, 'test_results_flops_step_scale.csv')):
    df_test = pd.read_csv(os.path.join(
        model_save_path, 'test_results_flops_step_scale.csv'), index_col=0)
else:
    df_test = pd.DataFrame(columns=['test_loss', 'test_acc', 'step_scale', 'flops percent', 'flops'])


# Load the model
if os.path.exists(os.path.join(model_save_path, 'best_valid_loss_model.pt')):
    model.load_state_dict(torch.load(os.path.join(
    model_save_path, f'best_valid_acc_model.pt')), strict=False)
else:
    warnings.warn('No model found, using random initialization')

print('Evaluate on test set')

with tqdm(enumerate(step_scale_list), total=len(step_scale_list), ncols=150) as pbar:
    for i, step_scale in pbar:
        # skip existing results
        if df_test['step_scale'].apply(lambda x: x == f"{step_scale}").any():
            pbar.write(f'Step Scale: {step_scale} already evaluated, updating params and macs')
           
            model.set_step_scale(step_scale, [1, *step_scale[:-1]])
            macs = model.get_number_of_MACs()
            macs_percent = model.get_number_of_MACs()/default_macs*100

            pos = df_test['step_scale'].apply(lambda x: x == f"{step_scale}")
            df_test.loc[pos, 'macs'] = macs
            df_test.loc[pos, 'macs percent'] = macs_percent
            df_test.loc[pos, 'params'] = number_of_parameters
            df_test.to_csv(os.path.join(model_save_path,f'test_results_flops_step_scale.csv'))

            continue
        
        model.set_step_scale(step_scale, [1, *step_scale[:-1]])
        macs = model.get_number_of_MACs()
        macs_percent = model.get_number_of_MACs()/default_macs*100
        test_loss, test_acc = evaluate(model, criterion, test_loader, transform=transform)
        pbar.write(f'Step Scale: {step_scale} macs:{macs_percent:5.2f}%  Test loss: {test_loss}, Test accuracy: {test_acc}')


        # save test results to csv
        tmp = pd.DataFrame({'test_loss': test_loss, 'test_acc': test_acc, 'step_scale': [step_scale], 'macs': macs, 'macs percent': macs_percent, 'params': number_of_parameters}, index=[0])
        df_test = pd.concat([df_test, tmp], ignore_index=True)
        df_test.reset_index(drop=True, inplace=True)
        df_test.to_csv(os.path.join(model_save_path,f'test_results_flops_step_scale.csv'))
