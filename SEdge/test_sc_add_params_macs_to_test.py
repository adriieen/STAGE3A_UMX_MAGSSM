import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import warnings


from torch.utils.data import Dataset, DataLoader
from src.model.classifier import SC_Model_classifier

from src.utils.train_test import train_one_epoch, evaluate
from src.utils.GoogleSpeechCommands import SubsetSC
from tqdm import tqdm

import matplotlib.pyplot as plt

from src.utils.experimentManager import ExperimentManagerLoadFunction, ExperimentManagerReadExistingEntry

import importlib.util
import json

results_path = './results_journal/'
# results_path = './results_vsc/'


parser = argparse.ArgumentParser(description='Keyword spotting')
# # Device
parser.add_argument('--device', default='cuda:0', type=str,
                    help='Device', choices=['cuda:0', 'cuda:1', 'cpu'])
# # Seed
parser.add_argument('--seed', default=1234, type=int, help='Seed')
# # Dataloader
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

def sweep(current_scale, remaining_depth, upper_bound=16):
    if remaining_depth == 0:
        return [[]]
    results = []
    for i in range(1, upper_bound+1):
        if current_scale*i>upper_bound+1:
            break
        results += [[current_scale*i,*ans] for ans in  sweep(current_scale*i, remaining_depth - 1, upper_bound=upper_bound)]
    return results

def clean_config_dict(config_dict):
    model_config_out = {}
    for key, value in config_dict.items():
        try:
            model_config_out[key] = json.loads(value)
            if type(model_config_out[key]) is list:
                if type(model_config_out[key][0]) is list:
                    model_config_out[key] = model_config_out[key][0]
                else:
                    model_config_out[key] = model_config_out[key]
        except:
            if value == 'True':
                model_config_out[key] = True
            elif value == 'False':
                model_config_out[key] = False
            elif value == 'None':
                model_config_out[key] = None
            elif value == '[]':
                model_config_out[key] = []
            else:
                model_config_out[key] = value

    return model_config_out


print("Start loading data")
dataset_path = "./data/SpeechCommands/"

test_set = SubsetSC(dataset_path, "testing")
print("End loading data")

print("Generate Datalaoder")
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)

for inputs, labels in test_loader:
    print(inputs.shape)
    break



with tqdm(os.listdir(results_path), ncols=150) as pbar:
    for path in pbar:
        if path == 'QUEUE':
            continue
        if not os.path.isdir(os.path.join(results_path, path)):
            continue
        
        if not os.path.exists(os.path.join(results_path, path, 'test_results.csv')):
            continue
        
        # model_save_path = ExperimentManagerLoadFunction(results_path, run=2)
        model_save_path = os.path.join(results_path, path)
        model_config = ExperimentManagerReadExistingEntry(model_save_path)
        # print(model_config)

        pbar.set_description(f'Processing {model_save_path}')

        model_config = clean_config_dict(model_config)

        hidden_sizes = model_config['hidden_sizes']


        # Dynamic import of the model, from backup folder in the model_save_path
        # spec = importlib.util.spec_from_file_location("ssm.model", os.path.join(model_save_path, 'backup', 'ssm', 'model.py'))
        # print(spec)
        # ssm_model = importlib.util.module_from_spec(spec)
        # print(ssm_model)
        # sys.modules["module.name"] = ssm_model
        # spec.loader.exec_module(ssm_model)

        # SC_Model_classifier = ssm_model.SC_Model_classifier

        device = torch.device(device if torch.cuda.is_available() else "cpu")

        # set seed for pytorch and numpy
        torch.manual_seed(seed)
        np.random.seed(seed)

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
        #

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

        number_of_parameters = model.get_number_of_parameters()
        params_per_layer = []
        for layer in model.seq:
            params_per_layer.append(layer.get_number_of_parameters())
            # print(layer.get_number_of_parameters())
        params_per_layer.append(model.decoder.in_features * model.decoder.out_features + model.decoder.out_features)
        params_per_layer = np.array(params_per_layer)
        print('Params per Layer', params_per_layer)

        plt.bar(x=np.arange(len(model.seq)+1),
                height=params_per_layer, label='params per layer')
        plt.savefig(os.path.join(model_save_path, 'params_per_layer.png'))
        plt.title('Params per layer')
        plt.xlabel('layer')
        plt.ylabel('params')
        plt.close()

        df_test2 = pd.read_csv(os.path.join(model_save_path, 'test_results.csv'), index_col=0)
        df_test2['params'] = number_of_parameters
        df_test2['macs'] = default_macs
        df_test2.to_csv(os.path.join(model_save_path, 'test_results.csv'))

        length = len(model_config['hidden_sizes'])

        # step_scale_list = sweep(1, length, upper_bound=32)

        step_scale_list = [
            [1]*length,
            [2]*length,
            [4]*length,
            ]


        print('Number of step scales:', len(step_scale_list))
        print('step scales:', step_scale_list)
        # sys.exit(0)

        if os.path.exists(os.path.join(model_save_path, 'test_results_flops_step_scale.csv')):
            df_test = pd.read_csv(os.path.join(
                model_save_path, 'test_results_flops_step_scale.csv'), index_col=0)
        else:
            df_test = pd.DataFrame(
                columns=['test_loss', 'test_acc', 'step_scale', 'flops percent', 'flops'])
        # step_scale_list.append([2**n for n in range(10)])


        if os.path.exists(os.path.join(model_save_path, 'best_valid_loss_model.pt')):
            model.load_state_dict(torch.load(os.path.join(
                # model_save_path, f'best_valid_loss_model.pt')), strict=False)
            model_save_path, f'best_valid_acc_model.pt'), map_location='cpu'), strict=False)
        else:
            warnings.warn('No model found, using random initialization')

        model = model.to(device)
        model.eval()

        with tqdm(enumerate(step_scale_list), total=len(step_scale_list), ncols=150, disable=True) as pbar2:
            for i, step_scale in pbar2:
                # skip existing results
                if 'step_scale' in df_test and len(df_test) > 0:
                    # print(df_test['step_scale'])
                    if df_test['step_scale'].apply(lambda x: x == f"{step_scale}").any():
                        pbar2.write(f'Step Scale: {step_scale} already evaluated, updating params and macs')
                    
                        model.set_step_scale(step_scale, [1, *step_scale[:-1]])
                        macs = model.get_number_of_MACs()
                        macs_percent = model.get_number_of_MACs()/default_macs*100

                        pos = df_test['step_scale'].apply(lambda x: x == f"{step_scale}")
                        df_test.loc[pos, 'macs'] = macs
                        df_test.loc[pos, 'macs percent'] = macs_percent
                        df_test.loc[pos, 'params'] = number_of_parameters
                        df_test.to_csv(os.path.join(model_save_path,f'test_results_flops_step_scale.csv'))

                        test_acc = df_test.loc[pos, 'test_acc'].values[0]
                        test_loss = df_test.loc[pos, 'test_loss'].values[0]
                        
                        if step_scale == [1]*length:
                            df_test2 = pd.read_csv(os.path.join(model_save_path, 'test_results.csv'), index_col=0)
                            df_test2['test_acc_16k'] = test_acc
                            df_test2['test_loss_16k'] = test_loss
                            df_test2.to_csv(os.path.join(model_save_path, 'test_results.csv'))
                        if step_scale == [2]*length:
                            df_test2 = pd.read_csv(os.path.join(model_save_path, 'test_results.csv'), index_col=0)
                            df_test2['test_acc_8k'] = test_acc
                            df_test2['test_loss_8k'] = test_loss
                            df_test2.to_csv(os.path.join(model_save_path, 'test_results.csv'))
                        if step_scale == [4]*length:
                            df_test2 = pd.read_csv(os.path.join(model_save_path, 'test_results.csv'), index_col=0)
                            df_test2['test_acc_4k'] = test_acc
                            df_test2['test_loss_4k'] = test_loss
                            df_test2.to_csv(os.path.join(model_save_path, 'test_results.csv'))

                        continue
                
                # sampling_rate = 16000
                # new_sample_rate = 16000
                # transform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=new_sample_rate)
                transform = None  # transform.to(device)

                criterion = torch.nn.CrossEntropyLoss()

                # pbar2.write('Evaluate on test set')

                # step_scale = [2 for n in range(10)]
                # flops = np.mean([1/x for x in step_scale])*100
                model.set_step_scale(step_scale, [1, *step_scale[:-1]])
                # pbar2.write(step_scale)
                macs = model.get_number_of_MACs()
                macs_percent = model.get_number_of_MACs()/default_macs*100
                # pbar2.write(f"About {flops_percent:5.2f}%flops")
                # model.set_step_scale(step_scale, step_scale)
                # model.set_step_scale([*step_scale[1:],step_scale[-1]], step_scale)
                # model.set_step_scale([2,2,2,2,2,2])
                # model.set_step_scale()
                # test_loss, test_acc = 0,0
                test_loss, test_acc = evaluate(
                    model, criterion, test_loader, transform=transform)
                pbar2.write(f'Step Scale: {step_scale} macs:{macs_percent:5.2f}%  Test loss: {test_loss}, Test accuracy: {test_acc}')
                # save test results to csv
                tmp = pd.DataFrame({'test_loss': test_loss, 'test_acc': test_acc, 'step_scale': [step_scale], 'macs': macs, 'macs percent': macs_percent, 'params': number_of_parameters}, index=[0])
                df_test = pd.concat([df_test, tmp], ignore_index=True)
                df_test.reset_index(drop=True, inplace=True)
                df_test.to_csv(os.path.join(model_save_path,f'test_results_flops_step_scale.csv'))

                if step_scale == [1]*length:
                    df_test2 = pd.read_csv(os.path.join(model_save_path, 'test_results.csv'), index_col=0)
                    df_test2['test_acc_16k'] = test_acc
                    df_test2['test_loss_16k'] = test_loss
                    df_test2.to_csv(os.path.join(model_save_path, 'test_results.csv'))
                if step_scale == [2]*length:
                    df_test2 = pd.read_csv(os.path.join(model_save_path, 'test_results.csv'), index_col=0)
                    df_test2['test_acc_8k'] = test_acc
                    df_test2['test_loss_8k'] = test_loss
                    df_test2.to_csv(os.path.join(model_save_path, 'test_results.csv'))
                if step_scale == [4]*length:
                    df_test2 = pd.read_csv(os.path.join(model_save_path, 'test_results.csv'), index_col=0)
                    df_test2['test_acc_4k'] = test_acc
                    df_test2['test_loss_4k'] = test_loss
                    df_test2.to_csv(os.path.join(model_save_path, 'test_results.csv'))
