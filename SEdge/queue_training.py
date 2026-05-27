import os
import yaml
import numpy as np

from src.utils.experimentManager import ExperimentManagerQueue

results_path = './results_vsc/'

train_history_csv = 'train_history.csv'
train_sub_epoch_history_csv = 'train_sub_epoch_history.csv'

hyperparameters_yaml_path = './hyperparams/Speech_Commands_Default.yaml'

save_files = ['src','queue_training.py']

if not os.path.exists(results_path):
    os.makedirs(results_path, exist_ok=True)

dropouts = [0.1, 0.0]
B_C_inits = ['orthogonal', 'S5']
norms = [False,True]
lrs = [0.001,0.002,0.003]

zeroOrderHoldRegularizations = [(lambda q: []), (lambda q: [round(x/len(q),2) for x in range(1,len(q)+1)])]  # first no reg, then reg

models = [
    './model_descriptions/S_Edge_full.yaml',                    # first no reg, then reg 
    './model_descriptions/S_Edge_naive_extra_large.yaml',       # all naive then reg all 
    './model_descriptions/S_Edge_naive_large.yaml',
    './model_descriptions/S_Edge_naive_medium.yaml',
    './model_descriptions/S_Edge_naive_small.yaml',
    './model_descriptions/S_Edge_large.yaml',                   # first no reg then reg  
    './model_descriptions/S_Edge_medium.yaml',                  # first no reg then reg
    './model_descriptions/S_Edge_small.yaml',                   # first no reg then reg
    './model_descriptions/S_Edge_tiny.yaml',                    # first no reg then reg
    ]

with open(hyperparameters_yaml_path, 'r') as f:
    hyperparameters_yaml = yaml.safe_load(f)







config = {
    'lr': -1,
    'epochs': hyperparameters_yaml['epochs'],
    'device': 'cuda:0',
    'seed': hyperparameters_yaml['seed'],
    'num_workers': 0,
    'batch_size': hyperparameters_yaml['batch_size'],
    'input_size': hyperparameters_yaml['input_size'],
    'classes': hyperparameters_yaml['classes'],
    'hidden_sizes': [],
    'output_sizes': [],
    'zeroOrderHoldRegularization': [],
    'trainable_skip_connections': None,
    'input_bias': None,
    'bias_init': 'None',
    'output_bias': None,
    'complex_output': None,
    'norm': None,
    'norm_type': 'bn',
    'B_C_init': 'None',
    'stability': 'None',
    'augments': 'None',
    'act': 'None',
    'weight_decay': hyperparameters_yaml['weight_decay'],
    'dropout': None,
}


def update_model_dict(config, model_yaml_path):
    with open(model_yaml_path, 'r') as f:
        model_yaml = yaml.safe_load(f)
    config['hidden_sizes'] = [model_yaml['hidden_sizes']]
    config['output_sizes'] = [model_yaml['output_sizes']]
    config['zeroOrderHoldRegularization'] = [model_yaml['zeroOrderHoldRegularization']]
    config['trainable_skip_connections'] = model_yaml['trainable_skip_connections']
    config['input_bias'] = model_yaml['input_bias']
    config['bias_init'] = model_yaml['bias_init']
    config['output_bias'] = model_yaml['output_bias']
    config['complex_output'] = model_yaml['complex_output']
    config['norm'] = model_yaml['norm']
    config['norm_type'] = model_yaml['norm_type']
    config['B_C_init'] = model_yaml['B_C_init']
    config['stability'] = model_yaml['stability']
    config['augments'] = hyperparameters_yaml['augments']
    config['act'] = model_yaml['act']
    return config




for norm in norms:
    for B_C_init in B_C_inits:
        for droput in dropouts:
            for lr in lrs:
                config = update_model_dict(config,models[0])  # first model is the default one
                config['lr'] = lr
                config['B_C_init'] = B_C_init
                config['norm'] = norm
                config['dropout'] = droput
                config['zeroOrderHoldRegularization'] = [zeroOrderHoldRegularizations[0](config['output_sizes'][0])]  # first no reg
                ExperimentManagerQueue(path=results_path, config_dict=config, saveFiles=save_files)
                
                config['zeroOrderHoldRegularization'] = [zeroOrderHoldRegularizations[1](config['output_sizes'][0])]  # first no reg
                ExperimentManagerQueue(path=results_path, config_dict=config, saveFiles=save_files)

                for ZOH in zeroOrderHoldRegularizations:
                    for model_yaml_path in models[1:5]:
                        config = update_model_dict(config, model_yaml_path)
                        config['lr'] = lr
                        config['B_C_init'] = B_C_init
                        config['norm'] = norm
                        config['dropout'] = droput
                        config['zeroOrderHoldRegularization'] = [ZOH(config['output_sizes'][0])]
                        ExperimentManagerQueue(path=results_path, config_dict=config, saveFiles=save_files) 
                for model_yaml_path in models[5:]:
                    for ZOH in zeroOrderHoldRegularizations:
                        config = update_model_dict(config, model_yaml_path)
                        config['lr'] = lr
                        config['B_C_init'] = B_C_init
                        config['norm'] = norm
                        config['dropout'] = droput
                        config['zeroOrderHoldRegularization'] = [ZOH(config['output_sizes'][0])]
                        ExperimentManagerQueue(path=results_path, config_dict=config, saveFiles=save_files)