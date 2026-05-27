import os
import argparse
import torch
import numpy as np
import pandas as pd
import sys

from time import time as get_time

from torch.utils.data import Dataset, DataLoader
# from ssm.model import SC_Model_classifier

from src.utils.train_test import train_one_epoch, evaluate
from src.utils.GoogleSpeechCommands import SubsetSC
from src.utils.LogFile import Echo_STDIO_to_File
from src.utils.experimentManager import ExperimentManagerQueuePop, ExperimentManagerReadExistingEntry
from src.utils.augments import augments_weak, augments_strong

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


import importlib.util
import json


results_path = './results_vsc/'

train_history_csv = 'train_history.csv'
train_sub_epoch_history_csv = 'train_sub_epoch_history.csv'


if not os.path.exists(results_path):
    os.makedirs(results_path, exist_ok=True)


parser = argparse.ArgumentParser(description='Keyword spotting')
# Device
parser.add_argument('--device', default='cuda:0', type=str, help='Device', choices=['cuda:0', 'cuda:1', 'cpu'])
# Dataloader
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers to use for dataloader')
args = parser.parse_args()
device = args.device
pin_memory = True if (device == 'cuda:0') or (device == 'cuda:1') else False

num_workers = args.num_workers

model_save_path = ExperimentManagerQueuePop(path=results_path)
model_config = ExperimentManagerReadExistingEntry(model_save_path)

print(model_config)

seed = model_config['seed']
batch_size = model_config['batch_size']
augments = model_config['augments']
lr = model_config['lr']
epochs = model_config['epochs']
epochs = 1

print("Model config after json load")
print(model_config)



# Dynamic import of the model, from backup folder in the model_save_path
spec = importlib.util.spec_from_file_location("src.model.classifier", os.path.join(model_save_path, 'backup', 'src', 'model', 'classifier.py'))
print(spec)
ssm_model = importlib.util.module_from_spec(spec)
print(ssm_model)
sys.modules["module.name"] = ssm_model
spec.loader.exec_module(ssm_model)

SC_Model_classifier = ssm_model.SC_Model_classifier

echo_stdio = Echo_STDIO_to_File(os.path.join(model_save_path, 'output.txt'))
sys.stdout = echo_stdio

echo_sterr = Echo_STDIO_to_File(os.path.join(model_save_path, 'error.txt'))
sys.stderr = echo_sterr

print("Echoing to file start")


device = torch.device(device if torch.cuda.is_available() else "cpu")


# set seed for pytorch and numpy
print(f"Set seed to {seed}")
torch.manual_seed(seed)
np.random.seed(seed)

# Load data
print("Start loading data")
dataset_path = "./data/SpeechCommands/"

train_set = SubsetSC(dataset_path, "training")
test_set = SubsetSC(dataset_path, "testing")
valid_set = SubsetSC(dataset_path, "validation")
print("End loading data")

print("Generate Datalaoder")
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,                   num_workers=num_workers, pin_memory=pin_memory)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin_memory)

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
                            dropout=model_config['dropout']
                            )
model.to(device)

params_ssm_lr = [param for name, param in model.named_parameters() if 'B' in name or 'C' in name or 'Lambda' in name or 'log_step' in name] 
params_other_lr = [param for name, param in model.named_parameters() if 'B' not in name and 'C' not in name and 'Lambda' not in name and 'log_step' not in name]


# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW([
    {'params': params_ssm_lr, 'lr': lr, 'weight_decay': 0},
    {'params': params_other_lr, 'lr': 4*lr, 'weight_decay': model_config['weight_decay']},
    ], lr=lr, weight_decay=model_config['weight_decay'])

scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=epochs*len(train_loader),
                                          cycle_mult=1.0,
                                          max_lr=[lr,4*lr],
                                          min_lr=[0,0],
                                          warmup_steps=100,
                                          gamma=1,
)


if augments == 'none':
    augments = None
elif augments == 'weak':
    augments = augments_weak
elif augments == 'strong':
    augments = augments_strong
else:
    raise ValueError(f"Augments: {augments} not supported")



# Train the model
best_val_loss = 1e3  # Init
best_val_loss_epoch = 0
best_val_acc = 0
best_val_acc_epoch = 0


# subsets of the data
df_metric = pd.DataFrame(columns=['train_loss', 'train_acc', 'valid_loss',
                         'valid_acc', 'epoch', 'learning_rate', 'training_time'])
df_sub_epoch = pd.DataFrame()

print("Start training")
start_time = get_time()
for epoch in range(epochs):
    train_loss, train_acc, sub_epoch_info = train_one_epoch(
        model, criterion, optimizer, train_loader, regularize=True, scheduler=scheduler, sub_epoch_documentation=10,
        augments_use=augments)
    valid_loss, val_acc = evaluate(model, criterion, valid_loader)
    # scheduler.step()
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        best_val_loss_epoch = epoch
        torch.save(model.state_dict(), os.path.join(model_save_path, 'best_valid_loss_model.pt'))
    if val_acc >= best_val_acc:
        best_val_acc_epoch = epoch
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(model_save_path, 'best_valid_acc_model.pt'))
    if epoch % 10 == 0:
        torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch_{epoch}_model.pt'))

    # lr_tmp = optimizer.param_groups[0]['lr']
    df_new_row = {'train_loss': train_loss,
                  'train_acc': train_acc,
                  'valid_loss': valid_loss,
                  'valid_acc': val_acc,
                  'epoch': epoch + 1,
                  #   'learning_rate': lr_tmp,
                  'training_time': get_time() - start_time,
                  'learning_rate': scheduler.max_lr
                  }
    df_metric.loc[epoch] = df_new_row
    df_metric.to_csv(os.path.join(model_save_path, train_history_csv))
    print(f"Epoch {epoch+1}, train_loss={train_loss:6.4f}, train_acc={train_acc:6.4f} val_loss={valid_loss:6.4f},  val_acc={val_acc:6.4f}")

    # save sub epoch info
    new_row = pd.DataFrame(sub_epoch_info)
    new_row['epoch'] = new_row['epoch'] + epoch
    df_sub_epoch = pd.concat([df_sub_epoch, new_row], ignore_index=True).reset_index(drop=True)
    df_sub_epoch.to_csv(os.path.join(model_save_path, train_sub_epoch_history_csv))

torch.save(model.state_dict(), os.path.join(model_save_path, f'last_model.pt'))

print('Evaluate on test set')
model.load_state_dict(torch.load(os.path.join(model_save_path, f'best_valid_loss_model.pt'), weights_only=True))
test_loss, test_acc = evaluate(model, criterion, test_loader)

# save test results to csv
df_test = pd.DataFrame(columns=['test_loss', 'test_acc', 'best_val_loss',
                       'best_val_loss_epoch', 'best_val_acc', 'best_val_acc_epoch'])
df_test.loc[0] = [test_loss, test_acc, best_val_loss, best_val_loss_epoch, best_val_acc, best_val_acc_epoch]
df_test.to_csv(os.path.join(model_save_path, 'test_results.csv'))
