
import os
import errno
import warnings
from pwd import getpwuid
import datetime
import json

import fcntl


import numpy as np
import pandas as pd


UN_PROCESSED_PREFIX = 'UN_PROCESSED'
EMPTY_PREFIX = 'EMPTY'
TIME_FORMAT = '%Y-%m-%d-%H:%M:%S'
FOLDER_NAME_DISPLAY_KEYS = ['lr', 'epochs']
IGNORE_PARAMETERS_FOR_DISTANCE = ['creator', 'timestamp', 'path', 'Run', 'device', 'num_workers']
BACKUP_NAME = 'backup'
HOMOGENIZE_SORT_KEYS = ['Run', 'timestamp']

QUEUE_PREFIX = 'QUEUE'
MUTEX_FILE = 'mutex.txt'


# Legacy code
def old_config_id(*,lr, epochs, batch_size, input_size, hidden_size, output_size, input_bias, bias_init, output_bias, n_layer, complex_output, norm, norm_type, B_C_init, stability): 
    return 'lr_{}_epochs_{}_batch_size_{}_input_size_{}_hidden_size_{}_output_size_{}_input_bias_{}_bias_init_{}_output_bias_{}_n_layer_{}_complex_output_{}_norm_{}_norm_type_{}_B_C_init_{}_stability_{}'.format(
            lr, epochs, batch_size, input_size, hidden_size, output_size, input_bias, bias_init, output_bias, n_layer, complex_output, norm, norm_type, B_C_init, stability
        )
 
# New Code
# save syntax
# Parameter Names may contain : [A-Za-z0-9\-\.\:]  (no spaces, no special characters except -)
# Parameter Values may contain : [A-Za-z0-9\-\.\:]  (no spaces, no special characters except - and .)
# Parameters are separated by _
# Values are separated by #
# There is an index at the beginning of the Name, by which the runs are sorted
# The index is followed by the parameter pairs
# The parameter pairs are sorted by the parameter name 

def ExperimentManagerPatternReader(name:str) -> dict:
    """
    Read the name of a file and return the parameters as a dictionary

    Parameters
    ----------
    name : str
        The name of the file

    Returns
    -------
    dict
        The parameters in the file name as a dictionary
    """
    Parameters = name.split('_')
    Parameters = {x.split('#')[0]:x.split('#')[1] for x in Parameters}
    return Parameters

def ExperimentManagerPatternWriter(Parameters:dict) -> str:
    """
    Write the parameters as a string

    Parameters
    ----------
    Parameters : dict
        The parameters as a dictionary

    Returns
    -------
    str
        The parameters as a string
    """
    run = Parameters['Run']
    del Parameters['Run']
    # creator = Parameters['creator']
    # del Parameters['creator']
    # timestamp = Parameters['timestamp']
    # del Parameters['timestamp']

    Parameters = [(x, Parameters[x]) for x in Parameters if x in FOLDER_NAME_DISPLAY_KEYS]
    Parameters = sorted(Parameters, key=lambda x: x[0])
    Parameters = [f'{x[0]}#{x[1]}' for x in Parameters]
    Parameters = '_'.join([f'Run#{run}', *Parameters])#, f'creator#{creator}',f'timestamp#{timestamp}'])
    return Parameters


def ExperimentManagerDictionarySanitizer(Parameters:dict) -> dict:
    """
    Sanatize the parameters dictionary, removing any special characters

    Parameters
    ----------
    Parameters : dict
        The parameters as a dictionary

    Returns
    -------
    dict
        The parameters as a dictionary, with the special characters removed
    """
    key_enabler = lambda key: key is not None and str(key) != 'None' and str(key) != 'NaN' and str(key) != 'nan'
    value_enabler = lambda value: value is not None and str(value) != 'None' and str(value) != 'NaN' and str(value) != 'nan'

    name_sanitizer = lambda name: ''.join(e for e in name.replace('_','-').replace(' ','-') if e.isalnum() or e in ['-', '.' , ':'])
    value_sanitizer = lambda value: ''.join(e for e in str(value) if e.isalnum() or e in ['-', '.' , ':'])
    Parameters = {name_sanitizer(x):value_sanitizer(Parameters[x]) for x in Parameters if key_enabler(x) and value_enabler(Parameters[x])}
    return Parameters

def ExperimentManagerDictionaryDistance(Parameters1:dict, Parameters2:dict) -> float:
    """
    Calculate the distance between two dictionaries

    Parameters
    ----------
    Parameters1 : dict
        The first dictionary
    Parameters2 : dict
        The second dictionary

    Returns
    -------
    float
        The distance between the two dictionaries.
        if the key is present in only one of the dictionaries the distance is 1 for each difference (this means that the value would be None in the other dictionary)
        If the key is present in both dictionaries and different the distance is 1 for each difference
        If the key is present in both dictionaries and the same the distance is 0
    """
    Parameters1 = {x:Parameters1[x] for x in Parameters1 if x not in IGNORE_PARAMETERS_FOR_DISTANCE}
    Parameters2 = {x:Parameters2[x] for x in Parameters2 if x not in IGNORE_PARAMETERS_FOR_DISTANCE}

    Parameters1 = ExperimentManagerDictionarySanitizer(Parameters1)
    Parameters2 = ExperimentManagerDictionarySanitizer(Parameters2)
    distance = 0
    key_error = []
    for key in Parameters1:
        if key not in Parameters2:
            distance += 1
            key_error.append(key)
            # print('not found', key)
        elif Parameters1[key].replace('.', '', 1).replace('-','',1).isdigit() and Parameters2[key].replace('.', '', 1).replace('-','',1).isdigit() :
            if not np.isclose(float(Parameters1[key]), float(Parameters2[key])):
                distance += 1
                key_error.append(key)
                # print('not close', key, Parameters1[key], Parameters2[key])
        elif Parameters1[key] != Parameters2[key]:
            distance += 1
            key_error.append(key)
            # print('not numeric', key, Parameters1[key], Parameters2[key])

    for key in Parameters2:
        if key not in Parameters1:
            distance += 1
            key_error.append(key)
            # print('not found', key)

    return distance, key_error

def ExperimentManagerReadExistingEntry(path:str) -> dict:
    """
    Read the existing entry in the directory and return it as a dictionary

    Parameters
    ----------
    path : str
        The path to the directory containing the tests

    Returns
    -------
    dict
        The dictionary containing the tests Parameters as keys and the values as the values, if a parameter is not present in a run, it is filled with None
    """
    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    if 'config.csv' not in os.listdir(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), os.path.join(path, 'config.csv'))
    content = pd.read_csv(os.path.join(path, 'config.csv'), index_col=0).T.iloc[0].to_dict()
    content['path'] = path
    content |= ExperimentManagerPatternReader(path.split('/')[-1])

    model_config_tmp = content.copy()
    content = {}
    for key, value in model_config_tmp.items():
        try:
            content[key] = json.loads(value)
            if type(content[key]) is list:
                if type(content[key][0]) is list:
                    content[key] = content[key][0]
                else:
                    content[key] = content[key]
        except:
            if value == 'True':
                content[key] = True
            elif value == 'False':
                content[key] = False
            elif value == 'None':
                content[key] = None
            elif value == '[]':
                content[key] = []
            else:
                content[key] = value

    return content


def ExperimentManagerReadExistingTests(path:str) -> pd.DataFrame:
    """
    Read the existing tests in the directory and return them as a DataFrame

    Parameters
    ----------
    path : str
        The path to the directory containing the tests

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the tests Parameters as columns and the index as the run number, sorted by the run number, columns are sorted by the parameter name, if a parameter is not present in a run, it is filled with None
    """
    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    content = os.listdir(path)
    if np.any(['Run#' not in x for x in content if x != EMPTY_PREFIX and x != UN_PROCESSED_PREFIX and x != QUEUE_PREFIX and x != MUTEX_FILE]):
        warnings.warn(f'Some files in the directory "{path}" do not follow the naming convention, check if they should be converted')
    if np.any(['config.csv' not in os.listdir(os.path.join(path, x)) for x in content if 'Run#' in x]):
        warnings.warn(f'Some files in the directory "{path}" do not have a config file, check if they should be deleted')
    if np.any([UN_PROCESSED_PREFIX in x for x in content]):
        warnings.warn(f'Some files in the directory "{path}" where not processed, check if they should be converted manually')
    
    content = [{**ExperimentManagerPatternReader(x),**pd.read_csv(os.path.join(path, x, 'config.csv'), index_col=0).T.iloc[0].to_dict(), 'path':x} for x in content if 'Run#' in x]
    content = pd.DataFrame(content)
    # print(content)
    if content.shape[0] == 0:
        return content

    content = content.astype({'Run':'int'})
    content['timestamp'] = pd.to_datetime(content['timestamp'], format=TIME_FORMAT)
    return content
    
        



def ExperimentManagerSaveFunction(path,*, config_dict={}, saveFiles=[]) -> str:
    """
    Save the experiment to the directory, makes runs unique by adding the index to the name and enforces the naming convention

    Parameters
    ----------
    path : str
        The path to the directory containing the tests
    config_dict : dict, optional
        The config dictionary to use for selection, by default {}
    saveFiles : list, optional
        The list of files to copy for a backup, by default []
    """
    existing = ExperimentManagerReadExistingTests(path)
    if existing.shape[0] == 0:
        run = 0
    else:
        run = np.max(existing['Run'].values) + 1


    config_dict['creator'] = str(getpwuid( os.getuid() ).pw_name)
    config_dict['timestamp'] = datetime.datetime.now().strftime(TIME_FORMAT)
    print(config_dict['timestamp'])
    
    config_file = pd.DataFrame(config_dict, index=[0]).T
    
    config_dict = ExperimentManagerDictionarySanitizer(config_dict)
    name = ExperimentManagerPatternWriter({**config_dict, 'Run':run})
    os.makedirs(os.path.join(path, name), exist_ok=True)
    config_file.to_csv(os.path.join(path, name, 'config.csv'))
    for file in saveFiles:
        if os.path.isdir(file):
            os.makedirs(os.path.join(path, name,BACKUP_NAME, file), exist_ok=True)
            os.system(f'cp -r {file}/* {os.path.join(path, name,BACKUP_NAME, file)}/')
            # print(f'cp -r {file}/* {os.path.join(path, name,BACKUP_NAME, file)}/')
        else:
            os.system(f'cp -r {file} {os.path.join(path, name, BACKUP_NAME )}')
    return os.path.join(path, name)


def ExperimentManagerLoadFunction(path,*,run=None, config_dict={}) -> str:
    """
    Load the experiment from the directory, Gives feedback if the run is not present or if there are multiple runs with the same index

    Parameters
    ----------
    path : str
        The path to the directory containing the tests
    run : int, optional
        The run number to load, if None the config is used for selection, by default None
    config_dict : dict, optional
        The config dictionary to use for selection, by default {}
    """
    existing = ExperimentManagerReadExistingTests(path)

    if run is not None:
        runs = existing['Run'].values
        if run == -1:
            run = np.max(runs)
        if run not in runs:
            raise ValueError(f'The run {run} is not present in the directory {path}')
        
        # check for multiple runs with the same index
        slice = existing.query(f'Run == {run}')
        if slice.shape[0] > 1:
            warnings.warn(f'There are multiple runs with index {run}, PLease use the dictionary for selection')
            # print the multiple runs
            print(slice)
            raise ValueError(f'There are multiple runs with index {run}, PLease use the dictionary for selection')

        return os.path.join(path, slice.iloc[0]['path'])
    else:
        # Check for the closest run
        slice = existing.copy()
        # slice.drop(columns=['path', 'creator', 'path', 'timestamp', 'Run'], inplace=True)
        slice.drop(columns=IGNORE_PARAMETERS_FOR_DISTANCE, inplace=True, errors='ignore')

        ret = slice.apply(lambda x: ExperimentManagerDictionaryDistance(x.to_dict(), config_dict), axis=1)
        existing['distance'] = [x[0] for x in ret]
        existing['key_error'] = [x[1] for x in ret]
        existing = existing.sort_values(by='distance')

        # distance 0
        slice = existing.query('distance == 0')
        if slice.shape[0] > 1:
            warnings.warn(f'Found multiple runs with the same parameters, please specify the run number')
            print(slice)
            raise ValueError(f'Found a run with the same parameters, please specify the run number')
        if slice.shape[0] == 0:
            warnings.warn(f'No run with the same parameters found listing the 10 closest runs')
            print(existing.head(10))
            raise ValueError(f'No run with the same parameters found listed the 10 closest runs')

        return os.path.join(path, slice.iloc[0]['path'])

def ExperimentManagerLock(path:str):
    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    if MUTEX_FILE not in os.listdir(path):
        open(os.path.join(path, MUTEX_FILE), 'w').close()
    mutex = open(os.path.join(path, MUTEX_FILE), 'r+')
    fcntl.flock(mutex, fcntl.LOCK_EX)

def ExperimentManagerUnlock(path:str):
    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    if MUTEX_FILE not in os.listdir(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), os.path.join(path, MUTEX_FILE))
    mutex = open(os.path.join(path, MUTEX_FILE), 'r+')
    fcntl.flock(mutex, fcntl.LOCK_UN)


def ExperimentManagerQueue(path:str,*, config_dict={}, saveFiles=[]) -> str:
    """
    Queue the experiment to the directory, makes runs unique by adding the index to the name and enforces the naming convention

    Parameters
    ----------
    path : str
        The path to the directory containing the tests
    config_dict : dict, optional
        The config dictionary to use for selection, by default {}
    saveFiles : list, optional
        The list of files to copy for a backup, by default []
    """
    existing = ExperimentManagerReadExistingTests(path)
    if os.path.exists(os.path.join(path, QUEUE_PREFIX)):
        existing = pd.concat([existing, ExperimentManagerReadExistingTests(os.path.join(path, QUEUE_PREFIX))], axis=0)
    if existing.shape[0] == 0:
        run = 0
    else:
        run = np.max(existing['Run'].values) + 1


    config_dict['creator'] = str(getpwuid( os.getuid() ).pw_name)
    config_dict['timestamp'] = datetime.datetime.now().strftime(TIME_FORMAT)
    
    config_file = pd.DataFrame(config_dict, index=[0]).T

    config_dict = ExperimentManagerDictionarySanitizer(config_dict)
    
    name = ExperimentManagerPatternWriter({**config_dict, 'Run':run})
    
    os.makedirs(os.path.join(path, QUEUE_PREFIX, name), exist_ok=True)
    config_file.to_csv(os.path.join(path, QUEUE_PREFIX, name, 'config.csv'))
    for file in saveFiles:
        if os.path.isdir(file):
            os.makedirs(os.path.join(path, QUEUE_PREFIX, name,BACKUP_NAME, file), exist_ok=True)
            os.system(f'cp -r {file}/* {os.path.join(path, QUEUE_PREFIX, name,BACKUP_NAME, file)}/')
            # print(f'cp -r {file}/* {os.path.join(path, name,BACKUP_NAME, file)}/')
        else:
            os.system(f'cp -r {file} {os.path.join(path, QUEUE_PREFIX, name, BACKUP_NAME )}')
    return os.path.join(path, name)


def ExperimentManagerQueuePop(path:str):
    # Lock
    ExperimentManagerLock(path)

    # Pop Queue
    existing = ExperimentManagerReadExistingTests(os.path.join(path,QUEUE_PREFIX))
    if existing.shape[0] == 0:
        raise ValueError('No experiments in the queue')
    existing = existing.sort_values(by='Run')
    
    # Move the queue to the results
    os.makedirs(path, exist_ok=True)    
    os.rename(os.path.join(path, QUEUE_PREFIX, existing.iloc[0]['path']), os.path.join(path, existing.iloc[0]['path']))

    # Unlock
    ExperimentManagerUnlock(path)

    
    return os.path.join(path, existing.iloc[0]['path'])
    


def ExperimentManagerHomogenizeResults(path:str):
    """
    Read the directory and homogenize the results, moving the directories without a config file to a subdirectory
    """
    logs_file = open('./ExperimentManagerHomogenizeResults_logs.txt', 'a+')
    logs_file.write('\n\n---------------------------------------------------------------------\n')
    logs_file.write(f'Homogenizing results in {path}\n')
    logs_file.write(f'Date: {datetime.datetime.now()}\n')

    # Read the directory
    results_table = []
    unprocessed = []
    empty = []
    content = os.listdir(path)
    content = [x for x in content if UN_PROCESSED_PREFIX not in x]
    for name in content:
        if name == EMPTY_PREFIX:
            continue
        elif name == UN_PROCESSED_PREFIX:
            continue
        elif name == 'tmp':
            raise ValueError('The tmp directory is present, please remove it before running the homogenize function, it indicates that the previous run was not completed')
        elif name == QUEUE_PREFIX:
            continue
        elif name == MUTEX_FILE:
            print(f'{name} is a mutex file, skipping, check if the previous runs were completed')
            continue
        # empty check
        elif os.listdir(os.path.join(path, name)) == ['backup','config.csv']:
            print(f'{name} is empty, moving it to empty')
            logs_file.write(f'{name} is empty, moving it to empty\n')
            empty.append(name)
        # existing
        elif 'Run#' in name:
            continue
        # newly read
        elif os.path.exists(os.path.join(path, name, 'config.csv')):
            logs_file.write(f'{name} has a config file, using it to rename the directory\n')
            print(f'{name} has a config file, using it to rename the directory')
            timestamp = os.path.getmtime(os.path.join(path, name, 'config.csv'))
            timestamp = datetime.datetime.fromtimestamp(timestamp)
            creator = getpwuid(os.stat(os.path.join(path, name, 'config.csv')).st_uid).pw_name 
            config = pd.read_csv(os.path.join(path, name, 'config.csv'), index_col=0).T
            config = config.to_dict(orient='records')[0]
            config = ExperimentManagerDictionarySanitizer(config)
            config = pd.DataFrame(config, index=[0])
            config['timestamp'] = timestamp
            config['creator'] = creator
            config['path'] = name
            results_table.append(config)
            continue
        else:
            warnings.warn(f'{name} does not have a config file, moving it to unprocessed')
            print(f'{name} does not have a config file, moving it to unprocessed')
            unprocessed.append(name)

    logs_file.write(f'\n\n')
    logs_file.write(f'Processed {len(results_table)} directories\n\n')

    # Move the unprocessed directories to a subdirectory
    if len(unprocessed) > 0:
        warnings.warn(f'The following directories do not have a config file: {unprocessed} moving them to {path}/{UN_PROCESSED_PREFIX}')
        logs_file.write(f'The following directories do not have a config file: {unprocessed} moving them to {path}/{UN_PROCESSED_PREFIX}')
        os.makedirs(os.path.join(path, UN_PROCESSED_PREFIX), exist_ok=True)
        for name in unprocessed:
            logs_file.write(f'Moving {name} to {path}/{UN_PROCESSED_PREFIX}\n')
            os.rename(os.path.join(path, name), os.path.join(path, UN_PROCESSED_PREFIX, name))
    
    # Move the empty directories to a subdirectory
    if len(empty) > 0:
        warnings.warn(f'The following directories are empty: {empty} moving them to {path}/{EMPTY_PREFIX}')
        logs_file.write(f'The following directories are empty: {empty} moving them to {path}/{EMPTY_PREFIX}')
        os.makedirs(os.path.join(path, EMPTY_PREFIX), exist_ok=True)
        for name in empty:
            logs_file.write(f'Moving {name} to {path}/{EMPTY_PREFIX}\n')
            os.rename(os.path.join(path, name), os.path.join(path, EMPTY_PREFIX, name))

    
    # Created the consistent results table
    existing = ExperimentManagerReadExistingTests(path)
    results_table = pd.concat([existing,*results_table])
    results_table = results_table.sort_values(by=HOMOGENIZE_SORT_KEYS).reset_index(drop=True)

    # Save the results table
    # wait user confirmation
    print(results_table)
    logs_file.write(f'\n\n')
    logs_file.write(results_table.to_string())

    results_table['Run'] = results_table.index

    print('Do you want move the directories to the new names? This is a destructive operation, removing the old run names and ids! [y/n]')
    logs_file.write('Do you want move the directories to the new names? This is a destructive operation, removing the old run names and ids! [y/n]')
    if len(unprocessed) > 0:
        print(f'There are {len(unprocessed)} unprocessed directories, please check the logs for more information')
        logs_file.write(f'There are {len(unprocessed)} unprocessed directories, please check the logs for more information')

    answer = input()
    logs_file.write(f'User input: {answer}\n')

    if answer == 'y':
        os.makedirs(os.path.join(path, 'tmp'), exist_ok=True)
        # move the folders initially to tmp
        individual_paths = results_table['path'].values
        for individual_path in individual_paths:
            os.rename(os.path.join(path,individual_path), os.path.join(path, 'tmp',individual_path))
            logs_file.write(f'Moving {individual_path} to tmp\n')
            pass
        
        # move the folders back to the original location, but with the new names
        for index, row in results_table.iterrows():
            old_path = row['path']
            config = row.to_dict()
            del config['path']
            if 'timestamp' not in config or config['timestamp'] is None or str(config['timestamp']) == 'NaT':
                print(f'{old_path} does not have a timestamp, using the file timestamp')
                logs_file.write(f'{old_path} does not have a timestamp, using the file timestamp\n')
                timestamp = os.path.getmtime(os.path.join(path, 'tmp', old_path, 'config.csv'))
                timestamp = datetime.datetime.fromtimestamp(timestamp)
                config['timestamp'] = timestamp
            config['timestamp'] = config['timestamp'].strftime(TIME_FORMAT)
            config = ExperimentManagerDictionarySanitizer(config)
            new_path = ExperimentManagerPatternWriter(config)
            os.rename(os.path.join(path,'tmp', old_path), os.path.join(path, new_path))
            print(f'{new_path}')
            logs_file.write(f'Moving tmp/{old_path} to {new_path}\n')

        # remove the tmp directory
        os.rmdir(os.path.join(path, 'tmp'))
    else:
        print('Exiting without changes')






if __name__ == "__main__":
    # print(ExperimentManagerSaveFunction('./results', config_dict={'lr':0.001, 'epochs':50.0}, saveFiles=['./ssm']))
    # print(ExperimentManagerLoadFunction('./results',run=-1, config_dict={'lr':0.001, 'epochs':50}))
    ExperimentManagerHomogenizeResults("./results_journal")	
    # print(ExperimentManagerLoadFunction('./results', config_dict={'test':6}))
    pass