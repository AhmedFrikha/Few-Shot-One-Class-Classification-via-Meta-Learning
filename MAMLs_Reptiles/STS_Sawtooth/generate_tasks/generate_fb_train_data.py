# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import StandardScaler
import random 
import pickle
import json
import os
import argparse



class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(config_file, args_dict):
    with open(config_file) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]

    return args_dict

def generate_sts_fb_train_data(dataset_id):
    

    """create meta-training, meta-validation and meta-testing tasks needed for the STS experiments, given a list of signals.

    Parameters
    ----------
    dataset_id : str
        can be 'sine' or 'sawtooth'.
    K_list : list
        size of the adaptation/finetuning sets in each of generated tasks/datasets.
    cir_inner_loop_list : list
        class-imbalance rate of the adaptation/finetuning sets in each of generated datasets
    n_finetune_sets : int
        number of adaptation/finetuning sets sampled from the test task


    Returns
    -------
    train_tasks_list : list
        contains the meta-training tasks (each divided in inner and outer loop data).
    val_tasks_list : list
        contains the meta-validation task (each divided in finetuning/adaptation and test sets).
    test_tasks_test_sets_list : list
        contains the test set of each of the meta-testing tasks.
    test_tasks_finetune_sets_list : list
        contains the finetuning/adaptation sets of each of meta-testing tasks, for each combination of adaptation set size (K) and class-imbalance rate (cir). 

    """


    base_path = '/home/USER/Documents'
    if (not (os.path.exists(base_path))):
        base_path = '/home/ubuntu/Projects'
    file = open(base_path + '/MAML/raw_data/SSR_data/ssr_' + dataset_id +'.txt', 'rb')
    sts = pickle.load(file)

    
    n_test_tasks = n_val_tasks = 5
    X_all = sts['X']
    Y_all = sts['Y']
    segment_length = X_all.shape[2]
    n_training_tasks = X_all.shape[0] - n_test_tasks - n_val_tasks

    X_train = sts['X'][:n_training_tasks]
    Y_train = sts['Y'][:n_training_tasks]

    
    fb_train_data_X, fb_train_data_Y = [], []

    for task_idx, (task_X, task_Y) in enumerate(zip(X_train, Y_train)):
        
        normal_indexes, anomaly_indexes = list(np.nonzero(task_Y == 0)[0]), list(np.nonzero(task_Y == 1)[0])
        
        fb_train_data_X += list(task_X[normal_indexes])    
        fb_train_data_Y += [2*task_idx]* len(normal_indexes)    

        fb_train_data_X += list(task_X[anomaly_indexes])    
        fb_train_data_Y += [2*task_idx+1]* len(anomaly_indexes)    


    s = np.arange(len(fb_train_data_X))
    fb_train_data_X, fb_train_data_Y = np.array(fb_train_data_X)[s], np.array(fb_train_data_Y)[s]
    train_data = {'X': fb_train_data_X,
                  'Y': fb_train_data_Y}


    return train_data


def main(args):

    seed = 123

    np.random.seed(seed)
    random.seed(seed)


    base_path = '/home/USER/Documents'
    if (not (os.path.exists(base_path))):
        base_path = '/home/ubuntu/Projects' 
    dir_path = base_path + '/MAML/input_data/new_SSR/'+ args.dataset_id + '_SSR'
    

    train_data = generate_sts_fb_train_data(dataset_id = args.dataset_id)

    if (not (os.path.exists(dir_path))):
        os.mkdir(dir_path)

    train_data_file = dir_path + '/fb_train_data.txt'
    with open(train_data_file, 'wb') as file:
        pickle.dump(train_data, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='generate SSR datasets')  
    
    parser.add_argument(
        '-dataset_id',
        type=str,
        metavar='',
        help='id of the SSR dataset to use')  
    parser.add_argument('-config_file', 
        type=str, 
        default="None")

    args = parser.parse_args()

    args_dict = vars(args)
    if args.config_file is not "None":
        args_dict = extract_args_from_json(args.config_file, args_dict)

    for key in list(args_dict.keys()):

        if str(args_dict[key]).lower() == "true":
            args_dict[key] = True
        elif str(args_dict[key]).lower() == "false":
            args_dict[key] = False


    args = Bunch(args_dict)
    main(args)
