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

def generate_sts_tasks(
        dataset_id,
        K_list,
        cir_inner_loop_list, 
        n_finetune_sets):
    

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

    test_task_norm_datapoints_per_K, val_task_norm_datapoints_per_K = {}, {}
    finetune_sets_per_K_cir = {}
    test_task_unnormalized, val_task_unnormalized = {}, {}
     

    train_task_list_inner, train_task_list_outer = [], []
    train_tasks_list, val_tasks_list, test_tasks_finetune_sets_list, test_tasks_test_sets_list = [], [], [], []


    for task_idx, (task_X, task_Y) in enumerate(zip(X_all, Y_all)):
        normal_indexes, anomaly_indexes = list(np.nonzero(task_Y == 0)[0]), list(np.nonzero(task_Y == 1)[0])
        
        n_inner_loop_normal = n_outer_loop_normal = int(len(normal_indexes)/2)
        n_inner_loop_anomaly = n_outer_loop_anomaly = int(len(anomaly_indexes)/2)

        inner_loop_normal_indexes = random.sample(normal_indexes, n_inner_loop_normal)
        inner_loop_anomaly_indexes = random.sample(anomaly_indexes, n_inner_loop_anomaly)
        inner_loop_indexes = []
        inner_loop_indexes += inner_loop_normal_indexes
        inner_loop_indexes += inner_loop_anomaly_indexes

        rest_normal_indexes = [index for index in normal_indexes if index not in inner_loop_normal_indexes]
        rest_anomaly_indexes = [index for index in anomaly_indexes if index not in inner_loop_anomaly_indexes]
          
        outer_loop_normal_indexes = rest_normal_indexes
        outer_loop_anomaly_indexes = rest_anomaly_indexes
        outer_loop_indexes = []
        outer_loop_indexes += outer_loop_normal_indexes
        outer_loop_indexes += outer_loop_anomaly_indexes


        if(task_idx >= n_training_tasks + n_val_tasks):

            test_task_unnormalized['test_X'], test_task_unnormalized['test_Y'] = task_X[outer_loop_indexes], np.expand_dims(task_Y[outer_loop_indexes],1)

            #shuffle
            s_test = np.arange(test_task_unnormalized['test_X'].shape[0])
            np.random.shuffle(s_test)
            test_task_unnormalized['test_X'], test_task_unnormalized['test_Y'] = test_task_unnormalized['test_X'][s_test], test_task_unnormalized['test_Y'][s_test]

            for K in K_list:
                finetune_sets_per_cir = {}
                normal_indexes_all_finetune_sets = []
                for cir in cir_inner_loop_list:

                    finetune_sets_list = []

                    n_finetune_normal = int(K*cir)
                    n_finetune_anomaly = K - n_finetune_normal
                    for i in range(n_finetune_sets):
                        finetune_normal_indexes = random.sample(inner_loop_normal_indexes, n_finetune_normal)
                        normal_indexes_all_finetune_sets += finetune_normal_indexes
                        finetune_anomaly_indexes = random.sample(inner_loop_anomaly_indexes, n_finetune_anomaly)
                        finetune_indexes = []
                        finetune_indexes += finetune_normal_indexes
                        finetune_indexes += finetune_anomaly_indexes
                        finetune_set = {}
                        finetune_set['finetune_X'], finetune_set['finetune_Y'] = task_X[finetune_indexes], np.expand_dims(task_Y[finetune_indexes],1)

                        #shuffle
                        s_finetune = np.arange(finetune_set['finetune_X'].shape[0])
                        np.random.shuffle(s_finetune)
                        finetune_set['finetune_X'], finetune_set['finetune_Y'] = finetune_set['finetune_X'][s_finetune], finetune_set['finetune_Y'][s_finetune]
                        finetune_sets_list.append(finetune_set)
                        
                    finetune_sets_per_cir[str(cir)] = finetune_sets_list

                finetune_sets_per_K_cir[str(K)] = finetune_sets_per_cir

                n_normalization_datapoints = K

                # delete doubles
                normal_indexes_all_finetune_sets = list(set(normal_indexes_all_finetune_sets))
                # sample normalization datapoints from the finetune sets
                normalization_set_indexes = random.sample(normal_indexes_all_finetune_sets, n_normalization_datapoints)
                normalization_X = task_X[normalization_set_indexes]  
                test_task_norm_datapoints_per_K[str(K)] = normalization_X

            #normalize test task
            test_task_test_sets = {}
            for K in K_list:
                test_task_test_sets[str(K)] = {}
                test_task_test_sets[str(K)]['test_Y'] = test_task_unnormalized['test_Y']

                norm_X = test_task_norm_datapoints_per_K[str(K)] 
                sc = StandardScaler()
                norm_X_flattened = norm_X.reshape(-1, 1)
                sc.fit(norm_X_flattened)

                test_task_norm_datapoints_normalized = sc.transform(test_task_norm_datapoints_per_K[str(K)].reshape(-1,1))
                test_task_norm_datapoints_per_K[str(K)] = test_task_norm_datapoints_normalized.reshape(test_task_norm_datapoints_per_K[str(K)].shape)
                
                test_task_test_set_normalized = sc.transform(test_task_unnormalized['test_X'].reshape(-1,1))
                test_task_test_sets[str(K)]['test_X'] = test_task_test_set_normalized.reshape(test_task_unnormalized['test_X'].shape)

                for key in finetune_sets_per_K_cir[str(K)].keys():
                    for idx, finetuneset in enumerate(finetune_sets_per_K_cir[str(K)][key]):
                        test_task_finetune_set_normalized = sc.transform(finetune_sets_per_K_cir[str(K)][key][idx]['finetune_X'].reshape(-1,1))
                        finetune_sets_per_K_cir[str(K)][key][idx]['finetune_X'] = test_task_finetune_set_normalized.reshape(finetune_sets_per_K_cir[str(K)][key][idx]['finetune_X'].shape)   

            test_tasks_test_sets_list.append(test_task_test_sets)
            test_tasks_finetune_sets_list.append(finetune_sets_per_K_cir)


        elif(task_idx >= n_training_tasks):
            val_task_unnormalized['test_X'], val_task_unnormalized['test_Y'] = task_X[outer_loop_indexes], np.expand_dims(task_Y[outer_loop_indexes],1)
            val_task_unnormalized['finetune_X'], val_task_unnormalized['finetune_Y'] = task_X[inner_loop_indexes], np.expand_dims(task_Y[inner_loop_indexes],1)

            for K in K_list:

                n_normalization_datapoints = K

                # sample normalization datapoints from the finetune set
                normalization_set_indexes = random.sample(inner_loop_normal_indexes, n_normalization_datapoints)
                normalization_X = task_X[normalization_set_indexes]
                val_task_norm_datapoints_per_K[str(K)] = normalization_X

            #shuffle
            s_val_finetune = np.arange(val_task_unnormalized['finetune_X'].shape[0])
            s_val_test = np.arange(val_task_unnormalized['test_X'].shape[0])
            np.random.shuffle(s_val_finetune)
            np.random.shuffle(s_val_test)

            val_task_unnormalized['finetune_X'], val_task_unnormalized['finetune_Y'] = val_task_unnormalized['finetune_X'][s_val_finetune], val_task_unnormalized['finetune_Y'][s_val_finetune]
            val_task_unnormalized['test_X'], val_task_unnormalized['test_Y'] = val_task_unnormalized['test_X'][s_val_test], val_task_unnormalized['test_Y'][s_val_test]

            #normalize val task
            val_task = {}
            for K in K_list:
                val_task[str(K)] = {}
                val_task[str(K)]['finetune_Y'] = val_task_unnormalized['finetune_Y']
                val_task[str(K)]['test_Y'] = val_task_unnormalized['test_Y']

                norm_X = val_task_norm_datapoints_per_K[str(K)] 
                sc = StandardScaler()
                norm_X_flattened = norm_X.reshape(-1, 1)
                sc.fit(norm_X_flattened)
                
                val_task_norm_datapoints_normalized = sc.transform(val_task_norm_datapoints_per_K[str(K)].reshape(-1,1))
                val_task_norm_datapoints_per_K[str(K)] = val_task_norm_datapoints_normalized.reshape(val_task_norm_datapoints_per_K[str(K)].shape)
                
                val_task_finetune_normalized = sc.transform(val_task_unnormalized['finetune_X'].reshape(-1,1)) 
                val_task[str(K)]['finetune_X'] = val_task_finetune_normalized.reshape(val_task_unnormalized['finetune_X'].shape)
                
                val_task_test_normalized = sc.transform(val_task_unnormalized['test_X'])
                val_task[str(K)]['test_X'] = val_task_test_normalized.reshape(val_task_unnormalized['test_X'].shape)

            val_tasks_list.append(val_task)


        else:

            #normalize
            sampled_normal_X = task_X[normal_indexes]
            sampled_normal_X_flattened = sampled_normal_X.reshape(-1,1)

            sc = StandardScaler()
            sc.fit(sampled_normal_X_flattened)
            task_X_flattened = sc.transform(task_X.reshape(-1,1))
            task_X = task_X_flattened.reshape(task_X.shape)

            s_inner = np.arange(task_X[inner_loop_indexes].shape[0])
            s_outer = np.arange(task_X[outer_loop_indexes].shape[0])
            np.random.shuffle(s_inner)
            np.random.shuffle(s_outer)
            
            train_task ={}
            train_task['X_inner'], train_task['Y_inner'] = task_X[inner_loop_indexes][s_inner], np.expand_dims(task_Y[inner_loop_indexes][s_inner],1)
            train_task['X_outer'], train_task['Y_outer'] = task_X[outer_loop_indexes][s_outer], np.expand_dims(task_Y[outer_loop_indexes][s_outer],1)

            train_tasks_list.append(train_task)

    return train_tasks_list, val_tasks_list, test_tasks_test_sets_list, test_tasks_finetune_sets_list

def main(args):

    seed = 123

    np.random.seed(seed)
    random.seed(seed)

    if(' ' in args.K_list):
        K_list = [int(i) for i in args.K_list.split(' ')]
    else:
        K_list = [int(args.K_list[0])]

    if(' ' in args.cir_inner_loop_list):
        cir_inner_loop_list = [float(i) for i in args.cir_inner_loop_list.split(' ')]
    else:
        cir_inner_loop_list = [float(args.cir_inner_loop_list[0])]
    

    base_path = '/home/USER/Documents'
    if (not (os.path.exists(base_path))):
        base_path = '/home/ubuntu/Projects' 
    dir_path = base_path + '/MAML/input_data/new_SSR/'+ args.dataset_id + '_SSR'
    

    train_tasks_list, val_tasks_list, test_tasks_test_sets_list, test_tasks_finetune_sets_list = generate_sts_tasks(
        dataset_id = args.dataset_id,
        K_list = K_list,
        cir_inner_loop_list = cir_inner_loop_list, 
        n_finetune_sets = args.n_finetune_sets)

    if (not (os.path.exists(dir_path))):
        os.mkdir(dir_path)

    train_tasks_file = dir_path + '/train_tasks.txt'
    with open(train_tasks_file, 'wb') as file:
        pickle.dump(train_tasks_list, file)

    val_task_file = dir_path + '/val_tasks.txt'
    with open(val_task_file, 'wb') as file:
        pickle.dump(val_tasks_list, file)

    test_task_test_sets_file = dir_path + '/test_tasks_test_sets.txt'
    with open(test_task_test_sets_file, 'wb') as file:
        pickle.dump(test_tasks_test_sets_list, file)

    test_task_finetune_sets_file = dir_path + '/test_tasks_finetune_sets.txt'
    with open(test_task_finetune_sets_file, 'wb') as file:
        pickle.dump(test_tasks_finetune_sets_list, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='generate SSR datasets')  
    parser.add_argument(
        '-K_list',
        type=str,
        metavar='',
        help='size of the adaptation/finetuning sets in each of generated datasets')    
    parser.add_argument(
        '-cir_inner_loop_list',
        type=str,
        metavar='',
        help=(' class-imbalance rate of the adaptation/finetuning sets in each of generated datasets'))
    parser.add_argument(
        '-dataset_id',
        type=str,
        metavar='',
        help='id of the SSR dataset to use')  
    parser.add_argument(
        '-n_finetune_sets',
        type=int,
        metavar='',
        help='number of adaptation/finetuning sets sampled from each test task') 
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
