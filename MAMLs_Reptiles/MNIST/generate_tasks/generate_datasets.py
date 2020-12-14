# -*- coding: utf-8 -*-
import numpy as np
from mlxtend.data import loadlocal_mnist
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

def generate_mnist_datasets(
        datapoints_per_task,
        K_list,
        cir_inner_loop_list, 
        test_task_idx, 
        val_task_idx,
        n_finetune_sets):


    """create meta-training, meta-validation and meta-testing tasks needed for a MT-MNIST experiment, given the validation and test digits.

    Parameters
    ----------
    datapoints_per_task : int
        total number of datapoints per task.
    K_list : list
        size of the adaptation/finetuning sets in each of generated tasks/datasets.
    cir_inner_loop_list : list
        class-imbalance rate of the adaptation/finetuning sets in each of generated datasets
    test_task_idx : int
        index of the meta-test task
    val_task_idx : int
        index of the meta-validation task
    n_finetune_sets : int
        number of adaptation/finetuning sets sampled from the test task


    Returns
    -------
    train_tasks : dict
        meta-training tasks (divided in inner and outer loop data).
    val_task : dict
        meta-validation task (divided in finetuning/adaptation and test sets).
    test_task_test_set : dict
        test set of the meta-testing task.
    finetune_sets_per_K_cir : dict
        20 finetuning/adaptation sets for each K and cir combination. 

    """

    # arbitrarily chosen, class-imbalance rate in outer and inner training loops
    cir_outer_loop = 0.5
    cir_inner_loop = 0.5
    # class-imbalance rate in the test sets of the test and validation tasks
    cir_test = 0.5
    # arbitrarily chosen, percentage of data that will be used in the inner training loop
    percent_data_inner_loop = 0.5

    percent_data_finetune_val = 0.8

    n_test_set = 4000

    test_task_idx, val_task_idx = test_task_idx, val_task_idx

    finetune_sets_per_K_cir = {}
    test_task_test_set, val_task = {}, {}
     

    train_task_list_inner, train_task_list_outer = [], []

    train_tasks_idxs = [i for i in range(0,10) if i not in [val_task_idx, test_task_idx]]

    base_path = '/home/USER/Documents'
    if (not (os.path.exists(base_path))):
        base_path = '/home/ubuntu/Projects'
    train_images, train_labels = loadlocal_mnist(
        images_path= base_path + '/MAML/raw_data/MNIST_data/train-images-idx3-ubyte', 
        labels_path= base_path + '/MAML/raw_data/MNIST_data/train-labels-idx1-ubyte')

    test_images, test_labels = loadlocal_mnist(
        images_path= base_path + '/MAML/raw_data/MNIST_data/t10k-images-idx3-ubyte', 
        labels_path= base_path + '/MAML/raw_data/MNIST_data/t10k-labels-idx1-ubyte')


    train_images, test_images = train_images.reshape((-1,28,28))/255.0, test_images.reshape((-1,28,28))/255.0
    images = np.concatenate((train_images, test_images))
    labels = np.concatenate((train_labels, test_labels))

    test_task_normal_indexes, val_task_normal_indexes = list(np.nonzero(labels == test_task_idx)[0]), list(np.nonzero(train_labels == val_task_idx)[0])
    test_task_X_normal, val_task_X_normal = images[test_task_normal_indexes],train_images[val_task_normal_indexes]
    test_task_Y_normal, val_task_Y_normal = np.zeros_like(labels[test_task_normal_indexes]), np.zeros_like(train_labels[val_task_normal_indexes])


    # val and test task have anomalies (samples of other numbers) that are not used for training
    # besides the two sets of anomalies (one for val task and one for test task are disjoint)
    test_task_anomalous_indexes = list(np.nonzero(test_labels[:5000] != test_task_idx)[0])
    val_task_anomalous_indexes= [index for index, element in enumerate(list(test_labels[5000:])) if element not in [val_task_idx, test_task_idx]]


    test_task_X_anomalous, val_task_X_anomalous = test_images[:5000][test_task_anomalous_indexes],test_images[5000:][val_task_anomalous_indexes]
    test_task_Y_anomalous, val_task_Y_anomalous = np.ones_like(test_labels[:5000][test_task_anomalous_indexes]), np.ones_like(test_labels[5000:][val_task_anomalous_indexes])

    test_task_X, val_task_X = np.concatenate((test_task_X_normal, test_task_X_anomalous)), np.concatenate((val_task_X_normal, val_task_X_anomalous))
    test_task_Y, val_task_Y = np.expand_dims(np.concatenate((test_task_Y_normal, test_task_Y_anomalous)),-1), np.expand_dims(np.concatenate((val_task_Y_normal, val_task_Y_anomalous)),-1)


    train_tasks_X_list, train_tasks_Y_list = [], []
    for task_idx in train_tasks_idxs:
        train_task_normal_indexes = list(np.nonzero(train_labels == task_idx)[0]) 
        train_task_anomalous_indexes = [index for index, element in enumerate(list(train_labels)) if element not in [task_idx, val_task_idx, test_task_idx]]
        assert(len(np.nonzero(train_labels[train_task_anomalous_indexes] == val_task_idx)[0]) == 0)
        assert(len(np.nonzero(train_labels[train_task_anomalous_indexes] == test_task_idx)[0]) == 0)
        train_task_X_normal, train_task_X_anomalous = train_images[train_task_normal_indexes], train_images[train_task_anomalous_indexes]
        train_task_Y_normal, train_task_Y_anomalous = np.zeros_like(train_labels[train_task_normal_indexes]), np.ones_like(train_labels[train_task_anomalous_indexes])
        train_task_X, train_task_Y = np.concatenate((train_task_X_normal, train_task_X_anomalous)), np.concatenate((train_task_Y_normal, train_task_Y_anomalous))
        train_tasks_X_list.append(train_task_X)
        train_tasks_Y_list.append(np.expand_dims(train_task_Y,-1))



    # building test task sets of data
    normal_indexes, anomaly_indexes = list(np.nonzero(test_task_Y == 0)[0]), list(np.nonzero(test_task_Y == 1)[0])
    n_test_set_normal = int(n_test_set*cir_test)
    test_set_normal_indexes = random.sample(normal_indexes, n_test_set_normal)
    test_set_anomaly_indexes = random.sample(anomaly_indexes, n_test_set - n_test_set_normal)
    test_set_indexes = []
    test_set_indexes += test_set_normal_indexes
    test_set_indexes += test_set_anomaly_indexes

    test_task_test_set['test_X'], test_task_test_set['test_Y'] = test_task_X[test_set_indexes], test_task_Y[test_set_indexes]


    #shuffle
    s_test = np.arange(test_task_test_set['test_X'].shape[0])
    np.random.shuffle(s_test)
    test_task_test_set['test_X'], test_task_test_set['test_Y'] = test_task_test_set['test_X'][s_test], test_task_test_set['test_Y'][s_test]

    rest_normal_indexes = [index for index in normal_indexes if index not in test_set_normal_indexes]
    rest_anomaly_indexes = [index for index in anomaly_indexes if index not in test_set_anomaly_indexes]


    for K in K_list:
        finetune_sets_per_cir = {}
        for cir in cir_inner_loop_list:

            rest_normal_indexes = [index for index in normal_indexes if index not in test_set_normal_indexes]
            rest_anomaly_indexes = [index for index in anomaly_indexes if index not in test_set_anomaly_indexes]
    
            finetune_sets_list = []

            disjoint = False
            if(cir*K*n_finetune_sets<len(rest_normal_indexes)):
                disjoint = True

            n_finetune_normal = int(K*cir)
            n_finetune_anomaly = K - n_finetune_normal
            for i in range(n_finetune_sets):
                # if enough for disjoint do that
                # else sample randomly
                # store in a dict with keys cir_K
                finetune_normal_indexes = random.sample(rest_normal_indexes, n_finetune_normal)
                finetune_anomaly_indexes = random.sample(rest_anomaly_indexes, n_finetune_anomaly)
                finetune_indexes = []
                finetune_indexes += finetune_normal_indexes
                finetune_indexes += finetune_anomaly_indexes
                finetune_set = {}
                finetune_set['finetune_X'], finetune_set['finetune_Y'] = test_task_X[finetune_indexes], test_task_Y[finetune_indexes]

                #shuffle
                s_finetune = np.arange(finetune_set['finetune_X'].shape[0])
                np.random.shuffle(s_finetune)
                finetune_set['finetune_X'], finetune_set['finetune_Y'] = finetune_set['finetune_X'][s_finetune], finetune_set['finetune_Y'][s_finetune]

                finetune_sets_list.append(finetune_set)
                
                if(disjoint):
                    rest_normal_indexes = [index for index in rest_normal_indexes if index not in finetune_normal_indexes]
                    rest_anomaly_indexes = [index for index in rest_anomaly_indexes if index not in finetune_anomaly_indexes]

            finetune_sets_per_cir[str(cir)] = finetune_sets_list

        finetune_sets_per_K_cir[str(K)] = finetune_sets_per_cir


    #building val task sets of data
    normal_indexes, anomaly_indexes = list(np.nonzero(val_task_Y == 0)[0]), list(np.nonzero(val_task_Y == 1)[0])
    n_val_finetune = int(percent_data_finetune_val*datapoints_per_task)
    n_val_test_set = datapoints_per_task - n_val_finetune
    n_val_test_set_normal = int(n_val_test_set*cir_test)
    val_test_set_normal_indexes = random.sample(normal_indexes, n_val_test_set_normal)


    val_test_set_anomaly_indexes = random.sample(anomaly_indexes, n_val_test_set - n_val_test_set_normal)
    val_test_set_indexes = []
    val_test_set_indexes += val_test_set_normal_indexes
    val_test_set_indexes += val_test_set_anomaly_indexes
    val_task['test_X'], val_task['test_Y'] = val_task_X[val_test_set_indexes], val_task_Y[val_test_set_indexes]


    rest_normal_indexes = [index for index in normal_indexes if index not in val_test_set_normal_indexes]
    rest_anomaly_indexes = [index for index in anomaly_indexes if index not in val_test_set_anomaly_indexes]

    n_val_finetune_normal = int(n_val_finetune*cir_inner_loop)
    val_finetune_normal_indexes = random.sample(rest_normal_indexes, n_val_finetune_normal)
    val_finetune_anomaly_indexes = random.sample(rest_anomaly_indexes, n_val_finetune - n_val_finetune_normal)
    val_finetune_indexes = []
    val_finetune_indexes += val_finetune_normal_indexes
    val_finetune_indexes += val_finetune_anomaly_indexes

    val_task['finetune_X'], val_task['finetune_Y'] = val_task_X[val_finetune_indexes], val_task_Y[val_finetune_indexes]

    #shuffle
    s_val_finetune = np.arange(val_task['finetune_X'].shape[0])
    s_val_test = np.arange(val_task['test_X'].shape[0])
    np.random.shuffle(s_val_finetune)
    np.random.shuffle(s_val_test)

    val_task['finetune_X'], val_task['finetune_Y'] = val_task['finetune_X'][s_val_finetune], val_task['finetune_Y'][s_val_finetune]
    val_task['test_X'], val_task['test_Y'] = val_task['test_X'][s_val_test], val_task['test_Y'][s_val_test]



    # building sets of data of the training tasks
    for task_X, task_Y in zip(train_tasks_X_list, train_tasks_Y_list):
        normal_indexes, anomaly_indexes = list(np.nonzero(task_Y == 0)[0]), list(np.nonzero(task_Y == 1)[0])

        n_inner_loop = int(percent_data_inner_loop*datapoints_per_task)
        n_inner_loop_normal = int(n_inner_loop*cir_inner_loop)
        n_outer_loop = datapoints_per_task - n_inner_loop
        n_outer_loop_normal = int(n_outer_loop*cir_outer_loop)
        
        inner_loop_normal_indexes = random.sample(normal_indexes, n_inner_loop_normal)
        inner_loop_anomaly_indexes = random.sample(anomaly_indexes, n_inner_loop - n_inner_loop_normal)
        inner_loop_indexes = []
        inner_loop_indexes += inner_loop_normal_indexes
        inner_loop_indexes += inner_loop_anomaly_indexes

        train_task_inner_X, train_task_inner_Y = task_X[inner_loop_indexes], task_Y[inner_loop_indexes]

        rest_normal_indexes = [index for index in normal_indexes if index not in inner_loop_normal_indexes]
        rest_anomaly_indexes = [index for index in anomaly_indexes if index not in inner_loop_anomaly_indexes]

        
        outer_loop_normal_indexes = random.sample(rest_normal_indexes, n_outer_loop_normal)
        outer_loop_anomaly_indexes = random.sample(rest_anomaly_indexes, n_outer_loop - n_outer_loop_normal)
        outer_loop_indexes = []
        outer_loop_indexes += outer_loop_normal_indexes
        outer_loop_indexes += outer_loop_anomaly_indexes

        train_task_outer_X, train_task_outer_Y = task_X[outer_loop_indexes], task_Y[outer_loop_indexes]


        s_inner = np.arange(train_task_inner_X.shape[0])
        s_outer = np.arange(train_task_outer_X.shape[0])
        np.random.shuffle(s_inner)
        np.random.shuffle(s_outer)
        train_task_list_inner.append([train_task_inner_X[s_inner],train_task_inner_Y[s_inner]])
        train_task_list_outer.append([train_task_outer_X[s_outer],train_task_outer_Y[s_outer]])



    train_tasks_inner_X = np.stack([train_task_list_inner[i][0]
                                for i in range(len(train_task_list_inner))], 0)
    train_tasks_inner_Y = np.stack([train_task_list_inner[i][1]
                                for i in range(len(train_task_list_inner))], 0)
    train_tasks_outer_X = np.stack([train_task_list_outer[i][0]
                                for i in range(len(train_task_list_outer))], 0)
    train_tasks_outer_Y = np.stack([train_task_list_outer[i][1]
                                for i in range(len(train_task_list_outer))], 0)

                        
    train_tasks = {'X_train_inner': train_tasks_inner_X,
               'Y_train_inner': train_tasks_inner_Y,
               'X_train_outer': train_tasks_outer_X,
               'Y_train_outer': train_tasks_outer_Y
               }


    return train_tasks, val_task, test_task_test_set, finetune_sets_per_K_cir

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
    dir_path = base_path + '/MAML/input_data/MNIST_val' + str(args.val_task_idx) + '_test' + str(args.test_task_idx)
    

    train_tasks, val_task, test_task_test_sets, test_task_finetune_sets = generate_mnist_datasets(
        datapoints_per_task = args.datapoints_per_task,
        K_list = K_list,
        cir_inner_loop_list = cir_inner_loop_list, 
        test_task_idx = args.test_task_idx, 
        val_task_idx = args.val_task_idx,
        n_finetune_sets = args.n_finetune_sets)

    if (not (os.path.exists(dir_path))):
        os.mkdir(dir_path)
    else:
        raise KeyboardInterrupt ('These datasets were already generated. Delete the folder: ', dir_path, ' to generate them again.')


    train_tasks_file = dir_path + '/train_tasks_val_' + str(args.val_task_idx) + '_test_' + str(args.test_task_idx) +'.txt'
    with open(train_tasks_file, 'wb') as file:
        pickle.dump(train_tasks, file)

    val_task_file = dir_path + '/val_task_val_' + str(args.val_task_idx) + '_test_' + str(args.test_task_idx) +'.txt'
    with open(val_task_file, 'wb') as file:
        pickle.dump(val_task, file)

    test_task_test_sets_file = dir_path + '/test_task_test_sets_val_' + str(args.val_task_idx) + '_test_' + str(args.test_task_idx) +'.txt'
    with open(test_task_test_sets_file, 'wb') as file:
        pickle.dump(test_task_test_sets, file)

    test_task_finetune_sets_file = dir_path + '/test_task_finetune_sets_val_' + str(args.val_task_idx) + '_test_' + str(args.test_task_idx) +'.txt'
    with open(test_task_finetune_sets_file, 'wb') as file:
        pickle.dump(test_task_finetune_sets, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='generate MT-MNIST datasets')
    parser.add_argument(
        '-datapoints_per_task',
        type=int,
        metavar='',
        help='number of datapoints per task')    
    parser.add_argument(
        '-K_list',
        type=str,
        metavar='',
        help='size of the adaptation/finetuning sets in each of generated datasets')    
parser.add_argument(
        '-cir_inner_loop_list',
        type=str,
        metavar='',
        help=('class-imbalance rate of the adaptation/finetuning sets in each of generated datasets'))
    parser.add_argument(
        '-test_task_idx',
        type=int,
        metavar='',
        help='index of the test task') 
    parser.add_argument(
        '-val_task_idx',
        type=int,
        metavar='',
        help='index of the val task')  
    parser.add_argument(
        '-n_finetune_sets',
        type=int,
        metavar='',
        help='number of adaptation/finetuning sets sampled from the test task') 
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
