# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import argparse
import os
import random 
import pickle
import json
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(config_file, args_dict):
    with open(config_file) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]

    return args_dict


def load_Omniglot():

    """ load the Omniglot tasks.

    Parameters
    ----------
    K : int
        size of the training set.
    test_task_idx : int
        index of the meta-testing task.
    val_task_idx : int
        index of the meta-validation task.


    Returns
    -------
    train_tasks: list
        meta-training tasks.
    val_task : list
        meta-validation tasks.
    test_tasks : list
        meta-testing tasks.

    """

    base_path = '/home/USER/Documents'
    if (not (os.path.exists(base_path))):
        base_path = '/home/ubuntu/Projects'
    if (not (os.path.exists(base_path))):
        base_path = '/home/USER/Projects'


    data_path = base_path + '/MAML/input_data/omniglot/30_20_split/'

    test_tasks_file = open(data_path + 'test_tasks.txt', 'rb')
    test_tasks = pickle.load(test_tasks_file)

    return test_tasks


def initialize_isoForest(seed, n_estimators, max_samples, contamination, **kwargs):

    isoForest = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
        contamination=contamination, n_jobs=-1, random_state=seed, behaviour='new', **kwargs)
    return isoForest

def train(isoForest, X_train):
    # print('X_train.shape', X_train.shape)


    if X_train.ndim > 2:
        X_train_shape = X_train.shape
        X_train = X_train.reshape(X_train_shape[0], -1)
    else:
        X_train = X_train

    # print('X_train.shape', X_train.shape)
    isoForest.fit(X_train.astype(np.float32))


def predict(isoForest, X,y):

    # reshape to 2D if input is tensor
    if X.ndim > 2:
        X_shape = X.shape
        X = X.reshape(X_shape[0], -1)

    scores = (-1.0) * isoForest.decision_function(X.astype(np.float32))  # compute anomaly score
    y_pred = isoForest.predict(X.astype(np.float32))
    y_pred[y_pred == 1.0] = 0.0
    y_pred[y_pred == -1.0] = 1.0


    #y_pred = (isoForest.predict(X.astype(np.float32)) == -1) * 1  # get prediction



    scores_flattened = scores.flatten()
    acc = 100.0 * sum(y == y_pred) / len(y)

    TP = np.count_nonzero(y_pred * y)
    TN = np.count_nonzero((y_pred - 1) * (y - 1))
    FP = np.count_nonzero(y_pred* (y - 1))
    FN = np.count_nonzero((y_pred-1) *y)

    if(TP+FP == 0):
        prec = 0.0
    else:
        prec = TP/(TP + FP) 

    rec = TP / (TP + FN)
    spec = TN / (TN + FP)

    if(prec+rec == 0):
        f1_score = 0.0
    else:
        f1_score = 2*prec*rec/(prec + rec)

    # if sum(y) > 0:
    auc_roc = roc_auc_score(y, scores.flatten())
    return acc, prec, rec, spec, f1_score, auc_roc


def main(args):


    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    K=args.K

    test_tasks = load_Omniglot()
    
    test_finetune_normal_indexes_list, test_finetune_anomalous_indexes_list = [], []
    for i in range (len(test_tasks)):
        test_finetune_normal_indexes_list.append(list(np.where(test_tasks[i]['Y_inner'] == 0)[0]))
        test_finetune_anomalous_indexes_list.append(list(np.where(test_tasks[i]['Y_inner'] == 1)[0]))
    

    acc_list, prec_list, rec_list, spec_list, f1_list, auc_roc_list = [], [], [], [], [], []


    n_estimators = 1000
    max_samples = 'auto'
    contamination = 0.1

    for test_task_idx, test_task in enumerate(test_tasks):
        normal_indexes = test_finetune_normal_indexes_list[test_task_idx]
        anomalous_indexes = test_finetune_normal_indexes_list[test_task_idx]
        finetune_X, finetune_Y = test_task["X_inner"], np.expand_dims(test_task["Y_inner"], -1)
        # sampled for the present finetune set

        finetune_normal_idxs = random.sample(normal_indexes, K) 
        finetune_indexes = []
      
        finetune_indexes += finetune_normal_idxs

        finetune_X = finetune_X[finetune_indexes]
        finetune_Y = finetune_Y[finetune_indexes]


        finetune_X = finetune_X.reshape(finetune_X.shape[0], -1)

        pca = PCA(0.95)
        # print(finetune_X.shape)
        pca.fit(finetune_X)
        finetune_X = pca.transform(finetune_X)
        # print(finetune_X.shape)
        test_X = np.reshape(test_task['X_outer'], (test_task['X_outer'].shape[0], -1))
        test_X = pca.transform(test_X)
        isoForest = initialize_isoForest(seed, n_estimators, max_samples, contamination)
        
        train(isoForest, finetune_X)
        acc, prec, rec, spec, f1_score, auc_roc = predict(isoForest, test_X, np.squeeze(test_task['Y_outer']))
        print('test_task: ', test_task_idx,
            ' acc ', acc,
            ' prec ', prec,
            ' rec ', rec,
            ' spec ', spec,
            ' f1_score ', f1_score, 
            ' auc_roc ', auc_roc)

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        spec_list.append(spec)
        f1_list.append(f1_score)
        auc_roc_list.append(auc_roc)

    
    test_results_dict = {}
    test_results_dict['acc'] = acc_list
    test_results_dict['prec'] = prec_list
    test_results_dict['rec'] = rec_list
    test_results_dict['spec'] = spec_list
    test_results_dict['f1'] = f1_list
    test_results_dict['auc_roc'] = auc_roc_list


    results_dir_path = './results/'
    if (not (os.path.exists(results_dir_path))):
        os.mkdir(results_dir_path)
    filename = args.summary_dir + '_K_' + str(K) +'.txt'
    with open(results_dir_path+filename, 'wb') as file:
        pickle.dump(test_results_dict, file)

    print('average metrics')

    print(
        ' acc : ',
        np.mean(acc_list),
        ' prec : ',
        np.mean(prec_list),
        ' recall : ',
        np.mean(rec_list),
        ' specificity : ',
        np.mean(spec_list),
        ' f1_score : ',
        np.mean(f1_list),
        ' auc_pr : ',
        np.mean(auc_roc_list))

    print('min metrics')

    print(
        ' acc : ',
        np.amin(acc_list),
        ' prec : ',
        np.amin(prec_list),
        ' recall : ',
        np.amin(rec_list),
        ' specificity : ',
        np.amin(spec_list),
        ' f1_score : ',
        np.amin(f1_list),
        ' auc_pr : ',
        np.amin(auc_roc_list))

    print('max metrics')

    print(
        ' acc : ',
        np.amax(acc_list),
        ' prec : ',
        np.amax(prec_list),
        ' recall : ',
        np.amax(rec_list),
        ' specificity : ',
        np.amax(spec_list),
        ' f1_score : ',
        np.amax(f1_list),
        ' auc_pr : ',
        np.amax(auc_roc_list))

    n_test_tasks = len(acc_list)
    print('ci95 metrics - number of test tasks :', n_test_tasks)

    print(
        ' acc : ',
        1.96*np.std(acc_list)/np.sqrt(n_test_tasks),
        ' prec : ',
        1.96*np.std(prec_list)/np.sqrt(n_test_tasks),
        ' recall : ',
        1.96*np.std(rec_list)/np.sqrt(n_test_tasks),
        ' specificity : ',
        1.96*np.std(spec_list)/np.sqrt(n_test_tasks),
        ' f1_score : ',
        1.96*np.std(f1_list)/np.sqrt(n_test_tasks),
        ' auc_pr : ',
        1.96*np.std(auc_roc_list)/np.sqrt(n_test_tasks)
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('supervised learning on one task (one MNIST digit) /' 
                     'purpose: benchmark with Model agnostic meta learning' 
                     ' with MNIST'))
    
    parser.add_argument(
        '-K',
        type=int,
        metavar='',
        help='number of data points sampled for training')
    parser.add_argument(
        '-cir_train',
        type=float,
        metavar='',
        help='percentage of positive examples')
    parser.add_argument(
        '-test_task_idx',
        type=int,
        metavar='',
        help='index of the task to be learned') 
    parser.add_argument(
        '-val_task_idx',
        type=int,
        metavar='',
        help='index of the val task - only needed to load the dataset') 
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
