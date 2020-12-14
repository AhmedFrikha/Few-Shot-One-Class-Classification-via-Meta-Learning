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
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances

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

def initialize_ocsvm(kernel, nu, gamma, **kwargs):

    if kernel in ('linear', 'poly', 'rbf', 'sigmoid'):
        kernel = kernel
    else:
        kernel = 'precomputed'

    ocsvm = svm.OneClassSVM(kernel=kernel, nu=nu, gamma=gamma,**kwargs)
    return ocsvm


def train(ocsvm, X_train, X_test, Y_test, kernel, nu, GridSearch=True, **kwargs):

    if X_train.ndim > 2:
        X_train_shape = X_train.shape
        X_train = X_train.reshape(X_train_shape[0], np.prod(X_train_shape[1:]))
    else:
        X_train = X_train

    if kernel in ('DegreeKernel', 'WeightedDegreeKernel'):
        # get_kernel_matrix(kernel=kernel, X_train=X_train, **kwargs)
        # svm.fit(K_train)
        print('unexpected behaviour')
    else:
        if GridSearch and kernel == 'rbf':

            # use grid search cross-validation to select gamma
            # print("Using GridSearchCV for hyperparameter selection...")

            # sample small hold-out set from test set for hyperparameter selection. Save as val set.
            
            n_test_set = len(X_test)
            n_val_set = int(0.1 * n_test_set)
            n_test_out = 0
            n_test_norm = 0
            n_val_out = 0
            n_val_norm = 0
            while (n_test_out == 0) | (n_test_norm == 0) | (n_val_out == 0) | (n_val_norm ==0):
                perm = np.random.permutation(n_test_set)
                X_val = X_test[perm[:n_val_set]]
                y_val = Y_test[perm[:n_val_set]]
                # only accept small test set if AUC can be computed on val and test set
                n_test_out = np.sum(Y_test[perm[:n_val_set]])
                n_test_norm = np.sum(Y_test[perm[:n_val_set]] == 0)
                n_val_out = np.sum(Y_test[perm[n_val_set:]])
                n_val_norm = np.sum(Y_test[perm[n_val_set:]] == 0)

            X_test = X_test[perm[n_val_set:]]
            Y_test = Y_test[perm[n_val_set:]]
            n_val = len(y_val)
            n_test_set = len(Y_test)

            val_scores = np.zeros((len(y_val), 1))
            test_scores = np.zeros((len(Y_test), 1))

            cv_auc = 0.0
            cv_acc = 0
            cv_f1 = 0

            g_best = 0.1
            for gamma in np.logspace(-10, -1, num=10, base=2):

                # train on selected gamma
                cv_svm = svm.OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
                cv_svm.fit(X_train)

                # predict on small hold-out set
                val_acc, _, _, _, val_f1_score, val_auc_roc = predict(cv_svm, X_val, y_val, kernel)

                # save model if AUC on hold-out set improved
                if val_f1_score > cv_f1:
 #                   print('gamma set to: ', g_best)
                    ocsvm = cv_svm
                    g_best = gamma
                    cv_auc = val_auc_roc
                    cv_f1 = val_f1_score

            # save results of best cv run
            # diag['val']['auc'] = cv_auc
            # diag['val']['acc'] = cv_acc

            oc_svm = svm.OneClassSVM(kernel='rbf', nu=nu, gamma=g_best)
 

            ocsvm.fit(X_train)


        else:
            # if rbf-kernel, re-initialize svm with gamma minimizing the
            # numerical error
            if kernel == 'rbf':
                gamma = 1 / (np.max(pairwise_distances(X_train)) ** 2)
                # ocsvm = svm.OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)

            ocsvm.fit(X_train)
            gamma = gamma

    return ocsvm



def predict(ocsvm, X, y, kernel, **kwargs):

    # reshape to 2D if input is tensor
    if X.ndim > 2:
        X_shape = X.shape
        X = X.reshape(X_shape[0], np.prod(X_shape[1:]))

    if kernel in ('DegreeKernel', 'WeightedDegreeKernel'):
        # get_kernel_matrix(kernel=kernel, which_set=which_set, **kwargs)
        # if which_set == 'train':
        #     scores = (-1.0) * ocsvm.decision_function(K_train)
        #     y_pred = (ocsvm.predict(K_train) == -1) * 1
        # if which_set == 'test':
        #     scores = (-1.0) * ocsvm.decision_function(K_test)
        #     y_pred = (ocsvm.predict(K_test) == -1) * 1
        print('unexpected behaviour')

    else:
        scores = (-1.0) * ocsvm.decision_function(X)
        y_pred = ocsvm.predict(X)

        y_pred[y_pred == 1.0] = 0.0
        y_pred[y_pred == -1.0] = 1.0

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
        
    # if which_set == 'test':
    #     rho = -svm.intercept_[0]

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


    kernel = 'rbf' 
    nu = 0.1
    GridSearch = True
    gamma = 'scale'

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
        ocsvm = initialize_ocsvm(kernel, nu, gamma)
        
        ocsvm = train(ocsvm, finetune_X, test_X, np.squeeze(test_task['Y_outer']), kernel, nu, GridSearch)
        # print('decision was made for gamma = ', ocsvm.gamma)
        acc, prec, rec, spec, f1_score, auc_roc = predict(ocsvm, test_X, np.squeeze(test_task['Y_outer']), kernel)
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
