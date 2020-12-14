# -*- coding: utf-8 -*-
import numpy as np
import random
import os
import random

from sklearn import svm

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import roc_auc_score

# code adapted from ”Deep One-Class Classification” ICML 2018 (Ruff, Lukas et al.)


def initialize_ocsvm(kernel, nu, gamma='scale', **kwargs):

    if kernel in ('linear', 'poly', 'rbf', 'sigmoid'):
        kernel = kernel
    else:
        kernel = 'precomputed'

    ocsvm = svm.OneClassSVM(kernel=kernel, nu=nu, gamma=gamma, **kwargs)
    return ocsvm


def train(
        ocsvm,
        X_train,
        X_test,
        Y_test,
        kernel,
        nu,
        GridSearch=True,
        **kwargs):

    if X_train.ndim > 2:
        X_train_shape = X_train.shape
        X_train = X_train.reshape(X_train_shape[0], np.prod(X_train_shape[1:]))
    else:
        X_train = X_train

    if GridSearch and kernel == 'rbf':

        n_test_set = len(X_test)
        n_val_set = int(0.5 * n_test_set)
        n_test_out = 0
        n_test_norm = 0
        n_val_out = 0
        n_val_norm = 0
        count = 0
        while (
            n_test_out == 0) | (
            n_test_norm == 0) | (
            n_val_out == 0) | (
                n_val_norm == 0):

            perm = np.random.permutation(n_test_set)
            X_val = X_test[perm[:n_val_set]]
            y_val = Y_test[perm[:n_val_set]]
            # only accept small test set if AUC can be computed on val and test
            # set
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
        for gamma in np.logspace(-10, -1, num=4, base=2):

            # train on selected gamma
            cv_svm = svm.OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
            cv_svm.fit(X_train)

            # predict on small hold-out set
            val_acc, _, _, _, val_f1_score, val_auc_roc = predict(
                cv_svm, X_val, y_val, kernel)

            # save model if ACC on hold-out set improved
            if val_acc > cv_acc:
                ocsvm = cv_svm
                g_best = gamma
                cv_acc = val_acc
                cv_f1 = val_f1_score

        oc_svm = svm.OneClassSVM(kernel='rbf', nu=nu, gamma=g_best)

        ocsvm.fit(X_train)
        best_hp = g_best

    elif GridSearch and kernel == 'linear':

        n_test_set = len(X_test)
        n_val_set = int(0.5 * n_test_set)
        n_test_out = 0
        n_test_norm = 0
        n_val_out = 0
        n_val_norm = 0
        count = 0
        while (
            n_test_out == 0) | (
            n_test_norm == 0) | (
            n_val_out == 0) | (
                n_val_norm == 0):

            perm = np.random.permutation(n_test_set)
            X_val = X_test[perm[:n_val_set]]
            y_val = Y_test[perm[:n_val_set]]
            # only accept small test set if AUC can be computed on val and test
            # set
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

        nu_best = 0.1
        for nu in [0.99, 0.9, 0.5, 0.25, 0.1, 0.01]:

            # train on selected gamma
            cv_svm = svm.OneClassSVM(kernel='linear', nu=nu)
            cv_svm.fit(X_train)

            # predict on small hold-out set
            val_acc, _, _, _, val_f1_score, val_auc_roc = predict(
                cv_svm, X_val, y_val, kernel)

            # save model if ACC on hold-out set improved
            if val_acc > cv_acc:
                ocsvm = cv_svm
                nu_best = nu
                cv_acc = val_acc
                cv_f1 = val_f1_score

        oc_svm = svm.OneClassSVM(kernel='linear', nu=nu_best)

        ocsvm.fit(X_train)
        best_hp = nu_best

    else:

        best_hp = 0
        ocsvm.fit(X_train)

    return ocsvm, best_hp


def predict(ocsvm, X, y, kernel, **kwargs):

    # reshape to 2D if input is tensor
    if X.ndim > 2:
        X_shape = X.shape
        X = X.reshape(X_shape[0], np.prod(X_shape[1:]))

    scores = (-1.0) * ocsvm.decision_function(X)
    y_pred = ocsvm.predict(X)

    y_pred[y_pred == 1.0] = 0.0
    y_pred[y_pred == -1.0] = 1.0

    scores_flattened = scores.flatten()
    acc = 100.0 * sum(y == y_pred) / len(y)

    TP = np.count_nonzero(y_pred * y)
    TN = np.count_nonzero((y_pred - 1) * (y - 1))
    FP = np.count_nonzero(y_pred * (y - 1))
    FN = np.count_nonzero((y_pred - 1) * y)

    if(TP + FP == 0):
        prec = 0.0
    else:
        prec = TP / (TP + FP)

    rec = TP / (TP + FN)
    spec = TN / (TN + FP)

    if(prec + rec == 0):
        f1_score = 0.0
    else:
        f1_score = 2 * prec * rec / (prec + rec)

    auc_roc = roc_auc_score(y, scores.flatten())

    return acc, prec, rec, spec, f1_score, auc_roc


class OCSVM:

    """

    A class for oc svm.

    """

    def __init__(self, seed, linear_only=False, GridSearch=True, nu=0.1):

        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.linear_only = linear_only
        self.GridSearch = GridSearch
        self.nu = nu

    def val_op(
            self,
            X_val_finetune,
            Y_val_finetune,
            val_task_test_X,
            val_task_test_Y):

        # We use OC-SVM with the linear kernel in our experiments

        if(self.linear_only):

            ocsvm = initialize_ocsvm('linear', self.nu)
            ocsvm, nu_best = train(ocsvm, X_val_finetune, val_task_test_X, np.squeeze(
                val_task_test_Y), 'linear', self.nu, self.GridSearch)
            acc_l, prec_l, rec_l, spec_l, f1_l, auc_roc_l = predict(
                ocsvm, val_task_test_X, np.squeeze(val_task_test_Y), 'linear')

            return acc_l, prec_l, rec_l, spec_l, f1_l, auc_roc_l, False, nu_best

        GridSearch = True
        nu = 0.1
        ocsvm = initialize_ocsvm('linear', nu)
        ocsvm, nu_best = train(ocsvm, X_val_finetune, val_task_test_X, np.squeeze(
            val_task_test_Y), 'linear', nu, GridSearch)
        acc_l, prec_l, rec_l, spec_l, f1_l, auc_roc_l = predict(
            ocsvm, val_task_test_X, np.squeeze(val_task_test_Y), 'linear')

        kernel = 'rbf'
        gamma = 'scale'
        nu = nu_best

        ocsvm = initialize_ocsvm(kernel, nu, gamma)
        ocsvm, g_best = train(ocsvm, X_val_finetune, val_task_test_X, np.squeeze(
            val_task_test_Y), kernel, nu, GridSearch)
        acc, prec, rec, spec, f1, auc_roc = predict(
            ocsvm, val_task_test_X, np.squeeze(val_task_test_Y), kernel)

        if(acc > acc_l):
            return acc, prec, rec, spec, f1, auc_roc, True, g_best
        else:
            return acc_l, prec_l, rec_l, spec_l, f1_l, auc_roc_l, False, nu_best
