# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import argparse
import os
import random
import pickle
import json


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(config_file, args_dict):
    with open(config_file) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]

    return args_dict


def load_MNIST(K, cir_inner_loop, test_task_idx, val_task_idx):
    """ load the required MT-MNIST dataset.

    Parameters
    ----------
    K : int
        size of the training set.
    cir_inner_loop : float
        class-imbalance rate of the adaptation/finetuning set.
    test_task_idx : int
        index of the meta-testing task.
    val_task_idx : int
        index of the meta-validation task.


    Returns
    -------
    train_tasks (or resampled_train_tasks): dict
        meta-training tasks.
    val_task : dict
        meta-validation tasks.
    relevant_test_task_finetune_sets : list
        list of different adaptation/finetuning sets sampled from the test task.
    test_task_test_set : dict
        test set.

    """

    base_path = '/home/USER/Documents'
    if (not (os.path.exists(base_path))):
        base_path = '/home/ubuntu/Projects'
    if (not (os.path.exists(base_path))):
        base_path = '/home/USER/Projects'

    data_path = base_path + '/MAML/input_data/MNIST_val' + \
        str(val_task_idx) + '_test' + str(test_task_idx) + '/'
    train_tasks_file = open(
        data_path +
        'train_tasks_val_' +
        str(val_task_idx) +
        '_test_' +
        str(test_task_idx) +
        '.txt',
        'rb')
    train_tasks = pickle.load(train_tasks_file)
    val_task_file = open(
        data_path +
        'val_task_val_' +
        str(val_task_idx) +
        '_test_' +
        str(test_task_idx) +
        '.txt',
        'rb')
    val_task = pickle.load(val_task_file)
    test_task_test_set_file = open(
        data_path +
        'test_task_test_sets_val_' +
        str(val_task_idx) +
        '_test_' +
        str(test_task_idx) +
        '.txt',
        'rb')
    test_task_test_set = pickle.load(test_task_test_set_file)
    test_task_finetune_sets_file = open(
        data_path +
        'test_task_finetune_sets_val_' +
        str(val_task_idx) +
        '_test_' +
        str(test_task_idx) +
        '.txt',
        'rb')
    test_task_finetune_sets = pickle.load(test_task_finetune_sets_file)

    relevant_test_task_finetune_sets = test_task_finetune_sets[str(
        K)][str(cir_inner_loop)]

    if(cir_inner_loop == 1.0):
        # Cir = 1.0 --> Moving the anomalies of the inner loop set of data to
        # the one of the outer loop'

        resampled_train_tasks = {}
        resampled_train_tasks['X_train_inner'], resampled_train_tasks['Y_train_inner'] = [
        ], []
        resampled_train_tasks['X_train_outer'], resampled_train_tasks['Y_train_outer'] = [
        ], []
        for task_idx in range(len(train_tasks['X_train_inner'])):
            normal_indexes_inner_loop = list(
                np.where(train_tasks['Y_train_inner'][task_idx] == 0)[0])
            anomalous_indexes_inner_loop = list(
                np.where(train_tasks['Y_train_inner'][task_idx] == 1)[0])
            resampled_train_tasks['X_train_outer'].append(
                np.concatenate(
                    (train_tasks['X_train_outer'][task_idx],
                     train_tasks['X_train_inner'][task_idx][anomalous_indexes_inner_loop])))
            resampled_train_tasks['Y_train_outer'].append(
                np.concatenate(
                    (train_tasks['Y_train_outer'][task_idx],
                     train_tasks['Y_train_inner'][task_idx][anomalous_indexes_inner_loop])))
            resampled_train_tasks['X_train_inner'].append(
                train_tasks['X_train_inner'][task_idx][normal_indexes_inner_loop])
            resampled_train_tasks['Y_train_inner'].append(
                train_tasks['Y_train_inner'][task_idx][normal_indexes_inner_loop])
        return resampled_train_tasks, val_task, relevant_test_task_finetune_sets, test_task_test_set
    else:
        return train_tasks, val_task, relevant_test_task_finetune_sets, test_task_test_set


def sample_train_val_data(
        train_tasks,
        val_task,
        n_val_finetune_sets,
        sample_tasks_indexes,
        K,
        n_queries,
        cir_inner_loop,
        tr_inner_normal_indexes_list,
        tr_inner_anomalous_indexes_list,
        tr_outer_normal_indexes_list,
        tr_outer_anomalous_indexes_list,
        finetune_val_normal_indexes,
        finetune_val_anomalous_indexes,
        get_val_data):
    """sample the inner and outer loop data from the sampled meta-training tasks as well as the adaptation set(s) from the meta-validation task(s).
       Note that here the adaptation/finetuning sets are sampled so that they have a class-imbalance rate that matches the one of the target task (cir inner loop).

    Parameters
    ----------
    train_tasks : dict
        contains features and labels of datapoints of meta-training tasks.
    val_task : dict
        contains features and labels of datapoints of the meta-validation task(s).
    sample_tasks_indexes : list
        indexes of the meta-training tasks sampled in the current metatraining iteration.
    n_val_finetune_sets : int
        number of adaptation/finetuning sets sampled from the meta-validation task(s).
    K : int
        adaptation/finetuning set size.
    n_queries : int
        size of the data batch sampled from the outer loop data for the meta-update.
    cir_inner_loop : int
        class-imbalance rate of the adaptation set of data.
    tr_inner_normal_indexes_list: list
        indices of the normal examples in the inner loop data of each meta-training task.
    tr_inner_anomalous_indexes_list: list
        indices of the anomalous examples in the inner loop data of each meta-training task.
    tr_outer_normal_indexes_list: list
        indices of the normal examples in the outer loop data of each meta-training task.
    tr_outer_anomalous_indexes_list: list
        indices of the anomalous examples in the outer loop data of each meta-training task.
    finetune_val_normal_indexes: list
        indices of the normal examples in the adaptation/finetuning data of the meta-validation task(s).
    finetune_val_anomalous_indexes: list
        indices of the anomalous examples in the adaptation/finetuning data of the meta-validation task(s).
    get_val_data : bool
        determines whether adaptation/finetuning sets from the meta-validation task(s) should be returned.

    Returns
    -------
    X_train_a_sampled : array
        features of the adaptation set sampled from each sampled meta-training task.
    Y_train_a_sampled : array
        labels of the adaptation set sampled from each sampled meta-training task.
    X_train_b_sampled : array
        features of the outer loop set sampled from each sampled meta-training task.
    Y_train_b_sampled : array
        labels of the outer loop set sampled from each sampled meta-training task.
    K_val_X_sampled_list : list
        features of the adaptation set(s) sampled from the meta-validation task(s).
    K_val_Y_sampled_list : list
        labels of the adaptation set(s) sampled from the meta-validation task(s).

    """

    X_train_a_sampled, X_train_b_sampled = [], []
    Y_train_a_sampled, Y_train_b_sampled = [], []

    n_needed_normal_train_inner = int(K * cir_inner_loop)
    n_needed_anomalous_train_inner = K - n_needed_normal_train_inner

    cir_train_outer = 0.5
    n_needed_normal_train_outer = int(n_queries * cir_train_outer)
    n_needed_anomalous_train_outer = n_queries - n_needed_normal_train_outer

    for task_idx in sample_tasks_indexes:
        task_X_train_a = train_tasks['X_train_inner'][task_idx]
        task_Y_train_a = train_tasks['Y_train_inner'][task_idx]
        K_tr_normal_idxs = random.sample(
            tr_inner_normal_indexes_list[task_idx],
            n_needed_normal_train_inner)
        K_tr_anomalous_idxs = random.sample(
            tr_inner_anomalous_indexes_list[task_idx],
            n_needed_anomalous_train_inner)
        K_tr_idxs = []
        K_tr_idxs += K_tr_normal_idxs
        K_tr_idxs += K_tr_anomalous_idxs
        X_train_a_sampled.append(task_X_train_a[K_tr_idxs])
        Y_train_a_sampled.append(task_Y_train_a[K_tr_idxs])

        task_X_train_b = train_tasks['X_train_outer'][task_idx]
        task_Y_train_b = train_tasks['Y_train_outer'][task_idx]
        K_tr_normal_idxs = random.sample(
            tr_outer_normal_indexes_list[task_idx],
            n_needed_normal_train_outer)
        K_tr_anomalous_idxs = random.sample(
            tr_outer_anomalous_indexes_list[task_idx],
            n_needed_anomalous_train_outer)
        K_tr_idxs = []
        K_tr_idxs += K_tr_normal_idxs
        K_tr_idxs += K_tr_anomalous_idxs
        X_train_b_sampled.append(task_X_train_b[K_tr_idxs])
        Y_train_b_sampled.append(task_Y_train_b[K_tr_idxs])

    X_train_a_sampled = np.array(X_train_a_sampled)
    Y_train_a_sampled = np.array(Y_train_a_sampled)
    X_train_b_sampled = np.array(X_train_b_sampled)
    Y_train_b_sampled = np.array(Y_train_b_sampled)

    # sample K datapoints from the validation task per cir
    K_val_X_sampled_list, K_val_Y_sampled_list = [], []

    if(get_val_data):
        n_needed_normal_val_finetune = int(K * cir_inner_loop)
        n_needed_anomalous_val_finetune = K - n_needed_normal_val_finetune
        for i in range(n_val_finetune_sets):
            K_val_X_sampled, K_val_Y_sampled = [], []
            K_val_normal_idxs = random.sample(
                finetune_val_normal_indexes,
                n_needed_normal_val_finetune)
            K_val_anomalous_idxs = random.sample(
                finetune_val_anomalous_indexes,
                n_needed_anomalous_val_finetune)
            K_val_idxs = []
            K_val_idxs += K_val_normal_idxs
            K_val_idxs += K_val_anomalous_idxs

            K_val_X_sampled, K_val_Y_sampled = val_task["finetune_X"][
                K_val_idxs], val_task["finetune_Y"][K_val_idxs]
            K_val_X_sampled_list.append(K_val_X_sampled)
            K_val_Y_sampled_list.append(K_val_Y_sampled)

    return X_train_a_sampled, Y_train_a_sampled, X_train_b_sampled, Y_train_b_sampled, K_val_X_sampled_list, K_val_Y_sampled_list


def main(args):

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    K = args.K
    cir = args.cir_inner_loop

    train_tasks, val_task, test_task_finetune_sets, test_task_test_set = load_MNIST(
        K, cir, args.test_task_idx, args.val_task_idx)

    tr_inner_normal_indexes_list, tr_inner_anomalous_indexes_list = [], []

    for i in range(len(train_tasks['Y_train_inner'])):
        tr_inner_normal_indexes_list.append(
            list(np.where(train_tasks['Y_train_inner'][i] == 0)[0]))
        tr_inner_anomalous_indexes_list.append(
            list(np.where(train_tasks['Y_train_inner'][i] == 1)[0]))

    tr_outer_normal_indexes_list, tr_outer_anomalous_indexes_list = [], []

    for i in range(len(train_tasks['Y_train_outer'])):
        tr_outer_normal_indexes_list.append(
            list(np.where(train_tasks['Y_train_outer'][i] == 0)[0]))
        tr_outer_anomalous_indexes_list.append(
            list(np.where(train_tasks['Y_train_outer'][i] == 1)[0]))

    finetune_val_normal_indexes = list(
        np.where(val_task['finetune_Y'] == 0)[0])
    finetune_val_anomalous_indexes = list(
        np.where(val_task['finetune_Y'] == 1)[0])

    sess = tf.InteractiveSession()
    input_shape = train_tasks['X_train_inner'][0][0].shape
    n_train_tasks = len(train_tasks['X_train_inner'])
    if('MAML' in args.summary_dir):
        if(args.stop_grad):
            from fomaml_class import FOMAML
            model = FOMAML(sess, args, seed, n_train_tasks, input_shape)

        else:
            from maml_class import MAML
            model = MAML(sess, args, seed, n_train_tasks, input_shape)

    elif('REPTILE' in args.summary_dir):
        from reptile_class import REPTILE
        model = REPTILE(sess, args, seed, n_train_tasks, input_shape)

    else:
        print('model is unknown')
        assert(0)

    summary = False
    if(args.summary_dir):
        summary = True

    if(summary):
        loddir_path = './summaries_MAML'
        if (not (os.path.exists(loddir_path))):
            os.mkdir(loddir_path)
        if (not (os.path.exists(os.path.join(loddir_path, model.summary_dir)))):
            os.mkdir(os.path.join(loddir_path, model.summary_dir))

        train_writer = tf.summary.FileWriter(
            os.path.join(loddir_path, model.summary_dir) + '/train')
        val_writer = tf.summary.FileWriter(
            os.path.join(loddir_path, model.summary_dir) + '/val')
        val_tags = [
            'val_loss_avg',
            'val_acc_avg',
            'val_precision_avg',
            'val_recall_avg',
            'val_specificity_avg',
            'val_f1_score_avg',
            'val_auc_pr_avg']

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # intialization
    val_test_loss = 0
    min_val_epoch = -1
    min_val_test_loss = 10000
    min_metatrain_epoch = -1
    min_metatrain_loss = 10000

    n_val_finetune_sets = 10
    val_interval = 100

    for epoch in range(args.meta_epochs):
        sample_tasks_indexes = random.sample(
            range(0, model.n_train_tasks), model.n_sample_tasks)
        get_val_data = (
            (epoch %
             val_interval == 0) or (
                epoch == args.meta_epochs -
                1))
        X_train_a, Y_train_a, X_train_b, Y_train_b, K_val_X_list, K_val_Y_list = sample_train_val_data(train_tasks, val_task, n_val_finetune_sets, sample_tasks_indexes, model.K, model.n_queries, cir,
                                                                                                       tr_inner_normal_indexes_list, tr_inner_anomalous_indexes_list, tr_outer_normal_indexes_list, tr_outer_anomalous_indexes_list, finetune_val_normal_indexes, finetune_val_anomalous_indexes, get_val_data)

        metatrain_loss, train_summaries = model.metatrain_op(
            epoch, X_train_a, Y_train_a, X_train_b, Y_train_b)

        if(metatrain_loss < min_metatrain_loss):
            min_metatrain_loss = metatrain_loss
            min_metatrain_epoch = epoch

        if(summary and (epoch % model.summary_interval == 0)):
            train_writer.add_summary(train_summaries, epoch)
            train_writer.flush()

        if((epoch % val_interval == 0) or (epoch == args.meta_epochs - 1)):
            val_metrics_list = []

            for K_val_X, K_val_Y in zip(K_val_X_list, K_val_Y_list):
                val_loss, val_test_loss, acc, precision, recall, specificity, f1_score, auc_pr = model.val_op(
                    K_val_X, K_val_Y, val_task['test_X'], val_task['test_Y'])
                val_metrics_list.append(
                    [val_test_loss, acc, precision, recall, specificity, f1_score, auc_pr])

            avg_val_metrics = np.mean(val_metrics_list, axis=0)
            if(avg_val_metrics[0] < min_val_test_loss):
                model.saver.save(
                    model.sess,
                    model.checkpoint_path +
                    model.summary_dir +
                    "_restore_val_test_loss/model.ckpt")
                min_val_test_loss = avg_val_metrics[0]
                min_val_epoch = epoch
                print('model saved ',
                      ' epoch: ', epoch,
                      ' val_test_loss: ', avg_val_metrics[0],
                      ' acc : ', avg_val_metrics[1],
                      ' prec : ', avg_val_metrics[2],
                      ' recall : ', avg_val_metrics[3],
                      ' spec : ', avg_val_metrics[4],
                      ' F1 : ', avg_val_metrics[5],
                      ' auc_pr : ', avg_val_metrics[6])

            if(summary):
                val_summaries = []
                for i in range(len(avg_val_metrics)):
                    val_summaries.append(
                        tf.Summary(
                            value=[
                                tf.Summary.Value(
                                    tag=val_tags[i],
                                    simple_value=avg_val_metrics[i]),
                            ]))
                for smr in val_summaries:
                    val_writer.add_summary(smr, epoch)
                val_writer.flush()

    if(summary):
        train_writer.close()
        val_writer.close()

    # meta-testing
    test_feed_dict = {
        model.X_finetune: test_task_test_set["test_X"],
        model.Y_finetune: test_task_test_set["test_Y"]}
    loss_list, acc_list, prec_list, rec_list, spec_list, f1_list, auc_pr_list = [
    ], [], [], [], [], [], []

    for finetune_set_index, finetune_set in enumerate(test_task_finetune_sets):

        model.saver.restore(
            model.sess,
            model.checkpoint_path +
            model.summary_dir +
            "_restore_val_test_loss/model.ckpt")

        if(summary):
            test_writer_restore_val = tf.summary.FileWriter(
                os.path.join(
                    loddir_path,
                    args.summary_dir) +
                '/test_restore_val_' +
                str(finetune_set_index))
        K_finetune_X, K_finetune_Y = finetune_set["finetune_X"], finetune_set["finetune_Y"]

        if(finetune_set_index == 0):
            if(model.bn):
                # compute BN stats based on the available adaptation/finetuning
                # set and assign them to the BN layers
                sess.run(
                    model.updated_bn_model, {
                        model.X_finetune: K_finetune_X})
            if(summary):
                sess.run(tf.local_variables_initializer())
                test_summaries = model.sess.run(
                    model.merged_test_restore_val, test_feed_dict)
                test_writer_restore_val.add_summary(test_summaries, 0)
                test_writer_restore_val.flush()

        for finetune_epoch in range(1, model.num_updates + 1):

            model.finetune_op(K_finetune_X, K_finetune_Y)
            if(finetune_epoch == model.num_updates):
                if(model.bn):
                    sess.run(
                        model.updated_bn_model, {
                            model.X_finetune: K_finetune_X})
                sess.run(tf.local_variables_initializer())
                test_summaries, test_loss, acc, precision, recall, specificity, f1_score, auc_pr = model.sess.run(
                    [model.merged_test_restore_val, model.test_loss, model.my_acc, model.my_precision, model.my_recall, model.my_specificity, model.my_f1_score, model.my_auc_pr], feed_dict=test_feed_dict)
                loss_list.append(test_loss)
                acc_list.append(acc)
                prec_list.append(precision)
                rec_list.append(recall)
                spec_list.append(specificity)
                f1_list.append(f1_score)
                auc_pr_list.append(auc_pr)

                if(summary):
                    test_writer_restore_val.add_summary(
                        test_summaries, finetune_epoch)
                    test_writer_restore_val.flush()

        if(summary):
            test_writer_restore_val.close()

    test_results_dict = {}
    test_results_dict['test_loss'] = loss_list
    test_results_dict['acc'] = acc_list
    test_results_dict['prec'] = prec_list
    test_results_dict['rec'] = rec_list
    test_results_dict['spec'] = spec_list
    test_results_dict['f1'] = f1_list
    test_results_dict['auc_pr'] = auc_pr_list

    results_dir_path = './results/'
    if (not (os.path.exists(results_dir_path))):
        os.mkdir(results_dir_path)
    filename = args.summary_dir + '.txt'
    with open(results_dir_path + filename, 'wb') as file:
        pickle.dump(test_results_dict, file)

    print('average metrics')

    print(
        ' test_loss : ',
        np.mean(loss_list),
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
        np.mean(auc_pr_list))

    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Model agnostic meta learning with MNIST')
    parser.add_argument(
        '-filters',
        type=str,
        metavar='',
        help='number of filters for each convolutional layer e.g. "32 32 32 32"')
    parser.add_argument(
        '-kernel_sizes',
        type=str,
        metavar='',
        help='kernel sizes for each convolutional layer e.g. "3 3 3 3"')
    parser.add_argument(
        '-dense_layers',
        type=str,
        metavar='',
        help='size of each dense layer of the model e.g. "256 128 64 64"')
    parser.add_argument(
        '-meta_lr',
        type=float,
        metavar='',
        help=('learning rate of the optimizer used for meta-training'
              '(beta in original paper)'))
    parser.add_argument(
        '-lr',
        type=float,
        metavar='',
        help=('learning rate for the first update for each task in '
              'a meta-training step (alpha in original paper)'
              ', also used for finetuning'))
    parser.add_argument(
        '-meta_epochs',
        type=int,
        metavar='',
        help='number of meta-training epochs')
    parser.add_argument(
        '-K',
        type=int,
        metavar='',
        help='number of data points sampled for training and testing')
    parser.add_argument(
        '-cir_inner_loop',
        type=float,
        metavar='',
        help=(
            'percentage of positive examples of the dataset used for the inner'
            ' updatefor each training task'))
    parser.add_argument(
        '-num_updates',
        type=int,
        metavar='',
        help=('number of parameter updates used to compute '
              'theta prime (new_weights in code)'))
    parser.add_argument(
        '-stop_grad',
        type=str,
        metavar='',
        help=('Set to True if the computation of second  derivatives '
              'when  backpropagating the  meta-gradient should be prevented'))
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
        '-seed',
        type=int,
        metavar='',
        help='seed')
    parser.add_argument(
        '-summary_dir',
        type=str,
        metavar='',
        help=('name of the doirectory where the summaries should be saved. '
              'set to False, if summaries are not needed '))
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
    assert(args.num_updates > 0), ("at least one update is needed")

    main(args)
