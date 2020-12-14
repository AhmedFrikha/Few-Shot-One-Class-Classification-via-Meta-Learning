# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import argparse
import os
import random
import pickle
import json
from miniimagenet_tasks import create_miniimagenet_task_distribution
from omniglot_tasks import create_omniglot_allcharacters_task_distribution
from cifarfs_tasks import create_cifarfs_task_distribution
from fc100_tasks import create_fc100_task_distribution

import time


# set to True to perform a hyperparameter search (intervals can be
# determined below)
Ray_tune = False


tf.logging.set_verbosity(tf.logging.ERROR)


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(config_file, args_dict):
    with open(config_file) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]

    return args_dict


def main(args):

    if(Ray_tune):
        for key in list(args.keys()):
            if str(args[key]).lower() == "true":
                args[key] = True
            elif str(args[key]).lower() == "false":
                args[key] = False
        print(args)
        args = Bunch(args)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    cir = args.cir_inner_loop
    K = args.K
    if(cir == 0.5):
        train_occ = False
        num_training_samples_per_class = int(K / 2)
    elif(cir == 1.0):
        train_occ = True
        num_training_samples_per_class = K
    else:
        print('cir between 0.5 and 1.0 not implemented')
        assert(0)
    test_occ = True

    if(Ray_tune):
        args.summary_dir = args.summary_dir + '_K_' + str(K) + '_lr_' + str(
            args.lr) + '_updts_' + str(args.num_updates) + '_q_' + str(args.n_queries)
        args.summary_dir = args.summary_dir.replace(".", "_")
        if(args.bn):
            args.summary_dir += '_bn'
        args.summary_dir = args.dataset + '_' + args.summary_dir

    base_path = "/home/USER/Documents/"
    if not (os.path.exists(base_path)):
        base_path = "/home/ubuntu/Projects/"
    if not (os.path.exists(base_path)):
        base_path = "/home/USER/Projects/"
    basefolder = base_path + "MAML/raw_data/"

    if(args.dataset == 'MIN'):
        metatrain_task_distribution, metaval_task_distribution, metatest_task_distribution = create_miniimagenet_task_distribution(
            basefolder + "miniImageNet_data/miniimagenet.pkl",
            train_occ=train_occ,
            test_occ=test_occ,
            num_training_samples_per_class=num_training_samples_per_class,
            num_test_samples_per_class=int(args.n_queries / 2),
            num_training_classes=2,
            meta_batch_size=8,
            seq_length=0

        )
    elif(args.dataset == 'OMN'):
        metatrain_task_distribution, metaval_task_distribution, metatest_task_distribution = create_omniglot_allcharacters_task_distribution(
            basefolder + "omniglot/omniglot.pkl",
            train_occ=train_occ,
            test_occ=test_occ,
            num_training_samples_per_class=num_training_samples_per_class,
            num_test_samples_per_class=int(args.n_queries / 2),
            num_training_classes=2,
            meta_batch_size=8,
            seq_length=0
        )
    elif(args.dataset == 'CIFAR_FS'):
        metatrain_task_distribution, metaval_task_distribution, metatest_task_distribution = create_cifarfs_task_distribution(
            base_path + "MAML/cifar_fc100/data/CIFAR_FS/CIFAR_FS_train.pickle",
            base_path + "MAML/cifar_fc100/data/CIFAR_FS/CIFAR_FS_val.pickle",
            base_path + "MAML/cifar_fc100/data/CIFAR_FS/CIFAR_FS_test.pickle",

            train_occ=train_occ,
            test_occ=test_occ,
            num_training_samples_per_class=num_training_samples_per_class,
            num_test_samples_per_class=int(args.n_queries / 2),
            num_training_classes=2,
            meta_batch_size=8,
            seq_length=0

        )
    elif(args.dataset == 'FC100'):
        metatrain_task_distribution, metaval_task_distribution, metatest_task_distribution = create_fc100_task_distribution(
            base_path + "MAML/cifar_fc100/data/FC100/FC100_train.pickle",
            base_path + "MAML/cifar_fc100/data/FC100/FC100_val.pickle",
            base_path + "MAML/cifar_fc100/data/FC100/FC100_test.pickle",

            train_occ=train_occ,
            test_occ=test_occ,
            num_training_samples_per_class=num_training_samples_per_class,
            num_test_samples_per_class=int(args.n_queries / 2),
            num_training_classes=2,
            meta_batch_size=8,
            seq_length=0

        )

    sess = tf.InteractiveSession()
    meta_batch = metatrain_task_distribution.sample_batch()
    input_shape = meta_batch[0].get_train_set()[0][0].shape
    if('MAML' in args.summary_dir or 'ANIL' in args.summary_dir):
        if(args.stop_grad):
            from fomaml_class import FOMAML
            model = FOMAML(sess, args, seed, 64, input_shape)

        else:
            from maml_class import MAML
            model = MAML(sess, args, seed, 64, input_shape)

    else:
        print('model is unknown')
        assert(0)

    summary = False
    if(args.summary_dir):
        summary = True

    if(summary):
        loddir_path = './summaries_CIMAML'
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
    n_val_tasks_sampled = 2
    val_test_loss = 0
    min_val_epoch = -1
    min_val_test_loss = 10000
    min_metatrain_epoch = -1
    min_metatrain_loss = 10000
    val_interval = 100

    max_val_acc = -1
    max_val_f1 = -1

    if(Ray_tune):
        track.init()
    for epoch in range(args.meta_epochs):
        if((epoch % val_interval == 0) or (epoch == args.meta_epochs - 1)):
            val_metrics_list = []
            analysis_list = []
            for _ in range(n_val_tasks_sampled):
                val_meta_batch = metaval_task_distribution.sample_batch()

                X_val_a, Y_val_a, X_val_b, Y_val_b = [], [], [], []
                for task in val_meta_batch:
                    X_val_a.append(task.get_train_set()[0])
                    Y_val_a.append(np.expand_dims(task.get_train_set()[1], -1))
                    X_val_b.append(task.get_test_set()[0])
                    Y_val_b.append(np.expand_dims(task.get_test_set()[1], -1))

                for K_val_X, K_val_Y, test_val_X, test_val_Y in zip(
                        X_val_a, Y_val_a, X_val_b, Y_val_b):
                    val_summaries, val_test_loss, acc, precision, recall, specificity, f1_score, auc_pr = model.val_op(
                        K_val_X, K_val_Y, test_val_X, test_val_Y)
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
                print('model saved',
                      ' epoch: ', epoch,
                      ' val_test_loss: ', avg_val_metrics[0],
                      ' acc : ', avg_val_metrics[1],
                      ' prec : ', avg_val_metrics[2],
                      ' recall : ', avg_val_metrics[3],
                      ' spec : ', avg_val_metrics[4],
                      ' F1 : ', avg_val_metrics[5],
                      ' auc_pr : ', avg_val_metrics[6])

            if(avg_val_metrics[1] > max_val_acc):
                max_val_acc = avg_val_metrics[1]
            if(avg_val_metrics[5] > max_val_f1):
                max_val_f1 = avg_val_metrics[5]
            if(Ray_tune):
                track.log(
                    mean_loss=min_val_test_loss,
                    mean_accuracy=max_val_acc,
                    f1_score=max_val_f1,
                    training_iteration=epoch)

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

        meta_batch = metatrain_task_distribution.sample_batch()
        X_train_a, Y_train_a, X_train_b, Y_train_b = [], [], [], []
        for task in meta_batch:
            X_train_a.append(task.get_train_set()[0])
            Y_train_a.append(np.expand_dims(task.get_train_set()[1], -1))
            X_train_b.append(task.get_test_set()[0])
            Y_train_b.append(np.expand_dims(task.get_test_set()[1], -1))

        X_train_a = np.array(X_train_a)
        Y_train_a = np.array(Y_train_a)
        X_train_b = np.array(X_train_b)
        Y_train_b = np.array(Y_train_b)

        metatrain_loss, train_summaries = model.metatrain_op(
            epoch, X_train_a, Y_train_a, X_train_b, Y_train_b)

        if(min_metatrain_loss > metatrain_loss):
            min_metatrain_loss = metatrain_loss
        if(summary and (epoch % model.summary_interval == 0)):
            train_writer.add_summary(train_summaries, epoch)
            train_writer.flush()

    if(summary):
        train_writer.close()
        val_writer.close()

    if(not(Ray_tune)):
        # this test on n_test_ tasks * 8 test tasks
        n_test_tasks = 100
        test_metrics_list = []
        model.saver.restore(
            model.sess,
            model.checkpoint_path +
            model.summary_dir +
            "_restore_val_test_loss/model.ckpt")
        print('training ended, restored best model')
        for _ in range(n_test_tasks):
            test_meta_batch = metatest_task_distribution.sample_batch()

            X_test_a, Y_test_a, X_test_b, Y_test_b = [], [], [], []
            for task in test_meta_batch:
                X_test_a.append(task.get_train_set()[0])
                Y_test_a.append(np.expand_dims(task.get_train_set()[1], -1))
                X_test_b.append(task.get_test_set()[0])
                Y_test_b.append(np.expand_dims(task.get_test_set()[1], -1))

            for K_test_X, K_test_Y, test_test_X, test_test_Y in zip(
                    X_test_a, Y_test_a, X_test_b, Y_test_b):
                test_summaries, test_test_loss, acc, precision, recall, specificity, f1_score, auc_pr = model.val_op(
                    K_test_X, K_test_Y, test_test_X, test_test_Y)
                test_metrics_list.append(
                    [test_test_loss, acc, precision, recall, specificity, f1_score, auc_pr])

        avg_test_metrics = np.mean(test_metrics_list, axis=0)

        print('+++ Test metrics - loss: ', avg_test_metrics[0],
              ' acc : ', avg_test_metrics[1],
              ' prec : ', avg_test_metrics[2],
              ' recall : ', avg_test_metrics[3],
              ' spec : ', avg_test_metrics[4],
              ' F1 : ', avg_test_metrics[5],
              ' auc_pr : ', avg_test_metrics[6])

    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='One-Class Model agnostic meta learning')

    parser.add_argument('-config_file',
                        type=str,
                        default="None")

    if(Ray_tune):
        parser.add_argument('-hps',
                            type=str,
                            default="None")

        args = parser.parse_args()
        print(args)

        import ray
        from ray import tune

        from ray.tune import track
        from ray.tune.schedulers import AsyncHyperBandScheduler

        config = {}

        # example for a hyperparameter search
        config['CIFAR_ocmaml_BN'] = {
            "dataset": "CIFAR_FS",
            "filters": "32 32 32 32",
            "kernel_sizes": "3 3 3 3",
            "dense_layers": "",
            "meta_lr": 0.001,
            "lr": tune.grid_search([0.01, 0.002, 0.05]),
            "num_updates": tune.grid_search([10, 5, 3]),
            "bn": "True",
            "meta_epochs": 10000,
            "K": 2,
            "cir_inner_loop": 1.0,
            "n_queries": tune.grid_search([30, 60]),
            "stop_grad": "False",
            "anil": "False",
            "seed": 1,
            "summary_dir": "CIFAR_FS_OCMAML_BN_"
        }

        ray.init()

        ahb = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            metric="mean_loss",
            mode="min",
            grace_period=int(1000),
            max_t=int(20000))

        analysis = tune.run(
            main, config=config[args.hps],
            resources_per_trial={"cpu": 3, "gpu": 1},
            scheduler=ahb)

        df = analysis.dataframe()

        results_dir_path = './results/'
        if (not (os.path.exists(results_dir_path))):
            os.mkdir(results_dir_path)
        filename = args.hps + '_ray_results_df.pkl'
        with open(results_dir_path + filename, 'wb') as file:
            pickle.dump(df, file)
        filename = args.hps + '_ray_results_analysis.pkl'
        with open(results_dir_path + filename, 'wb') as file:
            pickle.dump(analysis, file)

        final_results = df.sort_values(by=['mean_loss'], ascending=True)[
            ['mean_loss', 'mean_accuracy', 'config/num_updates', 'config/lr', 'config/n_queries']]

        print('**** final_results of the hyperparameter search **** ')
        print(final_results)

    else:

        args = parser.parse_args()

        args_dict = vars(args)
        if args.config_file is not "None":
            args_dict = extract_args_from_json(args.config_file, args_dict)

        for key in list(args_dict.keys()):

            if str(args_dict[key]).lower() == "true":
                args_dict[key] = True
            elif str(args_dict[key]).lower() == "false":
                args_dict[key] = False

        print(args)
        args = Bunch(args_dict)
        main(args)
