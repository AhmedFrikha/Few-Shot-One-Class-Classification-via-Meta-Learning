# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import argparse
import os
import random 
import pickle
import json
from imblearn.over_sampling import RandomOverSampler
from mtl_class import MTL



class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(config_file, args_dict):
    with open(config_file) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]

    return args_dict


def load_STS():

    """ load the STS sawtooth dataset.

        Returns
        -------
        mtl_train_tasks : list
            training tasks.
        val_tasks : list
            validation tasks.
        test_task_finetune_sets : list
            list of different training sets sampled from each test task.
        test_task_test_set : list
            test sets of the test tasks.

    """

    base_path = '/home/USER/Documents'
    if (not (os.path.exists(base_path))):
        base_path = '/home/ubuntu/Projects' 
    if (not (os.path.exists(base_path))):
        base_path = '/home/USER/Projects' 
    dataset_id = 'sawtooth'
    data_path = base_path + '/MAML/input_data/new_SSR/'+dataset_id+'_SSR/'
    
    train_tasks_file = open(data_path + 'train_tasks.txt', 'rb')
    train_tasks = pickle.load(train_tasks_file)
    
    val_tasks_file = open(data_path + 'val_tasks.txt', 'rb')
    val_tasks = pickle.load(val_tasks_file)

    test_tasks_test_sets_file = open(data_path + 'test_tasks_test_sets.txt', 'rb')
    test_tasks_test_sets = pickle.load(test_tasks_test_sets_file)

    test_tasks_finetune_sets_file = open(data_path + 'test_tasks_finetune_sets.txt', 'rb')
    test_tasks_finetune_sets = pickle.load(test_tasks_finetune_sets_file)

    mtl_train_tasks = []
    for train_task_idx in range(len(train_tasks)):
        mtl_train_task = {}
        mtl_train_task['X'] = np.concatenate((train_tasks[train_task_idx]['X_inner'], train_tasks[train_task_idx]['X_outer']), axis=0)
        mtl_train_task['Y'] = np.concatenate((train_tasks[train_task_idx]['Y_inner'], train_tasks[train_task_idx]['Y_outer']), axis=0)
        mtl_train_tasks.append(mtl_train_task)

    return mtl_train_tasks, val_tasks, test_tasks_finetune_sets, test_tasks_test_sets 




def sample_random_train_batch(train_tasks, batch_size):
    """sample a batch from each sampled training task.

    Parameters
    ----------
    train_tasks : dict
        contains features and labels of datapoints of all training tasks.
    batch_size : int
        batch size.

    Returns
    -------
    X_batch : numpy array
        features of the batch sampled from each training task.
    Y_batch : numpy array
        labels of the batch sampled from each training task.

    """

    X_train_sampled = []
    Y_train_sampled = []

    for task_idx in range(len(train_tasks)):
        task_X_train = train_tasks[task_idx]['X']
        task_Y_train = train_tasks[task_idx]['Y']
        sampled_tr_idxs = random.sample(range(0, len(task_X_train)), batch_size)
        X_train_sampled.append(task_X_train[sampled_tr_idxs])
        Y_train_sampled.append(task_Y_train[sampled_tr_idxs])

    X_batch = np.array(X_train_sampled)
    Y_batch = np.array(Y_train_sampled)


    return X_batch, Y_batch


def sample_random_val_finetune_data(val_tasks, K, cir, val_normal_indexes, val_anomalous_indexes):
    """samples K datapoints from the validation task.

    Parameters
    ----------
    val_tasks : list
        contains the data of the validation tasks.
    K : int
        size of the finetuning set.
    cir : int
        class-imbalance rate (cir) of the target task (and therefore we sample the finetuning sets of the val tasks to have this same cir).
    val_normal_indexes : list
        indices of normal data samples of the validation task
    val_anomalous_indexes : list
        indices of anomalous data samples of the validation task

    Returns
    -------
    val_X_sampled : array
        features of the K datapoints sampled from the validation task 
        in the current multitask learning iteration.
    val_Y_sampled : array
        labels of the K datapoints sampled from the validation task 
        in the current multitask learning iteration.

    """

    n_needed_normal_val = int(K*cir)
    n_needed_anomalous_val = K - n_needed_normal_val

    list_val_X_sampled, list_val_Y_sampled = [], []

    for i in range(len(val_normal_indexes)):
        val_normal_idxs = random.sample(val_normal_indexes[i], n_needed_normal_val)
        val_anomalous_idxs = random.sample(val_anomalous_indexes[i], n_needed_anomalous_val)
        val_idxs = []
        val_idxs += val_normal_idxs
        val_idxs+=val_anomalous_idxs

        val_X_sampled, val_Y_sampled = val_tasks[i][str(K)]["finetune_X"][val_idxs], val_tasks[i][str(K)]["finetune_Y"][val_idxs]
        list_val_X_sampled.append(val_X_sampled)
        list_val_Y_sampled.append(val_Y_sampled)

    return list_val_X_sampled, list_val_Y_sampled


def main(args):

    seed = 123

    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    cir_inner_loop_list = [float(i) for i in args.cir_inner_loop_list.split(' ')]
    K_list = [int(i) for i in args.K_list.split(' ')]

    train_tasks, val_tasks, test_tasks_finetune_sets, test_tasks_test_sets = load_STS()

    val_normal_indexes, val_anomalous_indexes = {}, {}

    for K in K_list:
        val_normal_indexes[str(K)], val_anomalous_indexes[str(K)] = [], []
        for val_task_idx in range(len(val_tasks)):
            val_normal_indexes[str(K)].append(list(np.where(val_tasks[val_task_idx][str(K)]['finetune_Y'] == 0)[0]))
            val_anomalous_indexes[str(K)].append(list(np.where(val_tasks[val_task_idx][str(K)]['finetune_Y'] == 1)[0]))

    sess = tf.InteractiveSession()
    model = MTL(sess, args, seed, len(train_tasks))

    summary = False
    if(args.summary_dir):
        summary = True


    if(summary):
        loddir_path = './summaries_MTL'
        if (not (os.path.exists(loddir_path))):
            os.mkdir(loddir_path)
        if (not (os.path.exists(os.path.join(loddir_path, model.summary_dir)))):
            os.mkdir(os.path.join(loddir_path, model.summary_dir))
        train_writer = tf.summary.FileWriter(
            os.path.join(loddir_path, model.summary_dir) + '/train')

        val_task_writers = {}

        for K in K_list:
            for cir in cir_inner_loop_list:
                val_task_writers[str(K)+'_'+str(cir)] = tf.summary.FileWriter(
                    os.path.join(loddir_path, model.summary_dir) + '/val_task_K_' +str(K)+'_cir_'+str(cir))


    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    min_val_tasks_test_loss = {}
    min_val_tasks_test_loss_mtl_epoch = {}
    min_val_tasks_test_loss_finetune_epoch= {}

    for K in K_list:
        for cir in cir_inner_loop_list:
            min_val_tasks_test_loss[str(K)+'_'+str(cir)] =10000
            min_val_tasks_test_loss_mtl_epoch[str(K)+'_'+str(cir)] = -1
            min_val_tasks_test_loss_finetune_epoch[str(K)+'_'+str(cir)] = -1
            

    # MTL-training
    for epoch in range(args.train_epochs+1):

        X_train, Y_train = sample_random_train_batch(train_tasks, args.batch_size)
        mtl_train_loss, tr_summaries = model.train_op(X_train, Y_train, epoch)

        if(summary and (epoch % model.summary_interval == 0)):
            train_writer.add_summary(tr_summaries, epoch)
            train_writer.flush()        

        if(epoch % model.val_task_finetune_interval == 0):
            for K in K_list:
                for cir in cir_inner_loop_list:

                    if((K==10 and cir ==0.99) or (K<10 and cir not in [0.5, 1.0])):
                        pass
                    else:

                        X_val_finetune_list, Y_val_finetune_list = sample_random_val_finetune_data(val_tasks, K, cir, val_normal_indexes[str(K)], val_anomalous_indexes[str(K)])
                        val_metrics_list = []
                        for val_task_index, (X_val_finetune, Y_val_finetune) in enumerate(zip(X_val_finetune_list, Y_val_finetune_list)):
                            if(cir > 0.5 and cir < 1.0):
                                ros = RandomOverSampler(random_state=seed)
                                X_val_finetune, Y_val_finetune_reshaped = ros.fit_resample(X_val_finetune, np.squeeze(Y_val_finetune))
                                Y_val_finetune = np.expand_dims(Y_val_finetune_reshaped, -1)

                            min_val_task_epoch, val_test_loss = model.val_op(X_val_finetune, Y_val_finetune, val_tasks[val_task_index][str(K)]['test_X'], val_tasks[val_task_index][str(K)]['test_Y'], K, cir, epoch)
                            val_metrics_list.append([min_val_task_epoch, val_test_loss])

                        avg_val_metrics = np.mean(val_metrics_list, axis=0)
                        if(avg_val_metrics[1] < min_val_tasks_test_loss[str(K)+'_'+str(cir)]):
                            model.saver.save(
                                model.sess,
                                model.checkpoint_path +
                                model.summary_dir +
                                "_restore_val_task_test_loss_" + str(K) + '_' + str(cir) + "/model.ckpt")
                            min_val_tasks_test_loss[str(K)+'_'+str(cir)] = avg_val_metrics[1]
                            min_val_tasks_test_loss_mtl_epoch[str(K)+'_'+str(cir)] = epoch
                            min_val_tasks_test_loss_finetune_epoch[str(K)+'_'+str(cir)] = int(avg_val_metrics[0])

                        if(summary):
                            val_task_writer = val_task_writers[str(K)+'_'+str(cir)]
                            val_task_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='avg_val_loss', simple_value=avg_val_metrics[1]),]), epoch)
                            val_task_writer.flush()


    if(summary):
        train_writer.close()
        for K in K_list:
            for cir in cir_inner_loop_list:
                val_task_writers[str(K)+'_'+str(cir)].close()

    
    test_task_finetune_writers, test_task_test_writers = {}, {}


    for K in K_list:
        for cir in cir_inner_loop_list:
            if((K==10 and cir ==0.99) or (K<10 and cir not in [0.5, 1.0])):
                pass
            
            else:
                loss_list, acc_list, prec_list, rec_list, spec_list, f1_list, auc_pr_list, epoch_list = [], [], [], [], [], [], [], []

                for test_task_index, (test_task_finetune_sets_per_K_cir, test_task_test_sets_per_K) in enumerate(zip(test_tasks_finetune_sets, test_tasks_test_sets)):
                    test_set = test_task_test_sets_per_K[str(K)]
                    feed_dict_test_task_test = {model.X_finetune: test_set["test_X"], model.Y_finetune: test_set["test_Y"]}
                    finetune_sets = test_task_finetune_sets_per_K_cir[str(K)][str(cir)]

                    if(K in [100, 1000] and cir<1.0):
                        model.summary_interval = 200
                        print_interval = 200


                        for finetune_set_index, finetune_set in enumerate(finetune_sets):
                            finetune_X, finetune_Y = finetune_set["finetune_X"], finetune_set["finetune_Y"]

                            # splitting finetune data into finetune and val sets and perform early stopping
                            normal_indexes, anomalous_indexes = list(np.nonzero(finetune_Y == 0)[0]), list(np.nonzero(finetune_Y == 1)[0])
                            finetune_normal_idxs = random.sample(normal_indexes, int(len(normal_indexes)*model.finetune_data_percentage))
                            finetune_anomalous_idxs = random.sample(anomalous_indexes, np.maximum(1,int(len(anomalous_indexes)*model.finetune_data_percentage)))
                            finetune_indexes = []
                            finetune_indexes += finetune_normal_idxs
                            finetune_indexes += finetune_anomalous_idxs

                            val_normal_idxs = [index for index in normal_indexes if index not in finetune_normal_idxs]
                            val_anomalous_idxs = [index for index in anomalous_indexes if index not in finetune_anomalous_idxs]
                            val_indexes = []
                            val_indexes += val_normal_idxs
                            val_indexes += val_anomalous_idxs

                            if(len(val_anomalous_idxs) < 1 and len(finetune_anomalous_idxs) > 0):
                                val_indexes += finetune_anomalous_idxs

                            val_X = finetune_X[val_indexes]
                            val_Y = finetune_Y[val_indexes]

                            finetune_X = finetune_X[finetune_indexes]
                            finetune_Y = finetune_Y[finetune_indexes]

                            if(cir > 0.50 and cir < 1.0):
                                ros = RandomOverSampler(random_state=seed)
                                finetune_X, finetune_Y_reshaped = ros.fit_resample(finetune_X, np.squeeze(finetune_Y))
                                finetune_Y = np.expand_dims(finetune_Y_reshaped, -1)

                                ros = RandomOverSampler(random_state=seed)
                                val_X, val_Y_reshaped = ros.fit_resample(val_X, np.squeeze(val_Y))
                                val_Y = np.expand_dims(val_Y_reshaped, -1)

                            val_feed_dict = {model.X_finetune: val_X, model.Y_finetune: val_Y}

                            if(summary):
                                finetune_writer_test_task = tf.summary.FileWriter(
                                    os.path.join(loddir_path, args.summary_dir) + '/finetune_with_valset_finetune_test_task_K_' + str(K) + '_cir_' + str(cir) + '_test_task_' + str(test_task_index) + '_fset_' + str(finetune_set_index))
                                val_writer_test_task = tf.summary.FileWriter(
                                    os.path.join(loddir_path, args.summary_dir) + '/finetune_with_valset_val_test_task_K_' + str(K) + '_cir_' + str(cir) + '_test_task_' + str(test_task_index) + '_fset_' + str(finetune_set_index))
                                test_writer_test_task = tf.summary.FileWriter(
                                    os.path.join(loddir_path, args.summary_dir) + '/finetune_with_valset_test_test_task_K_' + str(K) + '_cir_' + str(cir) + '_test_task_' + str(test_task_index) + '_fset_' + str(finetune_set_index))
                           
                            model.saver.restore(
                                model.sess,
                                model.checkpoint_path +
                                model.summary_dir +
                                "_restore_val_task_test_loss_" + str(K) + '_' + str(cir) + "/model.ckpt")

                            if(summary):
                                sess.run(tf.local_variables_initializer()) 
                                val_summaries = model.sess.run(model.merged_test, val_feed_dict)
                                val_writer_test_task.add_summary(val_summaries, 0)
                                val_writer_test_task.flush()
                                test_summaries = model.sess.run(model.merged_test, feed_dict_test_task_test)
                                test_writer_test_task.add_summary(test_summaries, 0)
                                test_writer_test_task.flush()

                            min_val_loss = 10000
                            min_val_loss_epoch = -1
                            early_stopping = 0

                            for finetune_epoch in range(1, args.finetune_epochs + 1):
                                        
                                if (model.batch_size < finetune_X.shape[0]):
                                    datapoints_idxs = random.sample(range(0, finetune_X.shape[0]), model.batch_size)
                                    finetune_loss, finetune_summaries = model.finetune_op(finetune_X[datapoints_idxs], finetune_Y[datapoints_idxs])
                                else:
                                    finetune_loss, finetune_summaries = model.finetune_op(finetune_X, finetune_Y)

                                if(summary and finetune_epoch%model.summary_interval==0):
                                    finetune_writer_test_task.add_summary(finetune_summaries, finetune_epoch)
                                    finetune_writer_test_task.flush()
                                    sess.run(tf.local_variables_initializer()) 
                                    val_summaries = model.sess.run(model.merged_test, val_feed_dict)
                                    val_writer_test_task.add_summary(val_summaries, finetune_epoch)
                                    val_writer_test_task.flush()
                                    test_summaries = model.sess.run(model.merged_test, feed_dict_test_task_test)
                                    test_writer_test_task.add_summary(test_summaries, finetune_epoch)
                                    test_writer_test_task.flush()

                                val_loss = model.test_loss.eval(val_feed_dict) 
                                if(val_loss < min_val_loss):
                                    early_stopping = 0
                                    min_val_loss = val_loss
                                    min_val_loss_epoch = finetune_epoch
                                    model.saver.save(
                                        model.sess,
                                        model.checkpoint_path +
                                        model.summary_dir +
                                        "_restore_val_set_test_task_" + str(test_task_index) + "_finetune_with_valset_" + str(finetune_set_index) +"/model.ckpt")

                                else:
                                    early_stopping+=1


                                if(early_stopping > model.early_stopping_val):
                                    break

                                if(val_loss < 0.0001):
                                    val_acc = model.test_acc.eval(val_feed_dict) 
                                    if(val_acc >= 1.0):
                                        # stopping training because accuracy on val set is 1.0 --- further training would just lead to further overfitting on the val set
                                        break

                            model.saver.restore(
                                model.sess,
                                model.checkpoint_path +
                                model.summary_dir +
                                "_restore_val_set_test_task_" + str(test_task_index) + "_finetune_with_valset_" + str(finetune_set_index) +"/model.ckpt")

                            sess.run(tf.local_variables_initializer()) 
                            test_loss, acc, precision, recall, specificity, f1_score, auc_pr = model.sess.run([model.test_loss, model.test_acc, model.test_precision, model.test_recall, model.test_specificity, model.test_f1_score, model.test_auc_pr], feed_dict=feed_dict_test_task_test) 

                            loss_list.append(test_loss)
                            acc_list.append(acc)
                            prec_list.append(precision)
                            rec_list.append(recall)
                            spec_list.append(specificity)
                            f1_list.append(f1_score)
                            auc_pr_list.append(auc_pr)
                            epoch_list.append(min_val_loss_epoch)


                            if(summary):
                                finetune_writer_test_task.close()
                                val_writer_test_task.close()
                                test_writer_test_task.close()

                    else:
                        if(min_val_tasks_test_loss_finetune_epoch[str(K)+'_'+str(cir)] <= 20):
                            finetune_summary_interval = 1
                        else:
                            finetune_summary_interval = int(min_val_tasks_test_loss_finetune_epoch[str(K)+'_'+str(cir)]/10)
                        
                        for finetune_set_index, finetune_set in enumerate(finetune_sets):

                            if(summary):
                                test_task_finetune_writers[str(K)+'_'+str(cir)+'_'+str(test_task_index)+'_'+str(finetune_set_index)] = tf.summary.FileWriter(
                                    os.path.join(loddir_path, model.summary_dir) + '/test_task_finetune_K_' +str(K)+'_cir_'+str(cir) + '_test_task_' + str(test_task_index) + '_fset_'+str(finetune_set_index))
                                test_task_test_writers[str(K)+'_'+str(cir)+'_'+str(test_task_index)+'_'+str(finetune_set_index)] = tf.summary.FileWriter(
                                    os.path.join(loddir_path, model.summary_dir) + '/test_task_test_K_' +str(K)+'_cir_'+str(cir) + '_test_task_' + str(test_task_index) + '_fset_'+str(finetune_set_index))
                
                            model.saver.restore(
                                model.sess,
                                model.checkpoint_path +
                                model.summary_dir +
                                "_restore_val_task_test_loss_" + str(K) + '_' + str(cir) + "/model.ckpt")

                            if(summary):
                                sess.run(tf.local_variables_initializer()) 
                                test_summaries = model.sess.run(model.merged_test, feed_dict_test_task_test)
                                test_task_test_writers[str(K)+'_'+str(cir)+'_'+str(test_task_index)+'_'+str(finetune_set_index)].add_summary(test_summaries, 0)
                                test_task_test_writers[str(K)+'_'+str(cir)+'_'+str(test_task_index)+'_'+str(finetune_set_index)].flush()

                            finetune_X, finetune_Y = finetune_set["finetune_X"], finetune_set["finetune_Y"]
                            if(cir > 0.5 and cir < 1.0):
                                ros = RandomOverSampler(random_state=seed)
                                finetune_X, finetune_Y_reshaped = ros.fit_resample(finetune_X, np.squeeze(finetune_Y))
                                finetune_Y = np.expand_dims(finetune_Y_reshaped, -1)

                            if(model.batch_size < K):
                                batch_idxs = random.sample(range(0, finetune_X.shape[0]), model.batch_size)

                                for finetune_epoch in range(1,min_val_tasks_test_loss_finetune_epoch[str(K)+'_'+str(cir)]+1):
                                    finetune_loss, finetune_summaries = model.finetune_op(finetune_X[batch_idxs], finetune_Y[batch_idxs])
                                    if(summary and finetune_epoch%finetune_summary_interval == 1):

                                        test_task_finetune_writers[str(K)+'_'+str(cir)+'_'+str(test_task_index)+'_'+str(finetune_set_index)].add_summary(finetune_summaries, finetune_epoch)
                                        test_task_finetune_writers[str(K)+'_'+str(cir)+'_'+str(test_task_index)+'_'+str(finetune_set_index)].flush()
                                        sess.run(tf.local_variables_initializer()) 
                                        test_summaries = model.sess.run(model.merged_test, feed_dict_test_task_test)
                                        test_task_test_writers[str(K)+'_'+str(cir)+'_'+str(test_task_index)+'_'+str(finetune_set_index)].add_summary(test_summaries, finetune_epoch)
                                        test_task_test_writers[str(K)+'_'+str(cir)+'_'+str(test_task_index)+'_'+str(finetune_set_index)].flush()
                                    
                            else:
                                for finetune_epoch in range(1,min_val_tasks_test_loss_finetune_epoch[str(K)+'_'+str(cir)]+1):
                                    finetune_loss, finetune_summaries = model.finetune_op(finetune_X, finetune_Y)
                                    if(summary and finetune_epoch%finetune_summary_interval == 1):
                
                                        test_task_finetune_writers[str(K)+'_'+str(cir)+'_'+str(test_task_index)+'_'+str(finetune_set_index)].add_summary(finetune_summaries, finetune_epoch)
                                        test_task_finetune_writers[str(K)+'_'+str(cir)+'_'+str(test_task_index)+'_'+str(finetune_set_index)].flush()
                                        sess.run(tf.local_variables_initializer()) 
                                        test_summaries = model.sess.run(model.merged_test, feed_dict_test_task_test)
                                        test_task_test_writers[str(K)+'_'+str(cir)+'_'+str(test_task_index)+'_'+str(finetune_set_index)].add_summary(test_summaries, finetune_epoch)
                                        test_task_test_writers[str(K)+'_'+str(cir)+'_'+str(test_task_index)+'_'+str(finetune_set_index)].flush()
                                    
                            sess.run(tf.local_variables_initializer()) 
                            test_loss, test_acc, test_precision, test_recall, test_specificity, test_f1_score, test_auc_pr = model.sess.run([model.test_loss, model.test_acc, model.test_precision, model.test_recall, model.test_specificity, model.test_f1_score, model.test_auc_pr], feed_dict=feed_dict_test_task_test) 

                            loss_list.append(test_loss)
                            acc_list.append(test_acc)
                            prec_list.append(test_precision)
                            rec_list.append(test_recall)
                            spec_list.append(test_specificity)
                            f1_list.append(test_f1_score)
                            auc_pr_list.append(test_auc_pr)
                            epoch_list.append(min_val_tasks_test_loss_finetune_epoch[str(K)+'_'+str(cir)])

                            if(summary):
                                sess.run(tf.local_variables_initializer()) 
                                test_summaries = model.sess.run(model.merged_test, feed_dict_test_task_test)
                                test_task_test_writers[str(K)+'_'+str(cir)+'_'+str(test_task_index)+'_'+str(finetune_set_index)].add_summary(test_summaries, args.finetune_epochs + 100)
                                test_task_test_writers[str(K)+'_'+str(cir)+'_'+str(test_task_index)+'_'+str(finetune_set_index)].flush()
                                test_task_test_writers[str(K)+'_'+str(cir)+'_'+str(test_task_index)+'_'+str(finetune_set_index)].close()

                test_results_dict = {}
                test_results_dict['test_loss'] = loss_list
                test_results_dict['acc'] = acc_list
                test_results_dict['prec'] = prec_list
                test_results_dict['rec'] = rec_list
                test_results_dict['spec'] = spec_list
                test_results_dict['f1'] = f1_list
                test_results_dict['auc_pr'] = auc_pr_list
                test_results_dict['epoch'] = epoch_list

                results_dir_path = './results/'
                if (not (os.path.exists(results_dir_path))):
                    os.mkdir(results_dir_path)
                filename = args.summary_dir + '_K_' + str(K) + '_cir_' + str(int(100*cir)) +'.txt'
                with open(results_dir_path+filename, 'wb') as file:
                    pickle.dump(test_results_dict, file)


                print('average metrics for K = ', K, ' cir = ', str(cir))

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
        description='Multi-task training on multiple tasks then transfer learning (finetuning) on target/test tasks')
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
        '-lr',
        type=float,
        metavar='',
        help='learning rate (for pretraining and finetuning')
    parser.add_argument(
        '-train_epochs',
        type=int,
        metavar='',
        help='number of training epochs for the training tasks')
    parser.add_argument(
        '-finetune_epochs',
        type=int,
        metavar='',
        help='number of finetuning epochs (only for test task)')
    parser.add_argument(
        '-batch_size',
        type=int,
        metavar='',
        help='number of data points sampled for training')
    parser.add_argument(
        '-K_list',
        type=str,
        metavar='',
        help='number of finetuning examples in the test task')
    parser.add_argument(
        '-cir_inner_loop_list',
        type=str,
        metavar='',
        help='percentage of positive examples in the test task')
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

    main(args)
