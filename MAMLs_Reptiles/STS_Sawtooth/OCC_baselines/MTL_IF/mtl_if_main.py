# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import argparse
import os
import random 
import pickle
import json
from imblearn.over_sampling import RandomOverSampler
from mtl_if_class import MTL_IF

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

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


def initialize_isoForest(seed, n_estimators, max_samples, contamination, **kwargs):

    isoForest = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
        contamination=contamination, n_jobs=-1, random_state=seed, behaviour='new', **kwargs)
    return isoForest

def train(isoForest, X_train):


    if X_train.ndim > 2:
        X_train_shape = X_train.shape
        X_train = X_train.reshape(X_train_shape[0], -1)
    else:
        X_train = X_train

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
    print(cir_inner_loop_list)
    K_list = [int(i) for i in args.K_list.split(' ')]

    train_tasks, val_tasks, test_tasks_finetune_sets, test_tasks_test_sets = load_STS()

    val_normal_indexes, val_anomalous_indexes = {}, {}

    for K in K_list:
        val_normal_indexes[str(K)], val_anomalous_indexes[str(K)] = [], []
        for val_task_idx in range(len(val_tasks)):
            val_normal_indexes[str(K)].append(list(np.where(val_tasks[val_task_idx][str(K)]['finetune_Y'] == 0)[0]))
            val_anomalous_indexes[str(K)].append(list(np.where(val_tasks[val_task_idx][str(K)]['finetune_Y'] == 1)[0]))

    sess = tf.InteractiveSession()
    input_shape = train_tasks[0]['X'][0].shape
    n_train_tasks = len(train_tasks)
    model = MTL_IF(sess, args, seed, input_shape, n_train_tasks)

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


    max_val_f1_score = {}
    max_val_f1_score_mtl_epoch = {}

    for K in K_list:
        for cir in cir_inner_loop_list:
            max_val_f1_score[str(K)+'_'+str(cir)] =-1
            max_val_f1_score_mtl_epoch[str(K)+'_'+str(cir)] = -1
            

    cir_val_IF = 1.0
    print('cir_val_IF set to ', cir_val_IF)

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

                        X_val_finetune_list, Y_val_finetune_list = sample_random_val_finetune_data(val_tasks, K, cir_val_IF, val_normal_indexes[str(K)], val_anomalous_indexes[str(K)])
                        val_metrics_list = []
                        for val_task_idx, (X_val_finetune, Y_val_finetune) in enumerate(zip(X_val_finetune_list, Y_val_finetune_list)):
                            # if(cir > 0.5 and cir < 1.0):
                            #     ros = RandomOverSampler(random_state=seed)
                            #     X_val_finetune_reshaped = np.reshape(X_val_finetune, (X_val_finetune.shape[0], -1))
                            #     X_val_finetune_reshaped, Y_val_finetune_reshaped = ros.fit_resample(X_val_finetune_reshaped, np.squeeze(Y_val_finetune))
                            #     X_val_finetune = np.reshape(X_val_finetune_reshaped, (-1, 84, 84, 3))
                            #     Y_val_finetune = np.expand_dims(Y_val_finetune_reshaped, -1)

                            val_acc, _, _, _, val_f1_score, val_auc_roc = model.val_op(X_val_finetune, Y_val_finetune, val_tasks[val_task_idx][str(K)]['test_X'], val_tasks[val_task_idx][str(K)]['test_Y'], K, cir, epoch)
                            val_metrics_list.append([val_acc, val_f1_score, val_auc_roc])

                        avg_val_metrics = np.mean(val_metrics_list, axis=0)

                        if(avg_val_metrics[1] > max_val_f1_score[str(K)+'_'+str(cir)]):

                            model.saver.save(
                                model.sess,
                                model.checkpoint_path +
                                model.summary_dir +
                                "_restore_val_task_test_f1_" + str(K) + '_' + str(cir_val_IF) + "/model.ckpt")
                            max_val_f1_score[str(K)+'_'+str(cir)] = avg_val_metrics[1]
                            max_val_f1_score_mtl_epoch[str(K)+'_'+str(cir)] = epoch
                            print('Epoch : ', epoch, ' model saved for K = ', K, ' val_f1_score = ', max_val_f1_score[str(K)+'_'+str(cir)])

                        # if(summary):
                        #     val_task_writer = val_task_writers[str(K)+'_'+str(cir)]
                        #     val_task_writer.add_summary(val_task_summaries, epoch)
                        #     val_task_writer.flush()


    if(summary):
        train_writer.close()
        for K in K_list:
            for cir in cir_inner_loop_list:
                val_task_writers[str(K)+'_'+str(cir)].close()

    
    n_estimators = 1000
    max_samples = 'auto'
    contamination = 0.1

    for K in K_list:
        
        acc_list, prec_list, rec_list, spec_list, f1_list, auc_roc_list = [], [], [], [], [], []
        model.saver.restore(
            model.sess,
            model.checkpoint_path +
            model.summary_dir +
            "_restore_val_task_test_f1_" + str(K) + '_' + str(cir_val_IF) + "/model.ckpt")
        for test_task_index, (test_task_finetune_sets_per_K_cir, test_task_test_sets_per_K) in enumerate(zip(test_tasks_finetune_sets, test_tasks_test_sets)):
            test_set = test_task_test_sets_per_K[str(K)]
            feed_dict_test_task_test = {model.X_finetune: test_set["test_X"], model.Y_finetune: test_set["test_Y"]}
            encoding_test = sess.run(model.val_test_shared_out, feed_dict=feed_dict_test_task_test)


            finetune_sets = test_task_finetune_sets_per_K_cir[str(K)][str(cir)]
            for finetune_set_index, finetune_set in enumerate(finetune_sets):
                finetune_X, finetune_Y = finetune_set["finetune_X"], finetune_set["finetune_Y"]

                feed_dict_test_task_finetune = {model.X_finetune : finetune_X, model.Y_finetune : finetune_Y}

                encoding_finetune = sess.run(model.val_finetune_shared_out, feed_dict=feed_dict_test_task_finetune)

                isoForest = initialize_isoForest(seed, n_estimators, max_samples, contamination)
                train(isoForest, encoding_finetune)
                acc, prec, rec, spec, f1_score, auc_roc = predict(isoForest, encoding_test, np.squeeze(test_set["test_Y"]))

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
        filename = args.summary_dir + '_K_' + str(K) + '_cir_' + str(int(100*cir)) +'.txt'
        with open(results_dir_path+filename, 'wb') as file:
            pickle.dump(test_results_dict, file)


        print('K = ', K, ' average metrics')

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
            ' auc_roc : ',
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
            ' auc_roc : ',
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
            ' auc_roc : ',
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
            ' auc_roc : ',
            1.96*np.std(auc_roc_list)/np.sqrt(n_test_tasks)
            )

    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-task training on multiple tasks then transfer learning (finetuning) on a test task/ purpose: benchmark with Model-Agnostic Meta-Learning (MAML)')
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
