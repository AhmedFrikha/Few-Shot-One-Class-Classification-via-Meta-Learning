# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import random 
from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

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

class MTL_IF:
    """
    
    A class for a neural network that is trained in a multi-task learning setting (common feature extractor, separate classfiers).
    
    """
    def __init__(self, sess, args, seed, input_shape, n_train_tasks):


        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.seed = seed
        # get parsed args
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.sess = sess
        self.summary = False
        self.summary_dir = args.summary_dir
        
        if(self.summary_dir):
            self.summary = True
            self.summary_interval = 200
            summaries_list_train = []
            summaries_list_val_test = []
            summaries_list_finetune = []
            summaries_list_test = []
        else:
            self.summary_dir = "no_summary"


        self.val_task_finetune_interval = 200
        self.val_task_finetune_epochs = 1000
        self.early_stopping_val = 100

        self.finetune_data_percentage = 0.8

        self.n_classes = 1

        self.dim_input = input_shape[0]
        self.n_train_tasks = n_train_tasks

        self.flatten = tf.keras.layers.Flatten()

        if(args.filters == ""):
            self.filter_sizes=[]

        else:
            self.filter_sizes = [int(i) for i in args.filters.split(' ')]
            self.kernel_sizes = [int(i) for i in args.kernel_sizes.split(' ')]
        
        
        # build model
        self.shared_layers = []

        if(len(self.filter_sizes) > 0):
            self.shared_layers.append(
                tf.keras.layers.Conv1D(
                        filters=self.filter_sizes[0],
                        kernel_size=self.kernel_sizes[0], 
                        input_shape=(None, self.dim_input, 1),
                        strides=2,
                        padding='same',
                        activation='relu',
                        name='conv0_shared'))
            # self.shared_layers.append(tf.keras.layers.BatchNormalization(name='bn0_shared'))
            for i in range(1, len(self.filter_sizes)):
                self.shared_layers.append(
                    tf.keras.layers.Conv1D(
                        filters=self.filter_sizes[i],
                        kernel_size=self.kernel_sizes[i],
                        strides=2,
                        padding='same',
                        activation='relu',
                        name='conv'+str(i)+'_shared'))
                # self.shared_layers.append(tf.keras.layers.BatchNormalization(name='bn'+str(i)+'_shared'))

        else:
            for i in range(0, len(self.dense_sizes)-2):
                self.shared_layers.append(tf.keras.layers.Dense(units=self.dense_sizes[i], activation='relu',name='dense_shared_'+str(i)))

        if(args.dense_layers == ""):
            self.dense_sizes=[]
        elif(' ' not in args.dense_layers):
            self.dense_sizes = []
            self.dense_sizes.append(int(args.dense_layers))
        else:
            self.dense_sizes = [int(i) for i in args.dense_layers.split(' ')]

        self.task_layers = []
        for i in range(self.n_train_tasks + 2):


            for j in range(0, len(self.dense_sizes)):
                self.task_layers.append(tf.keras.layers.Dense(units=self.dense_sizes[j], activation='relu',name='dense'+str(j)+'_separate'+str(i)))
            
                # self.task_layers.append(tf.keras.layers.BatchNormalization(name='bn'+str(j)+'_separate'+str(i)))
            self.task_layers.append(tf.keras.layers.Dense(units=self.n_classes, name='output_layer'+str(i)))


        # loss function
        self.loss_fct = tf.nn.sigmoid_cross_entropy_with_logits 

        self.X_train = tf.placeholder(
            tf.float32, (None, None, self.dim_input), name='X_train')
        self.Y_train = tf.placeholder(
            tf.float32, (None, None, self.n_classes), name='Y_train')
        
        self.X_finetune = tf.placeholder(
            tf.float32, (None, self.dim_input), name='X_finetune')
        self.Y_finetune = tf.placeholder(
            tf.float32, (None, self.n_classes), name='Y_finetune')

        self.construct_forward_shared = tf.make_template(
            'construct_forward_shared', self.feed_forward_shared)

        self.train_losses = self.compute_losses(update_batchnorm=True)


        shared_vars = tf.global_variables(scope="construct_forward_shared/")

        
        
        self.train_update_ops = []
        for i in range(self.n_train_tasks):
            task_vars = tf.global_variables(scope="task"+str(i)+"/")
            self.train_update_ops.append(tf.train.AdamOptimizer(
            self.lr).minimize(self.train_losses[i], var_list=shared_vars+task_vars))


        self.mean_train_loss = tf.reduce_mean(self.train_losses)

        if(self.summary):
            summaries_list_train.append(
                tf.summary.scalar('mean_train_loss', self.mean_train_loss))
            self.merged_train = tf.summary.merge(
                summaries_list_train)


        self.val_finetune_shared_out = self.construct_forward_shared(self.X_finetune, True)
        self.val_test_shared_out = self.construct_forward_shared(self.X_finetune, False)

        with tf.variable_scope('task8_val', reuse=tf.AUTO_REUSE):
            self.val_finetune_output = self.feed_forward_task(self.val_finetune_shared_out, True, -2)


        self.val_finetune_loss = tf.reduce_mean(self.loss_fct(
                labels=self.Y_finetune,
                logits=self.val_finetune_output))



        self.val_finetune_update_op = []
        finetune_val_task_vars = tf.global_variables(scope="task8_val/")
        self.val_finetune_update_op.append(tf.train.AdamOptimizer(
            self.lr).minimize(self.val_finetune_loss, var_list=finetune_val_task_vars+shared_vars))

        
        self.val_test_output = self.feed_forward_task(self.val_test_shared_out, False, -2)
        
        self.val_test_loss = tf.reduce_mean(self.loss_fct(
                labels=self.Y_finetune,
                logits=self.val_test_output))

        self.val_test_acc, self.val_test_precision, self.val_test_recall, self.val_test_specificity, self.val_test_f1_score, self.val_test_auc_pr= self.compute_metrics(
        self.val_test_output, self.Y_finetune)


        finetune_shared_out = self.construct_forward_shared(self.X_finetune, True)
        test_shared_out = self.construct_forward_shared(self.X_finetune, False)



        with tf.variable_scope('task9_test', reuse=tf.AUTO_REUSE):
            self.finetune_output = self.feed_forward_task(finetune_shared_out, True, -1)
        self.finetune_loss = tf.reduce_mean(self.loss_fct(
                labels=self.Y_finetune,
                logits=self.finetune_output))


        self.finetune_update_op = []
        finetune_task_vars = tf.global_variables(scope="task9_test/")
        

        self.finetune_update_op.append(tf.train.AdamOptimizer(
            self.lr).minimize(self.finetune_loss, var_list=finetune_task_vars+shared_vars))

        self.test_output = self.feed_forward_task(test_shared_out, False, -1)
        
        self.test_loss = tf.reduce_mean(self.loss_fct(
                labels=self.Y_finetune,
                logits=self.test_output))

        self.test_acc, self.test_precision, self.test_recall, self.test_specificity, self.test_f1_score, self.test_auc_pr= self.compute_metrics(
        self.test_output, self.Y_finetune)


        if(self.summary):
            summaries_list_finetune.append(
                tf.summary.scalar('finetune_loss', self.finetune_loss))
            summaries_list_test.append(
                tf.summary.scalar('test_loss', self.test_loss))
            summaries_list_test.append(
                tf.summary.scalar('test_acc', self.test_acc))
            summaries_list_test.append(
                tf.summary.scalar(
                    'test_precision', self.test_precision))
            summaries_list_test.append(
                tf.summary.scalar('test_recall', self.test_recall))
            summaries_list_test.append(
                tf.summary.scalar('test_specificity', self.test_specificity))
            summaries_list_test.append(
                tf.summary.scalar('test_f1_score', self.test_f1_score))
            summaries_list_test.append(
                tf.summary.scalar('test_auc_pr', self.test_auc_pr))
            self.merged_finetune = tf.summary.merge(
                summaries_list_finetune)
            self.merged_test = tf.summary.merge(
                summaries_list_test)
            summaries_list_val_test.append(
                tf.summary.scalar('val_test_loss', self.val_test_loss))
            summaries_list_val_test.append(
                tf.summary.scalar('val_test_acc', self.val_test_acc))
            summaries_list_val_test.append(
                tf.summary.scalar(
                    'val_test_precision', self.val_test_precision))
            summaries_list_val_test.append(
                tf.summary.scalar('val_test_recall', self.val_test_recall))
            summaries_list_val_test.append(
                tf.summary.scalar('val_test_specificity', self.val_test_specificity))
            summaries_list_val_test.append(
                tf.summary.scalar('val_test_f1_score', self.val_test_f1_score))
            summaries_list_val_test.append(
                tf.summary.scalar('val_test_auc_pr', self.val_test_auc_pr))
            self.merged_val_test = tf.summary.merge(
                summaries_list_val_test)
        
        self.saver = tf.train.Saver(max_to_keep=250)

        base_path = '/home/USER/Documents'
        if (not (os.path.exists(base_path))):
            base_path = '/home/ubuntu/Projects' 


        self.checkpoint_path = base_path + '/MAML/checkpoints_MTL/'
        
        if (not (os.path.exists(self.checkpoint_path))):
            os.mkdir(self.checkpoint_path)
        if (not (os.path.exists(os.path.join(self.checkpoint_path, self.summary_dir)))):
            os.mkdir(os.path.join(self.checkpoint_path, self.summary_dir))


    def compute_metrics(self, logits, labels):
        """compute non-running performance metrics.

        Parameters
        ----------
        logits : tensor
        labels : tensor
            

        Returns
        -------
        acc : tensor
            accuracy.
        precision : tensor
            precision.
        recall : tensor
            recall.
        specificity : tensor
            specificity.
        f1_score : tensor
            F1 score.
        auc_pr : tensor
            AUC-PR.

        """

        predictions = tf.cast(tf.greater(tf.nn.sigmoid(logits), 0.5), tf.float32)
        TP = tf.count_nonzero(predictions * labels)
        TN = tf.count_nonzero((predictions - 1) * (labels - 1))
        FP = tf.count_nonzero(predictions* (labels - 1))
        FN = tf.count_nonzero((predictions-1) *labels)
        acc = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))

        precision = tf.cond(tf.math.equal((TP+FP), 0), true_fn=lambda:tf.cast(0.0, tf.float64), false_fn=lambda: TP/(TP + FP))
        recall = TP / (TP + FN)
        specificity = TN / (TN + FP)
        f1_score = tf.cond(tf.math.equal((precision + recall), 0), true_fn=lambda:tf.cast(0.0, tf.float64), false_fn=lambda: 2*precision*recall/(precision + recall))

        auc_pr = tf.metrics.auc(labels=labels, predictions=tf.nn.sigmoid(logits), curve='PR',
                        summation_method='careful_interpolation')[1]

        return [
            acc, tf.cast(
                precision, tf.float32), tf.cast(
                recall, tf.float32), tf.cast(
                specificity, tf.float32), tf.cast(
                f1_score, tf.float32), auc_pr]

    def feed_forward_shared(self, inp, update_batchnorm):
        """computes an output tensor by feeding the input through the shared layers.

        Parameters
        ----------
        inp : tensor
            input tensor.
        update_batchnorm : bool
            argument for Batch normalization layers.

        Returns
        -------
        flattened : tensor
            output tensor of the shared layers.

        """
        if(len(self.filter_sizes) >0):
            h = tf.expand_dims(inp, -1)
        else:
            h = inp

        for i in range(0, len(self.shared_layers), 2):
            h = self.shared_layers[i](h)
            h = tf.layers.max_pooling1d(h, pool_size=2, strides=2, padding='same')
            # h = self.shared_layers[i + 1](h, training=update_batchnorm)

        flattened = self.flatten(h)
        return flattened


    def feed_forward_task(self, inp, update_batchnorm, task):
        """ compute an output tensor by feeding the input tensor through the given separate layer.

        Parameters
        ----------
        inp : tensor
            input tensor.
        update_batchnorm : bool
            argument for Batch normalization layers.
        task : int
            index of the task.
        Returns
        -------
        out : tensor
            output tensor of the separate layer.

        """
        h = inp

        n_denses = len(self.dense_sizes) +1
        for i in range(0,n_denses):
            h = self.task_layers[n_denses*task+i](h)     
        
        out = h
        return out

    def compute_losses(self, update_batchnorm):
        """ compute the losses of each training task.

        Parameters
        ----------
        update_batchnorm : bool
            argument for Batch normalization layers.

        Returns
        -------
        losses : list
            list containing the loss of each training task.

        """
        losses = []
        for i in range (self.n_train_tasks):
            shared_out = self.construct_forward_shared(self.X_train[i], update_batchnorm)
            with tf.variable_scope('task'+str(i), reuse=tf.AUTO_REUSE):
                task_output = self.feed_forward_task(shared_out, update_batchnorm, i)
            task_loss = tf.reduce_mean(self.loss_fct(
                labels=self.Y_train[i],
                logits=task_output))
            losses.append(task_loss)
        return losses 


    def train_op(self, X_train, Y_train, epoch):
        """update model parameters (minimizing the multi task learning loss).

        Parameters
        ----------
        X_train : numpy array
            features of the training batch.
        Y_train : numpy array
            labels of the training batch.
        epoch : int
            number of the current epoch.

        Returns
        -------
        train_loss : float
            training loss.
        summaries : list
            training summaries.

        """
        feed_dict_train = {self.X_train : X_train, self.Y_train : Y_train}

        train_loss, _= self.sess.run(
                [self.mean_train_loss, self.train_update_ops], feed_dict_train)

        if(self.summary and (epoch % self.summary_interval == 0)):
            train_summaries = self.sess.run(self.merged_train, feed_dict_train)
        else:
            train_summaries = None
            
        return train_loss, train_summaries

    def finetune_op(self, X_finetune, Y_finetune):
        """finetune model parameters to adapt to the target task.

        Parameters
        ----------
        X_finetune : numpy array
            features of the finetuning set of the target task.
        Y_finetune : numpy array
            labels of the finetuning set of the target task.

        Returns
        -------
        finetune_loss : float
            finetuning loss.
        finetune_summaries : list
            finetuning summaries.

        """

        feed_dict_finetune = {self.X_finetune : X_finetune, self.Y_finetune : Y_finetune}

        finetune_loss, _ = self.sess.run(
                [self.finetune_loss, self.finetune_update_op], feed_dict_finetune)

        if (self.summary):
            finetune_summaries = self.sess.run(self.merged_finetune, feed_dict_finetune)
        else:
            finetune_summaries = None
            
        return finetune_loss, finetune_summaries

    def val_op(self, X_val_finetune, Y_val_finetune, val_task_test_X, val_task_test_Y, K, cir, mtl_epoch):
        """ finetune the model using finetune set of the validation task an dthen test on its test set.

        Parameters
        ----------
        X_val_finetune : numpy array
            features of the validation finetune set.
        Y_val_finetune : numpy array
            labels of the validation finetune set.
        val_task_test_X : numpy array
            features of the validation test set.
        val_task_test_Y : numpy array
            labels of the validation test set.
        K : int
            size of the finetuning set.
        cir : int
            class-imbalance rate of the finetuning set.
        mtl_epoch : int
            number of the current multi task learning epoch.

        Returns
        -------
        train_loss : float
            training loss.
        summaries : list
            training summaries.

        """


        feed_dict_val_task_finetune = {self.X_finetune : X_val_finetune, self.Y_finetune : Y_val_finetune}
        feed_dict_val_task_test = {self.X_finetune : val_task_test_X, self.Y_finetune : val_task_test_Y}

        encoding_finetune = self.sess.run(self.val_finetune_shared_out, feed_dict=feed_dict_val_task_finetune)
        encoding_test = self.sess.run(self.val_test_shared_out, feed_dict=feed_dict_val_task_test)

        # print('encoding_finetune.shape =', encoding_finetune.shape)
        # print('encoding_test.shape =', encoding_test.shape)


        n_estimators = 1000
        max_samples = 'auto'
        contamination = 0.1
        isoForest = initialize_isoForest(self.seed, n_estimators, max_samples, contamination)
        train(isoForest, encoding_finetune)
        acc, prec, rec, spec, f1_score, auc_roc = predict(isoForest, encoding_test, np.squeeze(val_task_test_Y))

        return acc, prec, rec, spec, f1_score, auc_roc


        # saving model parameters
        # old_vars = []
        # for layer_idx in range(0, len(self.shared_layers)):
        #     layer_weights = self.shared_layers[layer_idx].get_weights()
        #     old_vars.append(layer_weights)

        # if('bn' in self.task_layers[-2].name):
        #     n_denses_bn = len(self.dense_sizes)*2 +1
        #     for layer_idx in range(0, n_denses_bn):
        #         layer_weights = self.task_layers[-2*n_denses_bn+layer_idx].get_weights()
        #         old_vars.append(layer_weights)

        # else:
        #     n_denses = len(self.dense_sizes)+1
        #     for layer_idx in range(0, n_denses):
        #         layer_weights = self.task_layers[-2*n_denses+layer_idx].get_weights()
        #         old_vars.append(layer_weights)


        # min_val_test_loss = 10000
        # min_val_test_loss_epoch = -1
        # early_stopping = 0
        
        # feed_dict_val_task_test = {self.X_finetune : val_task_test_X, self.Y_finetune : val_task_test_Y}

        # if(self.batch_size < K):
            
        #     for epoch in range(1,self.val_task_finetune_epochs+1):
                
        #         batch_idxs = random.sample(range(0, X_val_finetune.shape[0]), self.batch_size)
        #         X_val_finetune_batch, Y_val_finetune_batch = X_val_finetune[batch_idxs], Y_val_finetune[batch_idxs]
        #         feed_dict_val_task_finetune = {self.X_finetune : X_val_finetune_batch, self.Y_finetune : Y_val_finetune_batch}
                
        #         self.sess.run( self.val_finetune_update_op, feed_dict=feed_dict_val_task_finetune)
        #         val_test_loss = self.val_test_loss.eval(feed_dict_val_task_test)
        #         if(val_test_loss<min_val_test_loss):
        #             early_stopping = 0
        #             min_val_test_loss_epoch = epoch
        #             min_val_test_loss = val_test_loss
        #         else:
        #             early_stopping+=1

        #         if(early_stopping >= self.early_stopping_val):
        #             break

        # else:
        #     feed_dict_val_task_finetune = {self.X_finetune : X_val_finetune, self.Y_finetune : Y_val_finetune}
        #     #finetuning on the val task
        #     for epoch in range(1,self.val_task_finetune_epochs+1):
        #         self.sess.run( self.val_finetune_update_op, feed_dict=feed_dict_val_task_finetune)
        #         val_test_loss = self.val_test_loss.eval(feed_dict_val_task_test)
        #         if(val_test_loss<min_val_test_loss):
        #             early_stopping = 0
        #             min_val_test_loss_epoch = epoch
        #             min_val_test_loss = val_test_loss
        #         else:
        #             early_stopping+=1

        #         if(early_stopping >= self.early_stopping_val):
        #             break


        # if(self.summary):
        #     self.sess.run(tf.local_variables_initializer()) 
        #     val_task_summaries = self.sess.run(self.merged_val_test, feed_dict=feed_dict_val_task_test)           
        # else:
        #     val_task_summaries = None


        # # resetting previous model parameters
        # for layer_idx in range(0, len(self.shared_layers)):
        #     self.shared_layers[layer_idx].set_weights(old_vars[layer_idx])

        # n_shared_layers = len(self.shared_layers)
        # if('bn' in self.task_layers[-2].name):
        #     n_denses_bn = len(self.dense_sizes)*2 +1
        #     for layer_idx in range(0, n_denses_bn):
        #         self.task_layers[-2*n_denses_bn+layer_idx].set_weights(old_vars[layer_idx+n_shared_layers])
        # else:
        #     for layer_idx in range(0, n_denses):
        #         self.task_layers[-2*n_denses+layer_idx].set_weights(old_vars[layer_idx+n_shared_layers])

        # return val_task_summaries, min_val_test_loss_epoch, min_val_test_loss