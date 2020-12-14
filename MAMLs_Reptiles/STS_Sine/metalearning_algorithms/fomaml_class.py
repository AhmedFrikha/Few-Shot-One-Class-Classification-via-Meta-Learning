# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import os
import time

tf.logging.set_verbosity(tf.logging.ERROR)


class FOMAML:
    """ This class defines the model trained with the MAML algorithm.

    """

    def __init__(self, sess, args, seed, n_train_tasks, input_shape):
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.lr = args.lr
        self.meta_lr = args.meta_lr
        self.meta_epochs = args.meta_epochs
        self.K = args.K
        self.num_updates = args.num_updates
        self.bn = args.bn

        self.anil = False
        if(hasattr(args, 'anil')):
            self.anil = args.anil

        self.dataset = None
        if(hasattr(args, 'dataset')):
            self.dataset = args.dataset

        self.sess = sess
        self.summary = False
        self.summary_dir = args.summary_dir

        if(self.summary_dir):
            self.summary = True
            self.summary_interval = 100
            summaries_list_metatrain = []
            summaries_list_val = []
            summaries_list_test_restore_val = []
        else:
            self.summary_dir = "no_summary"

        self.stop_grad = args.stop_grad

        self.n_queries = args.n_queries
        # dataset specific variables
        self.n_classes = 1
        self.dim_input = input_shape[0]
        self.input_shape = input_shape
        self.n_train_tasks = n_train_tasks

        # number of tasks to sample per meta-training iteration
        self.n_sample_tasks = 8
        self.flatten = tf.keras.layers.Flatten()

        # build model
        self.layers = []
        if(args.filters == ""):
            self.filter_sizes = []
        else:
            self.filter_sizes = [int(i) for i in args.filters.split(' ')]
            self.kernel_sizes = [int(i) for i in args.kernel_sizes.split(' ')]
            if(len(self.filter_sizes) == 1):
                self.layers.append(
                    tf.keras.layers.Conv1D(
                        filters=self.filter_sizes[0],
                        kernel_size=self.kernel_sizes[0],
                        input_shape=(None, self.dim_input, 1),
                        strides=2,
                        padding='same',
                        activation='relu',
                        name='conv_last'))

                if(self.bn):
                    self.layers.append(
                        tf.keras.layers.BatchNormalization(
                            name='bn_c_last'))

            else:
                self.layers.append(
                    tf.keras.layers.Conv1D(
                        filters=self.filter_sizes[0],
                        kernel_size=self.kernel_sizes[0],
                        input_shape=(None, self.dim_input, 1),
                        strides=2,
                        padding='same',
                        activation='relu',
                        name='conv0'))
                if(self.bn):
                    self.layers.append(
                        tf.keras.layers.BatchNormalization(
                            name='bn_c0'))

            for i in range(1, len(self.filter_sizes)):
                if(i != len(self.filter_sizes) - 1):
                    self.layers.append(
                        tf.keras.layers.Conv1D(
                            filters=self.filter_sizes[i],
                            kernel_size=self.kernel_sizes[i],
                            strides=2,
                            padding='same',
                            activation='relu',
                            name='conv' + str(i)))
                    if(self.bn):
                        self.layers.append(
                            tf.keras.layers.BatchNormalization(
                                name='bn_c' + str(i)))

                else:
                    self.layers.append(
                        tf.keras.layers.Conv1D(
                            filters=self.filter_sizes[i],
                            kernel_size=self.kernel_sizes[i],
                            strides=2,
                            padding='same',
                            activation='relu',
                            name='conv_last'))
                    if(self.bn):
                        self.layers.append(
                            tf.keras.layers.BatchNormalization(
                                name='bn_c_last'))

        if(args.dense_layers == ""):
            self.dense_sizes = []
        else:
            self.dense_sizes = [int(i) for i in args.dense_layers.split(' ')]

        for i in range(0, len(self.dense_sizes)):
            if(len(self.filter_sizes) == 0 and i == 0):
                self.layers.append(
                    tf.keras.layers.Dense(
                        units=self.dense_sizes[i],
                        activation='relu',
                        input_shape=(
                            None,
                        ) + input_shape,
                        name='dense' + str(i)))
            else:
                self.layers.append(
                    tf.keras.layers.Dense(
                        units=self.dense_sizes[i],
                        activation='relu',
                        name='dense' + str(i)))
            if(self.bn):
                self.layers.append(tf.keras.layers.BatchNormalization(
                    name='bn_d' + str(i + len(self.filter_sizes))))

        self.layers.append(
            tf.keras.layers.Dense(
                units=self.n_classes,
                name='dense_last'))

        # loss function
        self.loss_fct = tf.nn.sigmoid_cross_entropy_with_logits

        self.X_finetune = tf.placeholder(
            tf.float32, (None, self.dim_input), name='X_finetune')
        self.Y_finetune = tf.placeholder(
            tf.float32, (None, self.n_classes), name='Y_finetune')

        self.X_outer_loop = tf.placeholder(
            tf.float32, (None, self.dim_input), name='X_outer_loop')
        self.Y_outer_loop = tf.placeholder(
            tf.float32, (None, self.n_classes), name='Y_outer_loop')

        self.construct_forward = tf.make_template(
            'construct_forward', self.feed_forward)

        finetune_output = self.construct_forward(
            self.X_finetune, training=True)

        self.finetune_loss = tf.reduce_mean(
            self.loss_fct(
                labels=self.Y_finetune,
                logits=finetune_output))

        self.inner_loop_optimizer = tf.train.GradientDescentOptimizer(
            self.lr)
        self.finetune_update_op = self.inner_loop_optimizer.minimize(
            self.finetune_loss)
        for i in range(1, self.num_updates):
            if(i == 1):
                with tf.control_dependencies([self.finetune_update_op]):

                    finetune_output = self.construct_forward(
                        self.X_finetune, training=True)

            else:

                with tf.control_dependencies([finetune_update_op]):

                    finetune_output = self.construct_forward(
                        self.X_finetune, training=True)
            finetune_loss = tf.reduce_mean(
                self.loss_fct(
                    labels=self.Y_finetune,
                    logits=finetune_output))

            finetune_update_op = self.inner_loop_optimizer.minimize(
                finetune_loss)

        if(self.bn):
            self.updated_bn_model = self.assign_stats(self.X_finetune)
            with tf.control_dependencies([finetune_update_op]):
                with tf.control_dependencies([self.updated_bn_model]):
                    self.outer_loop_output = self.construct_forward(
                        self.X_outer_loop, training=True)

        else:
            with tf.control_dependencies([finetune_update_op]):
                self.outer_loop_output = self.construct_forward(
                    self.X_outer_loop, training=True)

        self.outer_loop_loss = tf.reduce_mean(
            self.loss_fct(
                labels=self.Y_outer_loop,
                logits=self.outer_loop_output))

        self.meta_optimizer = tf.train.AdamOptimizer(self.meta_lr)

        self.meta_gradient = self.meta_optimizer.compute_gradients(
            self.outer_loop_loss)

        self.placeholder_gradients = []
        for grad_var in self.meta_gradient:
            self.placeholder_gradients.append(
                (tf.placeholder(tf.float32, shape=grad_var[0].get_shape()), grad_var[1]))

        self.meta_update_op = self.meta_optimizer.apply_gradients(
            self.placeholder_gradients)

        self.test_output = self.construct_forward(
            self.X_finetune, training=False)
        self.test_loss = tf.reduce_mean(self.loss_fct(
            labels=self.Y_finetune,
            logits=self.test_output))

        self.my_acc, self.my_precision, self.my_recall, self.my_specificity, self.my_f1_score, self.my_auc_pr = self.compute_metrics(
            self.test_output, self.Y_finetune)

        val_finetune_output = self.construct_forward(
            self.X_finetune, training=True)
        val_finetune_loss = tf.reduce_mean(
            self.loss_fct(
                labels=self.Y_finetune,
                logits=val_finetune_output))
        self.val_finetune_update_op = self.inner_loop_optimizer.minimize(
            val_finetune_loss, name='val_finetune_update_op')

        if(self.summary):

            summaries_list_val.append(
                tf.summary.scalar('val_test_loss', self.test_loss))
            summaries_list_val.append(
                tf.summary.scalar('val_accuracy', self.my_acc))
            summaries_list_val.append(
                tf.summary.scalar('val_precision', self.my_precision))
            summaries_list_val.append(
                tf.summary.scalar('val_recall', self.my_recall))
            summaries_list_val.append(
                tf.summary.scalar('val_specificity', self.my_specificity))
            summaries_list_val.append(
                tf.summary.scalar('val_f1_score', self.my_f1_score))
            summaries_list_val.append(
                tf.summary.scalar('val_auc_pr', self.my_auc_pr))
            summaries_list_test_restore_val.append(
                tf.summary.scalar('test_loss_1', self.test_loss))
            summaries_list_test_restore_val.append(
                tf.summary.scalar('accuracy_1', self.my_acc))
            summaries_list_test_restore_val.append(
                tf.summary.scalar('precision_1', self.my_precision))
            summaries_list_test_restore_val.append(
                tf.summary.scalar('recall_1', self.my_recall))
            summaries_list_test_restore_val.append(
                tf.summary.scalar('specificity_1', self.my_specificity))
            summaries_list_test_restore_val.append(
                tf.summary.scalar('f1_score_1', self.my_f1_score))
            summaries_list_test_restore_val.append(
                tf.summary.scalar('auc_pr_1', self.my_auc_pr))
            self.merged_test_restore_val = tf.summary.merge(
                summaries_list_test_restore_val)
            self.merged_val = tf.summary.merge(
                summaries_list_val)

        self.saver = tf.train.Saver()

        base_path = '/home/USER/Documents'
        if (not (os.path.exists(base_path))):
            base_path = '/home/ubuntu/Projects'
        if (not (os.path.exists(base_path))):
            base_path = '/home/USER/Projects'
        if (not (os.path.exists(base_path))):
            base_path = '/home/ceesgniewyk/Projects'

        self.checkpoint_path = base_path + '/MAML/checkpoints_MAML/'
        if (not (os.path.exists(self.checkpoint_path))):
            os.mkdir(self.checkpoint_path)
        if (not (os.path.exists(os.path.join(self.checkpoint_path, self.summary_dir)))):
            os.mkdir(os.path.join(self.checkpoint_path, self.summary_dir))

        self.m_vars = []

        for layer_idx in range(0, len(self.layers)):
            self.m_vars.append(self.layers[layer_idx].weights[0])
            self.m_vars.append(self.layers[layer_idx].weights[1])

    def feed_forward(self, inp, training, no_head=False):
        """computes an output tensor by feeding the input through the network.

        Parameters
        ----------
        inp : tensor
            input tensor.
        training : bool
            argument for Batch normalization layers.

        Returns
        -------
        out : tensor
            output tensor.

        """

        h = tf.expand_dims(inp, -1)
        n_layers_no_head = len(self.layers) - len(self.dense_sizes) - 1
        if(self.bn):
            n_layers_no_head = len(self.layers) - len(self.dense_sizes) * 2 - 1

        for i in range(n_layers_no_head):
            if('conv' in self.layers[i].name):
                h = self.layers[i](h)
                # if(self.dataset == 'MIN'):
                h = tf.layers.max_pooling1d(
                    h, pool_size=2, strides=2, padding='same')
            elif('bn' in self.layers[i].name):
                h = self.layers[i](h, training=training)

            if(self.bn and 'bn_c_last' in self.layers[i].name):
                h = self.flatten(h)

            elif(not(self.bn) and 'conv_last' in self.layers[i].name):
                h = self.flatten(h)

        if(no_head):
            return h
        else:
            if(n_layers_no_head < 1):
                i = -1
            for j in range(i + 1, len(self.layers)):
                h = self.layers[j](h)
            return h

    def metatrain_op(self, epoch, X_train_a, Y_train_a, X_train_b, Y_train_b):
        """performs one meta-training iteration.

        Parameters
        ----------
        X_train_a : tensor
            contains features of the K datapoints sampled for the inner loop (adaptation) updates of each meta-training task.
        Y_train_a : tensor
            contains labels of the K datapoints sampled for the inner loop (adaptation) updates of each meta-training task.
        X_train_b : tensor
            contains features sampled for the outer loop updates of each meta-training task.
        Y_train_b : tensor
            contains labels sampled for the outer loop updates of each meta-training task.

        Returns
        -------
        metatrain_loss : float
            sum of the losses computed on the sampled outer loop batch of each sampled meta-training task.
        train_summaries : list
            training summaries.

        """

        meta_grads_list = []
        grads_list = []
        metatrain_loss = 0
        old_vars = []
        for layer_idx in range(0, len(self.layers)):
            layer_weights = self.layers[layer_idx].get_weights()
            old_vars.append(layer_weights)

        old_vars_trainable = []
        for layer_idx in range(0, len(self.layers)):
            layer_weights = self.layers[layer_idx].get_weights()[:2]
            old_vars_trainable.append(layer_weights)

        for i in range(0, self.n_sample_tasks):
            feed_dict = {
                self.X_finetune: X_train_a[i],
                self.Y_finetune: Y_train_a[i],
                self.X_outer_loop: X_train_b[i],
                self.Y_outer_loop: Y_train_b[i]}
            task_metagrads, outer_loss = self.sess.run(
                [self.meta_gradient, self.outer_loop_loss], feed_dict)

            metatrain_loss += outer_loss
            grads_list.append([task_metagrads[j][0]
                               for j in range(len(task_metagrads))])

            for layer_idx in range(0, len(self.layers)):
                self.layers[layer_idx].set_weights(old_vars[layer_idx])

        avg_meta_grads = np.mean(grads_list, axis=0)
        meta_feed_dict = {}
        for i in range(len(avg_meta_grads)):

            meta_feed_dict[self.placeholder_gradients[i]
                           [0]] = avg_meta_grads[i]
        self.sess.run(self.meta_update_op, meta_feed_dict)

        train_summaries = tf.Summary(
            value=[
                tf.Summary.Value(
                    tag='metatrain_loss',
                    simple_value=metatrain_loss),
            ])

        return metatrain_loss, train_summaries

    def val_op(self, K_X_val, K_Y_val, val_test_X, val_test_Y):
        """performs one validation episode.

        Parameters
        ----------
        K_X_val : array
            contains features of the K datapoints sampled for adaptation to the validation task(s).
        K_Y_val : array
            contains labels of the K datapoints sampled for adaptation to the validation task(s).
        val_test_X : array
            contains features of the test set(s) of the validation task(s).
        val_test_Y : array
            contains labels of the test set(s) of the validation task(s).

        Returns
        -------
        val_summaries : list
            validation summaries.
        val_test_loss : float
            loss computed on the test set after adaptation.
        acc : float
            accuracy computed on the test set after adaptation.
        precision : float
            precision computed on the test set after adaptation.
        recall : float
            recall computed on the test set after adaptation.
        specificity : float
            specificity computed on the test set after adaptation.
        f1_score : float
            F1 score computed on the test set after adaptation.
        auc_pr : float
            AUC-PR computed on the test set after adaptation.

        """

        # save current network parameters (including bn stats)

        old_vars = []
        for layer_idx in range(0, len(self.layers)):
            layer_weights = self.layers[layer_idx].get_weights()
            old_vars.append(layer_weights)

        val_test_feed_dict = {
            self.X_finetune: val_test_X, self.Y_finetune: val_test_Y
        }

        for i in range(self.num_updates):
            self.sess.run(self.val_finetune_update_op, {
                self.X_finetune: K_X_val, self.Y_finetune: K_Y_val})

        if(self.bn):
            # assign batch normalization stats using the available adaptation
            # set
            self.sess.run(self.updated_bn_model, {self.X_finetune: K_X_val})

        if(self.summary):
            self.sess.run(tf.local_variables_initializer())
            val_summaries, val_test_loss, acc, precision, recall, specificity, f1_score, auc_pr = self.sess.run(
                [self.merged_val, self.test_loss, self.my_acc, self.my_precision, self.my_recall, self.my_specificity, self.my_f1_score, self.my_auc_pr], feed_dict=val_test_feed_dict)
        else:
            self.sess.run(tf.local_variables_initializer())
            val_test_loss, acc, precision, recall, specificity, f1_score, auc_pr = self.sess.run(
                [self.test_loss, self.my_acc, self.my_precision, self.my_recall, self.my_specificity, self.my_f1_score, self.my_auc_pr], feed_dict=val_test_feed_dict)
            val_summaries = None

        # resetting old networks parameters (including bn stats)
        for layer_idx in range(0, len(self.layers)):
            self.layers[layer_idx].set_weights(old_vars[layer_idx])

        return val_summaries, val_test_loss, acc, precision, recall, specificity, f1_score, auc_pr

    def finetune_op(self, K_X_finetune, K_Y_finetune):
        """performs one adaptation/finetuning update.

        Parameters
        ----------
        K_X_finetune : tensor
            contains features of the K datapoints sampled for adaptation.
        K_Y_finetune : tensor
            contains labels of the K datapoints sampled for adaptation

        Returns
        -------
        finetune_loss : float
            adaptation/finetuning loss.

        """

        feed_dict_finetune = {
            self.X_finetune: K_X_finetune, self.Y_finetune: K_Y_finetune}
        finetune_loss, _ = self.sess.run(
            [self.finetune_loss, self.finetune_update_op], feed_dict_finetune)
        return finetune_loss

    def assign_stats(self, K_finetune_samples):
        """ compute BN stats (mean and variance) using the given adaptation set and assign them to the BN layers in the network.

        Parameters
        ----------
        K_finetune_samples : tensor
            conatins the features of the K datapoints sampled for adaptation.

        Returns
        -------
        out : tensor
            output of the last batch normalization layer (when it is computed using session.run, the BN stats are assigned)

        """

        h = tf.expand_dims(K_finetune_samples, -1)

        for i in range(len(self.layers)):
            if('dense_last' in self.layers[i].name):
                out = h
                return out
            elif('bn_c' in self.layers[i].name):
                mean, var = tf.nn.moments(h, [0, 1])
                assign_op_1 = self.layers[i].variables[-1].assign(var)
                assign_op_2 = self.layers[i].variables[-2].assign(mean)
                with tf.control_dependencies([assign_op_1, assign_op_2]):
                    h = self.layers[i](h, training=False)
            elif('bn_d' in self.layers[i].name):
                mean, var = tf.nn.moments(h, 0)
                assign_op_3 = self.layers[i].variables[-1].assign(var)
                assign_op_4 = self.layers[i].variables[-2].assign(mean)
                with tf.control_dependencies([assign_op_3, assign_op_4]):
                    h = self.layers[i](h, training=False)
            elif('conv' in self.layers[i].name):
                h = self.layers[i](h)
                h = tf.layers.max_pooling1d(
                    h, pool_size=2, strides=2, padding='same')
            if('bn_c_last' in self.layers[i].name):
                h = self.flatten(h)
