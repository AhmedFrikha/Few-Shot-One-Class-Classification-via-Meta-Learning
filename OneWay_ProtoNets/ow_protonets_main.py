# Implementation of One-Way Prototypical Networks (Kruspe, 2019).

import tensorflow as tf
from PIL import Image
import numpy as np
import os
import glob
import argparse
from miniimagenet_tasks import create_miniimagenet_task_distribution
from omniglot_tasks import create_omniglot_allcharacters_task_distribution
from cifarfs_tasks import create_cifarfs_task_distribution
from fc100_tasks import create_fc100_task_distribution


# create batch-normalization layers
bn = []
bn.append(tf.keras.layers.BatchNormalization())
bn.append(tf.keras.layers.BatchNormalization())
bn.append(tf.keras.layers.BatchNormalization())
bn.append(tf.keras.layers.BatchNormalization())


def conv_block(inputs, out_channels, name='conv', training=False, block_idx=0):
    """ Feed-forward an input through a convolutional block of layers.

        Parameters
        ----------
        inputs : tensor (block inputs)
        out_channels : integer (number of filters of the convolutional layers)
        name : string
        training : bool (batchNorm argument)
        block_idx : integer (index of the conv block in the encoder
                    - used to determine which batchNorm layer to use)

        Returns
        -------
        out : tensor
            output of the convolutional block.

    """
    with tf.variable_scope(name):
        conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=3,
            padding='same')(inputs)
        conv = bn[block_idx](conv, training=training)
        conv = tf.nn.relu(conv)
        out = tf.contrib.layers.max_pool2d(conv, 2)
        return out


def conv_block_reordered(
        inputs,
        out_channels,
        name='conv',
        training=False,
        block_idx=0):
    """ Feed-forward an input through a convolutional block of with layers 
        reordered such that the batchNorm layer comes last.

        Parameters
        ----------
        inputs : tensor (block inputs)
        out_channels : integer (number of filters of the convolutional layers)
        name : string
        training : bool (batchNorm argument)
        block_idx : integer (index of the conv block in the encoder - 
                    used to determine which batchNorm layer to use)

        Returns
        -------
        out : tensor
            output of the convolutional block.

    """
    with tf.variable_scope(name):
        conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=3,
            padding='same')(inputs)
        conv = tf.nn.relu(conv)
        conv = tf.contrib.layers.max_pool2d(conv, 2)
        out = bn[block_idx](conv, training=training)
        return out


def encoder(x, h_dim, reuse=False, reorder_layers=True, training=False):
    """ Feed-forward an input through the encoder neural network.

        Parameters
        ----------
        x : tensor (input)
        h_dim : integer (number of filters of the convolutional layers)
        reuse : bool
        reorder_layers : bool
        training : bool (batchNorm argument)


        Returns
        -------
        embeddings : tensor
            embeddings of the given inputs.

    """
    if(reorder_layers):
        block = conv_block_reordered
    else:
        block = conv_block
    with tf.variable_scope('encoder', reuse=reuse):
        net = block(x, h_dim, name='conv_1', training=training, block_idx=0)
        net = block(net, h_dim, name='conv_2', training=training, block_idx=1)
        net = block(net, h_dim, name='conv_3', training=training, block_idx=2)
        net = block(net, h_dim, name='conv_4', training=training, block_idx=3)
        embeddings = tf.contrib.layers.flatten(net)
        return embeddings


def euclidean_distance(a, b):
    """ Compute non-running performance metrics.

        Parameters
        ----------
        a : tensor (embeddings of the query datapoints)
        b : tensor (embedding of the normal class prototype)


        Returns
        -------
        dists : tensor
            euclidean distance of each query datapoint to the normal class prototype
            and to the center of the embedding space (0).

    """

    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    dists_to_normal_class = tf.reduce_mean(tf.square(a - b), axis=2)
    dists_to_center = tf.reduce_mean(tf.square(a - 0.0), axis=2)
    dists = tf.concat([dists_to_normal_class, dists_to_center], 1)
    return


def main(args):

    # set the random seed
    seed = args.seed
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # hyperparameters
    n_episodes = 100
    n_way = 2  # because OCC is a binary classification problem
    h_dim = 64
    n_shot = args.n_shot
    n_query = args.n_query

    # load data
    base_path = "/home/USER/Documents/"
    if not (os.path.exists(base_path)):
        base_path = "/home/ubuntu/Projects/"
    if not (os.path.exists(base_path)):
        base_path = "/home/USER/Projects/"
    basefolder = base_path + "raw_data/"

    if(args.dataset == 'OMN'):
        n_epochs = 2000
        im_width, im_height, channels = 28, 28, 1
        metatrain_task_distribution, metaval_task_distribution, metatest_task_distribution = create_omniglot_allcharacters_task_distribution(
            basefolder + "omniglot/omniglot.pkl",
            train_occ=True,
            test_occ=True,
            num_training_samples_per_class=n_shot,
            num_test_samples_per_class=n_query,
            num_training_classes=2,
            meta_batch_size=1,
            seq_length=0
        )

    elif(args.dataset == 'CIFAR_FS'):
        n_epochs = 4000
        im_width, im_height, channels = 32, 32, 3

        metatrain_task_distribution, metaval_task_distribution, metatest_task_distribution = create_cifarfs_task_distribution(
            base_path + "data/CIFAR_FS/CIFAR_FS_train.pickle",
            base_path + "data/CIFAR_FS/CIFAR_FS_val.pickle",
            base_path + "data/CIFAR_FS/CIFAR_FS_test.pickle",
            train_occ=True,
            test_occ=True,
            num_training_samples_per_class=n_shot,
            num_test_samples_per_class=n_query,
            num_training_classes=2,
            meta_batch_size=1,
            seq_length=0
        )
    elif(args.dataset == 'FC100'):
        n_epochs = 4000
        im_width, im_height, channels = 32, 32, 3
        metatrain_task_distribution, metaval_task_distribution, metatest_task_distribution = create_fc100_task_distribution(
            base_path + "data/FC100/FC100_train.pickle",
            base_path + "data/FC100/FC100_val.pickle",
            base_path + "data/FC100/FC100_test.pickle",
            train_occ=True,
            test_occ=True,
            num_training_samples_per_class=n_shot,
            num_test_samples_per_class=n_query,
            num_training_classes=2,
            meta_batch_size=1,
            seq_length=0
        )

    else:
        n_epochs = 4000
        im_width, im_height, channels = 84, 84, 3

        metatrain_task_distribution, metaval_task_distribution, metatest_task_distribution = create_miniimagenet_task_distribution(
            basefolder + "miniImageNet_data/miniimagenet.pkl",
            train_occ=True,
            test_occ=True,
            num_training_samples_per_class=n_shot,
            num_test_samples_per_class=n_query,
            num_training_classes=2,
            meta_batch_size=1,
            seq_length=0
        )

    # batchNorm behavior
    support_training = True
    query_training = False

    # whether to reorder the layers such that batchNorm come last, as
    # mentioned in the original paper
    if(args.reorder == 'True'):
        reorder_layers = True
    else:
        reorder_layers = False

    # create placeholders for support and query inputs
    x = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])
    q = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])
    x_shape = tf.shape(x)
    q_shape = tf.shape(q)
    num_classes, num_support = x_shape[0], x_shape[1]
    num_classes_q, num_queries = q_shape[0], q_shape[1]
    y = tf.placeholder(tf.int64, [None, None])
    y_one_hot = tf.one_hot(y, depth=num_classes_q)

    # create the encoder network and feed inputs forward
    emb_x = encoder(tf.reshape(x,
                               [num_classes * num_support,
                                im_height,
                                im_width,
                                channels]),
                    h_dim,
                    h_dim,
                    reorder_layers=reorder_layers,
                    training=support_training)
    emb_dim = tf.shape(emb_x)[-1]

    # compute prototype for the normal class
    emb_x = tf.reduce_mean(
        tf.reshape(
            emb_x, [
                num_classes, num_support, emb_dim]), axis=1)

    # encode the query set
    emb_q = encoder(tf.reshape(q,
                               [num_classes_q * num_queries,
                                im_height,
                                im_width,
                                channels]),
                    h_dim,
                    h_dim,
                    reuse=True,
                    reorder_layers=reorder_layers,
                    training=query_training)

    # compute euclidean distances between query embeddings and normal class
    # prototype and center (0)
    dists = euclidean_distance(emb_q, emb_x)

    # compute loss and accuracy
    log_p_y = tf.reshape(tf.nn.log_softmax(-dists),
                         [num_classes_q, num_queries, -1])
    ce_loss = - \
        tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))
    acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), y)))

    # collect operations of batchNorm updates of moving mean and moving
    # variance
    bn_updates = []
    for i in [0, 1, 2, 3]:
        bn_updates += bn[i].updates

    # execute them before updating the model parameters
    with tf.control_dependencies(bn_updates):
        train_op = tf.train.AdamOptimizer().minimize(ce_loss)

    # summaries
    summary_dir = args.dataset + '_' + str(n_shot) + '_seed_' + str(seed)
    if(reorder_layers):
        summary_dir += '_R'
    summary_dir = summary_dir + '_Q_' + str(n_query)

    # model checkpoints
    saver = tf.train.Saver()
    checkpoint_path = base_path + 'MAML/OW_ProtoNets_checkpoints/'
    if (not (os.path.exists(checkpoint_path))):
        os.mkdir(checkpoint_path)
    if (not (os.path.exists(os.path.join(checkpoint_path, summary_dir)))):
        os.mkdir(os.path.join(checkpoint_path, summary_dir))

    # create session and initialize variables
    sess = tf.InteractiveSession()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # meta-training
    losses = []
    accs = []
    min_val_loss = 10000
    n_val_tasks = 50
    min_val_loss_epoch = -1
    early_stopping = False

    for ep in range(n_epochs):
        for epi in range(n_episodes):
            task = metatrain_task_distribution.sample_batch()[0]
            support = np.reshape(
                task.get_train_set()[0], (1, n_shot, im_height, im_width, channels))
            query = np.reshape(
                task.get_test_set()[0],
                (n_way,
                 n_query,
                 im_height,
                 im_width,
                 channels))
            labels = np.tile(
                np.arange(n_way)[
                    :, np.newaxis], (1, n_query)).astype(
                np.uint8)
            _, ls, ac = sess.run([train_op, ce_loss, acc], feed_dict={
                                 x: support, q: query, y: labels})
            losses.append(ls)
            accs.append(ac)
            if (epi + 1) % 100 == 0:
                print('TR: [epoch {}/{}, episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(
                    ep + 1, n_epochs, epi + 1, n_episodes, np.mean(losses), np.mean(accs)))
                losses, accs = [], []
                # val episode
                val_losses, val_accs = [], []
                for i in range(n_val_tasks):
                    task = metaval_task_distribution.sample_batch()[0]
                    support = np.reshape(
                        task.get_train_set()[0], (1, n_shot, im_height, im_width, channels))
                    query = np.reshape(
                        task.get_test_set()[0], (n_way, n_query, im_height, im_width, channels))
                    labels = np.tile(
                        np.arange(n_way)[
                            :, np.newaxis], (1, n_query)).astype(
                        np.uint8)
                    ls, ac = sess.run([ce_loss, acc], feed_dict={
                                      x: support, q: query, y: labels})
                    val_losses.append(ls)
                    val_accs.append(ac)
                mean_loss, mean_acc = np.mean(val_losses), np.mean(val_accs)
                if(mean_loss < min_val_loss):
                    min_val_loss = mean_loss
                    min_val_loss_epoch = ep
                    print('### model saved ###')
                    print('VAL: [epoch {}/{}, episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(
                        ep + 1, n_epochs, epi + 1, n_episodes, mean_loss, mean_acc))
                    saver.save(
                        sess,
                        checkpoint_path +
                        summary_dir +
                        "_restore_val_test_loss/model.ckpt")
                if(ep - min_val_loss_epoch > 300):
                    early_stopping = True
        if(early_stopping):
            print('##### EARLY STOPPING - NO IMPROVEMENT IN THE LAST 300 EPOCHS #####')
            break

    # restore best performing model on meta-validation set
    saver.restore(sess, checkpoint_path + summary_dir +
                  "_restore_val_test_loss/model.ckpt")
    print('### restored best model ###')

    # meta-testing
    n_test_episodes = 20000
    n_test_way = 2
    n_test_shot = n_shot
    n_test_query = n_query

    print('Testing...')
    avg_acc = 0.
    for epi in range(n_test_episodes):
        task = metatest_task_distribution.sample_batch()[0]
        support = np.reshape(
            task.get_train_set()[0],
            (1,
             n_shot,
             im_height,
             im_width,
             channels))
        query = np.reshape(
            task.get_test_set()[0],
            (n_way,
             n_query,
             im_height,
             im_width,
             channels))
        labels = np.tile(
            np.arange(n_test_way)[
                :, np.newaxis], (1, n_test_query)).astype(
            np.uint8)
        ls, ac = sess.run([ce_loss, acc], feed_dict={
                          x: support, q: query, y: labels})
        avg_acc += ac
        if (epi + 1) % 50 == 0:
            print('[test episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi +
                                                                             1, n_test_episodes, ls, ac))
    avg_acc /= n_test_episodes
    print('Average Test Accuracy: {:.5f}'.format(avg_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='One-Way ProtoNets')
    parser.add_argument('-dataset',
                        type=str,
                        default="None")
    parser.add_argument('-reorder',
                        type=str,
                        default="None")
    parser.add_argument('-n_shot',
                        type=int,
                        default="None")
    parser.add_argument('-n_query',
                        type=int,
                        default="None")
    parser.add_argument('-seed',
                        type=int,
                        default='None')

    args = parser.parse_args()
    main(args)
