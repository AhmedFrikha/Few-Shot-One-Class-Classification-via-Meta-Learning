"""
Giacomo Spigler. Meta-learnt priors slow down catastrophic forgetting in neural networks. arXiv preprint arXiv:1909.04170, 2019.
Utility functions to create Tasks from the Omniglot dataset.

The created tasks will be derived from ClassificationTask, and can be aggregated in a TaskDistribution object.
"""

import numpy as np
import pickle

from task import CB_OCCTask, OCCTask
from task_distribution import TaskDistribution

charomniglot_trainX = []
charomniglot_trainY = []

charomniglot_valX = []
charomniglot_valY = []

charomniglot_testX = []
charomniglot_testY = []


def create_omniglot_allcharacters_task_distribution(
    path_to_pkl,
    train_occ,
    test_occ,
    num_training_samples_per_class=10,
    num_test_samples_per_class=-1,
    num_training_classes=20,
    meta_batch_size=5,
    seq_length=0,
):
    """
    Returns a TaskDistribution that, on each reset, samples a different set of omniglot characters.

    Arguments:
    path_to_pkl: string
        Path to the pkl wrapped Omniglot dataset. This can be generated from the standard dataset using the supplied
        make_omniglot_dataset.py script.
    num_training_samples_per_class : int
        If -1, sample from the whole dataset. If >=1, the dataset will re-sample num_training_samples_per_class
        for each class at each reset, and sample minibatches exclusively from them, until the next reset.
        This is useful for, e.g., k-shot classification.
    num_test_samples_per_class : int
        Same as `num_training_samples_per_class'. Used to generate test sets for tasks on reset().
    num_training_classes : int
        If -1, use all the classes in `y'. If >=1, the dataset will re-sample `num_training_classes' at
        each reset, and sample minibatches exclusively from them, until the next reset.
    meta_batch_size : int
        Default size of the meta batch size.

    Returns:
    metatrain_task_distribution : TaskDistribution
        TaskDistribution object for use during training
    metaval_task_distribution : TaskDistribution
        TaskDistribution object for use during model validation
    metatest_task_distribution : TaskDistribution
        TaskDistribution object for use during testing
    """

    with open(path_to_pkl, "rb") as f:
        d = pickle.load(f)
        trainX_ = d["trainX"]
        trainY_ = d["trainY"]
        testX_ = d["testX"]
        testY_ = d["testY"]
    trainX_.extend(testX_)
    trainY_.extend(testY_)

    global charomniglot_trainX
    global charomniglot_trainY

    global charomniglot_valX
    global charomniglot_valY

    global charomniglot_testX
    global charomniglot_testY

    cutoff_tr, cutoff_val = 25, 30
    charomniglot_trainX = trainX_[:cutoff_tr]
    charomniglot_trainY = trainY_[:cutoff_tr]

    charomniglot_valX = trainX_[cutoff_tr:cutoff_val]
    charomniglot_valY = trainY_[cutoff_tr:cutoff_val]

    charomniglot_testX = trainX_[cutoff_val:]
    charomniglot_testY = trainY_[cutoff_val:]

    # Create a single large dataset with all characters, each for train and
    # test, and rename the targets appropriately
    trX = []
    trY = []

    valX = []
    valY = []

    teX = []
    teY = []

    cur_label_start = 0
    for alphabet_i in range(len(charomniglot_trainY)):
        charomniglot_trainY[alphabet_i] += cur_label_start
        trX.extend(charomniglot_trainX[alphabet_i])
        trY.extend(charomniglot_trainY[alphabet_i])
        cur_label_start += len(set(charomniglot_trainY[alphabet_i]))

    cur_label_start = 0
    for alphabet_i in range(len(charomniglot_valY)):
        charomniglot_valY[alphabet_i] += cur_label_start
        valX.extend(charomniglot_valX[alphabet_i])
        valY.extend(charomniglot_valY[alphabet_i])
        cur_label_start += len(set(charomniglot_valY[alphabet_i]))

    cur_label_start = 0
    for alphabet_i in range(len(charomniglot_testY)):
        charomniglot_testY[alphabet_i] += cur_label_start
        teX.extend(charomniglot_testX[alphabet_i])
        teY.extend(charomniglot_testY[alphabet_i])
        cur_label_start += len(set(charomniglot_testY[alphabet_i]))

    trX = np.asarray(trX, dtype=np.float32) / 255.0
    trY = np.asarray(trY, dtype=np.float32)
    valX = np.asarray(valX, dtype=np.float32) / 255.0
    valY = np.asarray(valY, dtype=np.float32)
    teX = np.asarray(teX, dtype=np.float32) / 255.0
    teY = np.asarray(teY, dtype=np.float32)

    charomniglot_trainX = trX
    charomniglot_valX = valX
    charomniglot_testX = teX
    charomniglot_trainY = trY
    charomniglot_valY = valY
    charomniglot_testY = teY

    if train_occ:
        metatrain_tasks_list = [
            OCCTask(
                charomniglot_trainX,
                charomniglot_trainY,
                num_training_samples_per_class,
                num_test_samples_per_class,
                num_training_classes,
                split_train_test=-1,
            )
        ]  # defaults to num_train / (num_train+num_test)

    else:

        metatrain_tasks_list = [
            CB_OCCTask(
                charomniglot_trainX,
                charomniglot_trainY,
                num_training_samples_per_class,
                num_test_samples_per_class,
                num_training_classes,
                split_train_test=-1,
            )
        ]  # defaults to num_train / (num_train+num_test)

    metatrain_task_distribution = TaskDistribution(
        tasks=metatrain_tasks_list,
        task_probabilities=[1.0],
        batch_size=meta_batch_size,
        sample_with_replacement=True,
    )

    if test_occ:

        metaval_tasks_list = [
            OCCTask(
                charomniglot_valX,
                charomniglot_valY,
                num_training_samples_per_class,
                num_test_samples_per_class,
                num_training_classes,
                split_train_test=-1,
            )
        ]

        metaval_task_distribution = TaskDistribution(
            tasks=metaval_tasks_list,
            task_probabilities=[1.0],
            batch_size=meta_batch_size,
            sample_with_replacement=True,
        )

        metatest_tasks_list = [
            OCCTask(
                charomniglot_testX,
                charomniglot_testY,
                num_training_samples_per_class,
                num_test_samples_per_class,
                num_training_classes,
                split_train_test=-1,
            )
        ]

        metatest_task_distribution = TaskDistribution(
            tasks=metatest_tasks_list,
            task_probabilities=[1.0],
            batch_size=meta_batch_size,
            sample_with_replacement=True,
        )
    else:

        metaval_tasks_list = [
            CB_OCCTask(
                charomniglot_valX,
                charomniglot_valY,
                num_training_samples_per_class,
                num_test_samples_per_class,
                num_training_classes,
                split_train_test=-1,
            )
        ]

        metaval_task_distribution = TaskDistribution(
            tasks=metaval_tasks_list,
            task_probabilities=[1.0],
            batch_size=meta_batch_size,
            sample_with_replacement=True,
        )

        metatest_tasks_list = [
            CB_OCCTask(
                charomniglot_testX,
                charomniglot_testY,
                num_training_samples_per_class,
                num_test_samples_per_class,
                num_training_classes,
                split_train_test=-1,
            )
        ]

        metatest_task_distribution = TaskDistribution(
            tasks=metatest_tasks_list,
            task_probabilities=[1.0],
            batch_size=meta_batch_size,
            sample_with_replacement=True,
        )

    return (
        metatrain_task_distribution,
        metaval_task_distribution,
        metatest_task_distribution,
    )
