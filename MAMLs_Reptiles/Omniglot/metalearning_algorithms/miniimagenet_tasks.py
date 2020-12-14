"""
Giacomo Spigler. Meta-learnt priors slow down catastrophic forgetting in neural networks. arXiv preprint arXiv:1909.04170, 2019.
Utility functions to create Tasks from the Mini-ImageNet dataset.

The created tasks will be derived from ClassificationTask, and can be aggregated in a TaskDistribution object.
"""

import numpy as np
import pickle

from task import CB_OCCTask, OCCTask, ClassificationTask
from task_distribution import TaskDistribution


# All data is pre-loaded in memory. This takes ~5GB if I recall correctly.
miniimagenet_trainX = []
miniimagenet_trainY = []

miniimagenet_valX = []
miniimagenet_valY = []

miniimagenet_testX = []
miniimagenet_testY = []


# TODO: allow for a custom train/test ratio split for each class!
def create_miniimagenet_task_distribution(
    path_to_pkl,
    train_occ,
    test_occ,
    num_training_samples_per_class=10,
    num_test_samples_per_class=15,
    num_training_classes=20,
    meta_batch_size=5,
    seq_length=0,
):
    """
    Returns a TaskDistribution that, on each reset, samples a different set of Mini-ImageNet classes.

    Arguments:
    path_to_pkl: string
        Path to the pkl wrapped Mini-ImageNet dataset. This can be generated from the standard dataset using the
        supplied make_miniimagenet_dataset.py script.
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

    global miniimagenet_trainX
    global miniimagenet_trainY

    global miniimagenet_valX
    global miniimagenet_valY

    global miniimagenet_testX
    global miniimagenet_testY

    with open(path_to_pkl, "rb") as f:
        d = pickle.load(f)
        miniimagenet_trainX, miniimagenet_trainY = d["train"]
        miniimagenet_valX, miniimagenet_valY = d["val"]
        miniimagenet_testX, miniimagenet_testY = d["test"]

    miniimagenet_trainX = miniimagenet_trainX.astype(np.float32) / 255.0
    miniimagenet_valX = miniimagenet_valX.astype(np.float32) / 255.0
    miniimagenet_testX = miniimagenet_testX.astype(np.float32) / 255.0

    del d

    if train_occ:

        train_tasks_list = [
            OCCTask(
                miniimagenet_trainX,
                miniimagenet_trainY,
                num_training_samples_per_class,
                num_test_samples_per_class,
                num_training_classes,
                split_train_test=0.5,
            )
        ]

    else:

        train_tasks_list = [
            CB_OCCTask(
                miniimagenet_trainX,
                miniimagenet_trainY,
                num_training_samples_per_class,
                num_test_samples_per_class,
                num_training_classes,
                split_train_test=0.5,
            )
        ]

    metatrain_task_distribution = TaskDistribution(
        tasks=train_tasks_list,
        task_probabilities=[1.0],
        batch_size=meta_batch_size,
        sample_with_replacement=True,
    )

    val_test_tasks_num_test_samples_per_class = num_test_samples_per_class
    if test_occ:

        val_tasks_list = [
            OCCTask(
                miniimagenet_valX,
                miniimagenet_valY,
                num_training_samples_per_class,
                val_test_tasks_num_test_samples_per_class,
                num_training_classes,
                split_train_test=0.5,
            )
        ]

        metaval_task_distribution = TaskDistribution(
            tasks=val_tasks_list,
            task_probabilities=[1.0],
            batch_size=meta_batch_size,
            sample_with_replacement=True,
        )

        test_tasks_list = [
            OCCTask(
                miniimagenet_testX,
                miniimagenet_testY,
                num_training_samples_per_class,
                val_test_tasks_num_test_samples_per_class,
                num_training_classes,
                split_train_test=0.5,
            )
        ]

        metatest_task_distribution = TaskDistribution(
            tasks=test_tasks_list,
            task_probabilities=[1.0],
            batch_size=meta_batch_size,
            sample_with_replacement=True,
        )

    else:

        val_tasks_list = [
            CB_OCCTask(
                miniimagenet_valX,
                miniimagenet_valY,
                num_training_samples_per_class,
                val_test_tasks_num_test_samples_per_class,
                num_training_classes,
                split_train_test=0.5,
            )
        ]

        metaval_task_distribution = TaskDistribution(
            tasks=val_tasks_list,
            task_probabilities=[1.0],
            batch_size=meta_batch_size,
            sample_with_replacement=True,
        )

        test_tasks_list = [
            CB_OCCTask(
                miniimagenet_testX,
                miniimagenet_testY,
                num_training_samples_per_class,
                val_test_tasks_num_test_samples_per_class,
                num_training_classes,
                split_train_test=0.5,
            )
        ]

        metatest_task_distribution = TaskDistribution(
            tasks=test_tasks_list,
            task_probabilities=[1.0],
            batch_size=meta_batch_size,
            sample_with_replacement=True,
        )

    return (
        metatrain_task_distribution,
        metaval_task_distribution,
        metatest_task_distribution,
    )
