"""
Giacomo Spigler. Meta-learnt priors slow down catastrophic forgetting in neural networks. arXiv preprint arXiv:1909.04170, 2019.
Specification of the base Task interfaces.

+ Task: base Task class
+ ClassificationTask: generic class with utilities to wrap datasets for classification problems.
+ RLTask: generic class with utilities to wrap reinforcement learning environments.

+ TaskAsTaskSequence: wrapper class that builds "tasks" that represent a sequence of tasks. Each sub-task can be queried
  independently, to allow for testing their final performance after all tasks have been learnt sequentially. This class
  is designed to be a starting point for continual learning research.
"""

import numpy as np
import tensorflow as tf

from copy import deepcopy


class Task:
    def reset():
        """
        Reset a task after using it. For example, in case of ClassificationTask this would sample a new subset
        of instances for each class. In case of RLTask, it would force a reset the environment. Note that the
        environment is automatically reset on sampling new data during training.
        In case of derived classes (e.g., OmniglotTaskSampler or TaskAsTaskSequence), reset would re-sample a new task
        from the appropriate distribution.
        """
        pass


class ClassificationTask(Task):
    def __init__(
        self,
        X,
        y,
        num_training_samples_per_class=-1,
        num_test_samples_per_class=-1,
        num_training_classes=-1,
        split_train_test=0.8,
    ):
        """
            X: ndarray [size, features, ...]
                Training dataset X.
            y: ndarray [size]
                Training dataset y.
            num_training_samples_per_class: int
                If -1, sample from the whole dataset. If >=1, the dataset will re-sample num_training_samples_per_class
                for each class at each reset, and only sample from them when queried, until the next reset.
                This is useful for, e.g., k-shot classification.
            num_test_samples_per_class: int
                If -1, sample from the whole dataset. If >=1, the dataset will re-sample num_test_samples_per_class
                for each class at each reset, and only sample from them when queried, until the next reset.
            num_training_classes: int
                If -1, use all the classes in `y'. If >=1, the dataset will re-sample num_training_classes at
                each reset, and only sample from them when queried, until the next reset.
            split_train_test : float [0,1], or <0
                On each reset, the instances in the dataset are first split into a train and test set. From those,
                num_training_samples_per_class and num_test_samples_per_class are sampled.
                If `split_train_test' < 0, then the split is automatically set to #train_samples / (#train_samples + #test_samples)
        """
        self.X = X
        self.y = y

        self._deepcopy_avoid_copying = ["X", "y"]

        self.num_training_classes = num_training_classes
        if self.num_training_classes >= len(set(self.y)):
            print('self.num_training_classes', self.num_training_classes)
            print('len(set(self.y))', len(set(self.y)))
            self.num_training_classes = -1
            print(
                "WARNING: more training classes than available in the dataset were requested. \
                   All the available classes (", len(self.y), ") will be used.", )

        self.split_train_test = split_train_test
        if self.split_train_test < 0:
            self.split_train_test = num_training_samples_per_class / (
                num_training_samples_per_class + num_test_samples_per_class
            )

        num_classes = (
            self.num_training_classes
            if self.num_training_classes > 0
            else len(set(self.y))
        )
        self.num_training_samples_per_class = num_training_samples_per_class
        if (
            self.num_training_samples_per_class * num_classes
            >= len(self.y) * self.split_train_test
        ):
            self.num_training_samples_per_class = -1
            print(
                "WARNING: more training samples per class than available training instances were requested. \
                   All the available instances (",
                int(len(self.y) * self.split_train_test),
                ") will be used.",
            )

        self.num_test_samples_per_class = num_test_samples_per_class
        if self.num_test_samples_per_class * num_classes >= len(self.y) * (
            1 - self.split_train_test
        ):
            self.num_test_samples_per_class = -1
            print(
                "WARNING: more test samples per class than available test instances were requested. \
                   All the available instances (",
                int(len(self.y) * (1 - self.split_train_test)),
                ") will be used.",
            )

        self.reset()

    def __deepcopy__(self, memo):
        """
        We override __deepcopy__ to prevent deep-copying the dataset (self.X and self.y), which will thus be shared
        between all deepcopied instances of the object.
        This is best for performance, as the dataset is not modifying by the Task instances, which rather only store
        the per-reset indices of the train and test samples.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if (
                hasattr(self, "_deepcopy_avoid_copying")
                and k not in self._deepcopy_avoid_copying
            ):
                setattr(result, k, deepcopy(v, memo))
        setattr(result, "X", self.X)
        setattr(result, "y", self.y)
        return result

    def reset(self):
        classes_to_use = list(set(self.y))
        if self.num_training_classes >= 1:
            classes_to_use = np.random.choice(
                classes_to_use, self.num_training_classes, replace=False
            )

        self.train_indices = []
        self.test_indices = []

        # For each class, take list of indices, sample k
        for c in classes_to_use:
            class_indices = np.where(self.y == c)[0]
            np.random.shuffle(class_indices)

            train_test_separation = int(
                len(class_indices) * self.split_train_test)
            all_train_indices = class_indices[0:train_test_separation]
            all_test_indices = class_indices[train_test_separation:]

            if self.num_training_samples_per_class >= 1:
                all_train_indices = np.random.choice(
                    all_train_indices,
                    self.num_training_samples_per_class,
                    replace=False,
                )
            if self.num_test_samples_per_class >= 1:
                all_test_indices = np.random.choice(
                    all_test_indices, self.num_test_samples_per_class, replace=False)

            self.train_indices.extend(all_train_indices)
            self.test_indices.extend(all_test_indices)

        np.random.shuffle(self.train_indices)
        np.random.shuffle(self.test_indices)

        # rename train and test indices so that they run in [0, num_of_classes)
        self.classes_ids = list(classes_to_use)

    # def evaluate(self, model, evaluate_on_train=False):
    #     """
    #     Evaluate a Keras `model' on the current Task, according to the metrices provided when building the model.
    #     """
    #     if evaluate_on_train:
    #         test_X, test_y = self.get_train_set()
    #     else:
    #         test_X, test_y = self.get_test_set()

    #     # Only valid with TFv1? (in this case, with the model defined using Keras function APIs
    #     # out = model.evaluate(test_X, test_y, batch_size=1000, verbose=0)
    #     # if not isinstance(out, list):
    #     #    out = [out]
    #     # out_dict = dict(zip(model.metrics_names, out))

    #     # TODO: wrap in a tf.function?
    #     # TODO: also: for... minibatch, run this code, which corresponds to .update_state
    #     out_dict = {}
    #     if model.metrics_names[0] == "loss":
    #         mets = [
    #             lambda true, pred: tf.reduce_mean(model.loss(true, pred))
    #             + (tf.add_n(model.losses) if len(model.losses) > 0 else 0)
    #         ] + model.metrics
    #     for i in range(len(model.metrics_names)):
    #         val = mets[i](test_y, model(test_X))
    #         if model.metrics_names[i] != "loss":
    #             mets[i].reset_states()

    #         out_dict[model.metrics_names[i]] = val

    #     return out_dict

    def sample_batch(self, batch_size):
        batch_indices = np.random.choice(
            self.train_indices, batch_size, replace=False)
        batch_X = self.X[batch_indices]
        batch_y = np.asarray([self.classes_ids.index(c)
                              for c in self.y[batch_indices]], dtype=np.int64)
        return batch_X, batch_y

    def get_train_set(self):
        return (
            self.X[self.train_indices],
            np.asarray(
                [self.classes_ids.index(c) for c in self.y[self.train_indices]],
                dtype=np.int64,
            ),
        )

    def get_test_set(self):
        return (
            self.X[self.test_indices],
            np.asarray(
                [self.classes_ids.index(c) for c in self.y[self.test_indices]],
                dtype=np.int64,
            ),
        )


class OCCTask(ClassificationTask):
    def __init__(
        self,
        X,
        y,
        num_training_samples_per_class=-1,
        num_test_samples_per_class=-1,
        num_training_classes=-1,
        split_train_test=0.8,
        input_parse_fn=None,
        num_parallel_processes=None,
    ):

        super().__init__(
            X=X,
            y=y,
            num_training_samples_per_class=num_training_samples_per_class,
            num_test_samples_per_class=num_test_samples_per_class,
            num_training_classes=num_training_classes,
            split_train_test=split_train_test,
        )

    def reset(self):

        classes_to_use = list(set(self.y))
        # print('classes_to_use', classes_to_use)

        self.normal_class = np.random.choice(classes_to_use, 1)[0]

        # if self.num_training_classes >= 1:
        #     classes_to_use = np.random.choice(classes_to_use, self.num_training_classes, replace=False)

        self.train_indices = []
        self.test_indices = []

        # add normal examples
        normal_class_indices = np.where(self.y == self.normal_class)[0]
        np.random.shuffle(normal_class_indices)
        all_train_indices = normal_class_indices[:
                                                 self.num_training_samples_per_class]
        all_test_normal_indices = np.random.choice(
            normal_class_indices[self.num_training_samples_per_class:],
            self.num_test_samples_per_class,
        )
        self.train_indices.extend(all_train_indices)
        self.test_indices.extend(all_test_normal_indices)

        all_test_anomalous_indices = []
        while len(all_test_anomalous_indices) < self.num_test_samples_per_class:
            c = np.random.choice(classes_to_use, 1)[0]
            if c != self.normal_class:
                class_indices = np.where(self.y == c)[0]
                np.random.shuffle(class_indices)
                anomalous_idx = np.random.choice(class_indices, 1)[0]
                all_test_anomalous_indices.append(anomalous_idx)

        self.test_indices.extend(all_test_anomalous_indices)

        np.random.shuffle(self.train_indices)
        # np.random.shuffle(self.test_indices)

        # rename train and test indices so that they run in [0, num_of_classes)
        self.classes_ids = list(classes_to_use)

        # make sure examples from only one class are available for training
        assert len(set(self.y[self.train_indices])) == 1

    def sample_batch(self, batch_size):
        batch_indices = np.random.choice(
            self.train_indices, batch_size, replace=False)
        batch_X = self.X[batch_indices]
        batch_y = np.asarray(
            [0 if c == self.normal_class else 1 for c in self.y[batch_indices]],
            dtype=np.int64,
        )
        return batch_X, batch_y

    def get_train_set(self):
        # print([
        #             0 if c == self.normal_class else 1
        #             for c in self.y[self.train_indices]
        #         ])
        return (
            self.X[self.train_indices],
            np.asarray(
                [
                    0 if c == self.normal_class else 1
                    for c in self.y[self.train_indices]
                ],
                dtype=np.int64,
            ),
        )

    def get_test_set(self):
        return (
            self.X[self.test_indices],
            np.asarray(
                [0 if c == self.normal_class else 1 for c in self.y[self.test_indices]],
                dtype=np.int64,
            ),
        )


class CB_OCCTask(OCCTask):
    def __init__(
        self,
        X,
        y,
        num_training_samples_per_class=-1,
        num_test_samples_per_class=-1,
        num_training_classes=-1,
        split_train_test=0.8,
        input_parse_fn=None,
        num_parallel_processes=None,
    ):

        super().__init__(
            X=X,
            y=y,
            num_training_samples_per_class=num_training_samples_per_class,
            num_test_samples_per_class=num_test_samples_per_class,
            num_training_classes=num_training_classes,
            split_train_test=split_train_test,
        )

    def reset(self):

        classes_to_use = list(set(self.y))
        # print('classes_to_use', classes_to_use)

        self.normal_class = np.random.choice(classes_to_use, 1)[0]

        # if self.num_training_classes >= 1:
        #     classes_to_use = np.random.choice(classes_to_use, self.num_training_classes, replace=False)

        self.train_indices = []
        self.test_indices = []

        # add normal examples
        normal_class_indices = np.where(self.y == self.normal_class)[0]
        np.random.shuffle(normal_class_indices)

        all_train_normal_indices = normal_class_indices[:
                                                        self.num_training_samples_per_class]
        self.train_indices.extend(all_train_normal_indices)

        all_train_anomalous_indices = []
        while len(
                all_train_anomalous_indices) < self.num_training_samples_per_class:
            c = np.random.choice(classes_to_use, 1)[0]
            if c != self.normal_class:
                class_indices = np.where(self.y == c)[0]
                np.random.shuffle(class_indices)
                anomalous_idx = np.random.choice(class_indices, 1)[0]
                all_train_anomalous_indices.append(anomalous_idx)

        all_test_normal_indices = np.random.choice(
            normal_class_indices[self.num_training_samples_per_class:],
            self.num_test_samples_per_class,
        )
        self.train_indices.extend(all_train_anomalous_indices)
        self.test_indices.extend(all_test_normal_indices)

        all_test_anomalous_indices = []
        while len(all_test_anomalous_indices) < self.num_test_samples_per_class:
            c = np.random.choice(classes_to_use, 1)[0]
            if c != self.normal_class:
                class_indices = np.where(self.y == c)[0]
                np.random.shuffle(class_indices)
                anomalous_idx = np.random.choice(class_indices, 1)[0]
                all_test_anomalous_indices.append(anomalous_idx)

        self.test_indices.extend(all_test_anomalous_indices)

        np.random.shuffle(self.train_indices)
        np.random.shuffle(self.test_indices)

        # rename train and test indices so that they run in [0, num_of_classes)
        self.classes_ids = list(classes_to_use)

        # make sure examples from more than one class are available for
        # training
        assert len(set(self.y[self.train_indices])) != 1


class TaskAsSequenceOfTasks(Task):
    """
    The definition of a `Task' is ambiguous in the real world, where task boundaries are not always well defined,
    and when tasks often have a hierarchical structure.

    The `TaskAsSequenceOfTasks' class addresses this concern by representing a single `Task' as a sequence of tasks
    sampled from a distribution.
    """

    def __init__(self, tasks_distribution, min_length, max_length):
        """
        tasks_distribution: TaskDistribution
            Task distribution object to be used to sample tasks from when generating the sequences.
        min_length / max_length: int
            Minimum and maximum number of tasks in the sequence, inclusive.
        """
        self.tasks_distribution = tasks_distribution

        self.current_task_sequence = []

        self.min_length = min_length
        self.max_length = max_length
        self.reset()

    def set_length_of_sequence(self, min_length, max_length):
        self.min_length = min_length
        self.max_length = max_length
        self.reset()

    def get_task_by_index(self, index):
        assert index >= -1 and index < len(
            self.current_task_sequence
        ), "INVALID TASK INDEX"
        return self.current_task_sequence[index]

    def get_sequence_length(self):
        return len(self.current_task_sequence)

    def reset(self):
        """
        Generate a new sequence of tasks.
        """
        new_length = np.random.randint(self.min_length, self.max_length + 1)
        self.current_task_sequence = self.tasks_distribution.sample_batch(
            batch_size=new_length
        )

    def sample_batch(self, batch_size):
        print(
            "NOT IMPLEMENTED. Fit and evaluate the task using `fit_n_iterations' and `evaluate'"
        )
        pass

    # def evaluate(self, model, evaluate_last_task_only=False):
    #     """
    #     Evaluate the performance of the model on all tasks in the sequence, unless evaluate_last_task_only=True.
    #     """

    #     tasks_to_evaluate = self.current_task_sequence
    #     if evaluate_last_task_only:
    #         tasks_to_evaluate = [self.current_task_sequence[-1]]

    #     out_dict = {}
    #     for task_i in range(len(tasks_to_evaluate)):
    #         task = tasks_to_evaluate[task_i]
    #         ret = task.evaluate(model)

    #         if len(tasks_to_evaluate) == 1:
    #             out_dict = ret
    #         else:
    #             for key in ret.keys():
    #                 out_dict[key + "_" + str(task_i)] = ret[key]

    #     return out_dict

    def get_test_set(self, task_index=-1):
        assert task_index >= -1 and task_index < len(
            self.current_task_sequence
        ), "Invalid task index"
        return self.current_task_sequence[task_index].get_test_set()
