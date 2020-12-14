# -*- coding: utf-8 -*-
import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import Normalizer, StandardScaler
import random 
import pickle
import json
import os
import argparse
from PIL import Image



def read_dataset(data_dir):
    """
    Iterate over the characters in a data directory.
    Args:
      data_dir: a directory of alphabet directories.
    Returns:
      An iterable over Characters.
    The dataset is unaugmented and not split up into
    training and test sets.
    """
    for alphabet_name in sorted(os.listdir(data_dir)):
        alphabet_dir = os.path.join(data_dir, alphabet_name)
        if not os.path.isdir(alphabet_dir):
            continue
        for char_name in sorted(os.listdir(alphabet_dir)):
            if not char_name.startswith('character'):
                continue
            yield Character(os.path.join(alphabet_dir, char_name), 0)

def split_dataset(dataset, num_train=1200, num_val=20):
    """
    Split the dataset into a training and test set.
    Args:
      dataset: an iterable of Characters.
    Returns:
      A tuple (train, test) of Character sequences.
    """
    all_data = list(dataset)
    random.shuffle(all_data)
    return all_data[:-num_val], all_data[-num_val:]

def augment_dataset(dataset):
    """
    Augment the dataset by adding 90 degree rotations.
    Args:
      dataset: an iterable of Characters.
    Returns:
      An iterable of augmented Characters.
    """
    for character in dataset:
        for rotation in [0, 90, 180, 270]:
            yield Character(character.dir_path, rotation=rotation)

# pylint: disable=R0903
class Character:
    """
    A single character class.
    """
    def __init__(self, dir_path, rotation=0):
        self.dir_path = dir_path
        self.rotation = rotation
        self._cache = {}

    def sample(self, num_images):
        """
        Sample images (as numpy arrays) from the class.
        Returns:
          A sequence of 28x28 numpy arrays.
          Each pixel ranges from 0 to 1.
        """
        names = [f for f in os.listdir(self.dir_path) if f.endswith('.png')]
        random.shuffle(names)
        images = []
        for name in names[:num_images]:
            images.append(self._read_image(os.path.join(self.dir_path, name)))
        return images

    def get_all_images(self):
        """
        Sample images (as numpy arrays) from the class.
        Returns:
          A sequence of 28x28 numpy arrays.
          Each pixel ranges from 0 to 1.
        """
        names = [f for f in os.listdir(self.dir_path) if f.endswith('.png')]
        random.shuffle(names)
        images = []
        for name in names:
            images.append(self._read_image(os.path.join(self.dir_path, name)))
        return images


    def _read_image(self, path):
        if path in self._cache:
            return self._cache[path]
        with open(path, 'rb') as in_file:
            img = Image.open(in_file).resize((28, 28)).rotate(self.rotation)
            self._cache[path] = np.array(img).astype('float32')
        return self._cache[path]



class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(config_file, args_dict):
    with open(config_file) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]

    return args_dict

def generate_omniglot_tasks():

    base_path = '/home/USER/Documents'
    if (not (os.path.exists(base_path))):
        base_path = '/home/ubuntu/Projects'
    data_dir_train_val = base_path + '/MAML/raw_data/omniglot/images_background'

    train_set, val_set= split_dataset(read_dataset(data_dir_train_val))
    train_set, val_set= list(augment_dataset(train_set)), list(augment_dataset(val_set))

    data_dir_test = base_path + '/MAML/raw_data/omniglot/images_evaluation'

    test_set= list(read_dataset(data_dir_test))

    train_set = list(train_set)
    val_set, test_set = list(val_set), list(test_set)

    train_tasks, val_tasks, test_tasks = [], [], []

    n_characters_train_set, n_characters_val_set, n_characters_test_set = len(train_set), len(val_set), len(test_set)
    for character_idx, character in enumerate(train_set):
        train_task = {}
        all_images = character.get_all_images()

        n_inner_loop_normal = n_inner_loop_anomalous = (int(len(all_images)/2))
        X_inner_loop_normal, X_outer_loop_normal = all_images[:n_inner_loop_normal], all_images[n_inner_loop_normal:]

        n = 0
        X_anomalous_list = []

        n_anomalies_needed = n_inner_loop_anomalous*2

        while n < n_anomalies_needed:
            sampled_character_index = random.sample(list(range(n_characters_train_set)), 1)[0]
            if(sampled_character_index == character_idx):
                pass
            else:
                X_anomalous = train_set[sampled_character_index].sample(1)[0]
                X_anomalous_list.append(X_anomalous)
                n+=1

        X_inner_loop = []
        X_inner_loop+=X_inner_loop_normal
        X_inner_loop+=X_anomalous_list[:n_inner_loop_anomalous]

        Y_inner_loop = []
        Y_inner_loop += list(np.zeros(len(X_inner_loop_normal)))
        Y_inner_loop += list(np.ones(len(X_anomalous_list[:n_inner_loop_anomalous])))

        X_outer_loop = []
        X_outer_loop+=X_outer_loop_normal
        X_outer_loop+=X_anomalous_list[n_inner_loop_anomalous:]

        Y_outer_loop = []
        Y_outer_loop += list(np.zeros(len(X_outer_loop_normal)))
        Y_outer_loop += list(np.ones(len(X_anomalous_list[n_inner_loop_anomalous:])))

        # SHUFFLE
        s_inner, s_outer = np.arange(len(X_inner_loop)), np.arange(len(X_outer_loop))
        np.random.shuffle(s_inner)
        np.random.shuffle(s_outer)

        train_task['X_inner'], train_task['X_outer'] = np.array(X_inner_loop)[s_inner], np.array(X_outer_loop)[s_outer]
        train_task['Y_inner'], train_task['Y_outer'] = np.array(Y_inner_loop)[s_inner], np.array(Y_outer_loop)[s_outer]

        train_tasks.append(train_task)


    for character_idx, character in enumerate(val_set):
        val_task = {}
        all_images = character.get_all_images()

        n_inner_loop_normal = n_inner_loop_anomalous = (int(len(all_images)/2))
        X_inner_loop_normal, X_outer_loop_normal = all_images[:n_inner_loop_normal], all_images[n_inner_loop_normal:]

        n = 0
        X_anomalous_list = []

        n_anomalies_needed = n_inner_loop_anomalous*2

        while n < n_anomalies_needed:
            sampled_character_index = random.sample(list(range(n_characters_val_set)), 1)[0]
            if(sampled_character_index == character_idx):
                pass
            else:
                X_anomalous = val_set[sampled_character_index].sample(1)[0]
                X_anomalous_list.append(X_anomalous)
                n+=1

        X_inner_loop = []
        X_inner_loop+=X_inner_loop_normal
        X_inner_loop+=X_anomalous_list[:n_inner_loop_anomalous]

        Y_inner_loop = []
        Y_inner_loop += list(np.zeros(len(X_inner_loop_normal)))
        Y_inner_loop += list(np.ones(len(X_anomalous_list[:n_inner_loop_anomalous])))

        X_outer_loop = []
        X_outer_loop+=X_outer_loop_normal
        X_outer_loop+=X_anomalous_list[n_inner_loop_anomalous:]

        Y_outer_loop = []
        Y_outer_loop += list(np.zeros(len(X_outer_loop_normal)))
        Y_outer_loop += list(np.ones(len(X_anomalous_list[n_inner_loop_anomalous:])))

        # SHUFFLE
        s_inner, s_outer = np.arange(len(X_inner_loop)), np.arange(len(X_outer_loop))
        np.random.shuffle(s_inner)
        np.random.shuffle(s_outer)
        val_task['X_inner'], val_task['X_outer'] = np.array(X_inner_loop)[s_inner], np.array(X_outer_loop)[s_outer]
        val_task['Y_inner'], val_task['Y_outer'] = np.array(Y_inner_loop)[s_inner], np.array(Y_outer_loop)[s_outer]

        val_tasks.append(val_task)

    for character_idx, character in enumerate(test_set):
        test_task = {}
        all_images = character.get_all_images()

        n_inner_loop_normal = n_inner_loop_anomalous = (int(len(all_images)/2))
        X_inner_loop_normal, X_outer_loop_normal = all_images[:n_inner_loop_normal], all_images[n_inner_loop_normal:]
        
        n = 0
        X_anomalous_list = []
        n_anomalies_needed = n_inner_loop_anomalous*2

        while n < n_anomalies_needed:
            sampled_character_index = random.sample(list(range(n_characters_test_set)), 1)[0]
            if(sampled_character_index == character_idx):
                pass
            else:
                X_anomalous = test_set[sampled_character_index].sample(1)[0]
                X_anomalous_list.append(X_anomalous)
                n+=1

        X_inner_loop = []
        X_inner_loop+=X_inner_loop_normal
        X_inner_loop+=X_anomalous_list[:n_inner_loop_anomalous]

        Y_inner_loop = []
        Y_inner_loop += list(np.zeros(len(X_inner_loop_normal)))
        Y_inner_loop += list(np.ones(len(X_anomalous_list[:n_inner_loop_anomalous])))


        X_outer_loop = []
        X_outer_loop+=X_outer_loop_normal
        X_outer_loop+=X_anomalous_list[n_inner_loop_anomalous:]


        Y_outer_loop = []
        Y_outer_loop += list(np.zeros(len(X_outer_loop_normal)))
        Y_outer_loop += list(np.ones(len(X_anomalous_list[n_inner_loop_anomalous:])))


        # SHUFFLE
        s_inner, s_outer = np.arange(len(X_inner_loop)), np.arange(len(X_outer_loop))
        np.random.shuffle(s_inner)
        np.random.shuffle(s_outer)

        test_task['X_inner'], test_task['X_outer'] = np.array(X_inner_loop)[s_inner], np.array(X_outer_loop)[s_outer]
        test_task['Y_inner'], test_task['Y_outer'] = np.array(Y_inner_loop)[s_inner], np.array(Y_outer_loop)[s_outer]
        test_tasks.append(test_task)

    return train_tasks, val_tasks, test_tasks


def main(args):

    seed = 123

    np.random.seed(seed)
    random.seed(seed)


    train_tasks, val_tasks, test_tasks = generate_omniglot_tasks()


    base_path = '/home/USER/Documents'
    if (not (os.path.exists(base_path))):
        base_path = '/home/ubuntu/Projects' 
    dir_path = base_path + '/MAML/input_data/omniglot/30_20_split/'
    

    if (not (os.path.exists(dir_path))):
        os.mkdir(dir_path)

    train_tasks_file = dir_path + 'train_tasks.txt'
    with open(train_tasks_file, 'wb') as file:
        pickle.dump(train_tasks, file)

    val_tasks_file = dir_path + 'val_tasks.txt'
    with open(val_tasks_file, 'wb') as file:
        pickle.dump(val_tasks, file)

    test_tasks_file = dir_path + 'test_tasks.txt'
    with open(test_tasks_file, 'wb') as file:
        pickle.dump(test_tasks, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='generate omniglot datasets')

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
