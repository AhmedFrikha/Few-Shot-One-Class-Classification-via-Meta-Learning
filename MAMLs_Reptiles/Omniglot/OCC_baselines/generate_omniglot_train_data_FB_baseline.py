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

def read_dataset_5_char_per_alphabet(data_dir):
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
        sampled_indices = np.random.choice(np.arange(len(sorted(os.listdir(alphabet_dir)))), 5, replace=False)
        # print('chosen indices: ', sampled_indices)
        for char_idx, char_name in enumerate(sorted(os.listdir(alphabet_dir))):
            if(char_idx in sampled_indices):
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

# def _sample_mini_dataset(dataset, num_classes, num_shots):
#     """
#     Sample a few shot task from a dataset.
#     Returns:
#       An iterable of (input, label) pairs.
#     """
#     shuffled = list(dataset)
#     random.shuffle(shuffled)
#     for class_idx, class_obj in enumerate(shuffled[:num_classes]):
#         for sample in class_obj.sample(num_shots):
#     yield (sample, class_idx)


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(config_file, args_dict):
    with open(config_file) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]

    return args_dict

def generate_omniglot_fb_train_data():

    base_path = '/home/USER/Documents'
    if (not (os.path.exists(base_path))):
        base_path = '/home/ubuntu/Projects'
    data_dir_train = base_path + '/MAML/raw_data/omniglot/images_background'

    train_set= list(read_dataset_5_char_per_alphabet(data_dir_train))
  
    train_images, train_labels = [], []

    n_characters_train_set= len(train_set)
    print('n_characters_train_set', n_characters_train_set)
    for character_idx, character in enumerate(train_set):
        all_images = character.get_all_images()
        train_images += all_images
        train_labels += [character_idx]*len(train_images)


    s = np.arange(len(train_images))
    np.random.shuffle(s)


    train_data = {'X':np.array(train_images)[s],
                  'Y':np.array(train_labels)[s]}
    

    return train_data






def main(args):

    seed = 123

    np.random.seed(seed)
    random.seed(seed)
    print('seed set to: ', seed)


    train_data = generate_omniglot_fb_train_data()


    base_path = '/home/USER/Documents'
    if (not (os.path.exists(base_path))):
        base_path = '/home/ubuntu/Projects' 
    dir_path = base_path + '/MAML/input_data/omniglot/30_20_split/'
    
    if (not (os.path.exists(dir_path))):
        os.mkdir(dir_path)

    train_data_file = dir_path + 'fb_train_data.txt'
    with open(train_data_file, 'wb') as file:
        pickle.dump(train_data, file)

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='generate omniglot datasets')
    # parser.add_argument(
    #     '-datapoints_per_task',
    #     type=int,
    #     metavar='',
    #     # required=True,
    #     help='number of datapoints per task')    
    # parser.add_argument(
    #     '-K_list',
    #     type=str,
    #     metavar='',
    #     # required=True,
    #     help='number of data points sampled for training and testing')    
    # parser.add_argument(
    #     '-cir_inner_loop_list',
    #     type=str,
    #     metavar='',
    #     # required=True,
    #     help=(
    #         'percentage of positive examples of the dataset used for the inner'
    #         ' updatefor each training task'))
    # parser.add_argument(
    #     '-test_task_idx',
    #     type=int,
    #     metavar='',
    #     # required=True,
    #     help='index of the test task') 
    # parser.add_argument(
    #     '-val_task_idx',
    #     type=int,
    #     metavar='',
    #     # required=True,
    #     help='index of the val task')  
    # parser.add_argument(
    #     '-n_finetune_sets',
    #     type=int,
    #     metavar='',
    #     # required=True,
    #     help='number of sets for finetuning - test task') 
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
