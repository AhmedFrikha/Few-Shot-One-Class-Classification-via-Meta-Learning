# -*- coding: utf-8 -*-
import numpy as np
import random 
import pickle
import json
import os
import argparse
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_dataset(data_dir):
    """
    Read the Mini-ImageNet dataset (taken from https://github.com/openai/supervised-reptile/).
    Args:
      data_dir: directory containing Mini-ImageNet.
    Returns:
      A tuple (train, val, test) of sequences of
        ImageNetClass instances.
    """
    return tuple(_read_classes(os.path.join(data_dir, x)) for x in ['train', 'val', 'test'])

def _read_classes(dir_path):
    """
    Read the WNID directories in a directory (taken from https://github.com/openai/supervised-reptile/).
    """
    return [ImageNetClass(os.path.join(dir_path, f)) for f in os.listdir(dir_path)
            if f.startswith('n')]

# pylint: disable=R0903
class ImageNetClass:
    """
    A single image class (taken from https://github.com/openai/supervised-reptile/).
    """
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self._cache = {}

    def sample(self, num_images):
        """
        Sample images (as numpy arrays) from the class.
        Returns:
          A sequence of 84x84x3 numpy arrays.
          Each pixel ranges from 0 to 1.
        """
        names = [f for f in os.listdir(self.dir_path) if f.endswith('.JPEG')]
        random.shuffle(names)
        images = []
        for name in names[:num_images]:
            images.append(self._read_image(name))
        return images

    def get_all_images(self):
        """
        Sample images (as numpy arrays) from the class.
        Returns:
          A sequence of 84x84x3 numpy arrays.
          Each pixel ranges from 0 to 1.
        """
        names = [f for f in os.listdir(self.dir_path) if f.endswith('.JPEG')]
        random.shuffle(names)
        images = []
        for name in names:
            images.append(self._read_image(name))
        return images


    def _read_image(self, name):
        if name in self._cache:
            return self._cache[name].astype('float32') / 0xff
        with open(os.path.join(self.dir_path, name), 'rb') as in_file:
            img = Image.open(in_file).resize((84, 84)).convert('RGB')
            self._cache[name] = np.array(img)
        return self._read_image(name)

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(config_file, args_dict):
    with open(config_file) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]

    return args_dict

def generate_miniImageNet_tasks():

    base_path = '/home/USER/Documents'
    if (not (os.path.exists(base_path))):
        base_path = '/home/ubuntu/Projects'
    data_dir = base_path + '/MAML/raw_data/miniImageNet_data'

    train_set, val_set, test_set = read_dataset(data_dir)
    train_tasks, val_tasks, test_tasks = [], [], []
    n_image_classes_train_set, n_image_classes_val_set, n_image_classes_test_set = len(train_set), len(val_set), len(test_set)
    
    #generate meta-training tasks
    for image_class_idx, image_class in enumerate(train_set):
        train_task = {}
        all_images = image_class.get_all_images()
        n_inner_loop_normal = n_inner_loop_anomalous = (int(len(all_images)/2))
        X_inner_loop_normal, X_outer_loop_normal = all_images[:n_inner_loop_normal], all_images[n_inner_loop_normal:]
        
        # for meta-training tasks: sample 5 times as many anomalous examples (i.e. examples from other classes) as normal examples are available 
        # n_inner_loop_anomalous = n_inner_loop_normal*5
        X_anomalous_list = []
        n_anomalies_needed = n_inner_loop_anomalous*2
        n=0
        while n < n_anomalies_needed:
            sampled_image_class_index = random.sample(list(range(n_image_classes_train_set)), 1)[0]
            if(sampled_image_class_index == image_class_idx):
                pass
            else:
                X_anomalous = train_set[sampled_image_class_index].sample(1)[0]
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

        # shuffle
        s_inner, s_outer = np.arange(len(X_inner_loop)), np.arange(len(X_outer_loop))
        np.random.shuffle(s_inner)
        np.random.shuffle(s_outer)
        train_task['X_inner'], train_task['X_outer'] = np.array(X_inner_loop)[s_inner], np.array(X_outer_loop)[s_outer]
        train_task['Y_inner'], train_task['Y_outer'] = np.array(Y_inner_loop)[s_inner], np.array(Y_outer_loop)[s_outer]
        train_tasks.append(train_task)

    # generate meta-validation tasks
    for image_class_idx, image_class in enumerate(val_set):
        val_task = {}
        all_images = image_class.get_all_images()
        n_inner_loop_normal = n_inner_loop_anomalous = (int(len(all_images)/2))
        X_inner_loop_normal, X_outer_loop_normal = all_images[:n_inner_loop_normal], all_images[n_inner_loop_normal:]
        X_anomalous_list = []
        n_anomalies_needed = n_inner_loop_anomalous*2
        n=0
        while n < n_anomalies_needed:
            sampled_image_class_index = random.sample(list(range(n_image_classes_val_set)), 1)[0]
            if(sampled_image_class_index == image_class_idx):
                pass
            else:
                X_anomalous = val_set[sampled_image_class_index].sample(1)[0]
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

        # shuffle
        s_inner, s_outer = np.arange(len(X_inner_loop)), np.arange(len(X_outer_loop))
        np.random.shuffle(s_inner)
        np.random.shuffle(s_outer)
        val_task['X_inner'], val_task['X_outer'] = np.array(X_inner_loop)[s_inner], np.array(X_outer_loop)[s_outer]
        val_task['Y_inner'], val_task['Y_outer'] = np.array(Y_inner_loop)[s_inner], np.array(Y_outer_loop)[s_outer]
        val_tasks.append(val_task)

    # generate meta-testing tasks
    for image_class_idx, image_class in enumerate(test_set):
        test_task = {}
        all_images = image_class.get_all_images()
        n_inner_loop_normal = n_inner_loop_anomalous = (int(len(all_images)/2))
        X_inner_loop_normal, X_outer_loop_normal = all_images[:n_inner_loop_normal], all_images[n_inner_loop_normal:]
        X_anomalous_list = []
        n_anomalies_needed = n_inner_loop_anomalous*2
        n=0
        while n < n_anomalies_needed:
            sampled_image_class_index = random.sample(list(range(n_image_classes_test_set)), 1)[0]
            if(sampled_image_class_index == image_class_idx):
                pass
            else:
                X_anomalous = test_set[sampled_image_class_index].sample(1)[0]
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

        # shuffle
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

    train_tasks, val_tasks, test_tasks = generate_miniImageNet_tasks()

    base_path = '/home/USER/Documents'
    if (not (os.path.exists(base_path))):
        base_path = '/home/ubuntu/Projects' 
    dir_path = base_path + '/MAML/input_data/2020_02_19_miniImageNet/'
    if (not (os.path.exists(dir_path))):
        os.mkdir(dir_path)
    
    train_tasks_file = dir_path + 'miniImageNet_train_tasks.txt'
    with open(train_tasks_file, 'wb') as file:
        pickle.dump(train_tasks, file)

    val_tasks_file = dir_path + 'miniImageNet_val_tasks.txt'
    with open(val_tasks_file, 'wb') as file:
        pickle.dump(val_tasks, file)

    test_tasks_file = dir_path + 'miniImageNet_test_tasks.txt'
    with open(test_tasks_file, 'wb') as file:
        pickle.dump(test_tasks, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='generate miniImageNet tasks')
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
