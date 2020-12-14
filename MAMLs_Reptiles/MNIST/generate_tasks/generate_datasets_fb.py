# -*- coding: utf-8 -*-
import numpy as np
from mlxtend.data import loadlocal_mnist
import random 
import pickle
import json
import os
import argparse



class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(config_file, args_dict):
    with open(config_file) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]

    return args_dict

def generate_multi_class_mnist_dataset(
        test_task_idx, 
        val_task_idx):


   
    train_tasks_idxs = [i for i in range(0,10) if i not in [val_task_idx, test_task_idx]]

    base_path = '/home/USER/Documents'
    if (not (os.path.exists(base_path))):
        base_path = '/home/ubuntu/Projects'
    if (not (os.path.exists(base_path))):
        base_path = '/home/USER/Projects'
    if (not (os.path.exists(base_path))):
        base_path = '/home/USER/Projects'
    train_images, train_labels = loadlocal_mnist(
        images_path= base_path + '/MAML/raw_data/MNIST_data/train-images-idx3-ubyte', 
        labels_path= base_path + '/MAML/raw_data/MNIST_data/train-labels-idx1-ubyte')


    images = train_images.reshape((-1,28,28))/255.0
    labels = train_labels

    print('images:',images.shape, ' labels :', labels.shape)

    train_digits_indexes = [index for index, element in enumerate(list(labels)) if element not in [val_task_idx, test_task_idx]] 


    images, labels = images[train_digits_indexes], labels[train_digits_indexes]
    print('images:',images.shape, ' labels :', labels.shape)


    unique_labels = list(set(labels))
    print('unique labels before ', unique_labels)

    new_label = 0

    for unique_label in unique_labels:
        for image_idx in range(len(images)):
            if(labels[image_idx] == unique_label):
                labels[image_idx] = new_label

        new_label +=1

    unique_labels = list(set(labels))
    print('unique labels after ', unique_labels)

    s = np.arange(images.shape[0])
    np.random.shuffle(s)
    images, labels = images[s], labels[s]

                        
    train_data = {'X': images,
                  'Y': labels}


    return train_data

def main(args):

    seed = 123

    np.random.seed(seed)
    random.seed(seed)

    base_path = '/home/USER/Documents'
    if (not (os.path.exists(base_path))):
        base_path = '/home/ubuntu/Projects' 
    if (not (os.path.exists(base_path))):
        base_path = '/home/USER/Projects'
    if (not (os.path.exists(base_path))):
        base_path = '/home/USER/Projects'
    dir_path = base_path + '/MAML/input_data/FB_MNIST_val' + str(args.val_task_idx) + '_test' + str(args.test_task_idx)
    

    train_data = generate_multi_class_mnist_dataset( 
        test_task_idx = args.test_task_idx, 
        val_task_idx = args.val_task_idx)

    if (not (os.path.exists(dir_path))):
        os.mkdir(dir_path)
    else:
        raise KeyboardInterrupt ('These datasets were already generated. Delete the folder: ', dir_path, ' to generate them again.')


    train_data_file = dir_path + '/fb_train_data_val_' + str(args.val_task_idx) + '_test_' + str(args.test_task_idx) +'.txt'
    with open(train_data_file, 'wb') as file:
        pickle.dump(train_data, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='generate FB-MNIST training dataset')
    
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
