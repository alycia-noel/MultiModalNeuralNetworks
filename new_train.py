# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:20:49 2021

@author: ancarey
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import transforms
from new_model import create_multiview_data, SVHN_Network, MNIST_Network

''' ================= Helper Methods ================= '''
def train_test_data_split(dataset, splits):
    datasets = {}
    
    first_idx, second_idx = train_test_split(list(range(dataset.__len__())), test_size=splits[1])
    datasets['first'] = Subset(dataset, first_idx)
    datasets['second']= Subset(dataset, second_idx)

    return datasets

''' ================= Set Variables ================= '''
batch_size = 64
lr = 0.001
epochs = 500 
num_classes = 2
drouput = False
fully_connected = False
GPU = True
shape = (64, 256, 28, 28)
file = ''
param = False
freeze = False
weighted_sum = False
train_split = .7
test_split = .3
splits=[train_split, test_split]

''' ================= Create Datasets ================='''
file_MNIST = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\MNISTandSVHN\\MNIST\\'
file_SVHN = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\MNISTandSVHN\\SVHN\\'

transform = transforms.Compose([transforms.ToTensor()])
data_class_zero = create_multiview_data(file_MNIST, file_SVHN, 0, transform)
zero_split = train_test_data_split(data_class_zero, splits)
zero_train = zero_split['first']
zero_test = zero_split['second']

data_class_one = create_multiview_data(file_MNIST, file_SVHN, 1, transform)
one_split = train_test_data_split(data_class_one, splits)
one_train = one_split['first']
one_test = one_split['second']

data_class_two = create_multiview_data(file_MNIST, file_SVHN, 2, transform)
two_split = train_test_data_split(data_class_two, splits)
two_train = two_split['first']
two_test = two_split['second']

data_class_three = create_multiview_data(file_MNIST, file_SVHN, 3, transform)
three_split = train_test_data_split(data_class_three, splits)
three_train = three_split['first']
three_test = three_split['second']

data_class_four = create_multiview_data(file_MNIST, file_SVHN, 4, transform)
four_split = train_test_data_split(data_class_four, splits)
four_train = four_split['first']
four_test = four_split['second']

data_class_five = create_multiview_data(file_MNIST, file_SVHN, 5, transform)
five_split = train_test_data_split(data_class_five, splits)
five_train = five_split['first']
five_test = five_split['second']

data_class_six = create_multiview_data(file_MNIST, file_SVHN, 6, transform)
six_split = train_test_data_split(data_class_six, splits)
six_train = six_split['first']
six_test = six_split['second']

data_class_seven = create_multiview_data(file_MNIST, file_SVHN, 7, transform)
seven_split = train_test_data_split(data_class_seven, splits)
seven_train = seven_split['first']
seven_test = seven_split['second']

data_class_eight = create_multiview_data(file_MNIST, file_SVHN, 8, transform)
eight_split = train_test_data_split(data_class_eight, splits)
eight_train = eight_split['first']
eight_test = eight_split['second']

data_class_nine = create_multiview_data(file_MNIST, file_SVHN, 9, transform)
nine_split = train_test_data_split(data_class_nine, splits)
nine_train = nine_split['first']
nine_test = nine_split['second']

''' ================= Put Data into Dataloaders ================= '''
train_dataset = zero_train + one_train + two_train + three_train + four_train + five_train + six_train + seven_train + eight_train + nine_train
test_dataset = zero_test + one_test + two_test + three_test + four_test + five_test + six_test + seven_test + eight_test + nine_test

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle = False)

# for i, data in enumerate(train_loader, 0):
#     #print(data[2])
#     c1_image = data[0]
#     c2_image= data[1]
#     label = data[2]
#     print(label, type(c1_image), type(c2_image))

''' ================= Initialize Network ================= '''
mnist_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\MNISTandSVHN\\MNIST.pth'
mnist_net = MNIST_Network()
mnist_net.load_state_dict(torch.load(mnist_path))


svhn_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\MNISTandSVHN\\SVHN.pth'
svhn_net = SVHN_Network()
svhn_net.load_state_dict(torch.load(svhn_path))
