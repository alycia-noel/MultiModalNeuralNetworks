# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:20:49 2021

@author: ancarey
"""

import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.optim as optim 
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import transforms
from new_model import create_multiview_data, SVHN_Network, MNIST_Network, Post_Fusion_Layer, Full_Network
from SingleModalityMNIST import M_Network
from SingleModalitySVHN import S_Network
from PIL import Image
''' ================= Helper Methods ================= '''
def train_test_data_split(dataset, splits):
    datasets = {}
    
    first_idx, second_idx = train_test_split(list(range(dataset.__len__())), test_size=splits[1])
    datasets['first'] = Subset(dataset, first_idx)
    datasets['second']= Subset(dataset, second_idx)

    return datasets

def show_image(img):
    print(img[0].shape)
    img = img[0].permute(1, 2, 0)
    plt.imshow(img)
    
''' ================= Set Variables ================= '''
batch_size = 1#64
lr = 0.001
epochs = 500 
train_split = .7
test_split = .3
splits=[train_split, test_split]

''' ================= Create Datasets ================='''
file_MNIST = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\MNISTandSVHN\\MNIST\\'
file_SVHN = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\MNISTandSVHN\\SVHN\\'

transform = transforms.Compose([transforms.Resize((28, 28), interpolation=Image.NEAREST),
                                transforms.ToTensor()
                                ])

data_class_zero = create_multiview_data(file_MNIST, file_SVHN, 0, transform)
zero_split = train_test_data_split(data_class_zero, splits)
zero_train = zero_split['first']
zero_test = zero_split['second']
#show_image(zero_train[0])

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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle = False)


''' ================= Initialize Network ================= '''
print("Loading Networks:")
##### MNIST Pre Fusion #####
mnist_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\MNISTandSVHN\\MNIST.pth'
mnist_pre_net = M_Network()
mnist_pre_net_dict = mnist_pre_net.state_dict()
mnist_pre_net.load_state_dict(torch.load(mnist_path))
mnist_net = MNIST_Network()
mnist_dict = mnist_net.state_dict()
pretrained_dict = {k: v for k, v in mnist_pre_net_dict.items() if k in mnist_dict}
mnist_dict.update(pretrained_dict) 
mnist_net.load_state_dict(pretrained_dict)
print("*** MNIST Loaded")

##### SVHN Pre Fusion #####
svhn_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\MNISTandSVHN\\SVHN.pth'
svhn_pre_net = S_Network()
svhn_pre_net_dict = svhn_pre_net.state_dict()
svhn_pre_net.load_state_dict(torch.load(svhn_path))
svhn_net = SVHN_Network()
svhn_dict = svhn_net.state_dict()
pretrained_dict_2 = {k: v for k, v in svhn_pre_net_dict.items() if k in svhn_dict}
svhn_dict.update(pretrained_dict_2) 
svhn_net.load_state_dict(pretrained_dict_2)
print("*** SVHN Loaded")

##### Post Fusion #####
post_fusion_net = Post_Fusion_Layer()
post_fusion_dict = post_fusion_net.state_dict()
pretrained_dict_3 = {k: v for k, v in svhn_pre_net_dict.items() if k in post_fusion_dict}
post_fusion_dict.update(pretrained_dict_3) 
post_fusion_net.load_state_dict(pretrained_dict_3)
print("*** Post Fusion Loaded")

##### Full Network #####
full_net = Full_Network(train_pre_net = False)
full_net.to('cuda')
print("*** Full Network Loaded")

optimizer = optim.Adam(full_net.parameters(), lr= lr)
print("*** Optimizer: ", optimizer)

''' ================= Training ================= '''
print("\n*** Beginning Training ***\n")
train_loss = []

for epoch in range(epochs):
    training_loss = 0.0
    full_net.train()
    
    for i, data in enumerate(train_loader):
        mnist_img, svhn_img, labels = data

        optimizer.zero_grad()
        
        outputs = full_net(mnist_img.to('cuda'), svhn_img.to('cuda'))
        print(outputs.detach().cpu().numpy(), labels.item())
        loss = F.cross_entropy(outputs, labels.to('cuda'))
        
        loss.backward()
        optimizer.step()
        
        training_loss += loss.item()
        
    average_loss = training_loss/len(train_loader.dataset)
    train_loss.append(average_loss)
    print(f'Average loss on Epoch {epoch}: {average_loss}')
    
plt.plot(train_loss)
plt.title("Training Loss over Epochs")
plt.xlabel("epoch")
plt.ylabel("accuracy")
