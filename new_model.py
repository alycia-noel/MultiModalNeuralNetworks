# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:20:39 2021

@author: ancarey
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image

class create_multiview_data(Dataset):
    def __init__(self, root_one, root_two, data_class, transform):
        self.root_one = root_one + str(data_class) + '\\'
        self.root_two = root_two + str(data_class) + '\\'
        self.data_class = data_class
        self.dataset_one = os.listdir(self.root_one)
        self.dataset_two = os.listdir(self.root_two)
        self.is_transform = transform
        
    def __len__(self):
        return len(self.dataset_one)
    
    def __getitem__(self, idx):
        label = self.data_class
        imagePath_one = self.root_one + self.dataset_one[idx]
        imagePath_two = self.root_two + self.dataset_two[idx]
        inp_one = Image.open(imagePath_one)
        inp_two = Image.open(imagePath_two)
        if self.is_transform:
            inp_one = self.is_transform(inp_one)
            inp_two = self.is_transform(inp_two)
        return(inp_one, inp_two, label)
            
class SVHN_Network(nn.Module):
    def __init__(self):
        super(SVHN_Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2)
        self.conv6 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2)
        self.conv8 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=1, padding=5)
        
        self.batchnorm1 = nn.BatchNorm2d(num_features=48)
        self.batchnorm2 = nn.BatchNorm2d(num_features=64)
        self.batchnorm3 = nn.BatchNorm2d(num_features=128)
        self.batchnorm4 = nn.BatchNorm2d(num_features=160)
        self.batchnorm5 = nn.BatchNorm2d(num_features=192)
        self.batchnorm6 = nn.BatchNorm2d(num_features=192)
        self.batchnorm7 = nn.BatchNorm2d(num_features=192)
        self.batchnorm8 = nn.BatchNorm2d(num_features=256)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.maxpool6 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        self.maxpool7 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        
        self.dropout = nn.Dropout(0.2)
      
        self.fc1 = nn.Linear(50176, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1(x)        # [64, 48, 28, 28]
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.maxpool1(x)     # [64, 48, 15, 15]
        x = self.dropout(x)

        x = self.conv2(x)        # [64, 64, 15, 15]
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.maxpool2(x)     # [64, 64, 16, 16]
        x = self.dropout(x)
        
        x = self.conv3(x)        # [64, 128, 16, 16]
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.maxpool3(x)     # [64, 128, 9, 9]
        x = self.dropout(x)
        
        x = self.conv4(x)        # [64, 160, 9, 9]
        x = self.batchnorm4(x)
        x = F.relu(x)
        x = self.maxpool4(x)     # [64, 160, 10, 10]
        x = self.dropout(x)
        
        x = self.conv5(x)        # [64, 192, 10, 10]
        x = self.batchnorm5(x)
        x = F.relu(x)
        x = self.maxpool5(x)     # [64, 192, 6, 6]
        x = self.dropout(x)
        
        x = self.conv6(x)        # [64, 192, 6, 6]      
        x = self.batchnorm6(x)
        x = F.relu(x)
        x = self.maxpool6(x)     # [64, 192, 7, 7]
        x = self.dropout(x)
        
        x = self.conv7(x)        # [64, 192, 7, 7]
        x = self.batchnorm7(x)
        x = F.relu(x)
        x = self.maxpool7(x)     # [64, 192, 4, 4]
        x = self.dropout(x)
        
        x = self.conv8(x)        # [64, 256, 14, 14]
        x = self.batchnorm8(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Fusion will happen here 
        
        # x = torch.flatten(x, 1)   # [64, 50176]  
        # x = self.fc1(x)           # [64, 128]       
        # x = F.relu(x)
        # x = self.dropout(x)             
        # x = self.fc2(x)           # [64, 10]          
        
        # output = F.log_softmax(x, dim=1) # [64, 10]
        
        
        return x    
    
class MNIST_Network(nn.Module):
    def __init__(self):
        super(MNIST_Network, self).__init__()
        self.conv0 = nn.Conv2d(1, 16, kernel_size = 3, stride = 1, padding = 1)
        self.conv1 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(50176, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x): 
        x = self.conv0(x)                #[64, 16, 28, 28]
        x = F.relu(x)
        x = self.conv1(x)                #[64, 32, 28, 28]
        x = F.relu(x)
        x = self.conv2(x)                #[64, 64, 28, 28]
        x = F.relu(x)
        x = self.conv3(x)                #[64, 128, 28, 28]
        x = F.relu(x)
        x = self.conv4(x)                #[64, 256, 28, 28]
        x = F.relu(x)
        x = F.max_pool2d(x, 2)           #[64, 256, 14, 14]
        x = self.dropout1(x)             #[64, 256, 14, 14]
        
        # Fusion will happen here 
        
        # x = torch.flatten(x, 1)          #[64, 50176]
        # x = self.fc1(x)                  #[64, 128]
        # x = F.relu(x)
        # x = self.dropout2(x)             #[64, 128]
        # x = self.fc2(x)                  #[64, 10]
        
        # output = F.log_softmax(x, dim=1) #[64, 10]
      
        return x
    
#     root_one = '/data/MNISTandSVHN/MNIST/' + data_class + '/'
#     root_two = '/data/MNISTandSVHN/SVHN/' + data_class + '/'
    
#     label = data_class
    
#     data

        
        
# create_multiview_dataset()