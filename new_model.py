# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:20:39 2021

@author: ancarey
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
#import numpy as np

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
        inp_one = Image.open(imagePath_one).convert('L') 
        inp_two = Image.open(imagePath_two).convert('RGB') 
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

        self.batchnorm1 = nn.BatchNorm2d(num_features=48)
        self.batchnorm2 = nn.BatchNorm2d(num_features=64)
        self.batchnorm3 = nn.BatchNorm2d(num_features=128)
        self.batchnorm4 = nn.BatchNorm2d(num_features=160)
        self.batchnorm5 = nn.BatchNorm2d(num_features=192)

        
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        
        self.dropout = nn.Dropout(0.2)
        
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

        return x    
    
class MNIST_Network(nn.Module):
    def __init__(self):
        super(MNIST_Network, self).__init__()
        self.conv0 = nn.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(128, 192, kernel_size = 3, stride = 1, padding = 4)
        self.dropout = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(6912, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x): 
        x = self.conv0(x)                #[64, 16, 28, 28]
        x = F.relu(x)
        x = F.max_pool2d(x, 2)           #[64, 16, 14, 14]
        x = self.conv2(x)                #[64, 64, 14, 14]
        x = F.relu(x)
        x = self.conv3(x)                #[64, 128, 14, 14]
        x = F.relu(x)
        x = F.max_pool2d(x, 2)           #[64, 128, 7, 7]
        x = self.conv4(x)                #[64, 192, 13, 13]
        x = F.relu(x)
        x = F.max_pool2d(x, 2)           #[64, 192, 6, 6]
        x = self.dropout(x)              #[64, 192, 6, 6]
        
        return x
    
class Same_Dim_Fusion_Layer(nn.Module):
    def __init__(self):
        super(Same_Dim_Fusion_Layer, self).__init__()
        self.feature_weight_one = nn.Parameter(torch.randn(192, 6, 6).uniform_(0,1).view(-1, 192*6*6), requires_grad=True)
        self.feature_weight_two = nn.Parameter(torch.randn(192, 6, 6).uniform_(0,1).view(-1, 192*6*6), requires_grad=True)

    def forward(self, class_one, class_two):
        x = (self.feature_weight_one * class_one.view(-1, 192*6*6)) + (self.feature_weight_two * class_two.view(-1, 192*6*6))
        
        return x
    
class FC_Fusion_Layer(nn.Module):
    def __init__(self):
        super(FC_Fusion_Layer, self).__init__()
        self.fusion = nn.Linear(in_features = 2 * 192 * 6 * 6, out_features = 192 * 6 * 6)
        
    def forward(self, class_one, class_two):
        flatten_class_one = torch.flatten(class_one, 1)          
        flatten_class_two = torch.flatten(class_two, 1)
        combined_classes = torch.cat((flatten_class_one, flatten_class_two), 1)
        
        x = self.fusion(combined_classes)
        
        return x
    
class Post_Fusion_Layer(nn.Module):
    def __init__(self):
        super(Post_Fusion_Layer, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(6912, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)                  #[64, 128]
        x = F.relu(x)
        x = self.dropout(x)              #[64, 128]
        x = self.fc2(x)                  #[64, 10]
        
        output = F.log_softmax(x, dim=1) #[64, 10]
      
        return output
    
class Full_Network(nn.Module):
    def __init__(self, mnist_net, svhn_net, post_fusion_net, train_pre_net = False):
        super(Full_Network, self).__init__()
        self.pre_net_one = mnist_net
        self.pre_net_two = svhn_net
        self.fuse = Same_Dim_Fusion_Layer()
        self.post_net = post_fusion_net
        self.train_pre_net = train_pre_net
        
        if self.train_pre_net == False:
            freeze(self.pre_net_one)
            freeze(self.pre_net_two)
    
    def forward(self, class_mnist, class_svhn):
        pre_c1_x = self.pre_net_one(class_mnist)
        pre_c2_x = self.pre_net_two(class_svhn)
        fusion = self.fuse(pre_c1_x, pre_c2_x)
        output = self.post_net(fusion)

        return output
        
def freeze(model):
    for params in model.parameters():
        params.requires_grad = False   
    
def Algorithm_One(model):
    params = model.state_dict()
   
    fw_1 = params['fuse.feature_weight_one'] #[192, 6, 6]
    fw_2 = params['fuse.feature_weight_two'] #[192, 6, 6]
    f_weight_one, f_weight_two = sort_rows(fw_1, fw_2)
    params['fuse.feature_weight_one'].copy_(f_weight_one)
    params['fuse.feature_weight_two'].copy_(f_weight_two)


def projection_weight_cal(element_list):
    for j in range(0,len(element_list)):
        value = element_list[j] + (1-sum(element_list[0:j+1]))/(j+1)
        if value > 0:
            index = j+1
    lam = 1/(index)*(1-sum(element_list[0:index]))
    for i in range(0 , len(element_list)):
        element_list[i] = max(element_list[i]+lam,0)
    return element_list

def sort_rows(T_1, T_2):
    T_1 = T_1.t()
    T_2 = T_2.t()
    T_1 = T_1.cpu()
    T_2 = T_2.cpu()
    tensor_one = T_1.numpy()
    tensor_two = T_2.numpy()
    for i in range(0 , len(tensor_one)):
        tensor_one_element = tensor_one[i]
        tensor_two_element = tensor_two[i]
        element_list = [tensor_one_element, tensor_two_element]
        element_list.sort(reverse=True)
        updated_weights = projection_weight_cal(element_list)
        tensor_one[i] = updated_weights[0]
        tensor_two[i] = updated_weights[1]
    T_1 = T_1.t()
    T_2 = T_2.t()
    
    return (T_1, T_2)

    