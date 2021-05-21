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
import numpy as np
from torch.autograd import Variable
import torch.fft as afft

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
        inp_one = Image.open(imagePath_one).convert('RGB') 
        inp_two = Image.open(imagePath_two).convert('RGB') 
        if self.is_transform:
            inp_one = self.is_transform(inp_one)
            inp_two = self.is_transform(inp_two)
        return(inp_one, inp_two, label)

class exp2_create_multiview_data(Dataset):
     def __init__(self, root_one, root_two, data_class, transform):
        self.root_one = root_one
        self.root_two = root_two
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
        inp_one = Image.open(imagePath_one).convert('RGB') 
        inp_two = Image.open(imagePath_two).convert('RGB') 
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
    
class Generic_CNN(nn.Module):
    def __init__(self):
        super(Generic_CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=0)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=0)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=0)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, padding=0)
      
        
        self.batchnorm1 = nn.BatchNorm2d(num_features=32)
        self.batchnorm2 = nn.BatchNorm2d(num_features=32)
        self.batchnorm3 = nn.BatchNorm2d(num_features=64)
        self.batchnorm4 = nn.BatchNorm2d(num_features=128)
        self.batchnorm5 = nn.BatchNorm2d(num_features=256)
        self.batchnorm6 = nn.BatchNorm2d(num_features=256)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.dropout = nn.Dropout(0.2)
      
       
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv3(x)       
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.conv4(x)       
        x = self.batchnorm4(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
       
        
        x = self.conv5(x)        
        x = self.batchnorm5(x)
        x = F.relu(x)
        x = self.maxpool(x)

        
        x = self.conv6(x)      
        x = self.batchnorm6(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        return x
    
class Same_Dim_Fusion_Layer(nn.Module):
    def __init__(self):
        super(Same_Dim_Fusion_Layer, self).__init__()
        self.feature_weight_one = nn.Parameter(torch.ones(192, 6, 6).view(-1, 192*6*6), requires_grad=True)
        self.feature_weight_two = nn.Parameter(torch.zeros(192, 6, 6).view(-1, 192*6*6), requires_grad=True)

    def forward(self, class_one, class_two):
        x = (self.feature_weight_one * class_one.view(-1, 192*6*6)) + (self.feature_weight_two * class_two.view(-1, 192*6*6))
        
        return x

class EXP2_Same_Dim_Fusion_Layer(nn.Module):
    def __init__(self):
        super(EXP2_Same_Dim_Fusion_Layer, self).__init__()
        self.feature_weight_one = nn.Parameter(torch.ones(256, 1, 1), requires_grad=True)
        self.feature_weight_two = nn.Parameter(torch.zeros(256, 1, 1).uniform_(0,1), requires_grad=True)

    def forward(self, class_one, class_two):
        x = (self.feature_weight_one * class_one) + (self.feature_weight_two * class_two)
        #print(x)
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
    
class EXP_2_FC_Fusion_Layer(nn.Module):
    def __init__(self):
        super(EXP_2_FC_Fusion_Layer, self).__init__()
        self.fusion = nn.Linear(in_features = 2 * 256 * 1 * 1, out_features = 256 * 1 * 1)
        
    def forward(self, class_one, class_two):
        flatten_class_one = torch.flatten(class_one, 1)          
        flatten_class_two = torch.flatten(class_two, 1)
        combined_classes = torch.cat((flatten_class_one, flatten_class_two), 1)
        
        x = self.fusion(combined_classes)
        
        return x

class fc_bilinear_fusion(nn.Module):
    def __init__(self):
        super(fc_bilinear_fusion, self).__init__()
        self.fusion = nn.Bilinear(256, 256, 256)

    def forward(self, class_one, class_two):
        c_one = torch.flatten(class_one, 1)
        c_two = torch.flatten(class_two, 1)

        output = self.fusion(c_one, c_two)
        return output
    
class compact_bilinear_fusion(nn.Module):
    def __init__(self):
        super(compact_bilinear_fusion, self).__init__()
        self.input_dim1 = 256
        self.input_dim2 = 256
        self.output_dim = 256
        self.sum_pool = True
        rand_h_1=None
        rand_s_1=None
        rand_h_2=None
        rand_s_2=None
        cuda = True
        
        if rand_h_1 is None:
            np.random.seed(1)
            rand_h_1 = np.random.randint(self.output_dim, size=self.input_dim1)
        if rand_s_1 is None:
            np.random.seed(3)
            rand_s_1 = 2 * np.random.randint(2, size=self.input_dim1) - 1

        self.sparse_sketch_matrix1 = Variable(self.generate_sketch_matrix(rand_h_1, rand_s_1, self.output_dim))

        if rand_h_2 is None:
            np.random.seed(5)
            rand_h_2 = np.random.randint(self.output_dim, size=self.input_dim2)
        if rand_s_2 is None:
            np.random.seed(7)
            rand_s_2 = 2 * np.random.randint(2, size=self.input_dim2) - 1

        self.sparse_sketch_matrix2 = Variable(self.generate_sketch_matrix(
            rand_h_2, rand_s_2, self.output_dim))

        if cuda:
            self.sparse_sketch_matrix1 = self.sparse_sketch_matrix1.cuda()
            self.sparse_sketch_matrix2 = self.sparse_sketch_matrix2.cuda()
            
    def generate_sketch_matrix(self, rand_h, rand_s, output_dim):
        rand_h = rand_h.astype(np.int64)
        rand_s = rand_s.astype(np.float32)
        assert(rand_h.ndim == 1 and rand_s.ndim ==
               1 and len(rand_h) == len(rand_s))
        assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))

        input_dim = len(rand_h)
        indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                                  rand_h[..., np.newaxis]), axis=1)
        indices = torch.from_numpy(indices)
        rand_s = torch.from_numpy(rand_s)
        sparse_sketch_matrix = torch.sparse.FloatTensor(
            indices.t(), rand_s, torch.Size([input_dim, output_dim]))
        return sparse_sketch_matrix.to_dense()
        
    def forward(self, class_one, class_two):
        batch_size, depth, height, width = class_one.size()
        
        class_one_flat = class_one.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim1)
        class_two_flat = class_two.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim2)
        
        sketch_1 = class_one_flat.mm(self.sparse_sketch_matrix1)
        sketch_2 = class_two_flat.mm(self.sparse_sketch_matrix2)
        
        fft1_real = afft.fft(sketch_1)
        fft1_imag = afft.fft(Variable(torch.zeros(sketch_1.size())).cuda())
        fft2_real = afft.fft(sketch_2)
        fft2_imag = afft.fft(Variable(torch.zeros(sketch_2.size())).cuda())

        fft_product_real = fft1_real.mul(fft2_real) - fft1_imag.mul(fft2_imag)
        fft_product_imag = fft1_real.mul(fft2_imag) + fft1_imag.mul(fft2_real)

        cbp_flat = afft.ifft(fft_product_real)
        #cbd_flat_2 = afft.ifft(fft_product_imag)[0]

        cbp = cbp_flat.view(batch_size, height, width, self.output_dim)

        if self.sum_pool:
            cbp = cbp.sum(dim=1).sum(dim=1)
        
        return cbp.float()

    
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

class EXP2_Post_Fusion_Layer(nn.Module):
    def __init__(self):
        super(EXP2_Post_Fusion_Layer, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(100, 9)
        
    def forward(self, x):
        x = torch.flatten(x, 1) 
        #print(x.shape)
        x = self.fc1(x)              
        x = F.relu(x)
        x = self.dropout(x)   
        x = self.fc3(x)                   
       
        
        output = F.log_softmax(x, dim=1) 
        
        return output

class EXP2_Full_Network(nn.Module):
    def __init__(self, ir_net, rgb_net, post_fusion_net, fusion_type, train_pre_net):
        super(EXP2_Full_Network, self).__init__()
        self.pre_net_one = ir_net
        self.pre_net_two = rgb_net
        self.fusion_type = fusion_type
        if self.fusion_type == 'FC-BF':
            self.fuse = fc_bilinear_fusion()
        elif self.fusion_type == 'C-BF':
            self.fuse = compact_bilinear_fusion()
        elif self.fusion_type == 'SD':
            self.fuse = EXP2_Same_Dim_Fusion_Layer()
        elif self.fusion_type == 'FC':
            self.fuse = EXP_2_FC_Fusion_Layer()
        self.post_net = post_fusion_net
        self.train_pre_net = train_pre_net
        
        if self.train_pre_net == False:
            freeze(self.pre_net_one)
            freeze(self.pre_net_two)
    
    def forward(self, class_ir, class_rgb):
        pre_c1_x = self.pre_net_one(class_ir)
        pre_c2_x = self.pre_net_two(class_rgb)
        fusion = self.fuse(pre_c1_x, pre_c2_x)
        output = self.post_net(fusion)

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
   # print(params)
    fw_1 = params['fuse.feature_weight_one'] #[192, 6, 6]
    fw_2 = params['fuse.feature_weight_two'] #[192, 6, 6]
    #print(fw_1.shape)
    f_weight_one, f_weight_two = sort_rows(fw_1.view(-1, 256*1*1), fw_2.view(-1, 256*1*1))
    params['fuse.feature_weight_one'].copy_(f_weight_one.view(256, 1, 1))
    params['fuse.feature_weight_two'].copy_(f_weight_two.view(256, 1, 1))


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