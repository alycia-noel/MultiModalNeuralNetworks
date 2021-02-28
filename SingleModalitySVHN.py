# -*- coding: utf-8 -*-
"""
Solo modality classification network for the SVHN dataset
CNN based

Created on Wed Feb  3 11:42:23 2021

@author: ancarey
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class S_Network(nn.Module):
    def __init__(self):
        super(S_Network, self).__init__()
        
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
        
        x = torch.flatten(x, 1)   # [64, 50176]  
        x = self.fc1(x)           # [64, 128]       
        x = F.relu(x)
        x = self.dropout(x)             
        x = self.fc2(x)           # [64, 10]          
        
        output = F.log_softmax(x, dim=1) # [64, 10]
        
        
        return output

# def train(log_interval, model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.cross_entropy(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'. format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
            
# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     accuracy = 0
    
#     with torch.no_grad():
#         for data, target in test_loader: 
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.cross_entropy(output, target).item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
            
#     test_loss /= len(test_loader.dataset)
#     accuracy = 100. * correct / len(test_loader.dataset)
    
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
    
#     return accuracy 

# batch_size = 64
# test_batch_size = 64
# epochs = 14
# lr = .001
# log_interval = 10
# acc = []    
       
# transform = transforms.Compose([
#     transforms.CenterCrop((28,28)),
#     transforms.ToTensor()
#     #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ])        

# train_set = torchvision.datasets.SVHN('../data', split='train', transform=transform, target_transform=None, download=True) 
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)  

# test_a_set = torchvision.datasets.SVHN('../data', split='test', transform=transforms.ToTensor(),download=True)
# test_a_loader = torch.utils.data.DataLoader(test_a_set, batch_size=batch_size, shuffle=True)

# test_set = torchvision.datasets.SVHN('../data', split='test', transform=transform, target_transform=None, download=True)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# images_train, labels_train = next(iter(train_loader)) #[128, 3, 28, 28], [128]
# images_test, labels_test = next(iter(test_loader))    #[128, 3, 28, 28], [128]
# images_test_a, labels_test_a = next(iter(test_a_set))

# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")

# model = S_Network().to(device)
# optimizer = optim.Adam(model.parameters(), lr= lr)

# for epoch in range(1, epochs + 1):
#     train(log_interval, model, device, train_loader, optimizer, epoch)
#     acc.append(test(model, device, test_loader))

# print(acc)
    
# PATH = './models/MNISTandSVHN/SVHN.pth'
# torch.save(model.state_dict(), PATH)

# plt.plot(acc)
# plt.title("SVHN Accuracy over Epochs")
# plt.xlabel("epoch")
# plt.ylabel("accuracy")



