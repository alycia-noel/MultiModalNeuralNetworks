# -*- coding: utf-8 -*-
"""
Solo modality classification network for the MNIST dataset
CNN based

Created on Wed Feb  3 11:42:23 2021

@author: ancarey
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class M_Network(nn.Module):
    def __init__(self):
        super(M_Network, self).__init__()
        self.conv0 = nn.Conv2d(1, 16, kernel_size = 3, stride = 1, padding = 1)
        self.conv1 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1)
        self.dropout = nn.Dropout(0.2)
        
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
        x = self.dropout(x)             #[64, 256, 14, 14]
        
        # Fusion will happen here 
        
        x = torch.flatten(x, 1)          #[64, 50176]
        x = self.fc1(x)                  #[64, 128]
        x = F.relu(x)
        x = self.dropout(x)             #[64, 128]
        x = self.fc2(x)                  #[64, 10]
        
        output = F.log_softmax(x, dim=1) #[64, 10]
      
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
#             test_loss += F.cross_entropy(output, target, reduction='sum').item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
            
#     test_loss /= len(test_loader.dataset)
#     accuracy = 100. * correct / len(test_loader.dataset)
    
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
    
#     return accuracy 
 
# batch_size = 64
# test_batch_size = 1000
# epochs = 14
# lr = .006
# seed = 1
# log_interval = 10
# save_model = False
# accuracy = []

# use_cuda = torch.cuda.is_available()
# torch.manual_seed(seed)
# device = torch.device("cuda" if use_cuda else "cpu")
    
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3981,))
#     ])

# train_set = datasets.MNIST('../data', train=True, download=True, transform=transform)
# test_set = datasets.MNIST('../data', train=False, transform=transform)

# train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size)

# model = M_Network().to(device)
# optimizer = optim.Adam(model.parameters(), lr= lr)

# for epoch in range(1, epochs + 1):
#     train(log_interval, model, device, train_loader, optimizer, epoch)
#     accuracy.append(test(model, device, test_loader))

# print(accuracy)

# PATH = './models/MNISTandSVHN/MNIST.pth'
# torch.save(model.state_dict(), PATH)

# plt.plot(accuracy)
# plt.title("MNIST Accuracy over Epochs")
# plt.xlabel("epoch")
# plt.ylabel("accuracy")
