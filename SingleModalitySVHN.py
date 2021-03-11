# -*- coding: utf-8 -*-
"""
Solo modality classification network for the SVHN dataset
CNN based

Created on Wed Feb  3 11:42:23 2021

@author: ancarey
"""
from __future__ import print_function

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch import optim 
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from helper_functions import plot_confusion_matrix, plot_roc, train, test

class S_Network(nn.Module):
    def __init__(self):
        super(S_Network, self).__init__()
        
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
      
        self.fc1 = nn.Linear(6912, 128)
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
        
        x = torch.flatten(x, 1)  # [64, 50176]  
        x = self.fc1(x)          # [64, 128]       
        x = F.relu(x)
        x = self.dropout(x)             
        x = self.fc2(x)          # [64, 10]          
        
        output = F.log_softmax(x, dim=1) # [64, 10]
        
        return output

def run_svhn_training(batch_size, epochs, lr, log_interval, device, file):
    accuracy = []
    acc = 0   
           
    transform = transforms.Compose([transforms.CenterCrop((28,28)),
                                    transforms.ToTensor()
                                    ])        
    
    train_set = datasets.SVHN('../data', split='train', transform=transform, target_transform=None, download=False) 
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)  

    test_set = datasets.SVHN('../data', split='test', transform=transform, target_transform=None, download=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    model = S_Network().to(device)
    optimizer = optim.SGD(model.parameters(), lr= lr)
    
    for epoch in range(1, epochs + 1):
        accuracy.append(train(log_interval, model, device, train_loader, optimizer, epoch))

    PATH = './models/MNISTandSVHN/SVHN.pth'
    torch.save(model.state_dict(), PATH)

    #Generating Confusion Matrix for test
    _, true, pred = test(model, device, test_loader)
    
    cm = confusion_matrix(true, pred)
    names = ('0','1','2','3','4','5','6','7','8','9')
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(cm, names, t='SVHN Confusion Matrix' )

    f1 = f1_score(true, pred, average='micro')
    acc = accuracy_score(true, pred)

    #Write test acc to file
    file1 = open(file, "a")
    write_string = "SVHN Accuracy: " + str(acc)+ "\t F1: "+ str(f1) + "\n"
    file1.write(write_string)
    file1.close()
    
    #Plotting ROC 
    plot_roc(model, device, test_loader, num_classes=10, t='SVHN ROC', mode='single')
    
    return accuracy