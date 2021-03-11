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
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch import optim 
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from helper_functions import plot_confusion_matrix, plot_roc, train, test

class M_Network(nn.Module):
    def __init__(self):
        super(M_Network, self).__init__()
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
        
        x = torch.flatten(x, 1)          #[64, 6912]
        x = self.fc1(x)                  #[64, 128]
        x = F.relu(x)
        x = self.dropout(x)              #[64, 128]
        x = self.fc2(x)                  #[64, 10]
        
        output = F.log_softmax(x, dim=1) #[64, 10]
      
        return output
    
def run_mnist_training(batch_size, epochs, lr, log_interval, device, file):    
    test_batch_size = 1000
    accuracy = []
    acc = 0
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3981,))
                                    ])

    train_set = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('../data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size)
    
    PATH = './models/MNISTandSVHN/MNIST.pth'

    model = M_Network().to(device)
    optimizer = optim.SGD(model.parameters(), lr= lr)

    #Training the model
    for epoch in range(1, epochs + 1):
        accuracy.append(train(log_interval, model, device, train_loader, optimizer, epoch))

    torch.save(model.state_dict(), PATH)
    
    #Generating Confusion Matrix for test
    _, true, pred = test(model, device, test_loader)
    
    cm = confusion_matrix(true, pred)
    names = ('0','1','2','3','4','5','6','7','8','9')
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(cm, names, t='MNIST Confusion Matrix' )

    f1 = f1_score(true, pred, average='micro')
    acc = accuracy_score(true, pred)

    #Write test acc to file
    file1 = open(file, "a")
    write_string = "MNIST Accuracy: " + str(acc)+ "\t F1: "+ str(f1) + "\n"
    file1.write(write_string)
    file1.close()
    
    #Plotting ROC 
    plot_roc(model, device, test_loader, num_classes=10, t='MNIST ROC', mode='single')
    
    return accuracy



