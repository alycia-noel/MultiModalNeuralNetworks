# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:33:10 2021

@author: ancarey
"""

import numpy as np
import torch
import torchvision
from torchvision import transforms#, datasets
import matplotlib.pyplot as plt

plt.ioff()

def save(image, i, data_class, train_or_test = 'train'):
    if train_or_test == 'train':
        folder = folder_train
    else: 
        folder = folder_test
    plt.axis('off')
    plt.imshow((image*255).astype(np.uint8))
    save_dir = folder + str(data_class) + '/' + str(i) + '.png'
    plt.savefig(save_dir, bbox_inches='tight', pad_inches = 0)
    plt.close()

# folder_train = './data/NumberClassification/MNIST/train/'
# folder_test = './data/NumberClassification/MNIST/test/'
folder_train = './data/NumberClassification/SVHN/train/'
folder_test = './data/NumberClassification/SVHN/test/'

# '''Data for MNIST'''
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3981,))
#     ])

# train_set = datasets.MNIST('../data', train=True, download=True, transform=transform)
# test_set = datasets.MNIST('../data', train=False, transform=transform)

# train_loader = torch.utils.data.DataLoader(train_set, batch_size=60000) #73257
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000)   #26032 , 531131

# images_train, labels_train = next(iter(train_loader))
# images_test, labels_test = next(iter(test_loader))


'''Data for SVHN'''
transform = transforms.Compose([
    transforms.CenterCrop((28,28)),
    transforms.ToTensor()
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]) 

train_set = torchvision.datasets.SVHN('../data', split='train', transform=transform, target_transform=None, download=True) 
train_loader = torch.utils.data.DataLoader(train_set, batch_size=73257)  

test_set = torchvision.datasets.SVHN('../data', split='test', transform=transforms.ToTensor(),download=True)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=26032)

images_train, labels_train = next(iter(train_loader)) 
images_test, labels_test = next(iter(test_loader))   

counter0 = 0
counter1 = 0
counter2 = 0
counter3 = 0
counter4 = 0
counter5 = 0
counter6 = 0
counter7 = 0 
counter8 = 0 
counter9 = 0

for i in range (0, 73257): #60000 for MNIST and 73257 for SVHN
    c = 0
    img = images_train[i]
    data_class = labels_train[i].item()
    #print(data_class)
    if data_class == 0:    
        counter0 = counter0 + 1
        c = counter0
    elif data_class == 1:
        counter1 = counter1 + 1
        c = counter1
    elif data_class == 2:
        counter2 = counter2 + 1
        c = counter2
    elif data_class == 3:
        counter3 = counter3 + 1
        c = counter3
    elif data_class == 4:
        counter4 = counter4 + 1
        c = counter4
    elif data_class == 5:
        counter5 = counter5 + 1
        c = counter5
    elif data_class == 6:
        counter6 = counter6 + 1
        c = counter6
    elif data_class == 7:
        counter7 = counter7 + 1
        c = counter7
    elif data_class == 8:
        counter8 = counter8 + 1
        c = counter8
    else:
        counter9 = counter9 + 1
        c = counter9
    #print(data_class, c, counter0, counter1, counter2, counter3, counter4, counter5, counter6, counter7, counter8, counter9)
    img = img.numpy().transpose((1,2,0))
    save(img, c, data_class, train_or_test = 'train')
    
    if i % 1000 == 0:
        print('Done: {} \tLeft: {}'. format(i, 73257 - i))

counter0 = 0
counter1 = 0
counter2 = 0
counter3 = 0
counter4 = 0
counter5 = 0
counter6 = 0
counter7 = 0 
counter8 = 0 
counter9 = 0
        
for i in range (0, 26032): #10000 for MNIST and 26032 for SVHN
    c = 0
    img = images_test[i]
    data_class = labels_test[i].item()
    if data_class == 0: #10 for SVHN or 0 for MNIST
        counter0 = counter0 + 1
        c = counter0
    elif data_class == 1:
        counter1 = counter1 + 1
        c = counter1
    elif data_class == 2:
        counter2 = counter2 + 1
        c = counter2
    elif data_class == 3:
        counter3 = counter3 + 1
        c = counter3
    elif data_class == 4:
        counter4 = counter4 + 1
        c = counter4
    elif data_class == 5:
        counter5 = counter5 + 1
        c = counter5
    elif data_class == 6:
        counter6 = counter6 + 1
        c = counter6
    elif data_class == 7:
        counter7 = counter7 + 1
        c = counter7
    elif data_class == 8:
        counter8 = counter8 + 1
        c = counter8
    else:
        counter9 = counter9 + 1
        c = counter9
   
    img = img.numpy().transpose((1,2,0))
    save(img, c, data_class, train_or_test = 'test')
    
    if i % 1000 == 0:
        print('Done: {} \tLeft: {}'. format(i, 26032 - i))       