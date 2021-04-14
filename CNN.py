# -*- coding: utf-8 -*-
"""
Generic CNN for Experiment 2
Labels: 
0 - Buildings
1 - Forest
2 - Glacier
3 - Mountain
4 - Sea
5 - Street

Created on Mon Mar 22 14:41:23 2021

@author: ancarey
"""

from __future__ import print_function
import os
from PIL import Image
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch import optim 
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from helper_functions import plot_confusion_matrix, plot_roc

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
       init.xavier_normal_(m.weight.data)
       init.normal_(m.bias.data)
       
def train_val_test_split(dataset, splits):
    datasets = {}
    first_idx, second_idx = train_test_split(list(range(dataset.__len__())), test_size=splits[1])
    datasets['train'] = Subset(dataset, first_idx)
    datasets['val'] = Subset(dataset, second_idx)
        
    return datasets

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    accuracy = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
            
        loss.backward()
        optimizer.step()
        
    incorrect = len(train_loader.dataset) - correct
    accuracy = 100. * correct / len(train_loader.dataset)
    lo = 100. * incorrect / len(train_loader.dataset)
    print(epoch, ":", accuracy, lo)
    return accuracy, lo

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    accuracy = 0
    predictions = []
    true = []

    with torch.no_grad():
        for data, target in test_loader: 
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            true.extend(target.view_as(pred))
            predictions.extend(pred)
            
    #test_loss /= len(test_loader.dataset)
    incorrect = len(test_loader.dataset) - correct
    #print(test_loss)
    accuracy = 100. * correct / len(test_loader.dataset)
    test_loss = 100. * incorrect / len(test_loader.dataset)
    #print(test_loss)
    return accuracy, [i.item() for i in true], [i.item() for i in predictions], test_loss

class image_Dataset(torch.utils.data.Dataset):
  def __init__(self, root, data_class, transform):
        self.root = root 
        self.data_class = data_class 
        self.dataset = os.listdir(self.root)
        self.is_transform = transform 
        
  def __len__(self):
        return len(self.dataset)

  def __getitem__(self, idx):
        label = self.data_class
        imagePath = self.root + self.dataset[idx]
        img = Image.open(imagePath).convert('RGB')
        img = self.is_transform(img)
        return img, label
    
class CNN(nn.Module):
     def __init__(self):
        super(CNN, self).__init__()
        
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
      
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(100, 9)
        
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
        #print(x.shape)
         
        x = torch.flatten(x, 1) 
        
        x = self.fc1(x)              
        x = F.relu(x)
        x = self.dropout(x)   
        x = self.fc3(x)                   
       
        
        output = F.log_softmax(x, dim=1) 
        
        return output
# ''' ================= Create Datasets ================='''

# #Paths
# building_rgb_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\train\\buildings\\'
# forest_rgb_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\train\\forest\\'
# glacier_rgb_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\train\\glacier\\'
# mountain_rgb_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\train\\mountain\\'
# sea_rgb_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\train\\sea\\'
# street_rgb_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\train\\street\\'

# rgb_val_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\val\\'

# building_rgb_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\test\\buildings\\'
# forest_rgb_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\test\\forest\\'
# glacier_rgb_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\test\\glacier\\'
# mountain_rgb_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\test\\mountain\\'
# sea_rgb_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\test\\sea\\'
# street_rgb_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\test\\street\\'

building_grayscale_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\train\\buildings\\'
forest_grayscale_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\train\\forest\\'
glacier_grayscale_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\train\\glacier\\'
mountain_grayscale_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\train\\mountain\\'
sea_grayscale_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\train\\sea\\'
street_grayscale_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\train\\street\\'

building_grayscale_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\test\\buildings\\'
forest_grayscale_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\test\\forest\\'
glacier_grayscale_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\test\\glacier\\'
mountain_grayscale_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\test\\mountain\\'
sea_grayscale_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\test\\sea\\'
street_grayscale_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\test\\street\\'

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,))#transforms.Normalize((0.1307,), (0.3981,))
    ])

#Split into train and test
batch_size = 64
epochs = 50
lr = 0.001

#best
#epochs = 50
#lr = 0.001 gray2

building_train = image_Dataset(building_grayscale_train_path, 0, transform)
forest_train = image_Dataset(forest_grayscale_train_path, 1, transform)
glacier_train = image_Dataset(glacier_grayscale_train_path, 2, transform)
mountain_train = image_Dataset(mountain_grayscale_train_path, 3, transform)
sea_train = image_Dataset(sea_grayscale_train_path, 4, transform)
street_train = image_Dataset(street_grayscale_train_path, 5, transform)

building_test = image_Dataset(building_grayscale_test_path, 0, transform)
forest_test = image_Dataset(forest_grayscale_test_path, 1, transform)
glacier_test = image_Dataset(glacier_grayscale_test_path, 2, transform)
mountain_test = image_Dataset(mountain_grayscale_test_path, 3, transform)
sea_test = image_Dataset(sea_grayscale_test_path, 4, transform)
street_test = image_Dataset(street_grayscale_test_path, 5, transform)

# building_train = image_Dataset(building_rgb_train_path, 0, transform)
# forest_train = image_Dataset(forest_rgb_train_path, 1, transform)
# glacier_train = image_Dataset(glacier_rgb_train_path, 2, transform)
# mountain_train = image_Dataset(mountain_rgb_train_path, 3, transform)
# sea_train = image_Dataset(sea_rgb_train_path, 4, transform)
# street_train = image_Dataset(street_rgb_train_path, 5, transform)

# building_test = image_Dataset(building_rgb_test_path, 0, transform)
# forest_test = image_Dataset(forest_rgb_test_path, 1, transform)
# glacier_test = image_Dataset(glacier_rgb_test_path, 2, transform)
# mountain_test = image_Dataset(mountain_rgb_test_path, 3, transform)
# sea_test = image_Dataset(sea_rgb_test_path, 4, transform)
# street_test = image_Dataset(street_rgb_test_path, 5, transform)


train_split = .95
val_split = .05

building_split = train_val_test_split(building_train, splits=[train_split, val_split])
building_train = building_split['train']
building_val = building_split['val']

forest_split = train_val_test_split(forest_train, splits=[train_split, val_split])
forest_train = forest_split['train']
forest_val = forest_split['val']

glacier_split = train_val_test_split(glacier_train, splits=[train_split, val_split])
glacier_train = glacier_split['train']
glacier_val = glacier_split['val']

mountain_split = train_val_test_split(mountain_train, splits=[train_split, val_split])
mountain_train = mountain_split['train']
mountain_val = mountain_split['val']

sea_split = train_val_test_split(sea_train, splits=[train_split, val_split])
sea_train = sea_split['train']
sea_val = sea_split['val']

street_split = train_val_test_split(street_train, splits=[train_split, val_split])
street_train = street_split['train']
street_val = street_split['val']

train_set = building_train + forest_train + glacier_train +  mountain_train + sea_train + street_train 
test_set = building_test + forest_test + glacier_test +  mountain_test + sea_test + street_test 
val_set = building_val + forest_val + glacier_val + mountain_val + sea_val + street_val

train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_set, batch_size, shuffle = True)

model = CNN()
model.apply(weight_init)
model.to('cuda')

log_interval = 1
device = 'cuda'

optimizer = optim.SGD(model.parameters(), lr = lr)

accuracy = []
loss = []
acc = 0
val_loss = []
val_acc = []
#train the model
for epoch in range(1, epochs + 1):
    a, l = train(log_interval, model, device, train_loader, optimizer, epoch)
    accuracy.append(a)
    loss.append(l)
    _, true, pred, lo = test(model, device, val_loader)
    acc_val = accuracy_score(true, pred) * 100.
    print(acc_val)
    val_loss.append(lo)
    val_acc.append(acc_val)

#save_path = "C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\GrayandRGB\RGB.pth"
save_path = "C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\GrayandRGB\Gray4.pth" #gray2 best
torch.save(model.state_dict(), save_path) 

print(accuracy)

plt.plot(accuracy)
plt.plot(loss)
plt.plot(val_loss)
plt.plot(val_acc)
plt.title("Gray Training Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy %")

#Generate Confusion Matrix for test 
_, true, pred, _ = test(model, device, test_loader)


cm = confusion_matrix(true, pred)
names = ('building','forest','glacier','mountain','sea','street')
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, names, t='Gray Confusion Matrix')

f1 = f1_score(true, pred, average='micro')
acc = accuracy_score(true, pred)
print('Gray Test accuracy:', acc)

file = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\results\\gray_and_rgb\\ind_acc.txt'
file1 = open(file, "a")
write_string = "Gray Testing Accuracy: " + str(acc)+ "\t F1: "+ str(f1) + "\n"
file1.write(write_string)
file1.close()

plot_roc(model, device, test_loader, num_classes=6, t='Gray ROC', mode='single')