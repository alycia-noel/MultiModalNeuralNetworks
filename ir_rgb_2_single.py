# -*- coding: utf-8 -*-
"""
Generic CNN for Experiment 3
Labels: 
0 - Country
1 - Field
2 - Forest
3 - Indoor
4 - Mountain
5 - OldBuilding
6 - Street
7 - Urban
8 - Water

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
    datasets['test'] = Subset(dataset, second_idx)
        
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
torch.manual_seed(0)

# #Paths
country_rgb = "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\rgbdata\\countryRGB\\"
field_rgb = "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\rgbdata\\fieldRGB\\"
forest_rgb = "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\rgbdata\\forestRGB\\"
indoor_rgb = "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\rgbdata\\indoorRGB\\"
mountain_rgb = "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\rgbdata\\mountainRGB\\"
oldbuilding_rgb = "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\rgbdata\\oldbuildingRGB\\"
street_rgb = "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\rgbdata\\streetRGB\\"
urban_rgb = "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\rgbdata\\urbanRGB\\"
water_rgb = "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\rgbdata\\waterRGB\\"

# country_ir= "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\irdata\\countryIR\\"
# field_ir= "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\irdata\\fieldIR\\"
# forest_ir= "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\irdata\\forestIR\\"
# indoor_ir= "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\irdata\\indoorIR\\"
# mountain_ir= "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\irdata\\mountainIR\\"
# oldbuilding_ir= "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\irdata\\oldbuildingIR\\"
# street_ir= "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\irdata\\streetIR\\"
# urban_ir= "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\irdata\\urbanIR\\"
# water_ir= "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\irdata\\waterIR\\"

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,))#transforms.Normalize((0.1307,), (0.3981,))
    ])

#Split into train and test
batch_size = 64
epochs = 75 #50 is best so far
lr = 0.005

#best
#epochs = 50
#lr = 0.001 gray2


country = image_Dataset(country_rgb, 0, transform)
field = image_Dataset(field_rgb, 1, transform)
forest = image_Dataset(forest_rgb, 2, transform)
indoor = image_Dataset(indoor_rgb, 3, transform)
mountain = image_Dataset(mountain_rgb, 4, transform)
oldbuilding = image_Dataset(oldbuilding_rgb, 5, transform)
street = image_Dataset(street_rgb, 6, transform)
urban = image_Dataset(urban_rgb, 7, transform)
water = image_Dataset(water_rgb, 8, transform)

# country = image_Dataset(country_ir, 0, transform)
# field = image_Dataset(field_ir, 1, transform)
# forest = image_Dataset(forest_ir, 2, transform)
# indoor = image_Dataset(indoor_ir, 3, transform)
# mountain = image_Dataset(mountain_ir, 4, transform)
# oldbuilding = image_Dataset(oldbuilding_ir, 5, transform)
# street = image_Dataset(street_ir, 6, transform)
# urban = image_Dataset(urban_ir, 7, transform)
# water = image_Dataset(water_ir, 8, transform)

train_split = .8
test_split = .2

country_split = train_val_test_split(country, splits=[train_split, test_split])
country_train = country_split['train']
country_test= country_split['test']

field_split = train_val_test_split(field, splits=[train_split, test_split])
field_train = field_split['train']
field_test= field_split['test']

forest_split = train_val_test_split(forest, splits=[train_split, test_split])
forest_train = forest_split['train']
forest_test= forest_split['test']

indoor_split = train_val_test_split(indoor, splits=[train_split, test_split])
indoor_train = indoor_split['train']
indoor_test= indoor_split['test']

mountain_split = train_val_test_split(mountain, splits=[train_split, test_split])
mountain_train = mountain_split['train']
mountain_test= mountain_split['test']

oldbuilding_split = train_val_test_split(oldbuilding, splits=[train_split, test_split])
oldbuilding_train = oldbuilding_split['train']
oldbuilding_test= oldbuilding_split['test']

street_split = train_val_test_split(street, splits=[train_split, test_split])
street_train = street_split['train']
street_test= street_split['test']

urban_split = train_val_test_split(urban, splits=[train_split, test_split])
urban_train = urban_split['train']
urban_test= urban_split['test']

water_split = train_val_test_split(water, splits=[train_split, test_split])
water_train = water_split['train']
water_test= water_split['test']

train_set = country_train + field_train + forest_train + indoor_train +  mountain_train + oldbuilding_train + street_train + urban_train + water_train
test_set = country_test + field_test + forest_test + indoor_test +  mountain_test + oldbuilding_test + street_test + urban_test + water_test


train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False)


model = CNN()
#model.apply(weight_init)
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
    _, true, pred, lo = test(model, device, test_loader)
    acc_val = accuracy_score(true, pred) * 100.
    print(acc_val)
    val_loss.append(lo)
    val_acc.append(acc_val)

save_path = "C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\IRandRGB\\RGB.pth"
torch.save(model.state_dict(), save_path) 

print(accuracy)

plt.plot(accuracy)
plt.plot(loss)
plt.plot(val_loss)
plt.plot(val_acc)
plt.title("IR Training Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy %")

#Generate Confusion Matrix for test 
_, true, pred, _ = test(model, device, test_loader)


cm = confusion_matrix(true, pred)
names = ('country','field', 'forest','indoor','mountain','old building','street', 'urban', 'water')
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, names, t='IR Confusion Matrix')

f1 = f1_score(true, pred, average='micro')
acc = accuracy_score(true, pred)
print('RGB Test accuracy:', acc)

file = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\results\\ir_and_rgb\\ind_acc.txt'
file1 = open(file, "a")
write_string = "IR Testing Accuracy: " + str(acc)+ "\t F1: "+ str(f1) + "\n"
file1.write(write_string)
file1.close()

plot_roc(model, device, test_loader, num_classes=9, t='IR ROC', mode='single')