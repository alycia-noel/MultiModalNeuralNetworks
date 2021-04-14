# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:20:49 2021

@author: ancarey
"""

import numpy as np
import torch
import torch.optim as optim 
import matplotlib.pyplot as plt
import torch.nn.functional as F
# sklearn.model_selection import train_test_split
#from torch.utils.data import Subset
from torchvision import transforms
from new_model import create_multiview_data, SVHN_Network, MNIST_Network, Post_Fusion_Layer, Full_Network, Algorithm_One
from SingleModalityMNIST import M_Network, run_mnist_training
from SingleModalitySVHN import S_Network, run_svhn_training
from PIL import Image
#import itertools
#from itertools import *
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from helper_functions import plot_acc_vs_epoch, plot_confusion_matrix, plot_roc,  plot_acc_vs_epoch_single
import pickle
''' ================= Methods ================= '''
def initialize_network(mode, load_weights, train_pre):
    torch.cuda.empty_cache()

    print("\nLoading Networks for", mode, "training: ")
    
    ##### MNIST Pre Fusion #####
    mnist_net = MNIST_Network()
    mnist_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\MNISTandSVHN\\MNIST.pth'
    if load_weights == True:
        mnist_pre_net = M_Network()
        mnist_pre_net_dict = mnist_pre_net.state_dict()
        mnist_pre_net.load_state_dict(torch.load(mnist_path))
        mnist_dict = mnist_net.state_dict()
        pretrained_dict = {k: v for k, v in mnist_pre_net_dict.items() if k in mnist_dict}
        mnist_dict.update(pretrained_dict) 
        mnist_net.load_state_dict(pretrained_dict)
    print("*** MNIST Loaded")
    
    ##### SVHN Pre Fusion #####
    svhn_net = SVHN_Network()
    svhn_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\MNISTandSVHN\\SVHN.pth'
    if load_weights == True:
        svhn_pre_net = S_Network()
        svhn_pre_net_dict = svhn_pre_net.state_dict()
        svhn_pre_net.load_state_dict(torch.load(svhn_path))
        svhn_dict = svhn_net.state_dict()
        pretrained_dict_2 = {k: v for k, v in svhn_pre_net_dict.items() if k in svhn_dict}
        svhn_dict.update(pretrained_dict_2) 
        svhn_net.load_state_dict(pretrained_dict_2)
    print("*** SVHN Loaded")

    ##### Post Fusion #####
    post_fusion_net = Post_Fusion_Layer()
    if load_weights == True:
        post_fusion_dict = post_fusion_net.state_dict()
        pretrained_dict_3 = {k: v for k, v in mnist_pre_net_dict.items() if k in post_fusion_dict}
        post_fusion_dict.update(pretrained_dict_3) 
        post_fusion_net.load_state_dict(pretrained_dict_3)
    print("*** Post Fusion Loaded")

    ##### Full Network #####
    full_net = Full_Network(mnist_net, svhn_net, post_fusion_net, train_pre_net = train_pre)
    full_net.to('cuda')
    print("*** Full Network Loaded")
    
    return full_net

def test_full_net(mode, model):
    print("\n*** Beginning Testing for", mode ," ***")
    test_loss = 0
    test_acc = []
    predictions = []
    true = []
    
    model.eval()
    with torch.no_grad():
        for batch, data in enumerate(test_loader):
            mnist_img, svhn_img, labels = data
            outputs = model(mnist_img.to('cuda'), svhn_img.to('cuda'))
            
            loss = F.cross_entropy(outputs, labels.to('cuda'))
            test_loss += loss.item()
        
            #_, pred = torch.max(outputs, 1)
    
            pred = outputs.argmax(dim=1, keepdim=True)
            true.extend(labels.view_as(pred))
            predictions.extend(pred.detach().cpu().numpy())
            
            test_acc.append((pred.detach().cpu() == labels).sum().item())
                
    lo = test_loss / len(test_loader.dataset)
    acc = 100. * np.sum(test_acc) / len(test_loader.dataset)

    return lo, acc, predictions, true

def train_full_net(mode, model, lr, save_path):
    print("\n*** Beginning Training for", mode, " ***")
    optimizer = optim.SGD(model.parameters(), lr= lr)
    train_loss = []
    train_acc = []
    
    for epoch in range(epochs):
        training_loss = 0.0
        cor = []
        accuracy = 0
        model.train()
        losses = []
        print("\n-- Beginning Epoch", epoch, "--")
        
        for batch, data in enumerate(train_loader):
            batch_cor = 0
            mnist_img, svhn_img, labels = data
    
            optimizer.zero_grad()
            
            outputs = model(mnist_img.to('cuda'), svhn_img.to('cuda'))
            loss = F.cross_entropy(outputs, labels.to('cuda'))
            pred = outputs.argmax(dim=1, keepdim=True).detach().cpu()
            batch_cor = pred.eq(labels.view_as(pred)).sum().item()
            cor.append(batch_cor)
            loss.backward()
            optimizer.step()
            Algorithm_One(model)
            losses.append(loss.item())
            training_loss += loss.item()
            
            if batch % 100 == 0: 
                print("Loss at Batch", batch, ":", losses[batch], "\tAccuracy at Batch", batch, ":", batch_cor/len(labels))
         
        accuracy = 100. * np.sum(cor) / len(train_loader.dataset)
        train_acc.append(accuracy)       
        average_loss = training_loss/len(train_loader.dataset)
        train_loss.append(average_loss)
        print(f'Average loss on Epoch {epoch }: {average_loss}', f'\tAverage accuracy on Epoch {epoch}: {accuracy}')
        
    torch.save(model.state_dict(), save_path) 
    
    return train_loss, train_acc

''' ================= Set Variables ================= '''
batch_size = 64
lr = 0.001
epochs = 200#100 #80 #10 #500 
file = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\results\\acc.txt'
open(file, 'w').close()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

''' ================= Create Datasets ================='''
train_MNIST = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\NumberClassification\\MNIST\\train\\'
train_SVHN = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\NumberClassification\\SVHN\\train\\'

test_MNIST = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\NumberClassification\\MNIST\\test\\'
test_SVHN = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\NumberClassification\\SVHN\\test\\'

transform = transforms.Compose([transforms.Resize((28, 28), interpolation=Image.NEAREST),
                                transforms.ToTensor()
                                ])

zero_train = create_multiview_data(train_MNIST, train_SVHN, 0, transform)
one_train = create_multiview_data(train_MNIST, train_SVHN, 1, transform)
two_train = create_multiview_data(train_MNIST, train_SVHN, 2, transform)
three_train = create_multiview_data(train_MNIST, train_SVHN, 3, transform)
four_train = create_multiview_data(train_MNIST, train_SVHN, 4, transform)
five_train = create_multiview_data(train_MNIST, train_SVHN, 5, transform)
six_train = create_multiview_data(train_MNIST, train_SVHN, 6, transform)
seven_train = create_multiview_data(train_MNIST, train_SVHN, 7, transform)
eight_train = create_multiview_data(train_MNIST, train_SVHN, 8, transform)
nine_train = create_multiview_data(train_MNIST, train_SVHN, 9, transform)

zero_test = create_multiview_data(test_MNIST, test_SVHN, 0, transform)
one_test = create_multiview_data(test_MNIST, test_SVHN, 1, transform)
two_test = create_multiview_data(test_MNIST, test_SVHN, 2, transform)
three_test = create_multiview_data(test_MNIST, test_SVHN, 3, transform)
four_test = create_multiview_data(test_MNIST, test_SVHN, 4, transform)
five_test = create_multiview_data(test_MNIST, test_SVHN, 5, transform)
six_test = create_multiview_data(test_MNIST, test_SVHN, 6, transform)
seven_test = create_multiview_data(test_MNIST, test_SVHN, 7, transform)
eight_test = create_multiview_data(test_MNIST, test_SVHN, 8, transform)
nine_test = create_multiview_data(test_MNIST, test_SVHN, 9, transform)

''' ================= Put Data into Dataloaders ================= '''
train_dataset = zero_train + one_train + two_train + three_train + four_train + five_train + six_train + seven_train + eight_train + nine_train
test_dataset = zero_test + one_test + two_test + three_test + four_test + five_test + six_test + seven_test + eight_test + nine_test

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle = False)

''' ================= Independent Training ================= '''
independent_batch_size = 64
MNIST_epochs = 14
SVHN_epochs = 50
MNIST_lr = .006
SVHN_lr =  .001
log_interval = 10

print("Independent Training:")
mnist_acc = run_mnist_training(batch_size, MNIST_epochs, MNIST_lr, log_interval, device, file)
print("*** MNIST Trained ")

# svhn_acc = run_svhn_training(batch_size, SVHN_epochs, SVHN_lr, log_interval, device, file)
# print("*** SVHN Trained ")

# ''' ================= With Pre-Fusion Training ================= '''
# mode = 'w/ pre-training'
# pre_train_path = "C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\MNISTandSVHN\\Pre_fusion_train_Full_Net.pth"
# pre_train_full_net = initialize_network(mode, load_weights = True, train_pre= True)
# with_pre_training_loss, with_pre_training_acc = train_full_net(mode, pre_train_full_net, lr, pre_train_path)

# pre_train_lo, pre_train_acc, pre_train_pred, pre_train_true = test_full_net(mode, pre_train_full_net)

# cm = confusion_matrix(pre_train_true, pre_train_pred)
# names = ('0','1','2','3','4','5','6','7','8','9')
# plt.figure(figsize=(10,10))
# plot_confusion_matrix(cm, names, t='With Pre-Fusion Training Confusion Matrix')

# f1 = f1_score(pre_train_true, pre_train_pred, average='micro')
# acc = accuracy_score(pre_train_true, pre_train_pred)

# file1 = open(file, "a")
# write_string = "With Pre-Fusion Training Accuracy: " + str(acc)+ "\t F1: "+ str(f1) + "\n"
# file1.write(write_string)
# file1.close()

# plot_roc(pre_train_full_net, device, test_loader, num_classes=10, t='With Pre-Fusion Training ROC', mode='multi')

# ''' ================= Without Pre-Fusion Training ================= '''
# mode = 'w/o pre-training'
# wo_pre_train_path = "C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\MNISTandSVHN\\Without_Pre_fusion_train_Full_Net.pth"
# wo_pre_train_full_net = initialize_network(mode, load_weights = True, train_pre= False)
# wo_pre_training_loss, wo_pre_training_acc = train_full_net(mode, wo_pre_train_full_net, lr, wo_pre_train_path)

# wo_pre_train_lo, wo_pre_train_acc, wo_pre_train_pred, wo_pre_train_true = test_full_net(mode, wo_pre_train_full_net)

# cm = confusion_matrix(wo_pre_train_true, wo_pre_train_pred)
# names = ('0','1','2','3','4','5','6','7','8','9')
# plt.figure(figsize=(10,10))
# plot_confusion_matrix(cm, names, t='Without Pre-Fusion Training Confusion Matrix')

# f1 = f1_score(wo_pre_train_true, wo_pre_train_pred, average='micro')
# acc = accuracy_score(wo_pre_train_true, wo_pre_train_pred)

# file1 = open(file, "a")
# write_string = "Without Pre-Fusion Training Accuracy: " + str(acc)+ "\t F1: "+ str(f1) + "\n"
# file1.write(write_string)
# file1.close()

# plot_roc(wo_pre_train_full_net, device, test_loader, num_classes=10, t='Without Pre-Fusion Training ROC', mode='multi')

# ''' ================= Joint Training ================= '''
# mode = 'joint'
# joint_train_path = "C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\MNISTandSVHN\\Joint_train_Full_Net.pth"
# joint_train_full_net = initialize_network(mode, load_weights = False, train_pre= True)
# joint_training_loss, joint_training_acc = train_full_net(mode, joint_train_full_net, lr, joint_train_path)

# joint_train_lo, joint_train_acc, joint_train_pred, joint_train_true = test_full_net(mode, joint_train_full_net)

# cm = confusion_matrix(joint_train_true, joint_train_pred)
# names = ('0','1','2','3','4','5','6','7','8','9')
# plt.figure(figsize=(10,10))
# plot_confusion_matrix(cm, names, t='Joint Training Confusion Matrix')

# f1 = f1_score(joint_train_true, joint_train_pred, average='micro')
# acc = accuracy_score(joint_train_true, joint_train_pred)

# file1 = open(file, "a")
# write_string = "Joint Training Accuracy: " + str(acc)+ "\t F1: "+ str(f1) + "\n"
# file1.write(write_string)
# file1.close()

# plot_roc(joint_train_full_net, device, test_loader, num_classes=10, t='Joint Training ROC', mode='multi')

''' ================= Plotting ================= '''
#plot_acc_vs_epoch(mnist_acc, svhn_acc, with_pre_training_acc, wo_pre_training_acc, joint_training_acc)
plot_acc_vs_epoch_single(mnist_acc, name='mnist')

# svhn_acc_file = "C:\\Users\\ancarey\\Documents\\FusionPaper\\results\\svhn_train_acc.data"
# mnist_acc_file = "C:\\Users\\ancarey\\Documents\\FusionPaper\\results\\mnist_train_acc.data"
# pre_train_acc_file = "C:\\Users\\ancarey\\Documents\\FusionPaper\\results\\pre_train_acc.data"
# wo_pre_train_acc_file = "C:\\Users\\ancarey\\Documents\\FusionPaper\\results\\wo_pre_train_acc.data"
# joint_train_acc_file = "C:\\Users\\ancarey\\Documents\\FusionPaper\\results\\joint_train_acc.data"

# with open(svhn_acc_file, 'wb') as filehandle:
#     pickle.dump(svhn_acc, filehandle)

# with open(mnist_acc_file, 'wb') as filehandle:
#     pickle.dump(mnist_acc, filehandle)
    
# with open(pre_train_acc_file, 'wb') as filehandle:
#     pickle.dump(with_pre_training_acc, filehandle)
    
# with open(wo_pre_train_acc_file, 'wb') as filehandle:
#     pickle.dump(wo_pre_training_acc, filehandle)
    
# with open(joint_train_acc_file, 'wb') as filehandle:
#     pickle.dump(joint_training_acc, filehandle)   