# -*- coding: utf-8 -*-
"""
Helper functions

Created on Wed Mar 10 18:02:13 2021

@author: ancarey
"""
from __future__ import print_function
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import itertools
from itertools import cycle
import numpy as np 
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def show_image(img):
    print(img[0].shape)
    img = img[0].permute(1, 2, 0)
    plt.imshow(img)

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
            
    accuracy = 100. * correct / len(train_loader.dataset)
    return accuracy

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
            
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return accuracy, [i.item() for i in true], [i.item() for i in predictions]

def plot_confusion_matrix(cm, classes, t, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(t)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_roc(model, device, test_loader, num_classes, t, mode):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    lw = 2
    n_classes = num_classes
    for i in range(num_classes):
        which_class = i
    
        if mode == 'single':
            actuals, class_probabilities = test_class_probabilities_single(model, device, test_loader, which_class)
        else:
            actuals, class_probabilities = test_class_probabilities(model, device, test_loader, which_class)
            
        fpr[i], tpr[i], _ = roc_curve(actuals, class_probabilities)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    plt.figure()
    plt.figure(figsize=(10,10))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                  label='Class {0} (area = {1:0.2f})'
                  ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
   
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(t)
    plt.legend(loc="lower right")
    plt.show()
    
def test_class_probabilities_single(model, device, test_loader, which_class):
    model.eval()
    actuals = []
    probabilities = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction) == which_class)
            probabilities.extend(np.exp(output[:, which_class].cpu().detach().numpy()))
    return [i.item() for i in actuals], [i.item() for i in probabilities]

def test_class_probabilities(model, device, test_loader, which_class):
    model.eval()
    actuals = []
    probabilities = []
    with torch.no_grad():
        for batch, data in enumerate(test_loader):
            ir_img, rgb_img, labels = data
            output = model(ir_img.to('cuda'), rgb_img.to('cuda'))
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(labels.view_as(prediction) == which_class)
            probabilities.extend(np.exp(output[:, which_class].cpu().detach().numpy()))
    return [i.item() for i in actuals], [i.item() for i in probabilities]

def plot_acc_vs_epoch(acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, acc_8):
    plt.plot(acc_1, label='Gray')
    plt.plot(acc_2, label='RGB')
    plt.plot(acc_3, label='Pre-Train')
    plt.plot(acc_4, label='W/O Pre-train')
    plt.plot(acc_5, label='Joint - SD')
    plt.plot(acc_6, label='Joint - FC')
    plt.plot(acc_7, label="Compact Bilinear")
    plt.plot(acc_8, label="FC Bilinear")
    plt.legend(loc = 4)
    plt.title("Training Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")
    
def plot_acc_vs_epoch_single(acc_1, name):
    plt.plot(acc_1, label=name)
    plt.legend(loc = 4)
    plt.title("Training Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")