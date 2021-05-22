# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:20:49 2021

@author: ancarey
"""

import numpy as np
import torch
import sys
import torch.optim as optim 
import matplotlib.pyplot as plt
import torch.nn.functional as F
# sklearn.model_selection import train_test_split
#from torch.utils.data import Subset
from torchvision import transforms
from new_model import exp2_create_multiview_data, Generic_CNN, EXP2_Post_Fusion_Layer, EXP2_Full_Network, Algorithm_One
from CNN import CNN
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from helper_functions import plot_acc_vs_epoch, plot_confusion_matrix, plot_roc, train_val_test_split
import pickle
from torchsummary import summary

sys.setrecursionlimit(10**6)

''' ================= Methods ================= '''
def initialize_network(mode, load_weights, train_pre, fusion):
    torch.cuda.empty_cache()

    print("\nLoading Networks for", mode, "training: ")
    
    ##### IR Pre Fusion #####
    ir_net = Generic_CNN()
    ir_path = "C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\IRandRGB\\IR.pth"
    if load_weights == True:
        ir_pre_net = CNN()
        ir_pre_net_dict = ir_pre_net.state_dict()
        ir_pre_net.load_state_dict(torch.load(ir_path))
        ir_dict = ir_net.state_dict()
        pretrained_dict = {k: v for k, v in ir_pre_net_dict.items() if k in ir_dict}
        ir_dict.update(pretrained_dict) 
        ir_net.load_state_dict(pretrained_dict)
    #print(ir_dict)
    print("*** IR Loaded")
    
    ##### RGB Pre Fusion #####
    rgb_net = Generic_CNN()
    rgb_path = "C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\IRandRGB\\RGB2.pth"
    if load_weights == True:
        rgb_pre_net = CNN()
        rgb_pre_net_dict = rgb_pre_net.state_dict()
        rgb_pre_net.load_state_dict(torch.load(rgb_path))
        rgb_dict = rgb_net.state_dict()
        pretrained_dict_2 = {k: v for k, v in rgb_pre_net_dict.items() if k in rgb_dict}
        rgb_dict.update(pretrained_dict_2) 
        rgb_net.load_state_dict(pretrained_dict_2)
    print("*** RGB Loaded")

    ##### Post Fusion #####
    post_fusion_net = EXP2_Post_Fusion_Layer()
    if load_weights == True:
        post_fusion_dict = post_fusion_net.state_dict()
        pretrained_dict_3 = {k: v for k, v in ir_pre_net_dict.items() if k in post_fusion_dict}
        post_fusion_dict.update(pretrained_dict_3) 
        post_fusion_net.load_state_dict(pretrained_dict_3)
    print("*** Post Fusion Loaded")

    ##### Full Network #####
    full_net = EXP2_Full_Network(ir_net, rgb_net, post_fusion_net, fusion_type = fusion, train_pre_net = train_pre)
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
            ir_img, rgb_img, labels = data
            outputs = model(ir_img.to('cuda'), rgb_img.to('cuda'))
            
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

def train_full_net(mode, model, lr, save_path, fusion_type):
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
            ir_img, rgb_img, labels = data
    
            optimizer.zero_grad()
            
            outputs = model(ir_img.to('cuda'), rgb_img.to('cuda'))
            loss = F.cross_entropy(outputs, labels.to('cuda'))
            pred = outputs.argmax(dim=1, keepdim=True).detach().cpu()
            batch_cor = pred.eq(labels.view_as(pred)).sum().item()
            cor.append(batch_cor)
            loss.backward()
            optimizer.step()
            if fusion_type == 'SD':
                Algorithm_One(model)
            losses.append(loss.item())
            training_loss += loss.item()
            if batch % 10 == 0:
                print("Loss at Batch", batch, ":", losses[batch], "\tAccuracy at Batch", batch, ":", batch_cor/len(labels))
         
        accuracy = 100. * np.sum(cor) / len(train_loader.dataset)
        train_acc.append(accuracy)       
        average_loss = training_loss/len(train_loader.dataset)
        train_loss.append(average_loss)
        print(f'Average loss on Epoch {epoch }: {average_loss}', f'\tAverage accuracy on Epoch {epoch}: {accuracy}')
        
    torch.save(model.state_dict(), save_path) 
    
    return train_loss, train_acc

''' ================= Set Variables ================= '''

torch.manual_seed(0)
batch_size = 64
epochs = 75
lr = 0.005
file = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\results\\ir_and_rgb\\ir_rgb_acc.txt'
open(file, 'w').close()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

''' ================= Create Datasets ================='''
#Paths
country_rgb = "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\rgbdata\\countryRGB\\"
field_rgb = "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\rgbdata\\fieldRGB\\"
forest_rgb = "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\rgbdata\\forestRGB\\"
indoor_rgb = "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\rgbdata\\indoorRGB\\"
mountain_rgb = "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\rgbdata\\mountainRGB\\"
oldbuilding_rgb = "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\rgbdata\\oldbuildingRGB\\"
street_rgb = "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\rgbdata\\streetRGB\\"
urban_rgb = "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\rgbdata\\urbanRGB\\"
water_rgb = "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\rgbdata\\waterRGB\\"

country_ir= "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\irdata\\countryIR\\"
field_ir= "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\irdata\\fieldIR\\"
forest_ir= "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\irdata\\forestIR\\"
indoor_ir= "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\irdata\\indoorIR\\"
mountain_ir= "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\irdata\\mountainIR\\"
oldbuilding_ir= "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\irdata\\oldbuildingIR\\"
street_ir= "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\irdata\\streetIR\\"
urban_ir= "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\irdata\\urbanIR\\"
water_ir= "C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\ir_rgb\\irdata\\waterIR\\"

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
    ])

country = exp2_create_multiview_data(country_rgb, country_ir, 0, transform)
field = exp2_create_multiview_data(field_rgb, field_ir, 1, transform)
forest = exp2_create_multiview_data(forest_rgb, forest_ir, 2, transform)
indoor = exp2_create_multiview_data(indoor_rgb, indoor_ir, 3, transform)
mountain = exp2_create_multiview_data(mountain_rgb, mountain_ir, 4, transform)
oldbuilding = exp2_create_multiview_data(oldbuilding_rgb, oldbuilding_ir, 5, transform)
street = exp2_create_multiview_data(street_rgb, street_ir, 6, transform)
urban = exp2_create_multiview_data(urban_rgb, urban_ir, 7, transform)
water = exp2_create_multiview_data(water_rgb, water_ir, 8, transform)

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

''' ================= Independent Training ================= '''

ir_acc = [16.71087533156499, 25.72944297082228, 31.56498673740053, 35.278514588859416, 37.40053050397878, 40.58355437665782, 40.84880636604775, 45.358090185676396, 47.48010610079576, 44.56233421750663, 47.48010610079576, 50.6631299734748, 50.6631299734748, 54.641909814323604, 52.785145888594165, 55.43766578249337, 56.23342175066313, 57.55968169761273, 61.273209549071616, 59.41644562334218, 64.45623342175067, 62.3342175066313, 64.19098143236074, 64.19098143236074, 68.16976127320955, 68.16976127320955, 69.23076923076923, 71.08753315649868, 71.35278514588859, 71.61803713527851, 74.27055702917772, 74.80106100795756, 77.71883289124668, 73.47480106100795, 77.45358090185677, 79.3103448275862, 78.77984084880637, 80.37135278514589, 78.51458885941645, 81.6976127320955, 82.75862068965517, 82.75862068965517, 83.0238726790451, 84.88063660477454, 84.88063660477454, 88.3289124668435, 90.45092838196287, 86.73740053050398, 89.65517241379311, 87.79840848806366, 89.38992042440319, 88.3289124668435, 87.26790450928382, 88.3289124668435, 91.77718832891247, 90.71618037135279, 88.85941644562334, 89.92042440318302, 91.24668435013263, 92.57294429708223, 90.9814323607427, 92.04244031830238, 94.16445623342175, 94.42970822281167, 94.42970822281167, 94.96021220159152, 94.6949602122016, 92.3076923076923, 95.49071618037135, 96.0212201591512, 95.75596816976127, 96.55172413793103, 96.55172413793103, 97.08222811671088, 96.28647214854111]
rgb_acc = [17.77188328912467, 25.9946949602122, 29.44297082228117, 35.54376657824934, 38.9920424403183, 40.58355437665782, 45.358090185676396, 46.684350132625994, 48.80636604774536, 53.05039787798408, 53.05039787798408, 56.23342175066313, 57.824933687002655, 62.3342175066313, 63.12997347480106, 61.80371352785146, 65.25198938992042, 66.84350132625995, 69.76127320954907, 70.55702917771883, 72.41379310344827, 72.94429708222812, 71.61803713527851, 73.20954907161804, 74.53580901856763, 72.41379310344827, 74.0053050397878, 78.77984084880637, 78.77984084880637, 80.37135278514589, 82.49336870026525, 81.16710875331565, 83.55437665782493, 85.14588859416446, 82.49336870026525, 82.49336870026525, 87.79840848806366, 88.06366047745358, 86.73740053050398, 87.26790450928382, 87.0026525198939, 86.20689655172414, 89.65517241379311, 90.18567639257294, 91.77718832891247, 92.3076923076923, 93.36870026525199, 92.04244031830238, 94.16445623342175, 92.83819628647215, 93.89920424403184, 94.42970822281167, 93.10344827586206, 95.49071618037135, 95.49071618037135, 94.6949602122016, 95.75596816976127, 96.28647214854111, 97.87798408488064, 95.49071618037135, 97.08222811671088, 94.96021220159152, 97.34748010610079, 97.61273209549071, 98.40848806366047, 97.87798408488064, 97.61273209549071, 98.40848806366047, 98.14323607427056, 98.6737400530504, 98.14323607427056, 98.40848806366047, 98.6737400530504, 98.6737400530504, 98.93899204244032]
''' ================= With Compact Bilinear Fusion Training ================= '''
mode = 'Compact Bilinear Fusion'
cbf_path = "C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\IRandRGB\\compact_bilinear_train_Full_Net.pth"
cbf_full_net = initialize_network(mode, load_weights = True, fusion='C-BF', train_pre= False)
net_details_1 = open("C:\\Users\\ancarey\\Documents\\FusionPaper\\cbf_params.txt", 'a')
for name, param in cbf_full_net.named_parameters():
    s = name + ' '+ str(param.shape) + '\n'
    net_details_1.write(s)
net_details_1.close()

cbf_training_loss, cbf_training_acc = train_full_net(mode, cbf_full_net, lr, cbf_path, fusion_type = 'C-BF')

cbf_lo, cbf_acc, cbf_pred, cbf_true = test_full_net(mode, cbf_full_net)

cm = confusion_matrix(cbf_true, cbf_pred)
names = ('country','field', 'forest','indoor','mountain','old building','street', 'urban', 'water')
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, names, t='Compact Bilinear Fusion Confusion Matrix')

f1 = f1_score(cbf_true, cbf_pred, average='micro')
acc = accuracy_score(cbf_true, cbf_pred)

file1 = open(file, "a")
write_string = "Compact Bilinear Fusion Training Accuracy: " + str(acc)+ "\t F1: "+ str(f1) + "\n"
file1.write(write_string)
file1.close()

plot_roc(cbf_full_net, device, test_loader, num_classes=9, t='Compact Bilinear Fusion Training ROC', mode='multi')


''' ================= With FC-Bilinear Fusion Training ================= '''
mode = 'Fully Connected Bilinear Fusion'
fcbf_path = "C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\IRandRGB\\fc_bilinear_train_Full_Net.pth"
fcbf_full_net = initialize_network(mode, load_weights = True, fusion='FC-BF', train_pre= False)
net_details_1 = open("C:\\Users\\ancarey\\Documents\\FusionPaper\\fcbf_params.txt", 'a')
for name, param in cbf_full_net.named_parameters():
    s = name + ' '+ str(param.shape) + '\n'
    net_details_1.write(s)
net_details_1.close()

fcbf_training_loss, fcbf_training_acc = train_full_net(mode, fcbf_full_net, lr, fcbf_path, fusion_type = 'FC-BF')

fcbf_lo, fcbf_acc, fcbf_pred, fcbf_true = test_full_net(mode, fcbf_full_net)

cm = confusion_matrix(fcbf_true, fcbf_pred)
names = ('country','field', 'forest','indoor','mountain','old building','street', 'urban', 'water')
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, names, t='Fully Conntexted Bilinear Fusion Confusion Matrix')

f1 = f1_score(fcbf_true, fcbf_pred, average='micro')
acc = accuracy_score(fcbf_true, fcbf_pred)

file1 = open(file, "a")
write_string = "Fully Connected Bilinear Fusion Training Accuracy: " + str(acc)+ "\t F1: "+ str(f1) + "\n"
file1.write(write_string)
file1.close()

plot_roc(fcbf_full_net, device, test_loader, num_classes=9, t='Fully Connected Bilinear Fusion Training ROC', mode='multi')

''' ================= With Pre-Fusion Training ================= '''
mode = 'w/ pre-training'
pre_train_path = "C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\IRandRGB\\Pre_fusion_train_Full_Net.pth"
pre_train_full_net = initialize_network(mode, load_weights = True, fusion='SD', train_pre= True)
net_details_1 = open("C:\\Users\\ancarey\\Documents\\FusionPaper\\wpt_params.txt", 'a')
for name, param in cbf_full_net.named_parameters():
    s = name + ' '+ str(param.shape) + '\n'
    net_details_1.write(s)
net_details_1.close()
with_pre_training_loss, with_pre_training_acc = train_full_net(mode, pre_train_full_net, lr, pre_train_path, fusion_type = 'SD')

pre_train_lo, pre_train_acc, pre_train_pred, pre_train_true = test_full_net(mode, pre_train_full_net)

cm = confusion_matrix(pre_train_true, pre_train_pred)
names = ('country','field', 'forest','indoor','mountain','old building','street', 'urban', 'water')
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, names, t='With Pre-Fusion Training Confusion Matrix')

f1 = f1_score(pre_train_true, pre_train_pred, average='micro')
acc = accuracy_score(pre_train_true, pre_train_pred)

file1 = open(file, "a")
write_string = "With Pre-Fusion Training Accuracy: " + str(acc)+ "\t F1: "+ str(f1) + "\n"
file1.write(write_string)
file1.close()

plot_roc(pre_train_full_net, device, test_loader, num_classes=9, t='With Pre-Fusion Training ROC', mode='multi')

''' ================= Without Pre-Fusion Training ================= '''
mode = 'w/o pre-training'
wo_pre_train_path = "C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\IRandRGB\\Without_Pre_fusion_train_Full_Net.pth"
wo_pre_train_full_net = initialize_network(mode, load_weights = True, fusion= 'SD', train_pre= False)
net_details_1 = open("C:\\Users\\ancarey\\Documents\\FusionPaper\\wopt_params.txt", 'a')
for name, param in cbf_full_net.named_parameters():
    s = name + ' '+ str(param.shape) + '\n'
    net_details_1.write(s)
net_details_1.close()
wo_pre_training_loss, wo_pre_training_acc = train_full_net(mode, wo_pre_train_full_net, lr, wo_pre_train_path, fusion_type = 'SD')

wo_pre_train_lo, wo_pre_train_acc, wo_pre_train_pred, wo_pre_train_true = test_full_net(mode, wo_pre_train_full_net)

cm = confusion_matrix(wo_pre_train_true, wo_pre_train_pred)
names = ('country','field', 'forest','indoor','mountain','old building','street', 'urban', 'water')
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, names, t='Without Pre-Fusion Training Confusion Matrix')

f1 = f1_score(wo_pre_train_true, wo_pre_train_pred, average='micro')
acc = accuracy_score(wo_pre_train_true, wo_pre_train_pred)

file1 = open(file, "a")
write_string = "Without Pre-Fusion Training Accuracy: " + str(acc)+ "\t F1: "+ str(f1) + "\n"
file1.write(write_string)
file1.close()

plot_roc(wo_pre_train_full_net, device, test_loader, num_classes=9, t='Without Pre-Fusion Training ROC', mode='multi')

''' ================= Joint Same Dim Training ================= '''
mode = 'joint'
lr=.001
joint_train_path = "C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\IRandRGB\\Joint_train_Full_Net_same_dim.pth"
joint_train_full_net = initialize_network(mode, load_weights = False, fusion = 'SD', train_pre= True)
net_details_1 = open("C:\\Users\\ancarey\\Documents\\FusionPaper\\jsd_params.txt", 'a')
for name, param in cbf_full_net.named_parameters():
    s = name + ' '+ str(param.shape) + '\n'
    net_details_1.write(s)
net_details_1.close()
joint_training_loss, joint_training_acc = train_full_net(mode, joint_train_full_net, lr, joint_train_path, fusion_type = 'SD')

joint_train_lo, joint_train_acc, joint_train_pred, joint_train_true = test_full_net(mode, joint_train_full_net)

cm = confusion_matrix(joint_train_true, joint_train_pred)
names = ('country','field', 'forest','indoor','mountain','old building','street', 'urban', 'water')
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, names, t='Joint Training Same Dimension Fusion Confusion Matrix')

f1 = f1_score(joint_train_true, joint_train_pred, average='micro')
acc = accuracy_score(joint_train_true, joint_train_pred)

file1 = open(file, "a")
write_string = "Joint Training Same Dimension Fusion Accuracy: " + str(acc)+ "\t F1: "+ str(f1) + "\n"
file1.write(write_string)
file1.close()

plot_roc(joint_train_full_net, device, test_loader, num_classes=9, t='Joint Training Same Dimension Fusion ROC', mode='multi')

''' ================= Joint FC Training ================= '''
mode = 'joint'
joint_fc_train_path = "C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\IRandRGB\\joint_fc_train_Full_Net_FC.pth"
joint_fc_train_full_net = initialize_network(mode, load_weights = False, fusion = 'FC', train_pre= True)
joint_fc_training_loss, joint_fc_training_acc = train_full_net(mode, joint_fc_train_full_net, lr, joint_fc_train_path, fusion_type = 'FC')
net_details_1 = open("C:\\Users\\ancarey\\Documents\\FusionPaper\\jfc_params.txt", 'a')
for name, param in cbf_full_net.named_parameters():
    s = name + ' '+ str(param.shape) + '\n'
    net_details_1.write(s)
net_details_1.close()
joint_fc_train_lo, joint_fc_train_acc, joint_fc_train_pred, joint_fc_train_true = test_full_net(mode, joint_fc_train_full_net)

cm = confusion_matrix(joint_fc_train_true, joint_fc_train_pred)
names = ('country','field', 'forest','indoor','mountain','old building','street', 'urban', 'water')
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, names, t='Joint Training Fully-Connected Fusion Confusion Matrix')

f1 = f1_score(joint_fc_train_true, joint_fc_train_pred, average='micro')
acc = accuracy_score(joint_fc_train_true, joint_fc_train_pred)

file1 = open(file, "a")
write_string = "Joint Training Fully-Connected Fusion Accuracy: " + str(acc)+ "\t F1: "+ str(f1) + "\n"
file1.write(write_string)
file1.close()

plot_roc(joint_fc_train_full_net, device, test_loader, num_classes=9, t='Joint Training Fully-Connected Fusion ROC', mode='multi')


''' ================= Plotting ================= '''
#with_pre_training_acc = [87.87943565626335, 87.87943565626335, 88.65612084936582, 88.82713410289297, 88.88413852073536, 89.0622773264928, 89.0836539831837, 89.32592275901382, 89.64657260937723, 89.3900527290865, 89.44705714692888, 89.43280604246829, 89.70357702721961, 90.095482399886, 90.35200228017672, 90.32350007125552, 90.18098902664957, 90.0669801909648, 90.41613225024939, 90.53726663816445, 90.30212341456463, 90.4090066980191, 91.25694741342454, 91.27119851788514, 90.83653983183697, 91.10018526435799, 91.49209063702438, 91.15006412997008, 91.49209063702438, 91.67735499501211, 91.59184836824853, 91.63460168163033, 91.79848938292717, 91.92674932307254, 92.20464586005416, 91.93387487530283, 91.86974490523015, 92.21177141228445, 92.44691463588428, 92.45404018811458, 92.19039475559356, 92.49679350149637, 92.66068120279321, 92.83882000855067, 92.99558215761722, 92.66780675502352, 92.79606669516888, 93.16659541114436, 93.0525865754596, 93.30198090352002]
#wo_pre_training_acc = [85.87715547954967, 86.26906085221604, 86.57545959811885, 86.51845518027648, 86.52558073250677, 86.63246401596123, 87.18113153769417, 87.12412711985179, 86.88898389625196, 87.0671227020094, 86.87473279179136, 87.00299273193673, 87.08137380647, 87.02436938862762, 86.98161607524582, 87.30226592560923, 86.92461165740345, 87.00299273193673, 87.28088926891834, 87.36639589568192, 87.4020236568334, 87.14550377654268, 87.21675929884566, 87.47327917913638, 87.34501923899103, 87.51603249251816, 87.23101040330626, 87.55166025366965, 87.22388485107597, 87.32364258230012, 87.60153911928174, 87.4091492090637, 87.63004132820294, 87.58728801482116, 87.38777255237281, 87.55878580589996, 87.56591135813025, 87.64429243266353, 87.68704574604531, 87.51603249251816, 87.25238705999715, 87.54453470143936, 87.38064700014252, 87.75830126834829, 87.85805899957246, 87.56591135813025, 87.61579022374234, 87.78680347726949, 87.82243123842098, 87.81530568619068]
plot_acc_vs_epoch(ir_acc, rgb_acc, with_pre_training_acc, wo_pre_training_acc, joint_training_acc, joint_fc_training_acc, cbf_training_acc, fcbf_training_acc)

# ir_acc_file = "C:\\Users\\ancarey\\Documents\\FusionPaper\\results\\ir_train_acc.data"
# rgb_acc_file = "C:\\Users\\ancarey\\Documents\\FusionPaper\\results\\rgb_train_acc.data"
# pre_train_acc_file = "C:\\Users\\ancarey\\Documents\\FusionPaper\\results\\ir_rgb_pre_train_acc.data"
# wo_pre_train_acc_file = "C:\\Users\\ancarey\\Documents\\FusionPaper\\results\\ir_rgb_wo_pre_train_acc.data"
# joint_train_acc_file = "C:\\Users\\ancarey\\Documents\\FusionPaper\\results\\ir_rgb_joint_train_acc.data"

# with open(ir_acc_file, 'wb') as filehandle:
#     pickle.dump(ir_acc, filehandle)

# with open(rgb_acc_file, 'wb') as filehandle:
#     pickle.dump(rgb_acc, filehandle)
    
# with open(pre_train_acc_file, 'wb') as filehandle:
#     pickle.dump(with_pre_training_acc, filehandle)
    
# with open(wo_pre_train_acc_file, 'wb') as filehandle:
#     pickle.dump(wo_pre_training_acc, filehandle)
    
# with open(joint_train_acc_file, 'wb') as filehandle:
#     pickle.dump(joint_training_acc, filehandle)   