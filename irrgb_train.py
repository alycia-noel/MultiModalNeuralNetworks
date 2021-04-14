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
from new_model import exp2_create_multiview_data, Generic_CNN, EXP2_Post_Fusion_Layer, EXP2_Full_Network, Algorithm_One
from CNN import CNN
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from helper_functions import plot_acc_vs_epoch, plot_confusion_matrix, plot_roc, train_val_test_split
import pickle
''' ================= Methods ================= '''
def initialize_network(mode, load_weights, train_pre, fusion):
    torch.cuda.empty_cache()

    print("\nLoading Networks for", mode, "training: ")
    
    ##### IR Pre Fusion #####
    ir_net = Generic_CNN()
    ir_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\GrayandRGB\\Gray3.pth'
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
    rgb_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\GrayandRGB\\RGB.pth'
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
        pretrained_dict_3 = {k: v for k, v in rgb_pre_net_dict.items() if k in post_fusion_dict}
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
            Algorithm_One(model, fusion_type)
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

batch_size = 64
epochs = 50
lr = 0.001
file = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\results\\gray_and_rgb\\gray_rgb_acc.txt'
open(file, 'w').close()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

''' ================= Create Datasets ================='''
#Paths
building_rgb_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\train\\buildings\\'
forest_rgb_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\train\\forest\\'
glacier_rgb_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\train\\glacier\\'
mountain_rgb_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\train\\mountain\\'
sea_rgb_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\train\\sea\\'
street_rgb_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\train\\street\\'

building_rgb_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\test\\buildings\\'
forest_rgb_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\test\\forest\\'
glacier_rgb_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\test\\glacier\\'
mountain_rgb_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\test\\mountain\\'
sea_rgb_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\test\\sea\\'
street_rgb_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\test\\street\\'

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
    transforms.ToTensor()
    ])

building_train = exp2_create_multiview_data(building_grayscale_train_path, building_rgb_train_path, 0, transform)
forest_train = exp2_create_multiview_data(forest_grayscale_train_path, forest_rgb_train_path, 1, transform)
glacier_train = exp2_create_multiview_data(glacier_grayscale_train_path, glacier_rgb_train_path, 2, transform)
mountain_train = exp2_create_multiview_data(mountain_grayscale_train_path, mountain_rgb_train_path, 3, transform)
sea_train = exp2_create_multiview_data(sea_grayscale_train_path, sea_rgb_train_path, 4, transform)
street_train = exp2_create_multiview_data(street_grayscale_train_path, street_rgb_train_path, 5, transform)

building_test = exp2_create_multiview_data(building_grayscale_test_path, building_rgb_test_path, 0, transform)
forest_test = exp2_create_multiview_data(forest_grayscale_test_path, forest_rgb_test_path, 1, transform)
glacier_test = exp2_create_multiview_data(glacier_grayscale_test_path, glacier_rgb_test_path, 2, transform)
mountain_test = exp2_create_multiview_data(mountain_grayscale_test_path, mountain_rgb_test_path, 3, transform)
sea_test = exp2_create_multiview_data(sea_grayscale_test_path, sea_rgb_test_path, 4, transform)
street_test = exp2_create_multiview_data(street_grayscale_test_path, street_rgb_test_path, 5, transform)

train_set = building_train + forest_train + glacier_train +  mountain_train + sea_train + street_train 
test_set = building_test + forest_test + glacier_test +  mountain_test + sea_test + street_test 

train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False)

''' ================= Independent Training ================= '''

ir_acc = [37.55720609197989, 51.279165728861884, 57.29612123940281, 61.27241353439868, 64.20586690674469, 
          66.44159351789331, 68.54977867807037, 69.81018831120114, 70.53792482556831, 70.80801260409633, 
          72.22597344136844, 72.78115387500938, 73.61392452547078, 73.67394403181034, 74.61925125665842, 
          75.28696826468602, 75.7821291919874, 76.25478280441143, 76.95250956560882, 77.07254857828795, 
          77.2976217270613, 78.13789481581514, 78.55052892189961, 78.61054842823918, 79.248255683097, 
          79.40580688723836, 80.11103608672819, 79.85595318478505, 80.1635531547753, 80.98131892865182, 
          81.05634331157626, 81.23640183059494, 81.69405056643409, 82.1441968639808, 82.35426513616926, 
          82.5118163403106, 83.08200165053643, 83.23205041638532, 83.13451871858354, 83.6296796458849, 
          84.31990396878986, 84.15485032635607, 84.68752344511967, 84.4924600495161, 84.36491859854452, 
          85.40025508290195, 85.46777702753396, 85.50528921899617, 85.93292820166555, 85.72285992947708]

rgb_acc = [47.49043439117713, 59.47933078250431, 64.30339860454647, 66.88423737714757, 69.11246155000376,
           71.01057843799235, 73.00622702378273, 73.32132943206543, 74.74679270762998, 75.15942681371446,
           76.449846200015, 77.25260709730662, 76.96001200390127, 77.35764123340086, 78.6480606197014,
           78.6480606197014, 79.28576787455923, 79.6308800360117, 79.95348488258684, 80.71873358841624,
           80.61369945232201, 81.0638457498687, 81.52149448570786, 82.10668467251857, 81.88911396203767,
           82.32425538299948, 82.45179683397104, 82.76689924225373, 83.27706504614, 83.45712356515868,
           83.53214794808312, 83.6296796458849, 83.75722109685648, 83.9447820541676, 84.38742591342186,
           84.53747467927076, 85.07014779803436, 85.42276239777928, 84.80756245779878, 85.6778452997224,
           85.64033310826018, 85.88791357191087, 86.30805011628779, 86.46560132042914, 86.48060619701403,
           86.33805986945757, 87.14082076674919, 87.38089879210743, 87.38089879210743, 87.26085977942832]


''' ================= Joint FC Training ================= '''
mode = 'joint'
joint_fc_train_path = "C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\IRandRGB\\joint_fc_train_Full_Net_FC.pth"
joint_fc_train_full_net = initialize_network(mode, load_weights = False, fusion = 'FC', train_pre= True)
joint_fc_training_loss, joint_fc_training_acc = train_full_net(mode, joint_fc_train_full_net, lr, joint_fc_train_path, fusion_type = 'FC')

joint_fc_train_lo, joint_fc_train_acc, joint_fc_train_pred, joint_fc_train_true = test_full_net(mode, joint_fc_train_full_net)

cm = confusion_matrix(joint_fc_train_true, joint_fc_train_pred)
names = ('building','forest' , 'glacier', 'mountain', 'sea', 'street')
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, names, t='Joint Training Fully-Connected Fusion Confusion Matrix')

f1 = f1_score(joint_fc_train_true, joint_fc_train_pred, average='micro')
acc = accuracy_score(joint_fc_train_true, joint_fc_train_pred)

file1 = open(file, "a")
write_string = "Joint Training Fully-Connected Fusion Accuracy: " + str(acc)+ "\t F1: "+ str(f1) + "\n"
file1.write(write_string)
file1.close()

plot_roc(joint_fc_train_full_net, device, test_loader, num_classes=6, t='Joint Training Fully-Connected Fusion ROC', mode='multi')

''' ================= With Pre-Fusion Training ================= '''
mode = 'w/ pre-training'
pre_train_path = "C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\IRandRGB\\Pre_fusion_train_Full_Net.pth"
pre_train_full_net = initialize_network(mode, load_weights = True, fusion='SD', train_pre= True)
with_pre_training_loss, with_pre_training_acc = train_full_net(mode, pre_train_full_net, lr, pre_train_path, fusion_type = 'SD')

pre_train_lo, pre_train_acc, pre_train_pred, pre_train_true = test_full_net(mode, pre_train_full_net)

cm = confusion_matrix(pre_train_true, pre_train_pred)
names = ('building','forest' , 'glacier', 'mountain', 'sea', 'street')
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, names, t='With Pre-Fusion Training Confusion Matrix')

f1 = f1_score(pre_train_true, pre_train_pred, average='micro')
acc = accuracy_score(pre_train_true, pre_train_pred)

file1 = open(file, "a")
write_string = "With Pre-Fusion Training Accuracy: " + str(acc)+ "\t F1: "+ str(f1) + "\n"
file1.write(write_string)
file1.close()

plot_roc(pre_train_full_net, device, test_loader, num_classes=6, t='With Pre-Fusion Training ROC', mode='multi')

''' ================= Without Pre-Fusion Training ================= '''
mode = 'w/o pre-training'
wo_pre_train_path = "C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\IRandRGB\\Without_Pre_fusion_train_Full_Net.pth"
wo_pre_train_full_net = initialize_network(mode, load_weights = True, fusion= 'SD', train_pre= False)
wo_pre_training_loss, wo_pre_training_acc = train_full_net(mode, wo_pre_train_full_net, lr, wo_pre_train_path, fusion_type = 'SD')

wo_pre_train_lo, wo_pre_train_acc, wo_pre_train_pred, wo_pre_train_true = test_full_net(mode, wo_pre_train_full_net)

cm = confusion_matrix(wo_pre_train_true, wo_pre_train_pred)
names = ('building','forest' , 'glacier', 'mountain', 'sea', 'street')
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, names, t='Without Pre-Fusion Training Confusion Matrix')

f1 = f1_score(wo_pre_train_true, wo_pre_train_pred, average='micro')
acc = accuracy_score(wo_pre_train_true, wo_pre_train_pred)

file1 = open(file, "a")
write_string = "Without Pre-Fusion Training Accuracy: " + str(acc)+ "\t F1: "+ str(f1) + "\n"
file1.write(write_string)
file1.close()

plot_roc(wo_pre_train_full_net, device, test_loader, num_classes=6, t='Without Pre-Fusion Training ROC', mode='multi')

''' ================= Joint Same Dim Training ================= '''
mode = 'joint'
joint_train_path = "C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\IRandRGB\\Joint_train_Full_Net_same_dim.pth"
joint_train_full_net = initialize_network(mode, load_weights = False, fusion = 'SD', train_pre= True)
joint_training_loss, joint_training_acc = train_full_net(mode, joint_train_full_net, lr, joint_train_path, fusion_type = 'SD')

joint_train_lo, joint_train_acc, joint_train_pred, joint_train_true = test_full_net(mode, joint_train_full_net)

cm = confusion_matrix(joint_train_true, joint_train_pred)
names = ('building','forest' , 'glacier', 'mountain', 'sea', 'street')
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, names, t='Joint Training Same Dimension Fusion Confusion Matrix')

f1 = f1_score(joint_train_true, joint_train_pred, average='micro')
acc = accuracy_score(joint_train_true, joint_train_pred)

file1 = open(file, "a")
write_string = "Joint Training Same Dimension Fusion Accuracy: " + str(acc)+ "\t F1: "+ str(f1) + "\n"
file1.write(write_string)
file1.close()

plot_roc(joint_train_full_net, device, test_loader, num_classes=6, t='Joint Training Same Dimension Fusion ROC', mode='multi')



''' ================= Plotting ================= '''
plot_acc_vs_epoch(ir_acc, rgb_acc, with_pre_training_acc, wo_pre_training_acc, joint_training_acc, joint_fc_train_acc)

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