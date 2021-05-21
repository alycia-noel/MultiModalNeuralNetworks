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
    ir_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\models\\GrayandRGB\\Gray4.pth'
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
epochs = 50
lr = 0.0005 #normally .001
file = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\results\\gray_and_rgb\\gray_rgb_acc2.txt'
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

ir_acc = [39.63538149898717, 52.224472953709956, 56.83847250356366, 60.882286743191536, 63.35058894140596, 65.90141796083728, 67.97209092955211, 69.16497861805087, 70.44039312776653, 71.29567109310526, 72.02340760747244, 73.07374896841473, 73.68894890839523, 74.04156350814014, 74.58173906519619, 75.35449020931803, 75.59456823467627, 76.02970965563809, 76.58489008927901, 77.20009002925951, 77.17758271438217, 77.62022657363643, 78.01785580313602, 78.39297771775827, 78.85812889188986, 78.80561182384275, 79.45832395528546, 79.77342636356816, 80.1635531547753, 80.22357266111486, 80.88378723085003, 80.79375797134068, 80.85377747768024, 81.85910420886788, 81.93412859179233, 81.76907494935854, 82.1967139320279, 82.39927976592392, 83.03698702078175, 82.99197239102709, 83.23955285467777, 83.61467476930002, 83.63718208417735, 83.89976742441293, 84.07232350513917, 84.58248930902543, 84.90509415560057, 85.3102258233926, 85.5428014104584, 85.2427038787606]

rgb_acc = [41.75857153574912, 54.71528246680171, 59.4268137144572, 63.32808162652862, 66.65916422837422, 68.98492009903218, 70.97306624653012, 72.64610998574537, 73.57641233400855, 74.28914397179084, 75.15192437542201, 75.78963163027984, 76.65241203391102, 76.97501688048615, 77.63523145022133, 78.1153875009378, 78.14539725410758, 78.52051916872983, 79.35328981919123, 79.88596293795483, 79.90096781453973, 80.27608972916198, 80.53867506939756, 81.03383599669893, 81.57401155375497, 81.66404081326431, 82.25673343836748, 82.65436266786706, 82.39177732763147, 82.95446019956486, 83.40460649711156, 83.30707479930977, 83.74971865856403, 83.86975767124315, 84.4924600495161, 84.52246980268588, 84.68002100682722, 85.27271363193037, 85.27271363193037, 85.58031360192062, 85.77537699752419, 86.00795258458999, 86.09047940580689, 86.60064520969316, 86.72068422237227, 86.87823542651361, 87.10330857528697, 86.99827443919274, 87.77852802160702, 87.62847925575812]

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
names = ('building','forest' , 'glacier', 'mountain', 'sea', 'street')
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, names, t='Compact Bilinear Fusion Confusion Matrix')

f1 = f1_score(cbf_true, cbf_pred, average='micro')
acc = accuracy_score(cbf_true, cbf_pred)

file1 = open(file, "a")
write_string = "Compact Bilinear Fusion Training Accuracy: " + str(acc)+ "\t F1: "+ str(f1) + "\n"
file1.write(write_string)
file1.close()

plot_roc(cbf_full_net, device, test_loader, num_classes=6, t='Compact Bilinear Fusion Training ROC', mode='multi')


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
names = ('building','forest' , 'glacier', 'mountain', 'sea', 'street')
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, names, t='Fully Conntexted Bilinear Fusion Confusion Matrix')

f1 = f1_score(fcbf_true, fcbf_pred, average='micro')
acc = accuracy_score(fcbf_true, fcbf_pred)

file1 = open(file, "a")
write_string = "Fully Connected Bilinear Fusion Training Accuracy: " + str(acc)+ "\t F1: "+ str(f1) + "\n"
file1.write(write_string)
file1.close()

plot_roc(fcbf_full_net, device, test_loader, num_classes=6, t='Fully Connected Bilinear Fusion Training ROC', mode='multi')

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
net_details_1 = open("C:\\Users\\ancarey\\Documents\\FusionPaper\\wopt_params.txt", 'a')
for name, param in cbf_full_net.named_parameters():
    s = name + ' '+ str(param.shape) + '\n'
    net_details_1.write(s)
net_details_1.close()
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