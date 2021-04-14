# -*- coding: utf-8 -*-
"""
Script for creating IR version of images 

Created on Thu Apr  8 12:46:14 2021

@author: ancarey
"""
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

building_rgb_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\train\\buildings\\*.jpg'
forest_rgb_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\train\\forest\\*.jpg'
glacier_rgb_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\train\\glacier\\*.jpg'
mountain_rgb_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\train\\mountain\\*.jpg'
sea_rgb_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\train\\sea\\*.jpg'
street_rgb_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\train\\street\\*.jpg'

rgb_val_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\val\\*.jpg'

building_rgb_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\test\\buildings\\*.jpg'
forest_rgb_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\test\\forest\\*.jpg'
glacier_rgb_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\test\\glacier\\*.jpg'
mountain_rgb_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\test\\mountain\\*.jpg'
sea_rgb_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\test\\sea\\*.jpg'
street_rgb_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\rgb\\test\\street\\*.jpg'

building_grayscale_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\train\\buildings\\'
forest_grayscale_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\train\\forest\\'
glacier_grayscale_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\train\\glacier\\'
mountain_grayscale_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\train\\mountain\\'
sea_grayscale_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\train\\sea\\'
street_grayscale_train_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\train\\street\\'

grayscale_val_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\val\\'

building_grayscale_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\test\\buildings\\'
forest_grayscale_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\test\\forest\\'
glacier_grayscale_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\test\\glacier\\'
mountain_grayscale_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\test\\mountain\\'
sea_grayscale_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\test\\sea\\'
street_grayscale_test_path = 'C:\\Users\\ancarey\\Documents\\FusionPaper\\data\\GrayandRGB\\grayscale\\test\\street\\'

building_rgb_train_img = glob.glob(building_rgb_train_path)
forest_rgb_train_img = glob.glob(forest_rgb_train_path)
glacier_rgb_train_img = glob.glob(glacier_rgb_train_path)
mountain_rgb_train_img = glob.glob(mountain_rgb_train_path)
sea_rgb_train_img = glob.glob(sea_rgb_train_path)
street_rgb_train_img = glob.glob(street_rgb_train_path)

rgb_val_img = glob.glob(rgb_val_path)

building_rgb_test_img = glob.glob(building_rgb_test_path)
forest_rgb_test_img = glob.glob(forest_rgb_test_path)
glacier_rgb_test_img = glob.glob(glacier_rgb_test_path)
mountain_rgb_test_img = glob.glob(mountain_rgb_test_path)
sea_rgb_test_img = glob.glob(sea_rgb_test_path)
street_rgb_test_img = glob.glob(street_rgb_test_path)

''' creating grayscale train images'''
for i in enumerate(building_rgb_train_img):
    _, img_num_full = os.path.split(i[1])
    img_num = img_num_full.split('.')[0]
    img = Image.open(i[1]).convert('LA')
    save_path = building_grayscale_train_path + img_num + '.png'
    img.save(save_path)
print('Train Building Done')

for i in enumerate(forest_rgb_train_img):
    _, img_num_full = os.path.split(i[1])
    img_num = img_num_full.split('.')[0]
    img = Image.open(i[1]).convert('LA')
    save_path = forest_grayscale_train_path + img_num + '.png'
    img.save(save_path)
print('Train Forest Done')

for i in enumerate(glacier_rgb_train_img):
    _, img_num_full = os.path.split(i[1])
    img_num = img_num_full.split('.')[0]
    img = Image.open(i[1]).convert('LA')
    save_path = glacier_grayscale_train_path + img_num + '.png'
    img.save(save_path)
print('Train Glacier Done')
   
for i in enumerate(mountain_rgb_train_img):
    _, img_num_full = os.path.split(i[1])
    img_num = img_num_full.split('.')[0]
    img = Image.open(i[1]).convert('LA')
    save_path = mountain_grayscale_train_path + img_num + '.png'
    img.save(save_path)
print('Train Mountain Done')
    
for i in enumerate(sea_rgb_train_img):
    _, img_num_full = os.path.split(i[1])
    img_num = img_num_full.split('.')[0]
    img = Image.open(i[1]).convert('LA')
    save_path = sea_grayscale_train_path + img_num + '.png'
    img.save(save_path)
print('Train Sea Done')
   
for i in enumerate(street_rgb_train_img):
    _, img_num_full = os.path.split(i[1])
    img_num = img_num_full.split('.')[0]
    img = Image.open(i[1]).convert('LA')
    save_path = street_grayscale_train_path + img_num + '.png'
    img.save(save_path)
print('Train Street Done')

''' creating grayscale val images '''    
for i in enumerate(rgb_val_img):
    _, img_num_full = os.path.split(i[1])
    img_num = img_num_full.split('.')[0]
    img = Image.open(i[1]).convert('LA')
    save_path = grayscale_val_path + img_num + '.png'
    img.save(save_path)
print('Validation Set Done')

''' creating grayscale testing images '''
for i in enumerate(building_rgb_test_img):
    _, img_num_full = os.path.split(i[1])
    img_num = img_num_full.split('.')[0]
    img = Image.open(i[1]).convert('LA')
    save_path = building_grayscale_test_path + img_num + '.png'
    img.save(save_path)
print('Test Building Done')
    
for i in enumerate(forest_rgb_test_img):
    _, img_num_full = os.path.split(i[1])
    img_num = img_num_full.split('.')[0]
    img = Image.open(i[1]).convert('LA')
    save_path = forest_grayscale_test_path + img_num + '.png'
    img.save(save_path)
print('Test Forest Done')
    
for i in enumerate(glacier_rgb_test_img):
    _, img_num_full = os.path.split(i[1])
    img_num = img_num_full.split('.')[0]
    img = Image.open(i[1]).convert('LA')
    save_path = glacier_grayscale_test_path + img_num + '.png'
    img.save(save_path)
print('Test Glacier Done')
    
for i in enumerate(mountain_rgb_test_img):
    _, img_num_full = os.path.split(i[1])
    img_num = img_num_full.split('.')[0]
    img = Image.open(i[1]).convert('LA')
    save_path = mountain_grayscale_test_path + img_num + '.png'
    img.save(save_path)
print('Test Mountain Done')    

for i in enumerate(sea_rgb_test_img):
    _, img_num_full = os.path.split(i[1])
    img_num = img_num_full.split('.')[0]
    img = Image.open(i[1]).convert('LA')
    save_path = sea_grayscale_test_path + img_num + '.png'
    img.save(save_path)
print('Test Sea Done')
    
for i in enumerate(street_rgb_test_img):
    _, img_num_full = os.path.split(i[1])
    img_num = img_num_full.split('.')[0] 
    img = Image.open(i[1]).convert('LA')
    save_path = street_grayscale_test_path + img_num + '.png'
    img.save(save_path)
    
print('Test Street Done')