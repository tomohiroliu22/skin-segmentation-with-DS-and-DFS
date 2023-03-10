import os
import random
import numpy as np
import cv2
import torch.utils.data as Data
from torch.utils.data import DataLoader,Dataset



def data_loader(opath, fold):
    """ Data loader to split training, validation, and testing set.
    INPUT:
        [opath]: path to your image files directory (ex: ./[your own path]/dataset/dataset/image/)
        [fold]: selcet the fold for cross-validation
    OUTPUT:
        [train_data_LD]: training set data loader
        [valid_data_LD]: validation set data loader
        [test_data_LD]: testing set data loader
    """
    n = int(len(os.listdir(opath))/5) # 5 folds cross-validation
    all_data_filename = sorted(os.listdir(opath))
    random.Random(42).shuffle(all_data_filename)   # seed = 42
    fold_list = [all_data_filename[i:i+n] for i in range(0,len(all_data_filename),n)]
    # validation data
    valid_data_list = fold_list[fold]
    # testing data
    if fold < 4:
        test_data_list  = fold_list[fold+1]
    else:
        test_data_list  = fold_list[0]    
    # training data
    train_data_list = []
    [train_data_list.append(x) for x in all_data_filename if x not in (valid_data_list+test_data_list)]
    
    print('train',len(train_data_list),'valid',len(valid_data_list),'test',len(test_data_list))
    
    train_data_LD = Data.DataLoader(dataset=train_data_list, batch_size=8, shuffle=True, num_workers=4)
    valid_data_LD = Data.DataLoader(dataset=valid_data_list, batch_size=8, shuffle=False, num_workers=4)
    test_data_LD  = Data.DataLoader(dataset=test_data_list, batch_size=8, shuffle=False, num_workers=4)
    return  train_data_LD, valid_data_LD, test_data_LD


def o_data(opath, batch_num, width, height):
    """ reading OCT images
    INPUT:
        [opath]: path to your image files directory (ex: ./[your own path]/dataset/dataset/image/)
        [batch_num]: the list containing the name of selected images in batch
        [width]: input images width
        [height]: input images height
    OUTPUT:
        [img_sub]: OCT images batch 
    """
    img_sub = np.zeros((len(batch_num),1,width,height))
    i = 0
    for name in np.array(batch_num):
        img_sub[i,0,:,6:506] = cv2.imread(opath + name, 0)/255
        i+=1 
    return img_sub

def g_data_cell(gpath, batch_num,width,height,USE_DS):
    """ reading cell labeling for deep feature sharing model
    INPUT:
        [gpath]: path to your cell labeling files directory (ex: ./[your own path]/dataset/dataset/cell/)
        [batch_num]: the list containing the name of selected images in batch
        [width]: input images width
        [height]: input images height
        [USE_DS]: using deep supervision
    OUTPUT:
        [img_sub]: OCT images batch 
        [img_sub2]: OCT images batch for down-sampling
    """
    NUM = len(batch_num)
    img_sub = np.zeros((NUM,1,width,height))
    if(USE_DS):
        half_width = int(width/2)
        half_height = int(height/2)
        img_sub2 = np.zeros((NUM,1,half_width,half_height))
    i = 0
    for name in np.array(batch_num):
        img = cv2.imread(gpath + name, 0)/255
        img_sub[i,0,:,6:506] = img
        if(USE_DS):
            img_half = cv2.resize(img,(250,half_width),interpolation=cv2.INTER_NEAREST)
            img_sub2[i,0,:,3:253] = img_half
        i+=1
    if(USE_DS):
        return img_sub, img_sub2
    else:
        return img_sub

def g_data_line(gpath, batch_num,width,height):
    """ reading layers labeling for deep feature sharing model
    INPUT:
        [gpath]: path to your layer labeling files directory (ex: ./[your own path]/dataset/dataset/layer/)
        [batch_num]: the list containing the name of selected images in batch
        [width]: input images width
        [height]: input images height
    OUTPUT:
        [img_sub]: air gap label batch 
        [img_sub2]: SC label batch 
        [img_sub3]: epidermis label batch  
        [img_sub4]: dermis label batch  
    """
    NUM = len(batch_num)
    img_sub = np.zeros((NUM,1,width,height))
    img_sub2 = np.zeros((NUM,1,width,height))
    img_sub3 = np.zeros((NUM,1,width,height))
    img_sub4 = np.zeros((NUM,1,width,height))
    i = 0
    for name in np.array(batch_num):
        img_temp = np.zeros((width,height))
        img_temp[:,6:506] = cv2.imread(gpath + name, 0)
        img_sub[i,0,img_temp==63] = 1  
        img_sub2[i,0,img_temp==127] = 1  
        img_sub3[i,0,img_temp==191] = 1
        img_sub4[i,0,img_temp==255] = 1
        i+=1
    return img_sub, img_sub2, img_sub3, img_sub4


def g_data_all(gpath_cell, gpath_line, batch_num, width, height,USE_DS):
    """ reading labeling for non deep feature sharing model
    INPUT:
        [gpath_cell]: path to your cell labeling files directory (ex: ./[your own path]/dataset/dataset/cell/)
        [gpath_line]: path to your layer labeling files directory (ex: ./[your own path]/dataset/dataset/layer/)
        [batch_num]: the list containing the name of selected images in batch
        [width]: input images width
        [height]: input images height
        [USE_DS]: using deep supervision
    OUTPUT:
        [img_sub]: all classes labeling batch
        [img_sub2]: all classes labeling batch for down-sampling
    """
    
    img_sub = np.zeros((len(batch_num),5,width,height))
    if(USE_DS):
        half_width = int(width/2)
        half_height = int(height/2)
        img_sub2 = np.zeros((len(batch_num),5,half_width,half_height))
    i = 0
    for name in np.array(batch_num):
        img_temp = np.zeros((width,height))
        img_sub_line = cv2.imread(gpath_line + name, 0)
        img_sub_cell = cv2.imread(gpath_cell + name, 0)
        img_sub_line[img_sub_cell==255] = 10
        img_temp[:,6:506] = img_sub_line
        img_sub[i,0,img_temp==63] = 1  
        img_sub[i,1,img_temp==127] = 1  
        img_sub[i,2,img_temp==191] = 1
        img_sub[i,3,img_temp==255] = 1
        img_sub[i,4,img_temp==10] = 1
        if(USE_DS):
            img_sub2[i,0,:,6:506] = cv2.resize(img_sub[i,0],(250,half_width),interpolation=cv2.INTER_NEAREST)
            img_sub2[i,1,:,6:506] = cv2.resize(img_sub[i,1],(250,half_width),interpolation=cv2.INTER_NEAREST)
            img_sub2[i,2,:,6:506] = cv2.resize(img_sub[i,2],(250,half_width),interpolation=cv2.INTER_NEAREST)
            img_sub2[i,3,:,6:506] = cv2.resize(img_sub[i,3],(250,half_width),interpolation=cv2.INTER_NEAREST)
            img_sub2[i,4,:,6:506] = cv2.resize(img_sub[i,4],(250,half_width),interpolation=cv2.INTER_NEAREST)
        i+=1 
    if(USE_DS):
        return img_sub, img_sub2
    else:
        return img_sub