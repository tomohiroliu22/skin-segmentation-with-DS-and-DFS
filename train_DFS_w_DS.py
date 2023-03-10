import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import loss as L
import pandas as pd
from data import data_loader, o_data, g_data_cell, g_data_line
import torch.optim as optim
from model import Optim_U_Net
from tool import IOUDICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_dfs_w_ds(opt): 
    fold = opt.fold
    print('\n U-Net with DFS + DS training: Fold:', fold)
    # Creating or loading the model 
    model = Optim_U_Net(img_ch=opt.input_nc,output_ch=opt.output_nc, USE_DS = True, USE_DFS = True)
    if(opt.load_model):
        model.load_state_dict(torch.load(opt.modelpath))
    model = model.to(device)
    number_of_parameters = sum(p.numel() for p in model.parameters())
    print(number_of_parameters)

    # Loading data
    gpath_cell = opt.dataroot + "/cell/"
    gpath_line = opt.dataroot + "/layer/"
    opath = opt.dataroot + "/image/"
    train_data_LD, valid_data_LD, _ = data_loader(opath, fold)

    # Loss function and optimizer
    loss_func2 = L.DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=0.1)

    train_epoch = opt.epoch
    # record the overall loss and dice 
    train_loss = np.zeros(train_epoch)
    valid_loss = np.zeros(train_epoch)
    train_mdice_cell = np.zeros(train_epoch)
    valid_mdice_cell = np.zeros(train_epoch)
    train_mdice_layer = np.zeros(train_epoch)
    valid_mdice_layer = np.zeros(train_epoch)
    
    height = 512
    width = 384
    for EPOCH in range(train_epoch):
        print('\nEPOCH:{} learning rate:{}====================================='.format(EPOCH,optimizer.param_groups[0]['lr']))
        start = time.time()
        model.train()
        # record the training loss and dice 
        train_loss_list = np.empty((0,1))
        train_dice_list_cell = np.empty((0,1))
        train_dice_list_layer = np.empty((0,1))
        # model training
        for _, t_batch_num in enumerate(train_data_LD):
            # loading training data
            img_sub  = o_data(opath, t_batch_num, width, height)
            t_gim_sub, t_gim_sub2 = g_data_cell(gpath_cell, t_batch_num, width, height, USE_DS=True)
            t_gim_sub3, t_gim_sub4, t_gim_sub5, t_gim_sub6 = g_data_line(gpath_line, t_batch_num, width, height)
            
            # Numpy to Tensor on GPU
            INPUT =  torch.from_numpy(img_sub.astype(np.float32)).to(device = device, dtype = torch.float)
            target = torch.from_numpy(t_gim_sub).to(device,dtype = torch.long)
            target2 = torch.from_numpy(t_gim_sub2).to(device,dtype = torch.long)
            target3 = torch.from_numpy(t_gim_sub3).to(device,dtype = torch.long)
            target4 = torch.from_numpy(t_gim_sub4).to(device,dtype = torch.long)
            target5 = torch.from_numpy(t_gim_sub5).to(device,dtype = torch.long)
            target6 = torch.from_numpy(t_gim_sub6).to(device,dtype = torch.long)

            # Model output and weight updating
            OUTPUT, OUTPUT2, OUTPUT3, OUTPUT4, OUTPUT5, OUTPUT6 = model(INPUT)
            loss = loss_func2(OUTPUT6, L.make_one_hot(target6,2).cuda()) + loss_func2(OUTPUT5, L.make_one_hot(target5,2).cuda()) + loss_func2(OUTPUT4, L.make_one_hot(target4,2).cuda()) + loss_func2(OUTPUT3, L.make_one_hot(target3,2).cuda()) + loss_func2(OUTPUT, L.make_one_hot(target,2).cuda()) + loss_func2(OUTPUT2, L.make_one_hot(target2,2).cuda())/2
            optimizer.zero_grad()              
            loss.backward()                     
            optimizer.step()

            # metrics recording
            train_loss_list = np.vstack((train_loss_list,loss.item()))
            t_out_img = np.argmax(OUTPUT.cpu().detach().numpy(), 1)
            t_out_img3 = np.argmax(OUTPUT3.cpu().detach().numpy(), 1)
            t_out_img4 = np.argmax(OUTPUT4.cpu().detach().numpy(), 1)
            t_out_img5 = np.argmax(OUTPUT5.cpu().detach().numpy(), 1)
            t_out_img6 = np.argmax(OUTPUT6.cpu().detach().numpy(), 1)
            NUM = len(t_batch_num)
            seg = np.zeros((NUM,384,500))
            seg[t_out_img6[:,:,6:506]==1] = 0
            seg[t_out_img5[:,:,6:506]==1] = 1
            seg[t_out_img4[:,:,6:506]==1] = 2
            seg[t_out_img3[:,:,6:506]==1] = 3
            seg[t_out_img[:,:,6:506]==1] = 4
            gt = np.zeros((NUM,384,500))
            gt[t_gim_sub6[:,0,:,6:506]==1] = 0
            gt[t_gim_sub5[:,0,:,6:506]==1] = 1
            gt[t_gim_sub4[:,0,:,6:506]==1] = 2
            gt[t_gim_sub3[:,0,:,6:506]==1] = 3
            gt[t_gim_sub[:,0,:,6:506]==1] = 4
            layer_DICE = np.array(IOUDICE(seg,gt,0))
            layer_DICE += np.array(IOUDICE(seg,gt,1))
            layer_DICE += np.array(IOUDICE(seg,gt,2))
            layer_DICE += np.array(IOUDICE(seg,gt,3))
            layer_DICE /= 4
            cell_DICE = np.array(IOUDICE(seg,gt,4))
            train_dice_list_cell = np.vstack((train_dice_list_cell,cell_DICE.item()))
            train_dice_list_layer = np.vstack((train_dice_list_layer,layer_DICE.item()))
            
        model.eval()
        valid_loss_list = np.empty((0,1))
        valid_dice_list_cell = np.empty((0,1))
        valid_dice_list_layer = np.empty((0,1))
        for _, v_batch_num in enumerate(valid_data_LD):
            # loading validation data
            val_sub  =  o_data(opath, v_batch_num, width, height)
            v_gim_sub, v_gim_sub2 = g_data_cell(gpath_cell, v_batch_num, width, height, USE_DS = True)
            v_gim_sub3, v_gim_sub4, v_gim_sub5, v_gim_sub6 = g_data_line(gpath_line, v_batch_num, width, height)
            
            # Numpy to Tensor on GPU
            INPUT =  torch.from_numpy(val_sub.astype(np.float32)).to(device = device, dtype = torch.float)
            target = torch.from_numpy(v_gim_sub).to(device,dtype = torch.long)
            target2 = torch.from_numpy(v_gim_sub2).to(device,dtype = torch.long)
            target3 = torch.from_numpy(v_gim_sub3).to(device,dtype = torch.long)
            target4 = torch.from_numpy(v_gim_sub4).to(device,dtype = torch.long)
            target5 = torch.from_numpy(v_gim_sub5).to(device,dtype = torch.long)
            target6 = torch.from_numpy(v_gim_sub6).to(device,dtype = torch.long)
            OUTPUT, OUTPUT2, OUTPUT3, OUTPUT4, OUTPUT5, OUTPUT6 = model(INPUT)
            
            # computing loss
            loss_v = loss_func2(OUTPUT6, L.make_one_hot(target6,2).cuda()) + loss_func2(OUTPUT5, L.make_one_hot(target5,2).cuda()) + loss_func2(OUTPUT4, L.make_one_hot(target4,2).cuda()) + loss_func2(OUTPUT3, L.make_one_hot(target3,2).cuda()) + loss_func2(OUTPUT, L.make_one_hot(target,2).cuda()) + loss_func2(OUTPUT2, L.make_one_hot(target2,2).cuda())/2
            
            # metrics recording
            valid_loss_list = np.vstack((valid_loss_list, loss_v.item()))
            v_out_img = np.argmax(OUTPUT.cpu().detach().numpy(), 1)
            v_out_img3 = np.argmax(OUTPUT3.cpu().detach().numpy(), 1)
            v_out_img4 = np.argmax(OUTPUT4.cpu().detach().numpy(), 1)
            v_out_img5 = np.argmax(OUTPUT5.cpu().detach().numpy(), 1)
            v_out_img6 = np.argmax(OUTPUT6.cpu().detach().numpy(), 1)
            NUM = len(v_batch_num)
            seg = np.zeros((NUM,384,500))
            seg[v_out_img6[:,:,6:506]==1] = 0
            seg[v_out_img5[:,:,6:506]==1] = 1
            seg[v_out_img4[:,:,6:506]==1] = 2
            seg[v_out_img3[:,:,6:506]==1] = 3
            seg[v_out_img[:,:,6:506]==1] = 4
            gt = np.zeros((NUM,384,500))
            gt[v_gim_sub6[:,0,:,6:506]==1] = 0
            gt[v_gim_sub5[:,0,:,6:506]==1] = 1
            gt[v_gim_sub4[:,0,:,6:506]==1] = 2
            gt[v_gim_sub3[:,0,:,6:506]==1] = 3
            gt[v_gim_sub[:,0,:,6:506]==1] = 4
            layer_DICE = np.array(IOUDICE(seg,gt,0))
            layer_DICE += np.array(IOUDICE(seg,gt,1))
            layer_DICE += np.array(IOUDICE(seg,gt,2))
            layer_DICE += np.array(IOUDICE(seg,gt,3))
            layer_DICE /= 4
            cell_DICE = np.array(IOUDICE(seg,gt,4))
            valid_dice_list_cell = np.vstack((valid_dice_list_cell,cell_DICE.item()))
            valid_dice_list_layer = np.vstack((valid_dice_list_layer,layer_DICE.item()))
            
        scheduler.step()

        # record overall result for each epoch
        train_loss[EPOCH],valid_loss[EPOCH] = train_loss_list.mean(),valid_loss_list.mean()
        train_mdice_cell[EPOCH],valid_mdice_cell[EPOCH] = train_dice_list_cell.mean(),valid_dice_list_cell.mean()
        train_mdice_layer[EPOCH],valid_mdice_layer[EPOCH] = train_dice_list_layer.mean(),valid_dice_list_layer.mean()
        
        
        print('%.3d'%EPOCH, "train loss:", '%.3f'%train_loss[EPOCH], "valid loss:", '%.3f'%valid_loss[EPOCH],
              "train DICE:",'%.3f'%train_mdice_cell[EPOCH], "valid DICE:",'%.3f'%valid_mdice_cell[EPOCH],
              "train DICE:",'%.3f'%train_mdice_layer[EPOCH], "valid DICE:",'%.3f'%valid_mdice_layer[EPOCH],
              round((time.time()-start),3))
    # save last epoch
    torch.save(model.state_dict() ,'{}.pkl'.format(opt.name))
    print("----saving successfully with name: {}.pkl----".format(opt.name))
    
    columns = ['training loss', "validation loss", 
    'training DICE (cell)', "validation DICE (cell)", 'training DICE (layer)', "validation DICE (layer)"]
    df = pd.DataFrame([train_loss,valid_loss,train_mdice_cell,valid_mdice_cell,train_mdice_layer,valid_mdice_layer],columns=columns)
    df.to_csv(opt.name+"_DFS_DS.csv",index=False)
    
    return 

def test_dfs_w_ds(opt):
    fold = opt.fold
    print('\n U-Net with DFS + DS testing: Fold:', fold)
    # Creating or loading the model 
    model = Optim_U_Net(img_ch=opt.input_nc,output_ch=opt.output_nc, USE_DS = True, USE_DFS = True)
    if(opt.load_model):
        model.load_state_dict(torch.load(opt.modelpath))
    model = model.to(device)
    number_of_parameters = sum(p.numel() for p in model.parameters())
    print(number_of_parameters)

    # Loading data
    gpath_cell = opt.dataroot + "/cell/"
    gpath_line = opt.dataroot + "/layer/"
    opath = opt.dataroot + "/image/"
    _, _, test_data_LD = data_loader(opath, fold)
    
    height = 512
    width = 384
    model.eval()
    test_dice_list_cell = np.empty((0,1))
    test_dice_list_layer = np.empty((0,1))
    for _, v_batch_num in enumerate(test_data_LD):
        # loading validation data
        test_sub  =  o_data(opath, v_batch_num, width, height)
        v_gim_sub = g_data_cell(gpath_cell, v_batch_num, width, height, USE_DS = False)
        v_gim_sub3, v_gim_sub4, v_gim_sub5, v_gim_sub6 = g_data_line(gpath_line, v_batch_num, width, height)
            
        # Numpy to Tensor on GPU
        INPUT =  torch.from_numpy(test_sub.astype(np.float32)).to(device = device, dtype = torch.float)
        OUTPUT, OUTPUT3, OUTPUT4, OUTPUT5, OUTPUT6 = model(INPUT)
                
        # metrics recording
        v_out_img = np.argmax(OUTPUT.cpu().detach().numpy(), 1)
        v_out_img3 = np.argmax(OUTPUT3.cpu().detach().numpy(), 1)
        v_out_img4 = np.argmax(OUTPUT4.cpu().detach().numpy(), 1)
        v_out_img5 = np.argmax(OUTPUT5.cpu().detach().numpy(), 1)
        v_out_img6 = np.argmax(OUTPUT6.cpu().detach().numpy(), 1)
        NUM = len(v_batch_num)
        seg = np.zeros((NUM,384,500))
        seg[v_out_img6[:,:,6:506]==1] = 0
        seg[v_out_img5[:,:,6:506]==1] = 1
        seg[v_out_img4[:,:,6:506]==1] = 2
        seg[v_out_img3[:,:,6:506]==1] = 3
        seg[v_out_img[:,:,6:506]==1] = 4
        gt = np.zeros((NUM,384,500))
        gt[v_gim_sub6[:,0,:,6:506]==1] = 0
        gt[v_gim_sub5[:,0,:,6:506]==1] = 1
        gt[v_gim_sub4[:,0,:,6:506]==1] = 2
        gt[v_gim_sub3[:,0,:,6:506]==1] = 3
        gt[v_gim_sub[:,0,:,6:506]==1] = 4
        for idx in range(len(v_batch_num)):
            layer_DICE = np.array(IOUDICE(seg[idx],gt[idx],0))
            layer_DICE += np.array(IOUDICE(seg[idx],gt[idx],1))
            layer_DICE += np.array(IOUDICE(seg[idx],gt[idx],2))
            layer_DICE += np.array(IOUDICE(seg[idx],gt[idx],3))
            layer_DICE /= 4
            cell_DICE = np.array(IOUDICE(seg[idx],gt[idx],4))
            test_dice_list_cell = np.vstack((test_dice_list_cell,cell_DICE.item()))
            test_dice_list_layer = np.vstack((test_dice_list_layer,layer_DICE.item()))
    print( "testing DICE (cell):",'%.3f'%test_dice_list_cell.mean(), "testing DICE (layer):",'%.3f'%test_dice_list_layer.mean())
    return