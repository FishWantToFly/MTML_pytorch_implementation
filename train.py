import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import os
import sys
import glob
import numpy as np
import random
import copy
from random import shuffle
import argparse
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
import h5py

from Dataset import Data_Configs as Data_Configs
from Dataset import Data_SCANNET as Data
from tqdm import tqdm
import importlib
import datetime, time
import logging
import lera
import math
import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import matplotlib.pyplot as plt
from model import MTML

def mkdir_p(dir_path):
    try:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def read_txt(filename):
    res= []
    with open(filename) as f:
        for line in f:
            res.append(line.strip())
    return res

parser = argparse.ArgumentParser('HTML')
parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs for training')
parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
parser.add_argument('--learning_rate', type=float, default= 5e-4, help='learning rate for training')
parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer')
parser.add_argument('--multi_gpu', type=str, default=None, help='whether use multi gpu training')
parser.add_argument('--model_name', type=str, default='Bo-net-revise', help='Name of model')
FLAGS = parser.parse_args()

checkpoint_dir = 'checkpoint'
mkdir_p(checkpoint_dir)

Weight_Decay = 1e-4
learning_rate = FLAGS.learning_rate

LOG_FOUT_train = open(os.path.join(checkpoint_dir, 'log_train.txt'), 'w')
LOG_FOUT_train.write(str(FLAGS)+'\n')
LOG_FOUT_test = open(os.path.join(checkpoint_dir, 'log_test.txt'), 'w')
LOG_FOUT_test.write(str(FLAGS)+'\n')

def log_string_train(out_str):
    LOG_FOUT_train.write(out_str+'\n')
    LOG_FOUT_train.flush()
    print(out_str)

def log_string_test(out_str):
    LOG_FOUT_test.write(out_str+'\n')
    LOG_FOUT_test.flush()
    print(out_str)

def get_fea_loss(feature_embedding, batch_group, batch_sem):
    '''
    Input : 
        feature_embedding [B, X, Y, Z, 3]
        batch_group [B, X, Y, Z, 1]
    '''
    batch_size = batch_group.shape[0]

    delta_var = 0.1
    delta_dist = 1.5
    cn = 0 # instace number of this scene

    # MTML setting
    gamma_var, gamma_dist, gamma_reg = 1, 1, 0.001
    loss_var_batch, loss_dist_batch, loss_reg_batch = 0, 0, 0

    for i in range(batch_size):
        pc_group = batch_group[i].squeeze(-1)
        pc_sem = batch_sem[i].squeeze(-1)
        pc_feature_embedding = feature_embedding[i] # [X, Y, Z, 3]
        pc_group_unique = torch.unique(pc_group)

        # decide size of average_feaures_embedding
        total_ins = 0
        for ins in pc_group_unique:
            if ins == -1: continue # invalid instance
            pos = (pc_group == ins).nonzero()
            if (pc_sem[pos[0][0]][pos[0][1]][pos[0][2]] <= 0): continue
            total_ins += 1
        average_feature_embeddings = torch.zeros((total_ins, 3)).cuda()
                    
        ##################
        # compute loss_var
        _id = 0
        loss_var_sum = 0
        for ins in pc_group_unique:
            pos = (pc_group == ins).nonzero()

            if ins == -1: continue
            if (pc_sem[pos[0][0]][pos[0][1]][pos[0][2]] <= 0): continue

            # feature embeddings of voxels belonged to this instance now
            _pc_embedding_feature = pc_feature_embedding[pos[:,0], pos[:, 1], pos[:, 2]].squeeze(1) # [num_voxel, 3]
            num_voxel = _pc_embedding_feature.shape[0]

            # compute averaege
            average_feature_embeddings_now = torch.mean(_pc_embedding_feature, dim = 0) # [3]
            average_feature_embeddings[_id] = average_feature_embeddings_now
            _id += 1

            # compute loss_var
            diff_embedding_feaures = average_feature_embeddings_now.repeat(num_voxel, 1) - _pc_embedding_feature # [num_voxel, 3]
            diff_embedding_feaures = torch.norm(diff_embedding_feaures, p = 2, dim = 1, keepdim = True)
            diff_embedding_feaures = torch.clamp(diff_embedding_feaures - delta_var, min = 0) ** 2
            loss_var_sum += torch.sum(diff_embedding_feaures) / num_voxel

        if (total_ins == 0) : 
            cn = 1
            loss_var_batch = torch.tensor(0)
        else : loss_var_batch += (loss_var_sum / total_ins)
        
        ##################
        # compute loss_dist
        C = total_ins
        loss_dist_sum = 0
        for i in range(C):
            for j in range(i + 1, C): # for non-repeated calculation
                diff_average_feaures = average_feature_embeddings[i] - average_feature_embeddings[j]
                diff_average_feaures = torch.norm(diff_average_feaures, p = 2)
                diff_average_feaures = torch.clamp(2 * delta_dist - diff_average_feaures, min = 0) ** 2
                loss_dist_sum += diff_average_feaures
        if (C == 0 or C == 1):
            loss_dist_batch  = torch.tensor(0)
            cn = 1
        else : loss_dist_batch += (loss_dist_sum / (C * (C - 1)))
        
        ##################
        # compute loss_reg
        loss_reg_sum = 0
        for i in range(C):
            diff_average_feaures = torch.norm(average_feature_embeddings[i], p = 2)
            loss_reg_sum += diff_average_feaures
        if (C == 0): loss_reg_batch = torch.tensor(0)
        else : loss_reg_batch += (loss_reg_sum / C)

    loss_var = loss_var_batch / batch_size
    loss_dist = loss_dist_batch / batch_size
    loss_reg = loss_reg_batch / batch_size

    return loss_var, loss_dist, loss_reg, cn

def get_dir_loss(dir_embedding, batch_group, batch_sem):
    '''
    Input : 
        dir_embedding [B, X, Y, Z, 3]
        batch_group [B, X, Y, Z, 1]
    '''
    batch_size = batch_group.shape[0]
    loss_dir_batch = 0

    for i in range(batch_size):
        pc_group = batch_group[i].squeeze(-1)
        pc_sem = batch_sem[i].squeeze(-1)
        pc_dir_embedding = dir_embedding[i] # [X, Y, Z, 3]
        pc_group_unique = torch.unique(pc_group)

        ##################
        # compute loss_dir
        total_ins = 0
        loss_dir_sum = 0
        for ins in pc_group_unique:
            pos = (pc_group == ins).nonzero()
            if ins == -1: continue
            if(pc_sem[pos[0][0]][pos[0][1]][pos[0][2]] <= 0): continue
            
            _pc_dir_bedding = pc_dir_embedding[pos[:,0], pos[:, 1], pos[:, 2]].squeeze(1) # [num_voxel, 3]
            num_voxel = _pc_dir_bedding.shape[0]

            ## Exception : num_voxel only 1
            ## Solution now : skip it
            if num_voxel <= 1 : continue

            # compute v_i and v_i_GT
            v_i = _pc_dir_bedding / torch.norm(_pc_dir_bedding, p = 2, dim = 1, keepdim = True) # normalized dir_embedding # [num_voxel, 3]
            x_i = pos.float() # [num_voxel, 3]
            x_center = torch.mean(pos.float(), dim = 0).unsqueeze(0) # center of this instance # [1, 3] 
            x_center = x_center.repeat(num_voxel, 1) # [num_voxel, 3]

            ## Exception : x_i equals to x_center -> cause Denominator to be 0
            ## Solution : ignore that center-like voxel
            # find which pixel is at the position of instance center
            check_center = torch.sum(torch.abs((x_i - x_center)), axis = 1) == 0

            v_i_GT = (x_i - x_center) / torch.norm((x_i - x_center), p = 2, dim = 1, keepdim = True) # [num_voxel, 3]
            v_i_GT[check_center, :] = 0
            loss_dir_sum += torch.sum(torch.mul(v_i, v_i_GT)) / num_voxel
            total_ins += 1

        if (total_ins == 0) :  loss_dir_batch += torch.tensor(0)
        else : loss_dir_batch += loss_dir_sum / total_ins

    loss_dir = -1 * loss_dir_batch / batch_size
    return loss_dir

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] ='0'

    dataset_path = './voxel'
    train_scene_txt = os.path.join(dataset_path ,'train.txt')
    val_scene_txt = os.path.join(dataset_path ,'val.txt')

    train_scenes = read_txt(train_scene_txt)
    val_scenes = read_txt(val_scene_txt)

    _dataset_path = os.path.join(dataset_path, 'voxel')
    train_data = Data(_dataset_path, train_scenes, val_scenes , mode = 'train')
    val_data = Data(_dataset_path, train_scenes, val_scenes , mode = 'val')
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=FLAGS.batchsize, shuffle=True, 
                    num_workers=10)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=FLAGS.batchsize, shuffle=False, 
                    num_workers=10)

    mtml = MTML().cuda()
    mtml = torch.nn.DataParallel(mtml, device_ids = [0])
    optim_params = [
        {'params' : mtml.parameters() , 'lr' : FLAGS.learning_rate , 'betas' : (0.9, 0.999) , 'eps' : 1e-08 },
    ]
    optimizer = optim.Adam(optim_params , lr=learning_rate ,weight_decay=Weight_Decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # Ratio of loss function
    fea_ratio, dir_ratio, sem_ratio = 1, 0.5, 1

    print("Start training.")
    for epoch in range(FLAGS.epoch):
        loss_list = []
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            rgb , sem , ins = data
            rgb , sem , ins = rgb.cuda() , sem.cuda() , ins.cuda()
            dir_embedding, feature_embedding = mtml(rgb)

            loss_var,  loss_dist, loss_reg , cn = get_fea_loss(feature_embedding, ins, sem)
            dir_loss = get_dir_loss(dir_embedding, ins, sem) * dir_ratio
            if (cn == 1):
                continue
            fea_loss = (loss_var + loss_dist + 0.001 * loss_reg)  * fea_ratio
            total_loss = fea_loss + dir_loss 

            total_loss.backward()
            optimizer.step()

            loss_list.append([total_loss.item(), loss_var.item(), loss_dist.item(), loss_reg.item(), fea_loss.item(), dir_loss.item()])

            if i % 20 == 0:
                print("Epoch %3d Iteration %3d (train)" % (epoch, i))
                print("%.3f %.3f %.3f %.3f %.3f %.3f" % (total_loss.item(), loss_var.item(), loss_dist.item(), loss_reg.item(), fea_loss.item(), dir_loss.item()))
                print('')

        loss_list_final = np.mean(loss_list, axis=0)
        log_string_train(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        log_string_train("Epoch %3d (train)" % (epoch))
        log_string_train("%.3f %.3f %.3f %.3f %.3f %.3f" % (loss_list_final[0], loss_list_final[1], loss_list_final[2], loss_list_final[3], loss_list_final[4], loss_list_final[5]))
        log_string_train('')
        scheduler.step(epoch)  

        # Save model
        if epoch % 5 == 0:
                torch.save(mtml.module.state_dict(), '%s/%s_%.3d.pth' % (checkpoint_dir, 'mtml', epoch))

        # Testing
        if epoch % 5 == 0:
            loss_list = []
            with torch.no_grad():
                for i, data in enumerate(val_dataloader):
                    rgb , sem , ins = data
                    rgb , sem , ins  = rgb.cuda() , sem.cuda() , ins.cuda()
                    dir_embedding, feature_embedding = mtml(rgb)

                    loss_var, loss_dist, loss_reg, cn = get_fea_loss(feature_embedding, ins, sem)
                    dir_loss = get_dir_loss(dir_embedding, ins, sem) * dir_ratio
                    if (cn == 1):
                        continue
                    fea_loss = (loss_var + loss_dist + 0.001 * loss_reg) * fea_ratio
                    total_loss = fea_loss + dir_loss 

                    loss_list.append([total_loss.item(), loss_var.item(), loss_dist.item(), loss_reg.item(), fea_loss.item(), dir_loss.item()])

                loss_list_final = np.mean(loss_list, axis=0)
                log_string_test(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                log_string_test("Epoch %3d (test)" % (epoch))
                log_string_test("%.3f %.3f %.3f %.3f %.3f %.3f" % (loss_list_final[0], loss_list_final[1], loss_list_final[2], loss_list_final[3], loss_list_final[4], loss_list_final[5]))
                log_string_test('')
            