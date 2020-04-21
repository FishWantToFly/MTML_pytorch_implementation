'''
For training
'''

import os
import sys
import glob
import numpy as np
import random
import h5py
import torch
from PIL import Image
from scipy.ndimage.interpolation import rotate

class Data_Configs:
    sem_names = ['background', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refridgerator', 
                'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture'
                ]
    sem_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    sem_num = len(sem_names)
    ins_max_num = 20 # for scannet maximum

class Data_SCANNET:
    def __init__(self, dataset_path, train_scenes, test_scenes , mode):
        self.root_folder_4_traintest = dataset_path
        if mode == 'train':
            self.train_files = train_scenes
            print('train files:', len(self.train_files))
        elif mode == 'val':
            self.test_files = test_scenes
            print('val files:', len(self.test_files))

        self.mode = mode


    def load_data_file_voxel(self, file_path):
        scene = file_path
        fin = h5py.File(os.path.join(self.root_folder_4_traintest, scene + '.h5'), 'r')
        rgbs = fin['rgbs'][:] # [H, W, D , 3] 
        sem = fin['sem_labels'][:] # [H, W, D, 1] 
        ins = fin['ins_labels'][:] # [H, W, D, 1]
        return rgbs, sem, ins

    
    def load_voxel(self, file_path):
        rgbs, sem, ins = self.load_data_file_voxel(file_path)
        return rgbs, sem, ins 


    def __getitem__(self , index):
        if self.mode == 'train':
            bat_files = self.train_files[index]
        elif self.mode == 'val':
            bat_files = self.test_files[index]
        rgbs, sem, ins = self.load_voxel(bat_files)

        # Data augmentation
        if self.mode == 'train':
            # rotate degree
            angle_list = [15 , 30 , 45 , 75 , 90 , 105 , 120 , 135 , 150 , 165 , 180] 

            # which operation needed to do now
            do_flip = np.random.uniform(0.0, 1.0) < 0.5 
            do_rotate = np.random.uniform(0.0, 1.0) < 0.5 
            # rotate which angle
            angles = random.randint(0, 10)
            
            # flip
            if (do_flip == 1):
                rgbs = np.ascontiguousarray(np.flip(rgbs , (0,1)))
                sem = np.ascontiguousarray(np.flip(sem , (0,1)))
                ins = np.ascontiguousarray(np.flip(ins , (0,1)))
            # rotate
            if (do_rotate == 1):
                rgbs = rotate(rgbs , angle = angle_list[angles], axes = (0,1) , mode = 'mirror', order=0 , reshape=False)
                sem = rotate(sem , angle = angle_list[angles], axes = (0,1) , mode = 'mirror', order=0 , reshape=False)
                ins = rotate(ins , angle = angle_list[angles], axes = (0,1) , mode = 'mirror', order=0 , reshape=False)

        rgbs = np.asarray(rgbs, dtype=np.float32)
        sem = np.asarray(sem, dtype=np.int32)
        ins = np.asarray(ins, dtype=np.int32)
        
        return rgbs, sem , ins

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_files)
        elif self.mode == 'val':
            return len(self.test_files)