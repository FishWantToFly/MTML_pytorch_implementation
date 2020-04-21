import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import os, sys, glob
import numpy as np
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(os.path.join(BASE_DIR, '../3D-BoNet/data_scannet/utils'))
from Dataset import Data_Configs as Data_Configs
from Dataset import Data_SCANNET as Data
import io_util
from tqdm import tqdm
import math
import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import matplotlib.pyplot as plt
from model import MTML
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pyntcloud import PyntCloud

def read_txt(filename):
    res= []
    with open(filename) as f:
        for line in f:
            res.append(line.strip())
    return res

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] ='0'

    epoch_num = '080' # which epoch you want to use
    MODEL_PATH = os.path.join('checkpoint')
    dataset_path ='voxel'

    train_scene_txt = os.path.join(dataset_path,'train.txt')
    val_scene_txt = os.path.join(dataset_path ,'val.txt')
    train_scenes = read_txt(train_scene_txt)
    val_scenes = read_txt(val_scene_txt)

    _dataset_path = os.path.join(dataset_path, 'voxel')
    val_data = Data(_dataset_path, train_scenes, val_scenes , mode = 'val')
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, 
                    num_workers=10)

    mtml = MTML().cuda().eval()
    mtml.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'mtml_%s.pth' % (epoch_num)))) 

    for i, data in enumerate(val_dataloader):
        rgb , sem , ins = data
        rgb , sem , ins  = rgb.cuda() , sem.cuda() , ins.cuda()  
        sem = sem.squeeze(0).squeeze(-1).view(-1)
        ins = ins.squeeze(0).squeeze(-1).view(-1)
        dir_embedding, feature_embedding = mtml(rgb)
        rgb = rgb.view(-1,3)
        loc = feature_embedding.view(-1,3)
        loc = loc[sem > 0]
        ins = ins[sem > 0]
        rgb = rgb[sem > 0]
        
        # choose first scene as visualization target
        if (i == 0):
            break
    
    # Prepare for plotting
    x = np.arange(10)
    ys = [i+x+(i*x)**2 for i in range(8)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Do mean shift on feature embedding
    ms = MeanShift()
    ms.fit(loc.detach().cpu().numpy())
    cluster_centers = ms.cluster_centers_
    labels = ms.labels_
    ax.scatter(cluster_centers[:,0], cluster_centers[:,1], cluster_centers[:,2], marker='x', color='red', s=300, linewidth=5, zorder=10)

    # visualization
    for i in np.unique(labels):
        ss = loc[labels == i]
        ss = ss.detach().cpu().numpy()
        ax.scatter(ss[:,0], ss[:,1], ss[:,2], marker='o' , color=colors[i])
    plt.show()