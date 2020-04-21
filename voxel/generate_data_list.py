import glob, os, copy, random
import numpy as np
from os import walk
from sklearn.model_selection import train_test_split

data_list_dir = "./voxel"
data_list = []
for scene in glob.glob("./voxel/*.h5"):
	_, _scene = os.path.split(scene)
	data_list.append(_scene[:-3])

random.seed()
train_list, val_list = train_test_split(data_list, test_size=0.2)
print("Train data len : %d" % (len(train_list)))
print("Val data len : %d" % (len(val_list)))

train_save_dir = os.path.join('train.txt')
val_save_dir = os.path.join('val.txt')
with open(train_save_dir, 'w') as f:
	for scene in train_list:
		f.write("%s\n" % scene)
with open(val_save_dir, 'w') as f:
	for scene in val_list:
		f.write("%s\n" % scene)
