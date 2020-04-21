import numpy as np
import glob, os, pickle, json, copy, sys, h5py

from statistics import mode
import plyfile
from pyntcloud import PyntCloud
from plyfile import PlyData, PlyElement
from collections import Counter

MTML_VOXEL_SIZE = 0.1 # size for voxel

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def read_label_ply(filename):
    plydata = PlyData.read(filename)
    x = np.asarray(plydata.elements[0].data['x'])
    y = np.asarray(plydata.elements[0].data['y'])
    z = np.asarray(plydata.elements[0].data['z'])
    label = np.asarray(plydata.elements[0].data['label'])
    return np.stack([x,y,z], axis=1), label

def read_color_ply(filename):
    plydata = PlyData.read(filename)
    x = np.asarray(plydata.elements[0].data['x'])
    y = np.asarray(plydata.elements[0].data['y'])
    z = np.asarray(plydata.elements[0].data['z'])
    r = np.asarray(plydata.elements[0].data['red'])
    g = np.asarray(plydata.elements[0].data['green'])
    b = np.asarray(plydata.elements[0].data['blue'])
    return np.stack([x,y,z,r,g,b], axis=1)

def collect_label(labelPath, scan): 
    aggregation = os.path.join(labelPath, scan+'.aggregation.json')
    segs = os.path.join(labelPath, scan+'_vh_clean_2.0.010000.segs.json')
    sem = os.path.join(labelPath, scan+'_vh_clean_2.labels.ply')
    # Load all labels
    fid = open(aggregation,'r')
    aggreData = json.load(fid)
    fid = open(segs,'r')
    segsData = json.load(fid)
    _, semLabel = read_label_ply(sem)

    # Convert segments to normal labels
    segGroups = aggreData['segGroups']
    segIndices = np.array(segsData['segIndices'])

    # outGroups is the output instance labels
    outGroups = np.zeros(np.shape(segIndices)) - 1

    for j in range(np.shape(segGroups)[0]):
        segGroup = segGroups[j]['segments']
        objectId = segGroups[j]['objectId']
        for k in range(np.shape(segGroup)[0]):
            segment = segGroup[k]
            ind = np.where(segIndices==segment)
            if all(outGroups[ind] == -1) != True:
                print('Error!')
            outGroups[ind] = int(objectId)

    outGroups = outGroups.astype(int)
    return semLabel, outGroups

def save_h5(h5_filename, rgbs, sem_labels, ins_labels, data_dtype='float32', label_dtype='int32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'rgbs', data=rgbs,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'sem_labels', data=sem_labels,
            compression='gzip', compression_opts=4,
            dtype=label_dtype)
    h5_fout.create_dataset(
            'ins_labels', data=ins_labels,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

def most_common(lst):
    data = Counter(lst)
    return max(lst, key=data.get)

# take into account wall and floor
VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
target_sem_idx = np.arange(40)
count = 0
for i in range(40):
    if i in VALID_CLASS_IDS:
        count += 1
        target_sem_idx[i] = count
    else:
        target_sem_idx[i] = 0
        
def changem(input_array, source_idx, target_idx):
    mapping = {}
    for i, sidx in enumerate(source_idx):
        mapping[sidx] = target_idx[i]
    input_array = np.array([mapping[i] for i in input_array])
    return input_array


if __name__ == '__main__':
    dataset_path = sys.argv[1]
    save_path = './voxel'

    anky_cloud_list = sorted(glob.glob(os.path.join(dataset_path, '*/*_vh_clean_2.ply')))

    save_map_dir = os.path.join(save_path, 'mapping')
    save_voxel_dir = os.path.join(save_path, 'voxel')
    make_dir(save_path)
    make_dir(save_map_dir)
    make_dir(save_voxel_dir)

    non_empty_voxel_num_list = []
    index = -1

    for anky_cloud_path in anky_cloud_list:
        index += 1
        anky_cloud_label_path = anky_cloud_path[:-3]+'labels.ply'
        scnen_name = os.path.basename(anky_cloud_path)[:12]
        anky_cloud = PyntCloud.from_file(anky_cloud_path)
        voxelgrid_id = anky_cloud.add_structure("voxelgrid", size_x=MTML_VOXEL_SIZE, size_y=MTML_VOXEL_SIZE, size_z=MTML_VOXEL_SIZE, \
            regular_bounding_box=False) # regular_bounding_box set false to allow different length of xyz
        voxelgrid = anky_cloud.structures[voxelgrid_id]
        pc_num = voxelgrid.voxel_x.shape[0]

        x_len, y_len, z_len = voxelgrid.x_y_z
        # create voxel->pc dict
        voxel_pc_mapping_dict = {}
        for i in range(pc_num):
            _x = voxelgrid.voxel_x[i]
            _y = voxelgrid.voxel_y[i]
            _z = voxelgrid.voxel_z[i]
            get_list = voxel_pc_mapping_dict.get((_x, _y, _z))
            if get_list == None :
                voxel_pc_mapping_dict[(_x, _y, _z)] = [i]
            else :
                temp_list = []
                for el in (get_list):
                    temp_list.append(el)
                temp_list.append(i)
                voxel_pc_mapping_dict[(_x, _y, _z)] = temp_list

        # save voxel -> point cloud mapping
        with open(os.path.join(save_map_dir, '%s.pkl' % (scnen_name)), 'wb') as fp:
            pickle.dump(voxel_pc_mapping_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        '''
        create rgb / ins_label / sem_label
        '''
        # pad zero if length is odd (for 3d conv/deconv)
        x_len_even, y_len_even, z_len_even = x_len, y_len, z_len
        if x_len_even % 2 == 1:
            x_len_even += 1 
        if y_len_even % 2 == 1:
            y_len_even += 1 
        if z_len_even % 2 == 1:
            z_len_even += 1 

        voxel_sem_label = -1 * np.zeros((x_len_even, y_len_even, z_len_even, 1), dtype = np.int32) 
        voxel_ins_label = -1 * np.zeros((x_len_even, y_len_even, z_len_even, 1), dtype = np.int32)
        voxel_rgb = np.zeros((x_len_even, y_len_even, z_len_even, 3))

        sem_label_gt, ins_label_gt = collect_label(os.path.dirname(anky_cloud_label_path), scnen_name)
        sem_label_gt[sem_label_gt>=40] = 0
        sem_label_gt[sem_label_gt<0] = 0
        sem_label_gt = changem(sem_label_gt, np.arange(40), target_sem_idx)
        rgb_label_gt = read_color_ply(anky_cloud_path)[:, 3:6]

        for i in range(x_len):
            for j in range(y_len):
                for k in range(z_len):
                    pc_list = voxel_pc_mapping_dict.get((i, j, k))
                    if pc_list != None :
                        this_voxel_sem_list = sem_label_gt[pc_list]
                        this_voxel_ins_list = ins_label_gt[pc_list]
                        this_voxel_rgb_list = rgb_label_gt[pc_list]
                        
                        # sem + ins
                        _sem = most_common(this_voxel_sem_list)
                        _ins = most_common(this_voxel_ins_list)
                        voxel_sem_label[i][j][k] = _sem
                        voxel_ins_label[i][j][k] = _ins

                        # rgb
                        r_sum, g_sum, b_sum = 0, 0, 0
                        for l in this_voxel_rgb_list:
                            r_sum += l[0] 
                            g_sum += l[1] 
                            b_sum += l[2] 
                        r_final = r_sum / len(this_voxel_rgb_list) / 255
                        g_final = g_sum / len(this_voxel_rgb_list) / 255
                        b_final = b_sum / len(this_voxel_rgb_list) / 255 
                        voxel_rgb[i][j][k] = (r_final, g_final, b_final)
        
        # store as .h5 file
        rgbs = copy.deepcopy(voxel_rgb)
        sem_labels = copy.deepcopy(voxel_sem_label)
        ins_labels = copy.deepcopy(voxel_ins_label)

        h5_filename = os.path.join(save_voxel_dir, '%s.h5' % scnen_name)
        print(index)
        print('{0}'.format(h5_filename))
        print()
        if not os.path.isfile(h5_filename):
            save_h5(h5_filename,
                    rgbs,
                    sem_labels,
                    ins_labels
            )
