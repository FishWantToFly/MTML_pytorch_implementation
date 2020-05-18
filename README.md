# Unofficial implementation of 3D Instance Segmentation via Multi-Task Metric Learning (MTML)
This is unofficial implementation of MTML written in Pytorch.
Notice : We just build the network and see its loss decrease untill epoch 100 while training, and we have not implemented any post-process yet (we just apply simple mean-shift to see its visualization); therefore, **we have no any experiments and performance report right now**.

### (1) Setup
* Ubuntu 16.04 + cuda 9.0
* Python 3.6 + Pytorch 1.2
* pyntcloud library

### (2) Data Download
We use ScanNet dataset to implement.
ScanNet official website : http://www.scan-net.org/ (for data download)

### (3) Data Preprocess (point cloud -> voxel)
```
# Generate voxel from point cloud
python generate_voxel.py
# Generate train / test data list
cd voxel
python generate_data_list.py
cd ..
```

### (3) Train/test
```
python train.py
```

### (4) Visualization
```
python main_train.py
```

### (5) Qualitative Results on ScanNet
1. Point cloud
![Arch Image](https://github.com/FishWantToFly/MTML_pytorch_implementation/blob/master/images/point%20cloud.png)
2. Ground truth segmentation
![Arch Image](https://github.com/FishWantToFly/MTML_pytorch_implementation/blob/master/images/gt.png)
3. Mean shift results of feature embedding

![Arch Image](https://github.com/FishWantToFly/MTML_pytorch_implementation/blob/master/images/mean%20shift.png)

4. Instance segmentation results from mean shift of feature embedding
![Arch Image](https://github.com/FishWantToFly/MTML_pytorch_implementation/blob/master/images/mean%20shift%20prediction.png)