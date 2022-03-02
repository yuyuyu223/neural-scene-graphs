from data_loader.load_kitti import load_kitti_data
from main import train
import sys

# load_kitti_data(basedir="./data/KITTI/data_tracking_image_2/training/image_02/0000", selected_frames=[[0,0],[5,6]])

if __name__ == '__main__':
    sys.argv = ['--config=example_configs/config_kitti_0006_example_train.txt']
                # '--first_frame=0,0',
                # '--last_frame=5,6',
                # '--dataset_type=kitti',
                # '--expname=yu_00',
                # '--datadir=./data/kitti/training/image_02/0006',
                # '--obj_only']
    train()
    