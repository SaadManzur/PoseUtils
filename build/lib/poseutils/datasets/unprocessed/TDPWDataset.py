from __future__ import print_function, absolute_import, division

import numpy as np
from tqdm import tqdm
from poseutils.constants import dataset_indices

parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14]
joints_left = [4, 5, 6, 10, 11, 12]
joints_right = [1, 2, 3, 13, 14, 15]
                        
skeleton_3DPW_joints_group = [[2, 3], [5, 6], [1, 4], [0, 7], [12, 13], [9, 10], [8, 11]]

NAMES_3DPW = ['']*14
NAMES_3DPW[0] = 'Hip'
NAMES_3DPW[1] = 'RHip'
NAMES_3DPW[2] = 'RKnee'
NAMES_3DPW[3] = 'RAnkle'
NAMES_3DPW[4] = 'LHip'
NAMES_3DPW[5] = 'LKnee'
NAMES_3DPW[6] = 'LAnkle'
# NAMES_3DPW[7] = 'Spine2'
NAMES_3DPW[7] = 'Neck'
# NAMES_3DPW[8] = 'Head'
NAMES_3DPW[8] = 'LUpperArm'
NAMES_3DPW[9] = 'LElbow'
NAMES_3DPW[10] = 'LWrist'
NAMES_3DPW[11] = 'RUpperArm'
NAMES_3DPW[12] = 'RElbow'
NAMES_3DPW[13] = 'RWrist'

# Human3.6m IDs for training and testing
TRAIN_SUBJECTS = ['S0']
TEST_SUBJECTS = ['S0']

class TDPWDataset(object):

    def __init__(self, path):
        super(TDPWDataset, self).__init__()
        
        self.cameras = None

        self._data_train = { "2d": None, "3d": None }
        self._data_valid = { "2d": None, "3d": None }

        self.load_data(path)

    def load_data(self, path):

        data = np.load(path, allow_pickle=True, encoding='latin1')['data'].item()

        data_train = data['train']
        data_valid = data['test']

        self._data_train['2d'] = data_train["combined_2d"]
        self._data_train['3d'] = data_train["combined_3d_cam"]*1000

        self._data_valid['2d'] = data_valid["combined_2d"]
        self._data_valid['3d'] = data_valid["combined_3d_cam"]*1000

        print("[PoseUtils] Loaded raw data")

    def get_2d_valid(self, jnts=14):

        to_select, to_sort = dataset_indices('3dpw', jnts)

        return self._data_valid['2d'][:, to_select, :][:, to_sort, :]

    def get_3d_valid(self, jnts=14):

        to_select, to_sort = dataset_indices('3dpw', jnts)

        return self._data_valid['3d'][:, to_select, :][:, to_sort, :]
    
    def get_2d_train(self, jnts=14):

        to_select, to_sort = dataset_indices('3dpw', jnts)

        return self._data_train['2d'][:, to_select, :][:, to_sort, :]

    def get_3d_train(self, jnts=14):

        to_select, to_sort = dataset_indices('3dpw', jnts)

        return self._data_train['3d'][:, to_select, :][:, to_sort, :]