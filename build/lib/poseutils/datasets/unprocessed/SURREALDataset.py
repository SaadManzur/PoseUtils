from __future__ import print_function, absolute_import, division

import numpy as np
from poseutils.constants import dataset_indices

parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14]
joints_left = [4, 5, 6, 10, 11, 12]
joints_right = [1, 2, 3, 13, 14, 15]
                        
skeleton_SURREAL_joints_group = [[2, 3], [5, 6], [1, 4], [0, 7], [12, 13], [9, 10], [8, 11]]

NAMES_SURREAL = ['']*14
NAMES_SURREAL[0] = 'Hip'
NAMES_SURREAL[1] = 'RHip'
NAMES_SURREAL[2] = 'RKnee'
NAMES_SURREAL[3] = 'RAnkle'
NAMES_SURREAL[4] = 'LHip'
NAMES_SURREAL[5] = 'LKnee'
NAMES_SURREAL[6] = 'LAnkle'
# NAMES_SURREAL[7] = 'Spine2'
NAMES_SURREAL[7] = 'Neck'
# NAMES_SURREAL[8] = 'Head'
NAMES_SURREAL[8] = 'LUpperArm'
NAMES_SURREAL[9] = 'LElbow'
NAMES_SURREAL[10] = 'LWrist'
NAMES_SURREAL[11] = 'RUpperArm'
NAMES_SURREAL[12] = 'RElbow'
NAMES_SURREAL[13] = 'RWrist'

# Human3.6m IDs for training and testing
TRAIN_SUBJECTS = ['S0']
TEST_SUBJECTS = ['S0']

class SURREALDataset(object):

    def __init__(self, path_train, path_valid):
        super(SURREALDataset, self).__init__()

        self.cameras = None

        self._data_train = { "2d": None, "3d": None }
        self._data_valid = { "2d": None, "3d": None }

        self.cameras = []

        self.load_data(path_train, path_valid)

    def load_data(self, path_train, path_valid):

        data_train = np.load(path_train, allow_pickle=True)
        data_valid = np.load(path_valid, allow_pickle=True)

        max_idx = data_train["data_3d"].shape[0]//6

        self._data_train['2d'] = data_train['data_2d'][max_idx:3*max_idx, :, :]
        self._data_train['3d'] = data_train['data_3d'][max_idx:3*max_idx, :, :]

        self._data_valid['2d'] = data_valid["data_2d"]
        self._data_valid['3d'] = data_valid["data_3d"]

        print("[PoseUtils] Loaded raw data")

    def get_2d_valid(self, jnts=14):

        to_select, to_sort = dataset_indices('surreal', jnts)

        return self._data_valid['2d'][:, to_select, :][:, to_sort, :]

    def get_3d_valid(self, jnts=14):

        to_select, to_sort = dataset_indices('surreal', jnts)

        return self._data_valid['3d'][:, to_select, :][:, to_sort, :]
    
    def get_2d_train(self, jnts=14):

        to_select, to_sort = dataset_indices('surreal', jnts)

        return self._data_train['2d'][:, to_select, :][:, to_sort, :]

    def get_3d_train(self, jnts=14):

        to_select, to_sort = dataset_indices('surreal', jnts)

        return self._data_train['3d'][:, to_select, :][:, to_sort, :]