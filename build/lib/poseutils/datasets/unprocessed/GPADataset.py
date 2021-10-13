from __future__ import print_function, absolute_import, division

import numpy as np
from poseutils.constants import dataset_indices

parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14]
joints_left = [4, 5, 6, 10, 11, 12]
joints_right = [1, 2, 3, 13, 14, 15]
                        
skeleton_GPA_joints_group = [[2, 3], [5, 6], [1, 4], [0, 7], [12, 13], [9, 10], [8, 11]]

NAMES_GPA = ['']*14
NAMES_GPA[0] = 'Hip'
NAMES_GPA[1] = 'RHip'
NAMES_GPA[2] = 'RKnee'
NAMES_GPA[3] = 'RAnkle'
NAMES_GPA[4] = 'LHip'
NAMES_GPA[5] = 'LKnee'
NAMES_GPA[6] = 'LAnkle'
# NAMES_GPA[7] = 'Spine2'
NAMES_GPA[7] = 'Neck'
# NAMES_GPA[8] = 'Head'
NAMES_GPA[8] = 'LUpperArm'
NAMES_GPA[9] = 'LElbow'
NAMES_GPA[10] = 'LWrist'
NAMES_GPA[11] = 'RUpperArm'
NAMES_GPA[12] = 'RElbow'
NAMES_GPA[13] = 'RWrist'

# Human3.6m IDs for training and testing
TRAIN_SUBJECTS = ['S0']
TEST_SUBJECTS = ['S0']

class GPADataset(object):

    def __init__(self, path):
        super(GPADataset, self).__init__()

        # TODO: Update camera later if needed
        self.cameras = None

        self._data_train = { "2d": None, "3d": None }
        self._data_valid = { "2d": None, "3d": None }

        self.cameras = []

        self.load_data(path)

    def load_data(self, path):

        data = np.load(path, allow_pickle=True, encoding='latin1')['data'].item()

        self._data_train["2d"] = data["train"]["2d"]
        self._data_train["3d"] = data["train"]["3d"]
        self._data_valid["3d"] = data["test"]["3d"]
        self._data_valid["2d"] = data["test"]["2d"]

        print("[PoseUtils] Loaded raw data")

    def get_2d_valid(self, jnts=14):

        to_select, to_sort = dataset_indices('gpa', jnts)

        return self._data_valid['2d'][:, to_select, :][:, to_sort, :]

    def get_3d_valid(self, jnts=14):

        to_select, to_sort = dataset_indices('gpa', jnts)

        return self._data_valid['3d'][:, to_select, :][:, to_sort, :]
    
    def get_2d_train(self, jnts=14):

        to_select, to_sort = dataset_indices('gpa', jnts)

        return self._data_train['2d'][:, to_select, :][:, to_sort, :]

    def get_3d_train(self, jnts=14):

        to_select, to_sort = dataset_indices('gpa', jnts)

        return self._data_train['3d'][:, to_select, :][:, to_sort, :]
