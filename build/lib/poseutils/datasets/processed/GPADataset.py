from __future__ import print_function, absolute_import, division

import os
import h5py
import glob
import copy
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from poseutils.view import draw_skeleton
from poseutils.transform import normalize_zscore
from poseutils.props import get_body_centered_axes
from poseutils.transform import normalize_skeleton
import poseutils.camera_utils as cameras

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

    def __init__(self, path, center_2d=False, load_metrics=None, skel_norm=False):
        # TODO: Update the fps here if needed
        super(GPADataset, self).__init__()

        # TODO: Update camera later if needed
        self.cameras = None

        self._data_train = { "2d": np.zeros((0, 14, 2), dtype=np.float32), "3d": np.zeros((0, 14, 3), dtype=np.float32), "axes": [] }
        self._data_valid = { "2d": np.zeros((0, 14, 2), dtype=np.float32), "3d": np.zeros((0, 14, 3), dtype=np.float32), "axes": [] }

        self.mean_2d = 0.0
        self.std_2d = 0.0
        self.mean_3d = 0.0
        self.std_3d = 0.0

        self.center_2d = center_2d

        self.skel_norm = skel_norm

        self.cameras = []

        self.load_data(path)

    def load_data(self, path, load_metrics=None):
        filename = os.path.splitext(os.path.basename(path))[0]

        indices_to_select = [0, 24, 25, 26, 29, 30, 31, 2, 5, 6, 7, 17, 18, 19, 9, 10, 11]
        indices_to_sort = [0, 1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 15, 16]

        data = np.load(path, allow_pickle=True, encoding='latin1')['data'].item()

        data_train = data['train']
        data_valid = data['test']

        self._data_train['2d'] = data_train["2d"][:, indices_to_select,  :][:, indices_to_sort, :]
        self._data_train['3d'] = data_train["3d"][:, indices_to_select,  :][:, indices_to_sort, :]

        self._data_valid['2d'] = data_valid["2d"][:, indices_to_select,  :][:, indices_to_sort, :]
        self._data_valid['3d'] = data_valid["3d"][:, indices_to_select,  :][:, indices_to_sort, :]

        self._data_train['3d'] -= self._data_train['3d'][:, :1, :]
        self._data_valid['3d'] -= self._data_valid['3d'][:, :1, :]

        if self.center_2d:
            self._data_train['2d'] -= self._data_train['2d'][:, :1, :]
            self._data_valid['2d'] -= self._data_valid['2d'][:, :1, :]

        if self.skel_norm:
            self._data_train['2d'] = normalize_skeleton(self._data_train['2d'])
            self._data_valid['2d'] = normalize_skeleton(self._data_valid['2d'])

        _, _, _, self._data_train['axes'] = get_body_centered_axes(self._data_train['3d'])
        _, _, _, self._data_valid['axes'] = get_body_centered_axes(self._data_valid['3d'])

        # self.plot_random()

        self.mean_3d = np.mean(self._data_train['3d'], axis=0)
        self.std_3d = np.std(self._data_train['3d'], axis=0)
        

        self._data_train['3d'] = normalize_zscore(self._data_train['3d'], self.mean_3d, self.std_3d, skip_root=True)
        self._data_valid['3d'] = normalize_zscore(self._data_valid['3d'], self.mean_3d, self.std_3d, skip_root=True)

        if not self.skel_norm:
            self.mean_2d = np.mean(self._data_train['2d'], axis=0)
            self.std_2d = np.std(self._data_train['2d'], axis=0)
            self._data_train['2d'] = normalize_zscore(self._data_train['2d'], self.mean_2d, self.std_2d, skip_root=self.center_2d)
            self._data_valid['2d'] = normalize_zscore(self._data_valid['2d'], self.mean_2d, self.std_2d, skip_root=self.center_2d)

    def define_actions(self, action=None):
        all_actions = ["N"]

        if action is None:
            return all_actions

        if action not in all_actions:
            raise (ValueError, "Undefined action: {}".format(action))

        return [action]

    def get_2d_valid(self):
        return [self._data_valid['2d'].reshape((-1, 14, 2))]

    def get_3d_valid(self):
        return [self._data_valid['3d'].reshape((-1, 14, 3))]
    
    def get_2d_train(self):
        return [self._data_train['2d'].reshape((-1, 14, 2))]

    def get_3d_train(self):
        return [self._data_train['3d'].reshape((-1, 14, 3))]

    def get_axes_train(self):
        return [self._data_train['axes'][:, :, :2]]

    def get_axes_valid(self):
        return [self._data_valid['axes'][:, :, :2]]

    def get_joints_group(self):
        return skeleton_GPA_joints_group

    def plot_random(self):

        idx = np.random.randint(0, high=self._data_train['3d'].shape[0])

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(121, projection='3d')
        bx = fig.add_subplot(122)
        draw_skeleton(self._data_train['3d'][idx, :, :]/1000, ax, is_3d=True)
        draw_skeleton(self._data_train['2d'][idx, :, :], bx, is_3d=False)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        ax.set_zlim((-1, 1))
        bx.set_xlim((-960, 960))
        bx.set_ylim((960, -960))
        plt.show()