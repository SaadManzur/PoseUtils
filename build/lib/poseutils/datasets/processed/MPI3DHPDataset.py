import copy
import numpy as np
from numpy.core.defchararray import center
from tqdm import tqdm
from poseutils.transform import normalize_zscore
from poseutils.view import draw_skeleton
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from poseutils.props import get_body_centered_axes
from poseutils.transform import normalize_skeleton

parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14]
joints_left = [4, 5, 6, 10, 11, 12]
joints_right = [1, 2, 3, 13, 14, 15]
                        
skeleton_MPI3DHP_joints_group = [[2, 3], [5, 6], [1, 4], [0, 7], [12, 13], [9, 10], [8, 11]]

NAMES_MPI3DHP = ['']*16
NAMES_MPI3DHP[0] = 'Hip'
NAMES_MPI3DHP[1] = 'RHip'
NAMES_MPI3DHP[2] = 'RKnee'
NAMES_MPI3DHP[3] = 'RAnkle'
NAMES_MPI3DHP[4] = 'LHip'
NAMES_MPI3DHP[5] = 'LKnee'
NAMES_MPI3DHP[6] = 'LAnkle'
NAMES_MPI3DHP[7] = 'Spine2'
NAMES_MPI3DHP[8] = 'Neck'
NAMES_MPI3DHP[9] = 'Head'
NAMES_MPI3DHP[10] = 'LUpperArm'
NAMES_MPI3DHP[11] = 'LElbow'
NAMES_MPI3DHP[12] = 'LWrist'
NAMES_MPI3DHP[13] = 'RUpperArm'
NAMES_MPI3DHP[14] = 'RElbow'
NAMES_MPI3DHP[15] = 'RWrist'

TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6]
TEST_SUBJECTS = [7, 8]

class MPI3DHPDataset(object):

    def __init__(self, path, center_2d=False, norm_pts=-1, skel_norm=False):
        super(MPI3DHPDataset, self).__init__()

        self._data_train = {"2d": np.zeros((0, 14, 2), dtype=np.float32), "3d": np.zeros((0, 14, 3), dtype=np.float32), "axes": []}
        self._data_valid = {"2d": np.zeros((0, 14, 2), dtype=np.float32), "3d": np.zeros((0, 14, 3), dtype=np.float32), "axes": []}

        self.mean_2d = 0.0
        self.std_2d = 0.0
        self.mean_3d = 0.0
        self.std_3d = 0.0

        self.center_2d = center_2d

        self.skel_norm = skel_norm

        self.load_data(path, norm_pts)

    def load_data(self, path, norm_pts=-1):

        data = np.load(path, allow_pickle=True, encoding='latin1')

        self.split_train_test(data['data'].item(), data['idx_14'])

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

        self.mean_3d = np.mean(self._data_train['3d'], axis=0)
        self.std_3d = np.std(self._data_train['3d'], axis=0)

        self._data_train['3d'] = normalize_zscore(self._data_train['3d'], self.mean_3d, self.std_3d, skip_root=True)        
        self._data_valid['3d'] = normalize_zscore(self._data_valid['3d'], self.mean_3d, self.std_3d, skip_root=True)

        if not self.skel_norm:
            self.mean_2d = np.mean(self._data_train['2d'], axis=0)
            self.std_2d = np.std(self._data_train['2d'], axis=0)
            self._data_train['2d'] = normalize_zscore(self._data_train['2d'], self.mean_2d, self.std_2d, skip_root=self.center_2d)
            self._data_valid['2d'] = normalize_zscore(self._data_valid['2d'], self.mean_2d, self.std_2d, skip_root=self.center_2d)

    def split_train_test(self, data, idx):

        data_3d = []
        data_2d = []

        camera_set = [0, 2, 4, 7, 8]

        for subj in TRAIN_SUBJECTS:
            for seq in range(2):
                stacked_3d = np.vstack([data[(subj, seq+1)]['3dc'][i] for i in camera_set])
                stacked_2d = np.vstack([data[(subj, seq+1)]['2d'][i] for i in camera_set])
                data_3d.append(stacked_3d[:, idx, :])
                data_2d.append(stacked_2d[:, idx, :])

        self._data_train['3d'] = np.vstack(data_3d)
        self._data_train['2d'] = np.vstack(data_2d)

        data_3d = []
        data_2d = []

        for subj in TEST_SUBJECTS:
            for seq in range(2):
                stacked_3d = np.vstack([data[(subj, seq+1)]['3dc'][i] for i in camera_set])
                stacked_2d = np.vstack([data[(subj, seq+1)]['2d'][i] for i in camera_set])
                data_3d.append(stacked_3d[:, idx, :])
                data_2d.append(stacked_2d[:, idx, :])

        self._data_valid['3d'] = np.vstack(data_3d)
        self._data_valid['2d'] = np.vstack(data_2d)

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
        return skeleton_MPI3DHP_joints_group

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