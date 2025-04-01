from __future__ import print_function, absolute_import, division

import os
import h5py
import glob
import copy
import numpy as np
from tqdm import tqdm
import poseutils.camera_utils as cameras
from poseutils.view import draw_skeleton
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from poseutils.transform import normalize_zscore
from poseutils.props import get_body_centered_axes
from poseutils.transform import normalize_skeleton

parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14]
joints_left = [4, 5, 6, 10, 11, 12]
joints_right = [1, 2, 3, 13, 14, 15]
                        
skeleton_H36M_joints_group = [[2, 3], [5, 6], [1, 4], [0, 7], [12, 13], [9, 10], [8, 11]]

NAMES_H36M = ['']*14
NAMES_H36M[0] = 'Hip'
NAMES_H36M[1] = 'RHip'
NAMES_H36M[2] = 'RKnee'
NAMES_H36M[3] = 'RAnkle'
NAMES_H36M[4] = 'LHip'
NAMES_H36M[5] = 'LKnee'
NAMES_H36M[6] = 'LAnkle'
# NAMES_H36M[7] = 'Spine2'
NAMES_H36M[7] = 'Neck'
# NAMES_H36M[8] = 'Head'
NAMES_H36M[8] = 'LUpperArm'
NAMES_H36M[9] = 'LElbow'
NAMES_H36M[10] = 'LWrist'
NAMES_H36M[11] = 'RUpperArm'
NAMES_H36M[12] = 'RElbow'
NAMES_H36M[13] = 'RWrist'

# Human3.6m IDs for training and testing
TRAIN_SUBJECTS = ['S0']
TEST_SUBJECTS = ['S0']

class H36MDataset(object):

    def __init__(self, path, center_2d=False, load_metrics=None, skip_cams=[], skel_norm=False):
        # TODO: Update the fps here if needed
        super(H36MDataset, self).__init__()

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

        self.skip_cams = skip_cams

        self.load_data(path)

    def load_data(self, path, load_metrics=None):
        filename = os.path.splitext(os.path.basename(path))[0]

        indices_to_select_2d = [0, 1, 2, 3, 6, 7, 8, 13, 17, 18, 19, 25, 26, 27]

        self.cameras = cameras.load_cameras(os.path.join(path, "cameras.h5"))

        TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
        TEST_SUBJECTS  = [9, 11]

        actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

        trainset = self.load_3d_data(path, TRAIN_SUBJECTS, actions)
        testset = self.load_3d_data(path, TEST_SUBJECTS, actions)

        d2d_train, _, d3d_train = self.project_to_cameras(trainset)
        d2d_valid, _, d3d_valid = self.project_to_cameras(testset)

        if self.center_2d:
            self._data_train['2d'] = self.root_center(np.array(d2d_train))[:, indices_to_select_2d, :]
            self._data_valid['2d'] = self.root_center(np.array(d2d_valid))[:, indices_to_select_2d, :]
        else:
            self._data_train['2d'] = np.array(d2d_train)[:, indices_to_select_2d, :]
            self._data_valid['2d'] = np.array(d2d_valid)[:, indices_to_select_2d, :]
        
        if self.skel_norm:
            print("==> Using skeleton normalization")
            self._data_train['2d'] = normalize_skeleton(self._data_train['2d'])
            self._data_valid['2d'] = normalize_skeleton(self._data_valid['2d'])

        self._data_train['3d'] = self.root_center(np.array(d3d_train))[:, indices_to_select_2d, :]
        self._data_valid['3d'] = self.root_center(np.array(d3d_valid))[:, indices_to_select_2d, :]

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

    def load_3d_data(self, path, subjects, actions):

        data = {}

        total_data_points = 0
        for subj in subjects:
            for action in actions:
                # print('Reading subject {0}, action {1}'.format(subj, action))

                dpath = os.path.join( path, 'S{0}'.format(subj), 'MyPoses/3D_positions', '{0}*.h5'.format(action) )

                fnames = glob.glob( dpath )

                loaded_seqs = 0

                for fname in fnames:

                    seqname = os.path.basename( fname )

                    if action == "Sitting" and seqname.startswith( "SittingDown" ):
                        continue

                    if seqname.startswith( action ):
                        loaded_seqs = loaded_seqs + 1

                        with h5py.File( fname, 'r' ) as h5f:
                            poses = h5f['3D_positions'][:]
                            poses = poses.T

                            data[( subj, action, seqname )] = poses.reshape((-1, 32, 3))

                            total_data_points += poses.shape[0]

        print("Total 3d points loaded: ", total_data_points)

        return data

    def root_center(self, data3d):

        return data3d - data3d[:, :1, :]

    def normalization_stats_3d(self, pose_set_3d):

        for key in pose_set_3d.keys():
            poses = pose_set_3d[key]
            
            for i in range(poses.shape[0]):
                poses[i, :, :] -= poses[i, 0, :]

            pose_set_3d[key] = poses

        complete_data = np.vstack(pose_set_3d.values())

        return np.mean(complete_data, axis=0), np.std(complete_data, axis=0)

    def project_to_cameras( self, poses_set):

        t2d = []
        t2dc = []
        t3d = []

        total_points = 0
        for key in poses_set.keys():
            (subj, action, sqename) = key
            t3dw = poses_set[key]

            for cam in range(4):
                R, T, f, c, k, p, name = self.cameras[ (subj, cam+1) ]

                if int(name) in self.skip_cams:
                    continue

                t3dc = cameras.world_to_camera_frame( np.reshape(t3dw, [-1, 3]), R, T)
                pts2d, _, _, _, _ = cameras.project_point_radial( np.reshape(t3dw, [-1, 3]), R, T, f, c, k, p )
                cam2d = np.divide(pts2d - c.T, f.T)
                pts2d = np.reshape( pts2d, [-1, 32, 2] )
                total_points += pts2d.shape[0]
                t2d.append(pts2d)
                t3d.append(t3dc.reshape((-1, 32, 3)))
                t2dc.append(cam2d.reshape((-1, 32, 2)))

        t2d = np.vstack(t2d)
        t2dc = np.vstack(t2dc)
        t3d = np.vstack(t3d)

        print("Projected points: ", total_points)

        return t2d, t2dc, t3d

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
        return skeleton_H36M_joints_group

    def plot_random(self):
        idx = np.random.randint(0, high=self._data_train['3d'].shape[0])

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(121, projection='3d')
        bx = fig.add_subplot(122)
        draw_skeleton(self._data_train['3d'][idx, :, :]/1000, ax)
        draw_skeleton(self._data_train['2d'][idx, :, :], bx)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        ax.set_zlim((-1, 1))
        bx.set_xlim((-500, 500))
        bx.set_ylim((500, -500))
        plt.show()