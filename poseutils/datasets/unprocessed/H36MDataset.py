from __future__ import print_function, absolute_import, division

import os
import h5py
import glob
import numpy as np
import poseutils.camera_utils as cameras
from poseutils.constants import dataset_indices

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

    def __init__(self, path):
        super(H36MDataset, self).__init__()

        self.cameras = None

        self._data_train = { "2d": None, "3d": None }
        self._data_valid = { "2d": None, "3d": None }

        self.cameras = []

        self.load_data(path)

    def load_data(self, path):

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

        self._data_train['2d'] = d2d_train
        self._data_train['3d'] = d3d_train
        self._data_valid['2d'] = d2d_valid
        self._data_valid['3d'] = d3d_valid

        print("[PoseUtils] Loaded raw data")

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

        return data

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

        return t2d, t2dc, t3d

    def get_2d_valid(self, jnts=14):

        to_select, to_sort = dataset_indices('h36m', jnts)

        return self._data_valid['2d'][:, to_select, :][:, to_sort, :]

    def get_3d_valid(self, jnts=14):

        to_select, to_sort = dataset_indices('h36m', jnts)

        return self._data_valid['3d'][:, to_select, :][:, to_sort, :]
    
    def get_2d_train(self, jnts=14):

        to_select, to_sort = dataset_indices('h36m', jnts)

        return self._data_train['2d'][:, to_select, :][:, to_sort, :]

    def get_3d_train(self, jnts=14):

        to_select, to_sort = dataset_indices('h36m', jnts)

        return self._data_train['3d'][:, to_select, :][:, to_sort, :]
