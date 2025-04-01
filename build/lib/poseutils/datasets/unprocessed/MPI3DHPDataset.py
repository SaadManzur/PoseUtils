
import numpy as np
from poseutils.constants import dataset_indices

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

    def __init__(self, path):
        super(MPI3DHPDataset, self).__init__()

        self._data_train = { "2d": None, "3d": None }
        self._data_valid = { "2d": None, "3d": None }

        self.load_data(path)

    def load_data(self, path):

        data = np.load(path, allow_pickle=True, encoding='latin1')

        self.split_train_test(data['data'].item())

        print("[PoseUtils] Loaded raw data")

    def split_train_test(self, data):

        data_3d = []
        data_2d = []

        camera_set = [0, 2, 4, 7, 8]

        for subj in TRAIN_SUBJECTS:
            for seq in range(2):
                stacked_3d = np.vstack([data[(subj, seq+1)]['3dc'][i] for i in camera_set])
                stacked_2d = np.vstack([data[(subj, seq+1)]['2d'][i] for i in camera_set])
                data_3d.append(stacked_3d)
                data_2d.append(stacked_2d)

        self._data_train['3d'] = np.vstack(data_3d)
        self._data_train['2d'] = np.vstack(data_2d)

        data_3d = []
        data_2d = []

        for subj in TEST_SUBJECTS:
            for seq in range(2):
                stacked_3d = np.vstack([data[(subj, seq+1)]['3dc'][i] for i in camera_set])
                stacked_2d = np.vstack([data[(subj, seq+1)]['2d'][i] for i in camera_set])
                data_3d.append(stacked_3d)
                data_2d.append(stacked_2d)

        self._data_valid['3d'] = np.vstack(data_3d)
        self._data_valid['2d'] = np.vstack(data_2d)


    def get_2d_valid(self, jnts=14):

        to_select, to_sort = dataset_indices('3dhp', jnts)

        return self._data_valid['2d'][:, to_select, :][:, to_sort, :]

    def get_3d_valid(self, jnts=14):

        to_select, to_sort = dataset_indices('3dhp', jnts)

        return self._data_valid['3d'][:, to_select, :][:, to_sort, :]
    
    def get_2d_train(self, jnts=14):

        to_select, to_sort = dataset_indices('3dhp', jnts)

        return self._data_train['2d'][:, to_select, :][:, to_sort, :]

    def get_3d_train(self, jnts=14):

        to_select, to_sort = dataset_indices('3dhp', jnts)

        return self._data_train['3d'][:, to_select, :][:, to_sort, :]
