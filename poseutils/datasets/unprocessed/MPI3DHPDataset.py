from __future__ import print_function, absolute_import, division

import numpy as np
from poseutils.logger import log
from poseutils.datasets.unprocessed.Dataset import Dataset

class MPI3DHPDataset(Dataset):

    def __init__(self, path):
        super(MPI3DHPDataset, self).__init__()

        self.load_data(path)

    def load_data(self, path):

        data = np.load(path, allow_pickle=True, encoding='latin1')

        self.split_train_test(data['data'].item())

        log("Loaded raw data")

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
