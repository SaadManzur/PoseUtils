from __future__ import print_function, absolute_import, division

import numpy as np
from poseutils.constants import dataset_indices

class Dataset(object):

    def __init__(self):
        super(Dataset, self).__init__()

        self.cameras = None

        self._data_train = { "2d": None, "3d": None, "raw": None }
        self._data_valid = { "2d": None, "3d": None, "raw": None }

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