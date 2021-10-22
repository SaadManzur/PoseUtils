from __future__ import print_function, absolute_import, division

import numpy as np
from poseutils.datasets.unprocessed.Dataset import Dataset
from poseutils.datasets.transformation.CalculateMetrics import CalculateMetrics

class TransformedDataset(object):

    def __init__(self, dataset=None, njnts=14):
        super(TransformedDataset, self).__init__()

        self._data_train = { "2d": None, "3d": None }
        self._data_valid = { "2d": None, "3d": None }

        if dataset is not None:
            self.set_train(dataset.get_2d_train(njnts), dataset.get_3d_train(njnts))
            self.set_valid(dataset.get_2d_valid(njnts), dataset.get_3d_valid(njnts))

        self.means = {"2d": 0, "3d": 0}
        self.stds = {"2d": 0, "3d": 0}

    def set_train(self, d2d, d3d):

        self._data_train["2d"] = d2d
        self._data_train["3d"] = d3d

    def set_valid(self, d2d, d3d):

        self._data_valid["2d"] = d2d
        self._data_valid["3d"] = d3d

    def calculate_metrics(self):

        d2d, d3d = self._data_train['2d'], self._data_train['3d']

        self.means["2d"], self.stds["2d"] = np.mean(d2d, axis=0), np.std(d2d, axis=0)
        self.means["3d"], self.stds["3d"] = np.mean(d3d, axis=0), np.std(d3d, axis=0)

    def apply2d(self, transformations):

        for transformation in transformations:

            if isinstance(transformation, CalculateMetrics):
                self.calculate_metrics()
            else:
                self._data_train["2d"] = transformation(self._data_train["2d"], mean=self.means["2d"], std=self.stds["2d"])
                self._data_valid["2d"] = transformation(self._data_valid["2d"], mean=self.means["2d"], std=self.stds["2d"])

    def apply3d(self, transformations):

        for transformation in transformations:

            if isinstance(transformation, CalculateMetrics):
                self.calculate_metrics()
            else:
                self._data_train["3d"] = transformation(self._data_train["3d"], mean=self.means["3d"], std=self.stds["3d"])
                self._data_valid["3d"] = transformation(self._data_valid["3d"], mean=self.means["3d"], std=self.stds["3d"])

    def get_2d_train(self):

        return self._data_train['2d']

    def get_2d_valid(self):
        
        return self._data_valid['2d']

    def get_3d_train(self):

        return self._data_train['3d']

    def get_3d_valid(self):

        return self._data_valid['3d']
