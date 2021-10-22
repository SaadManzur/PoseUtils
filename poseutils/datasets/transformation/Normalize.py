import numpy as np

from poseutils.logger import log
from poseutils.transform import normalize_zscore
from poseutils.datasets.transformation.Transformation import Transformation

class Normalize(Transformation):

    def __init__(self, skip_root=True):
        super(Normalize, self).__init__()

        self.skip_root = skip_root

    def __call__(self, X, mean, std, **kwds):

        log("Normalizing")

        return normalize_zscore(X, mean, std, self.skip_root)