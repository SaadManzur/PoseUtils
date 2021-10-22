import numpy as np

from poseutils.logger import log
from poseutils.transform import unnormalize_zscore
from poseutils.datasets.transformation.Transformation import Transformation

class Unnormalize(Transformation):

    def __init__(self, skip_root=True):
        super(Unnormalize, self).__init__()

        self.skip_root = skip_root

    def __call__(self, X, mean, std, **kwds):

        log("Unnormalizing")

        return unnormalize_zscore(X, mean, std, self.skip_root)