import numpy as np

from poseutils.logger import log
from poseutils.composite import scale_into_bounding_box_2d
from poseutils.datasets.transformation.Transformation import Transformation

class CropAndScale(Transformation):

    def __init__(self, low=0, high=256, *args, **kwds):
        super(CropAndScale, self).__init__(args, kwds)

        self.low = low
        self.high = high

    def __call__(self, X, **kwds):

        log("Applying crop and scale")

        return scale_into_bounding_box_2d(X, self.low, self.high)