from poseutils.logger import log
from poseutils.transform import root_center
from poseutils.datasets.transformation.Transformation import Transformation

class RootCenter(Transformation):

    def __init__(self, root_idx=0):
        super(RootCenter, self).__init__()

        self.root_idx = root_idx

    def __call__(self, X, **kwds):

        log("Applying root centering")
        
        return root_center(X, self.root_idx)