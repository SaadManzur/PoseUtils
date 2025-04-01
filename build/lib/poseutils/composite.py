import numpy as np
from poseutils.props import get_bounding_box_2d
from poseutils.transform import scale_bounding_area_to

def scale_into_bounding_box_2d(joints, low=0, high=256):

    lx, ly, rx, ry = get_bounding_box_2d(joints)
    stacked_bbox = np.vstack((lx, ly, rx, ry)).T
    joints_scaled = scale_bounding_area_to(joints, stacked_bbox, low, high)

    return joints_scaled