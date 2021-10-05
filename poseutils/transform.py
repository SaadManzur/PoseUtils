import numpy as np
from poseutils.constants import *

def normalize_zscore(X, mean, std, skip_root=False):

    for i in range(X.shape[0]):
        if not skip_root:
            X[i, :] = np.divide(X[i, :] - mean[:], std[:])
        else:
            X[i, 1:] = np.divide(X[i, 1:] - mean[1:], std[1:])
    
    return X

def unnormalize_zscore(X, mean, std, skip_root=False):

    XX = np.zeros(X.shape)

    for i in range(X.shape[0]):
        if not skip_root:
            XX[i, :] = np.multiply(X[i, :], std[:]) + mean[:]
        else:
            XX[i, 1:] = np.multiply(X[i, 1:], std[1:]) + mean[1:]

    return XX

def scale_bounding_area_to(X, bbox, low=0, high=256):

    assert len(X.shape) == 3
    assert X.shape[-1] == 2
    assert bbox.shape[-1] == 4

    half_max = (high - low)/2

    half_width = (bbox[:, 2] - bbox[:, 0])/2
    half_height = (bbox[:, 3] - bbox[:, 1])/2

    X_new = X - bbox[:, :2].reshape((-1, 1, 2))

    for i in range(X.shape[0]):
        
        scale_x = 1.0
        scale_y = 1.0

        offset_x = 0
        offset_y = 0

        if half_width[i] > half_height[i]:
            scale_x = half_max / half_width[i]
            scale_y = (half_height[i] / half_width[i]) * scale_x
            offset_y = half_max - half_height[i] * scale_y
        else:
            scale_y = half_max / half_height[i]
            scale_x = (half_width[i] / half_height[i]) * scale_y
            offset_x = half_max - half_width[i] * scale_x

        X_new[i, :, 0] = X_new[i, :, 0] * scale_x + offset_x
        X_new[i, :, 1] = X_new[i, :, 1] * scale_y + offset_y

    return X_new

def normalize_torso_2d(torso):

    #0: RH 1: LH 2: LS 3: RS

    assert len(torso.shape) == 3
    assert torso.shape[1] == 4 and torso.shape[-1] == 2

    torso_ = torso.copy()
    
    widths = [[], [], []]
    names = ["RH -> LH", "RH -> LS", "RH -> RS"]

    torso1_4u = torso_[:, 1, :] - torso_[:, 0, :]
    torso1_8u = torso_[:, 2, :] - torso_[:, 0, :]
    torso1_11u = torso_[:, 3, :] - torso_[:, 0, :]

    torso1_4l = np.linalg.norm(torso1_4u, axis=1).reshape(-1, 1)
    torso1_8l = np.linalg.norm(torso1_8u, axis=1).reshape(-1, 1)
    torso1_11l = np.linalg.norm(torso1_11u, axis=1).reshape(-1, 1)

    torso1_4u = torso1_4u / torso1_4l
    torso1_8u = torso1_8u / torso1_8l
    torso1_11u = torso1_11u / torso1_11l
    
    torso_[:, 0, :] = np.zeros((torso_.shape[0], 2))
    torso_[:, 1, :] = (torso1_4l / torso1_8l+1e-8)*torso1_4u
    torso_[:, 2, :] = torso1_8u
    torso_[:, 3, :] = (torso1_11l / torso1_8l+1e-8)*torso1_11u
    
    widths[0].append(torso1_4l)
    widths[1].append(torso1_8l)
    widths[2].append(torso1_11l)

    return torso_, np.array(widths), names

def normalize_skeleton(joints):

    assert len(joints.shape) == 3
    assert joints.shape[1] == 14 or joints.shape[1] == 16
    assert joints.shape[-1] == 2 or joints.shape[-1] == 3

    hip = 0

    if joints.shape[1] == 14:
        names = NAMES_14
    else:
        names = NAMES_16
    
    neck = names.index('Neck')

    joints_ = joints.copy()
    joints_ -= joints_[:, :1, :]

    spine = joints_[:, neck, :] - joints_[:, hip, :]
    spine_norm = np.linalg.norm(spine, axis=1).reshape(-1, 1)

    adjacency = adjacency_list(joints_.shape[1])

    queue = []

    queue.append(0)

    while len(queue) > 0:
        current = queue.pop(0)

        for child in adjacency[current]:
            queue.append(child)
            prnt_to_chld = joints[:, child, :] - joints[:, current, :]
            prnt_to_chld_norm = np.linalg.norm(prnt_to_chld, axis=1).reshape(-1, 1)
            prnt_to_chld_unit = prnt_to_chld / prnt_to_chld_norm
            joints_[:, child, :] = joints_[:, current, :] + (prnt_to_chld_unit * (prnt_to_chld_norm / (spine_norm + 1e-8)))

    return joints_