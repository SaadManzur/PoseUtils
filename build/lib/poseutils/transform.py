import numpy as np
from poseutils.constants import *

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