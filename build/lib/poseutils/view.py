from poseutils.constants import *

def draw_skeleton(pose, ax, jnts_14=True):

    assert len(pose.shape) == 2
    assert pose.shape[-1] == 3 or pose.shape[-1] == 2 

    if jnts_14:
        edges = EDGES_14
        lefts = LEFTS_14
        rights = RIGHTS_14

    is_3d = False
    if pose.shape[-1] == 3:
        is_3d = True

    col_right = 'b'
    col_left = 'r'

    if is_3d:
        ax.scatter(pose[:, 0], pose[:, 1], zs=pose[:, 2], color='k')
    else:
        ax.scatter(pose[:, 0], pose[:, 1], color='k')

    for u, v in edges:
        col_to_use = 'k'

        if u in lefts and v in lefts:
            col_to_use = col_left
        elif u in rights and v in rights:
            col_to_use = col_right

        if is_3d:
            ax.plot([pose[u, 0], pose[v, 0]], [pose[u, 1], pose[v, 1]], zs=[pose[u, 2], pose[v, 2]], color=col_to_use)
        else:
            ax.plot([pose[u, 0], pose[v, 0]], [pose[u, 1], pose[v, 1]], color=col_to_use)