import numpy as np
from tqdm import tqdm

from poseutils.constants import *
from poseutils.common import calc_angle_360
from poseutils.common import normalize_a_to_b

def calculate_avg_limb_lengths(jnts_xd, cvt_mm=False):

    assert len(jnts_xd.shape) == 3
    assert jnts_xd.shape[-1] == 2 or jnts_xd.shape[-1] == 3
    assert jnts_xd.shape[1] == 14 or jnts_xd.shape[1] == 16

    if jnts_xd.shape[1] == 14:
        edge_names = EDGE_NAMES_14JNTS
        edges = EDGES_14
    else:
        edge_names = EDGE_NAMES_16JNTS
        edges = EDGES_16
    
    edge_lengths = []
    
    for i_pt in tqdm(range(jnts_xd.shape[0])):
        jnts = jnts_xd[i_pt, :, :]
        edge_length = [0.0]*len(edge_names)
        
        for i, (u, v) in enumerate(edges):
            edge_length[i] = np.linalg.norm(jnts[u]-jnts[v])
        
        edge_lengths.append(np.array(edge_length))
        
    edge_lengths = np.vstack(edge_lengths)
    
    if cvt_mm:
        edge_lengths = edge_lengths*1000
    
    return np.mean(edge_lengths, axis=0), np.std(edge_lengths, axis=0), edge_names

def calculate_camera_angles(data):

    assert len(data.shape) == 3
    assert data.shape[-1] == 3
    assert data.shape[1] == 14 or data.shape[1] == 16
    
    if data.shape[1] == 14:
        ls_idx = 8
        rs_idx = 11
    else:
        ls_idx = 10
        rs_idx = 13

    angles = []
    
    for i in tqdm(range(data.shape[0])):

        hip = data[i, 0, :]
        ls = data[i, ls_idx, :]
        rs = data[i, rs_idx, :]

        h_ls = normalize_a_to_b(hip, ls)
        h_rs = normalize_a_to_b(hip, rs)

        fwd = np.cross(h_ls, h_rs)
        fwd /= np.linalg.norm(fwd)
        
        up = normalize_a_to_b(hip, (ls + rs)/2.0)
        
        right = normalize_a_to_b(np.zeros(3), np.cross(fwd, up))
        
        up = normalize_a_to_b(np.zeros(3), np.cross(right, fwd))
        
        camera_pos = normalize_a_to_b(np.zeros(3), -hip)
        
        camera_elev = 90 - np.degrees(np.arccos(np.dot(up, camera_pos)))
        
        camera_pos_proj_up = np.dot(up, camera_pos)*up
        
        camera_pos_proj_gnd = normalize_a_to_b(camera_pos_proj_up, camera_pos) 
        
        camera_azim = calc_angle_360(fwd, camera_pos_proj_gnd, up)

        angles.append(np.array([camera_elev, camera_azim]))
        
    return np.array(angles)