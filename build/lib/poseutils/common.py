import numpy as np

def normalize_vec(vec):
    return vec / np.linalg.norm(vec)

def normalize_a_to_b(a, b):
    vec = b - a
    return vec / np.linalg.norm(vec)

def calc_angle_360(v1, v2, n):
    
    x = np.cross(v1, v2)
    c = np.sign(np.dot(x, n)) * np.linalg.norm(x)
    a = np.arctan2(c, np.dot(v1, v2))
    
    return np.degrees(a)