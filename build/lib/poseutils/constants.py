NAMES_14 = ['']*14
NAMES_14[0] = 'Hip'
NAMES_14[1] = 'RHip'
NAMES_14[2] = 'RKnee'
NAMES_14[3] = 'RAnkle'
NAMES_14[4] = 'LHip'
NAMES_14[5] = 'LKnee'
NAMES_14[6] = 'LAnkle'
NAMES_14[7] = 'Neck'
NAMES_14[8] = 'LUpperArm'
NAMES_14[9] = 'LElbow'
NAMES_14[10] = 'LWrist'
NAMES_14[11] = 'RUpperArm'
NAMES_14[12] = 'RElbow'
NAMES_14[13] = 'RWrist'

NAMES_16 = ['']*16
NAMES_16[0] = 'Hip'
NAMES_16[1] = 'RHip'
NAMES_16[2] = 'RKnee'
NAMES_16[3] = 'RAnkle'
NAMES_16[4] = 'LHip'
NAMES_16[5] = 'LKnee'
NAMES_16[6] = 'LAnkle'
NAMES_16[7] = 'Spine'
NAMES_16[8] = 'Neck'
NAMES_16[9] = 'Head'
NAMES_16[10] = 'LUpperArm'
NAMES_16[11] = 'LElbow'
NAMES_16[12] = 'LWrist'
NAMES_16[13] = 'RUpperArm'
NAMES_16[14] = 'RElbow'
NAMES_16[15] = 'RWrist'

EDGES_14 = [[0, 1], [0, 4], [0, 7], [1, 2], [2, 3], [4, 5], [5, 6], [7, 8], [8, 9], [9, 10], [7, 11], [11, 12], [12, 13]]
LEFTS_14 = [4, 5, 6, 8, 9, 10]
RIGHTS_14 = [1, 2, 3, 11, 12, 13]

EDGES_16 = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 10], [10, 11], [11, 12], [8, 13], [13, 14], [14, 15]]
LEFTS_16 = [4, 5, 6, 8, 9, 10]
RIGHTS_16 = [1, 2, 3, 11, 12, 13]

EDGE_NAMES_16JNTS = ['HipRhip',
              'RFemur', 'RTibia', 'HipLHip',
              'LFemur', 'LTibia', 'LowerSpine',
              'UpperSpine', 'NeckHead',
              'LShoulder', 'LHumerus', 'LRadioUlnar',
              'RShoulder', 'RHumerus', 'RRadioUlnar']

EDGE_NAMES_14JNTS = ['HipRhip',
              'RFemur', 'RTibia', 'HipLHip',
              'LFemur', 'LTibia', 'HipNeck',
              'LShoulder', 'LHumerus', 'LRadioUlnar',
              'RShoulder', 'RHumerus', 'RRadioUlnar']

def adjacency_list(n_jnts):

    if n_jnts == 14:
        edges = EDGES_14
    elif n_jnts == 16:
        edges = EDGES_16
    else:
        raise ValueError("Only 14 or 16 joint cofigs")

    adjacency = []

    for _ in range(n_jnts):
        adjacency.append([])

    for u, v in edges:
        adjacency[u].append(v)

    return adjacency

def dataset_indices(dataset_name, n_jnts):

    assert n_jnts == 14 or n_jnts == 16

    to_select = None
    to_sort = None

    if dataset_name == "h36m":

        if n_jnts == 14:
            to_select = [0, 1, 2, 3,  6, 7, 8, 13, 17, 18, 19, 25, 26, 27]
        else:
            to_select = [0, 1, 2, 3, 6, 7, 8, 12, 13, 15, 17, 18, 19, 25, 26, 27]
    
    elif dataset_name == "gpa":

        to_select = [0, 24, 25, 26, 29, 30, 31, 2, 5, 6, 7, 17, 18, 19, 9, 10, 11]

        if n_jnts == 14:    
            to_sort = [0, 1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 15, 16]
        else:
            to_sort = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]
        
    elif dataset_name == "3dpw":

        if n_jnts == 14:
            to_select = [0, 2, 5, 8, 1, 4, 7, 12, 16, 18, 20, 17, 19, 21]
        else:
            to_select = [0, 2, 5, 8, 1, 4, 7, 6, 12, 15, 16, 18, 20, 17, 19, 21]

    elif dataset_name == "surreal":

        to_select = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]

        if n_jnts == 14:
            to_sort = [0, 2, 5, 8, 1, 4, 7, 12, 16, 18, 20, 17, 19, 21]
        else:
            to_sort = [0, 2, 5, 8, 1, 4, 7, 6, 12, 15, 16, 18, 20, 17, 19, 21]

    elif dataset_name == "3dhp":
        
        if n_jnts == 14:
            to_select = [4, 23, 24, 25, 18, 19, 20, 5, 9, 10, 11, 14, 15, 16]
        else:
            to_select = [4, 23, 24, 25, 18, 19, 20, 3, 5, 6, 9, 10, 11, 14, 15, 16]

    else:
        raise ValueError("Supports: h36m, gpa, 3dpw, surreal, 3dhp")

    return to_select, to_sort