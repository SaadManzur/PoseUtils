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