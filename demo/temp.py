import numpy as np
from poseutils.datasets.unprocessed import TDPWDataset
from poseutils.props import get_edm

dpath = "/home/saad/Personal/Research/Dataset/CrossDataset/3dpw_wo_invalid.npz"

dataset = TDPWDataset(dpath)

d3d_train = dataset.get_3d_train()

edm = get_edm(d3d_train[:1, :, :], True)

for i in range(14):
    for j in range(i, 14):

        dist = np.sqrt(np.sum((d3d_train[0, i, :] - d3d_train[0, j, :])**2))

        assert np.abs(dist - edm[0, i, j]) < 1e-5

print(edm[0, :3, :3])

edm = get_edm(d3d_train[:1, :, :])

print(edm.shape)