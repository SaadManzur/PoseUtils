import numpy as np

def calculate_jpe(pred, target):

    assert pred.shape == target.shape
    assert pred.shape[1] == 14 or pred.shape[1] == 16

    target_ = target.copy()
    pred_ = pred.copy()

    target_ -= target_[:, :1, :]
    
    sqerr = (pred_ - target_)**2
    dists = np.zeros((sqerr.shape[0], pred.shape[1]))

    dist_idx = 1
    for k in range(1, pred.shape[1]):
        dists[:, dist_idx] = np.sqrt(np.sum(sqerr[:, k, :], axis=1))
        dist_idx += 1
    
    mpjpe = np.mean(dists)
    pjpe = np.mean(dists, axis=0)
    ppmjpe = np.mean(dists, axis=1)

    return mpjpe, pjpe, ppmjpe, dists