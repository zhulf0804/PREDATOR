import numpy as np


def inlier_ratio_core(points_src, points_tgt, feats_src, feats_tgt, transf, inlier_threshold=0.1):

    R, t = transf[:3, :3], transf[:3, 3]
    dists = np.matmul(feats_src, feats_tgt.T) # (n, m)
    row_max_inds = np.argmax(dists, axis=1)
    col_max_inds = np.argmax(dists, axis=0)

    points_src = points_src @ R.T + t
    inlier_mask = np.sum((points_src - points_tgt[row_max_inds]) ** 2, axis=1) < inlier_threshold ** 2
    inlier_ratio = np.sum(inlier_mask) / len(inlier_mask)

    # mutual inlier ratio
    mutual_corrs = []
    for i in range(len(points_src)):
        if col_max_inds[row_max_inds[i]] == i:
            mutual_corrs.append([i, row_max_inds[i]])
    mutual_corrs = np.array(mutual_corrs, dtype=np.int)
    mutual_mask = np.sum((points_src[mutual_corrs[:, 0]] - points_tgt[mutual_corrs[:, 1]]) ** 2, axis=1) < inlier_threshold ** 2
    mutual_inlier_ratio = np.sum(mutual_mask) / len(mutual_corrs)

    return inlier_ratio, mutual_inlier_ratio


def registration_recall_core(points_src, points_tgt, gt_corrs, pred_T):
    '''

    :param points_src: (n, 3)
    :param points_tgt: (m, 3)
    :param gt_corrs: (n1, 2)
    :param pred_T: (4, 4)
    :return: float
    '''
    points_src = points_src[gt_corrs[:, 0]]
    points_tgt = points_tgt[gt_corrs[:, 1]]
    R, t = pred_T[:3, :3], pred_T[:3, 3]
    points_src = points_src @ R.T + t
    mse = np.mean(np.sum((points_src - points_tgt) ** 2, axis=1))
    rmse = np.sqrt(mse)

    return rmse
