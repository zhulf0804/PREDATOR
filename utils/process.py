import numpy as np
import torch


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def gather_points(points, inds):
    '''

    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]


def square_dists(points1, points2):
    '''
    Calculate square dists between two group points
    :param points1: shape=(B, N, C)
    :param points2: shape=(B, M, C)
    :return:
    '''
    B, N, C = points1.shape
    _, M, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, N, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, M)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.clamp(dists, min=1e-8)
    return dists.float()
