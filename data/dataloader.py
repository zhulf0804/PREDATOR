import numpy as np
import time
import torch
from functools import partial
# from utils import batch_grid_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling


def batch_neighbors(batch_queries, batch_supports, q_batches, s_batches, radius, max_nn):
    '''

    :param batch_queries: shape=(n, 3), n = n1 + n2 + ... + nk, k denotes batch size
    :param batch_supports: shape=(m, 3), m = m1 + m2 + ... + mk
    :param q_batches: shape=(k, ), values = (n1, n2, ..., nk)
    :param s_batches: shape=(k, ), values = (m1, m2, ..., mk)
    :param radius: float
    :return: shape=(n, max_nn)
    '''
    inds = cpp_neighbors.batch_query(batch_queries, batch_supports, q_batches, s_batches, radius=radius)
    if max_nn > 0:
        return torch.from_numpy(inds[:, :max_nn])
    return torch.from_numpy(inds)


def batch_grid_subsampling(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len)

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features)

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_labels)

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                              batches_len,
                                                                              features=features,
                                                                              classes=labels,
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features), torch.from_numpy(s_labels)


def collate_fn(list_data, config, neighborhood_limits):
    batched_points_list = []
    batched_feats_list = []
    batched_lengths_list = []
    batched_transf_list = []
    batched_coors_list = []
    for item in list_data:
        src_points, tgt_points = item['src_points'], item['tgt_points']
        src_feats, tgt_feats = item['src_feats'], item['tgt_feats']
        transf = item['transf']
        coors = item['coors']
        batched_points_list.append(src_points)
        batched_points_list.append(tgt_points)
        batched_feats_list.append(src_feats)
        batched_feats_list.append(tgt_feats)
        batched_lengths_list.append(len(src_points))
        batched_lengths_list.append(len(tgt_feats))
        batched_transf_list.append(transf)
        batched_coors_list.append(torch.from_numpy(coors))

    batched_points = torch.from_numpy(np.concatenate(batched_points_list, axis=0).astype(np.float32))
    batched_feats = torch.from_numpy(np.concatenate(batched_feats_list, axis=0).astype(np.float32))
    batched_lengths = torch.from_numpy(np.array(batched_lengths_list, dtype=np.int32))
    batched_transf = torch.from_numpy(np.array(batched_transf_list).astype(np.float32))
    batched_coors = batched_coors_list

    r_normal = config.first_subsampling_dl * config.conv_radius
    stack_points, stack_neighbors, stack_pools, stack_upsamples, stack_lengths = [], [], [], [], []
    layer = 0

    for block_i, block in enumerate(config.architecture):
        if 'upsample' in block:
            break
        flag_conv, flag_pool = False, False
        if 'strided' in block or 'upsample' in config.architecture[block_i + 1]:
            conv_i = batch_neighbors(batch_queries=batched_points,
                                     batch_supports=batched_points,
                                     q_batches=batched_lengths,
                                     s_batches=batched_lengths,
                                     radius=r_normal,
                                     max_nn=neighborhood_limits[layer])
            flag_conv = True

        if 'strided' in block:
            voxel_size = 2 * r_normal / config.conv_radius
            new_points, new_len = batch_grid_subsampling(points=batched_points,
                                                         batches_len=batched_lengths,
                                                         sampleDl=voxel_size)
            pool_i = batch_neighbors(batch_queries=new_points,
                                     batch_supports=batched_points,
                                     q_batches=new_len,
                                     s_batches=batched_lengths,
                                     radius=r_normal,
                                     max_nn=neighborhood_limits[layer])
            upsample_i = batch_neighbors(batch_queries=batched_points,
                                         batch_supports=new_points,
                                         q_batches=batched_lengths,
                                         s_batches=new_len,
                                         radius=2*r_normal,
                                         max_nn=neighborhood_limits[layer])
            flag_pool = True

        if flag_conv:
            stack_points.append(batched_points)
            stack_lengths.append(batched_lengths)
            stack_neighbors.append(conv_i.long())
        if flag_pool:
            stack_pools.append(pool_i.long())
            stack_upsamples.append(upsample_i.long())

            batched_points, batched_lengths = new_points, new_len
            r_normal *= 2
            layer += 1

    dict_inputs = {
        'points': stack_points,
        'neighbors': stack_neighbors,
        'pools': stack_pools,
        'upsamples': stack_upsamples,
        'stacked_lengths': stack_lengths,
        'feats': batched_feats,
        'coors': batched_coors,
        'transf': batched_transf,
    }
    return dict_inputs


def calibrate_neighbors(dataset, config, collate_fn, keep_ratio=0.8, samples_threshold=2000):
    # compute higher bound of neighbors number
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.deform_radius + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)

    for i in range(len(dataset)):
        dict_inputs = collate_fn(list_data=[dataset[i]],
                                 config=config,
                                 neighborhood_limits=[hist_n] * config.num_layers)

        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], axis=1).numpy() for neighb_mat in dict_inputs['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)

        # If we get enough points in every stack (downsampled layer), stop the loop.
        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists, axis=1)
    threshold = keep_ratio * cumsum[:, hist_n-1]
    percentiles = np.sum(cumsum < threshold[:, None], axis=1)
    return percentiles


def get_dataloader(config, dataset, batch_size, num_workers, shuffle, neighborhood_limits=None):
    if neighborhood_limits is None:
        neighborhood_limits = calibrate_neighbors(dataset=dataset,
                                                  config=config,
                                                  collate_fn=collate_fn)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_fn, config=config, neighborhood_limits=neighborhood_limits),
        drop_last=True,
    )

    return dataloader, neighborhood_limits
