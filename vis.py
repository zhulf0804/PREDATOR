import copy
import numpy as np
import os
import torch
from easydict import EasyDict as edict

from data import ThreeDMatch, get_dataloader
from models import architectures, PREDATOR
from utils import decode_config, npy2pcd, execute_global_registration, \
    npy2feat, vis_plys

CUR = os.path.dirname(os.path.abspath(__file__))


def main():
    config = decode_config(os.path.join(CUR, 'configs', 'threedmatch.yaml'))
    config = edict(config)
    config.architecture = architectures[config.dataset]
    config.num_workers = 1

    val_dataset = ThreeDMatch(root=config.root,
                              split='test',
                              aug=False,
                              overlap_radius=config.overlap_radius)
    val_dataloader, neighborhood_limits = get_dataloader(config=config,
                                                         dataset=val_dataset,
                                                         batch_size=config.batch_size,
                                                         num_workers=config.num_workers,
                                                         shuffle=False,
                                                         neighborhood_limits=[38, 36, 35, 38])

    print(neighborhood_limits)
    model = PREDATOR(config)
    if config.cuda:
        model = model.cuda()
        model.load_state_dict(torch.load(config.checkpoint))
    else:
        model.load_state_dict(
            torch.load(config.checkpoint, map_location=torch.device('cpu')))
    model.eval()

    for step, inputs in enumerate(val_dataloader):
        for k, v in inputs.items():
            if isinstance(v, list):
                for i in range(len(v)):
                    inputs[k][i] = inputs[k][i].cuda()
            else:
                inputs[k] = inputs[k].cuda()

        batched_feats = model(inputs)
        stack_points = inputs['points']
        stack_lengths = inputs['stacked_lengths']
        coords_src = stack_points[0][:stack_lengths[0][0]]
        coords_tgt = stack_points[0][stack_lengths[0][0]:]
        feats_src = batched_feats[:stack_lengths[0][0]]
        feats_tgt = batched_feats[stack_lengths[0][0]:]
        coors = inputs['coors'][0] # list, [coors1, coors2, ..], preparation for batchsize > 1
        transf = inputs['transf'][0] # (1, 4, 4), preparation for batchsize > 1

        source_npy = coords_src.detach().cpu().numpy()
        target_npy = coords_tgt.detach().cpu().numpy()
        source_npy_raw, target_npy_raw = copy.deepcopy(source_npy), copy.deepcopy(target_npy)
        source_feats_npy = feats_src[:, :-2].detach().cpu().numpy()
        target_feats_npy = feats_tgt[:, :-2].detach().cpu().numpy()

        source_overlap_scores = feats_src[:, -2].detach().cpu().numpy()
        target_overlap_scores = feats_tgt[:, -2].detach().cpu().numpy()
        source_saliency_scores = feats_src[:, -1].detach().cpu().numpy()
        target_saliency_scores = feats_tgt[:, -1].detach().cpu().numpy()

        source_scores = source_overlap_scores * source_saliency_scores
        target_scores = target_overlap_scores * target_saliency_scores

        npoints = 5000
        if source_npy.shape[0] > npoints:
            p = source_scores / np.sum(source_scores)
            idx = np.random.choice(len(source_npy), size=npoints, replace=False, p=p)
            source_npy = source_npy[idx]
            source_feats_npy = source_feats_npy[idx]
        if target_npy.shape[0] > npoints:
            p = target_scores / np.sum(target_scores)
            idx = np.random.choice(len(target_npy), size=npoints, replace=False, p=p)
            target_npy = target_npy[idx]
            target_feats_npy = target_feats_npy[idx]

        source, target = npy2pcd(source_npy), npy2pcd(target_npy)
        source_feats, target_feats = npy2feat(source_feats_npy), npy2feat(target_feats_npy)
        pred_T, estimate = execute_global_registration(source=source,
                                                       target=target,
                                                       source_feats=source_feats,
                                                       target_feats=target_feats,
                                                       voxel_size=config.first_subsampling_dl)

        source, target = npy2pcd(source_npy_raw), npy2pcd(target_npy_raw)
        estimate = copy.deepcopy(source)
        estimate.transform(pred_T)
        estimate_gt = copy.deepcopy(source)
        estimate_gt.transform(transf.detach().cpu().numpy())

        source.paint_uniform_color([0, 1, 0])
        target.paint_uniform_color([1, 0, 0])
        estimate.paint_uniform_color([0, 0, 1])
        estimate_gt.paint_uniform_color([0, 0, 1])
        vis_plys([source, target, estimate], need_color=False)
        # vis_plys([source, target, estimate_gt], need_color=False)


if __name__ == '__main__':
    main()
