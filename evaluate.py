import copy
import glob
import numpy as np
import os
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

from data import ThreeDMatch, get_dataloader
from models import architectures, PREDATOR
from utils import decode_config, npy2pcd, execute_global_registration, \
    execute_global_registration013, npy2feat, vis_plys, setup_seed
from metrics import inlier_ratio_core, registration_recall_core

CUR = os.path.dirname(os.path.abspath(__file__))


def get_scene_split(file_path):
    test_cats = ['7-scenes-redkitchen',
                 'sun3d-home_at-home_at_scan1_2013_jan_1',
                 'sun3d-home_md-home_md_scan9_2012_sep_30',
                 'sun3d-hotel_uc-scan3',
                 'sun3d-hotel_umd-maryland_hotel1',
                 'sun3d-hotel_umd-maryland_hotel3',
                 'sun3d-mit_76_studyroom-76-1studyroom2',
                 'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika']
    c = 0
    splits, ply_coors_ids = [], []
    for cat in test_cats:
        with open(os.path.join(file_path, cat, 'gt.log'), 'r') as f:
            lines = f.readlines()
            stride = len(lines) // 5
            for line in lines[::5]:
                item = list(map(int, line.strip().split('\t')))
                ply_coors_ids.append(item)
        splits.append([c, c + stride])
        c += stride
    return splits, np.array(ply_coors_ids, dtype=np.int)


def main():
    setup_seed(22)
    config = decode_config(os.path.join(CUR, 'configs', 'threedmatch.yaml'))
    config = edict(config)
    config.architecture = architectures[config.dataset]
    config.num_workers = 4
    test_dataset = ThreeDMatch(root=config.root,
                               split='test',
                               aug=False,
                               overlap_radius=config.overlap_radius)

    test_dataloader, neighborhood_limits = get_dataloader(config=config,
                                                          dataset=test_dataset,
                                                          batch_size=config.batch_size,
                                                          num_workers=config.num_workers,
                                                          shuffle=False,
                                                          neighborhood_limits=None)

    print(neighborhood_limits)
    model = PREDATOR(config)
    if config.cuda:
        model = model.cuda()
        model.load_state_dict(torch.load(config.checkpoint))
    else:
        model.load_state_dict(
            torch.load(config.checkpoint, map_location=torch.device('cpu')))
    model.eval()

    fmr_threshold = 0.05
    rmse_threshold = 0.2
    inlier_ratios, mutual_inlier_ratios = [], []
    mutual_feature_match_recalls, feature_match_recalls = [], []
    rmses = []
    with torch.no_grad():
        for inputs in tqdm(test_dataloader):
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
            source_npy_raw = copy.deepcopy(source_npy)
            target_npy_raw = copy.deepcopy(target_npy)
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

            inlier_ratio, mutual_inlier_ratio = inlier_ratio_core(points_src=source_npy,
                                                                  points_tgt=target_npy,
                                                                  feats_src=source_feats_npy,
                                                                  feats_tgt=target_feats_npy,
                                                                  transf=transf.detach().cpu().numpy())

            inlier_ratios.append(inlier_ratio)
            mutual_inlier_ratios.append(mutual_inlier_ratio)
            feature_match_recalls.append(inlier_ratio > fmr_threshold)
            mutual_feature_match_recalls.append(mutual_inlier_ratio > fmr_threshold)

            source, target = npy2pcd(source_npy), npy2pcd(target_npy)
            source_feats, target_feats = npy2feat(source_feats_npy), npy2feat(target_feats_npy)
            # pred_T, estimate = execute_global_registration(source=source,
            #                                                target=target,
            #                                                source_feats=source_feats,
            #                                                target_feats=target_feats,
            #                                                voxel_size=config.first_subsampling_dl)

            pred_T, estimate = execute_global_registration013(source=source,
                                                              target=target,
                                                              source_feats=source_feats,
                                                              target_feats=target_feats,
                                                              voxel_size=config.first_subsampling_dl)

            coors_filter = {}
            for i, j in coors.cpu().numpy():
                if i not in coors_filter:
                    coors_filter[i] = j
            coors_filter = np.array([[i, j] for i, j in coors_filter.items()])
            rmse = registration_recall_core(points_src=source_npy_raw,
                                            points_tgt=target_npy_raw,
                                            gt_corrs=coors_filter,
                                            pred_T=pred_T)
            rmses.append(rmse)
            # rmses.append(0.01)

            # estimate_gt = copy.deepcopy(source)
            # estimate_gt.transform(transf.detach().cpu().numpy())
            #
            # source.paint_uniform_color([0, 1, 0])
            # target.paint_uniform_color([1, 0, 0])
            # estimate.paint_uniform_color([0, 0, 1])
            # estimate_gt.paint_uniform_color([0, 0, 1])
            # vis_plys([source, target, estimate], need_color=False)
            # vis_plys([source, target, estimate_gt], need_color=False)

    file_path = os.path.join(CUR, 'data', 'gt')
    splits, ply_coors_ids = get_scene_split(file_path=file_path)
    valid_idx = np.abs(ply_coors_ids[:, 0] - ply_coors_ids[:, 1]) > 1
    n_valids = []
    cat_inlier_ratios, cat_mutual_inlier_ratios = [], []
    cat_mutual_feature_match_recalls, cat_feature_match_recalls = [], []
    cat_registration_recalls = []
    for split in splits:
        m_inlier_ratio = np.mean(inlier_ratios[split[0]:split[1]])
        m_mutual_inlier_ratio = np.mean(mutual_inlier_ratios[split[0]:split[1]])
        m_feature_match_recall = np.mean(feature_match_recalls[split[0]:split[1]])
        m_mutual_feature_match_recall = np.mean(mutual_feature_match_recalls[split[0]:split[1]])

        valid_idx_split = valid_idx[split[0]:split[1]]
        n_valids.append(np.sum(valid_idx_split))
        m_registration_recall = np.mean(np.array(rmses[split[0]:split[1]])[valid_idx_split] < rmse_threshold)
        cat_inlier_ratios.append(m_inlier_ratio)
        cat_mutual_inlier_ratios.append(m_mutual_inlier_ratio)
        cat_feature_match_recalls.append(m_feature_match_recall)
        cat_mutual_feature_match_recalls.append(m_mutual_feature_match_recall)
        cat_registration_recalls.append(m_registration_recall)

    weighted_reg_recall = n_valids * np.array(cat_registration_recalls) / np.sum(n_valids)
    print("Inlier ratio: ", np.mean(cat_inlier_ratios))
    print("Mutual inlier ratio: ", np.mean(cat_mutual_inlier_ratios))
    print("Feature match recall: ", np.mean(cat_feature_match_recalls))
    print("Mutual feature match recall: ", np.mean(cat_mutual_feature_match_recalls))

    print('valid pairs / total pairs: ', np.sum(n_valids), ' / ', len(valid_idx))
    print("Registration recall: ", np.mean(cat_registration_recalls))
    print("Weighted registration recall: ", np.sum(weighted_reg_recall))

 
if __name__ == '__main__':
    main()
