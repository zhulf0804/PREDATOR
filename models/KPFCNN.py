import numpy as np
import os
import torch
import torch.nn as nn

from models.KPConv import block_decider
from models import InformationInteractive


class PREDATOR(nn.Module):
    def __init__(self, config):
        super().__init__()
        r = config.first_subsampling_dl * config.conv_radius
        in_dim, out_dim = config.in_feats_dim, config.first_feats_dim
        K = config.num_kernel_points

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skips = [] # record the index of layers to be needed in decoder layer.
        self.encoder_skip_dims = [] # record the dims before pooling or strided-conv.
        block_i, layer_ind = 0, 0
        for block in config.architecture:
            if 'upsample' in block:
                break
            if np.any([skip_block in block for skip_block in ['strided', 'pool']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)
            self.encoder_blocks.append(block_decider(block_name=block,
                                                     radius=r,
                                                     in_dim=in_dim,
                                                     out_dim=out_dim,
                                                     use_bn=config.use_batch_norm,
                                                     bn_momentum=config.batch_norm_momentum,
                                                     layer_ind=layer_ind,
                                                     config=config))

            in_dim = out_dim // 2 if 'simple' in block else out_dim
            if np.any([skip_block in block for skip_block in ['strided', 'pool']]):
                r *= 2
                out_dim *= 2
                layer_ind += 1
            block_i += 1

        # bottleneck layer
        self.bottleneck = nn.Conv1d(out_dim, config.gnn_feats_dim, 1)

        # Information Interactive block
        self.info_interative = InformationInteractive(layer_names=config.nets,
                                                      feat_dims=config.gnn_feats_dim,
                                                      k=config.dgcnn_k,
                                                      nhead=config.num_head)
        self.pro_gnn = nn.Conv1d(config.gnn_feats_dim, config.gnn_feats_dim, 1)
        self.ol_score = nn.Conv1d(config.gnn_feats_dim, 1, 1)
        # self.epsilon = torch.tensor(-5.0) # how to set ?
        self.epsilon = nn.Parameter(torch.tensor(-5.0))  # how to set ?

        # Decoder blocks
        out_dim = config.gnn_feats_dim + 2
        self.decoder_blocks = nn.ModuleList()
        self.decoder_skips = []
        layer = len(self.encoder_skip_dims) - 1
        for block in config.architecture[block_i:]:
            if 'upsample' in block:
                layer_ind -= 1
                self.decoder_skips.append(block_i + 1)
            self.decoder_blocks.append(block_decider(block_name=block,
                                                     radius=r,
                                                     in_dim=in_dim,    # how to set for the first loop
                                                     out_dim=out_dim,
                                                     use_bn=config.use_batch_norm,
                                                     bn_momentum=config.batch_norm_momentum,
                                                     layer_ind=layer_ind,
                                                     config=config))
            in_dim = out_dim

            if 'upsample' in block:
                r *= 0.5
                in_dim += self.encoder_skip_dims[layer]
                layer -= 1
                out_dim = out_dim // 2
            block_i += 1

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        stack_points = inputs['points']
        # stack_neighbors = inputs['neighbors']
        # stack_pools = inputs['pools']
        # stack_upsamples = inputs['upsamples']
        stack_lengths = inputs['stacked_lengths']
        # batched_coors = inputs['coors']
        # batched_transf = inputs['transf']

        # 1. encoder
        batched_feats = inputs['feats']
        block_i = 0
        skip_feats = []
        for block in self.encoder_blocks:
            if block_i in self.encoder_skips:
                skip_feats.append(batched_feats)
            batched_feats = block(batched_feats, inputs)
            block_i += 1

        # 2.1 bottleneck layer
        batched_feats = self.bottleneck(batched_feats.transpose(0, 1).unsqueeze(0)) # (1, gnn_feats_dim, n)

        # 2.2 information interaction
        len_src, len_tgt = stack_lengths[-1][0], stack_lengths[-1][1]
        coords_src, coords_tgt = stack_points[-1][:len_src], stack_points[-1][len_src:]
        coords_src, coords_tgt = coords_src.transpose(0, 1).unsqueeze(0), \
                                 coords_tgt.transpose(0, 1).unsqueeze(0)
        feats_src, feats_tgt = batched_feats[:, :, :len_src], \
                               batched_feats[:, :, len_src:]
        feats_src, feats_tgt = self.info_interative(coords_src, feats_src, coords_tgt, feats_tgt)
        batched_feats = torch.cat([feats_src, feats_tgt], dim=-1)
        batched_feats = self.pro_gnn(batched_feats) # why this one ?

        # 2.3 overlap score
        ol_scores = self.ol_score(batched_feats).squeeze(0).transpose(0, 1) # (n, 1)

        # 2.4 saliency score
        temperature = torch.exp(self.epsilon) + 0.03
        batched_feats_norm = batched_feats / (torch.norm(batched_feats, dim=1, keepdim=True) + 1e-8)
        batched_feats_norm = batched_feats_norm.squeeze(0).transpose(0, 1) # (n, c)
        feats_norm_src, feats_norm_tgt = batched_feats_norm[:len_src], \
                                         batched_feats_norm[len_src:]
        inner_product = torch.matmul(feats_norm_src, feats_norm_tgt.transpose(0, 1)) # (n1, n2), n1 + n2
        ol_scores_src, ol_scores_tgt = ol_scores[:len_src], ol_scores[len_src:] # (n1, 1), (n2, 1)
        sali_scores_src = torch.matmul(torch.softmax(inner_product / temperature, dim=1), ol_scores_tgt) # (n1, 1)
        sali_scores_tgt = torch.matmul(torch.softmax(inner_product.transpose(0, 1) / temperature, dim=1), ol_scores_src) # (n2, 1)
        sali_scores = torch.cat([sali_scores_src, sali_scores_tgt], dim=0) # (n, 1)

        # 2.5 feats
        batched_feats_raw = batched_feats.squeeze(0).transpose(0, 1)  # (n, c)
        batched_feats = torch.cat([batched_feats_raw, ol_scores, sali_scores], dim=1)

        # 3. decoder
        for block in self.decoder_blocks:
            if block_i in self.decoder_skips:
                batched_feats = torch.cat([batched_feats, skip_feats.pop()], dim=-1)
            batched_feats = block(batched_feats, inputs)
            block_i += 1

        # batched_feats[:, -2:] = self.sigmoid(batched_feats[:, -2:])
        # batched_feats[:, :-2] = batched_feats[:, :-2] / torch.norm(batched_feats[:, :-2], dim=1, keepdim=True)
        overlap_scores = self.sigmoid(batched_feats[:, -2:-1])
        saliency_scores = self.sigmoid(batched_feats[:, -1:])
        batched_feats = batched_feats[:, :-2] / torch.norm(batched_feats[:, :-2], dim=1, keepdim=True)
        batched_feats = torch.cat([batched_feats, overlap_scores, saliency_scores], dim=-1)
        return batched_feats
