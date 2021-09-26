import copy
import torch
import torch.nn as nn
from utils import square_dists, gather_points


def get_graph_features(feats, coords, k=10):
    '''

    :param feats: (B, N, C)
    :param coords: (B, N, 3)
    :param k: float
    :return: (B, N, k, 2C)
    '''

    sq_dists = square_dists(coords, coords)
    inds = torch.topk(sq_dists, k+1, dim=-1, largest=False, sorted=True)[1]
    inds = inds[:, :, 1:] # (B, N, k)

    neigh_feats = gather_points(feats, inds) # (B, N, k, c)
    feats = torch.unsqueeze(feats, 2).repeat(1, 1, k, 1) # (B, N, k, c)
    return torch.cat([feats, neigh_feats - feats], dim=-1)


class GCN(nn.Module):
    def __init__(self, feats_dim, k):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(feats_dim * 2, feats_dim, 1, bias=False),
            nn.InstanceNorm2d(feats_dim),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(feats_dim * 2, feats_dim * 2, 1, bias=False),
            nn.InstanceNorm2d(feats_dim * 2),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(feats_dim * 4, feats_dim, 1, bias=False),
            nn.InstanceNorm1d(feats_dim),
            nn.LeakyReLU(0.2)
        )
        self.k = k

    def forward(self, coords, feats):
        '''

        :param coors: (B, 3, N)
        :param feats: (B, C, N)
        :param k: float
        :return: (B, C, N)
        '''
        feats1 = get_graph_features(feats=feats.permute(0, 2, 1).contiguous(),
                                    coords=coords.permute(0, 2, 1).contiguous(),
                                    k=self.k)
        feats1 = self.conv1(feats1.permute(0, 3, 1, 2).contiguous())
        feats1 = torch.max(feats1, dim=-1)[0]

        feats2 = get_graph_features(feats=feats1.permute(0, 2, 1).contiguous(),
                                    coords=coords.permute(0, 2, 1).contiguous(),
                                    k=self.k)
        feats2 = self.conv2(feats2.permute(0, 3, 1, 2).contiguous())
        feats2 = torch.max(feats2, dim=-1)[0]

        feats3 = torch.cat([feats, feats1, feats2], dim=1)
        feats3 = self.conv3(feats3)

        return feats3


def multi_head_attention(query, key, value):
    '''

    :param query: (B, dim, nhead, N)
    :param key: (B, dim, nhead, M)
    :param value: (B, dim, nhead, M)
    :return: (B, dim, nhead, N)
    '''
    dim = query.size(1)
    scores = torch.einsum('bdhn, bdhm->bhnm', query, key) / dim**0.5
    attention = torch.nn.functional.softmax(scores, dim=-1)
    feats = torch.einsum('bhnm, bdhm->bdhn', attention, value)
    return feats


class Cross_Attention(nn.Module):
    def __init__(self, feat_dims, nhead):
        super().__init__()
        assert feat_dims % nhead == 0
        self.feats_dim = feat_dims
        self.nhead = nhead
        # self.q_conv = nn.Conv1d(feat_dims, feat_dims, 1, bias=True)
        # self.k_conv = nn.Conv1d(feat_dims, feat_dims, 1, bias=True)
        # self.v_conv = nn.Conv1d(feat_dims, feat_dims, 1, bias=True)
        self.conv = nn.Conv1d(feat_dims, feat_dims, 1)
        self.q_conv, self.k_conv, self.v_conv = [copy.deepcopy(self.conv) for _ in range(3)] # a good way than better ?
        self.mlp = nn.Sequential(
            nn.Conv1d(feat_dims * 2, feat_dims * 2, 1),
            nn.InstanceNorm1d(feat_dims * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(feat_dims * 2, feat_dims, 1),
        )

    def forward(self, feats1, feats2):
        '''

        :param feats1: (B, C, N)
        :param feats2: (B, C, M)
        :return: (B, C, N)
        '''
        b = feats1.size(0)
        dims = self.feats_dim // self.nhead
        query = self.q_conv(feats1).reshape(b, dims, self.nhead, -1)
        key = self.k_conv(feats2).reshape(b, dims, self.nhead, -1)
        value = self.v_conv(feats2).reshape(b, dims, self.nhead, -1)
        feats = multi_head_attention(query, key, value)
        feats = feats.reshape(b, self.feats_dim, -1)
        feats = self.conv(feats)
        cross_feats = self.mlp(torch.cat([feats1, feats], dim=1))
        return cross_feats


class InformationInteractive(nn.Module):
    def __init__(self, layer_names, feat_dims, k, nhead):
        super().__init__()
        self.layer_names = layer_names
        self.k = k
        self.blocks = nn.ModuleList()
        for layer_name in layer_names:
            if layer_name == 'gcn':
                self.blocks.append(GCN(feat_dims, k))
            elif layer_name == 'cross_attn':
                self.blocks.append(Cross_Attention(feat_dims, nhead))
            else:
                raise NotImplementedError

    def forward(self, coords1, feats1, coords2, feats2):
        '''

        :param coords1: (B, 3, N)
        :param feats1: (B, C, N)
        :param coords2: (B, 3, M)
        :param feats2: (B, C, M)
        :return: feats1=(B, C, N), feats2=(B, C, M)
        '''
        for layer_name, block in zip(self.layer_names, self.blocks):
            if layer_name == 'gcn':
                feats1 = block(coords1, feats1)
                feats2 = block(coords2, feats2)
            elif layer_name == 'cross_attn':
                feats1 = feats1 + block(feats1, feats2)
                feats2 = feats2 + block(feats2, feats1)
            else:
                raise NotImplementedError

        return feats1, feats2
