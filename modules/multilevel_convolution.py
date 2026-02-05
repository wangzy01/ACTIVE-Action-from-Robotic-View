import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import point_utils
from typing import List
from knn_cuda import KNN
from timm.models.layers import DropPath
from ipdb import set_trace as st
from einops import rearrange, repeat
from typing import List, Dict

class MultilevelConv(nn.Module):
    def __init__(self,
                 in_planes: int,
                 mlp_planes: List[int],
                 mlp_batch_norm: List[bool],
                 mlp_activation: List[bool],
                 spatial_kernel_size: [float, int],
                 spatial_stride: int,
                 temporal_kernel_size: int,
                 temporal_stride: int = 1,
                 temporal_padding: [int, int] = [0, 0],
                 temporal_padding_mode: str = 'replicate',
                 operator: str = 'addition',
                 spatial_pooling: str = 'max',
                 temporal_pooling: str = 'sum',
                 bias: bool = False,
                 scale_xyz: List[float] = [1.0, 1.0, 1.0]
                 ):

        super().__init__()

        self.in_planes = in_planes
        self.mlp_planes = mlp_planes
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_activation = mlp_activation

        self.r, self.k = spatial_kernel_size
        self.spatial_stride = spatial_stride

        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_stride = temporal_stride
        self.temporal_padding = temporal_padding
        self.temporal_padding_mode = temporal_padding_mode

        self.operator = operator
        self.spatial_pooling = spatial_pooling
        self.temporal_pooling = temporal_pooling

        # Convolution layers for displacement
        conv_d = [nn.Conv2d(in_channels=4, out_channels=mlp_planes[0], kernel_size=1, stride=1, padding=0, bias=bias)]
        if mlp_batch_norm[0]:
            conv_d.append(nn.BatchNorm2d(num_features=mlp_planes[0]))
        if mlp_activation[0]:
            conv_d.append(nn.ReLU(inplace=True))
        self.conv_d = nn.Sequential(*conv_d)

        # Convolution layers for features if in_planes != 0
        if in_planes != 0:
            conv_f = [nn.Conv2d(in_channels=in_planes, out_channels=mlp_planes[0], kernel_size=1, stride=1, padding=0, bias=bias)]
            if mlp_batch_norm[0]:
                conv_f.append(nn.BatchNorm2d(num_features=mlp_planes[0]))
            if mlp_activation[0]:
                conv_f.append(nn.ReLU(inplace=True))
            self.conv_f = nn.Sequential(*conv_f)

        # MLP layers
        mlp = []
        for i in range(1, len(mlp_planes)):
            if mlp_planes[i] != 0:
                mlp.append(nn.Conv2d(in_channels=mlp_planes[i-1], out_channels=mlp_planes[i], kernel_size=1, stride=1, padding=0, bias=bias))
            if mlp_batch_norm[i]:
                mlp.append(nn.BatchNorm2d(num_features=mlp_planes[i]))
            if mlp_activation[i]:
                mlp.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp)
        self.scale_xyz = torch.nn.Parameter(torch.tensor(scale_xyz, dtype=torch.float32).view(1, 1, 3))



    def forward(self, xyzs: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            xyzs: torch.Tensor
                 (B, T, N, 3) tensor of sequence of the xyz coordinates
            features: torch.Tensor
                 (B, T, C, N) tensor of sequence of the features
        """
        device = xyzs.device

        B, T, N, _ = xyzs.shape

        assert (self.temporal_kernel_size % 2 == 1), "MultiP4DConv: Temporal kernel size should be odd!"
        assert ((T + sum(self.temporal_padding) - self.temporal_kernel_size) % self.temporal_stride == 0), "MultiP4DConv: Temporal length error!"

        # Split temporal frames
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]  # List of (B, N, 3)
        


        if self.temporal_padding_mode == 'zeros':
            xyz_padding = torch.zeros_like(xyzs[0])
            for i in range(self.temporal_padding[0]):
                xyzs = [xyz_padding] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyz_padding]
        else:
            for i in range(self.temporal_padding[0]):
                xyzs = [xyzs[0]] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyzs[-1]]

        if self.in_planes != 0:
            features = torch.split(tensor=features, split_size_or_sections=1, dim=1)
            features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in features]  # List of (B, C, N)

            if self.temporal_padding_mode == 'zeros':
                feature_padding = torch.zeros_like(features[0])
                for i in range(self.temporal_padding[0]):
                    features = [feature_padding] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [feature_padding]
            else:
                for i in range(self.temporal_padding[0]):
                    features = [features[0]] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [features[-1]]

        # Initialize lists for new_xyzs and new_features per level
        new_xyzs_levels = {1: [], 2: [], 3: []}
        new_features_levels = {1: [], 2: [], 3: []}

        npoints_hierarchy = [96, 48, 24]   # Fixed hierarchy for num_points=768
 
        for t in range(self.temporal_kernel_size//2, len(xyzs)-self.temporal_kernel_size//2, self.temporal_stride):
            temporal_window = range(t - self.temporal_kernel_size//2, t + self.temporal_kernel_size//2 + 1)

            window_xyz = [xyzs[i] for i in temporal_window]  # List of (B, N, 3)
            window_features = [features[i] for i in temporal_window] if self.in_planes != 0 else None  # List of (B, C, N) or None

            sampled_indices = point_utils.multilevel_neighborhood_sampling(window_xyz[t - temporal_window.start], npoints_hierarchy, k=16, radius=self.r)  # dict with 'level_1', 'level_2', 'level_3'

            for level in [1, 2, 3]:
                level_key = f"level_{level}"
                sampled_inds = sampled_indices[level_key]  # (B, n_points_level)
                sampled_inds = sampled_inds.to(device) 
                sampled_inds = sampled_inds.long()  


                gathered_xyz = window_xyz[t - temporal_window.start].transpose(1, 2).contiguous()  # (B, 3, N)
                sampled_xyz = torch.gather(
                    gathered_xyz,
                    2,
                    sampled_inds.unsqueeze(1).expand(-1, 3, -1)  # (B, 3, n_points_level)
                ).transpose(1, 2).contiguous()  # (B, n_points_level, 3)

                new_xyzs_levels[level].append(sampled_xyz)

                # Spatial anchor points
                anchor_xyz_flipped = sampled_xyz.transpose(1, 2).contiguous()  # (B, 3, n_points_level)
                anchor_xyz_expanded = sampled_xyz.unsqueeze(3)  # (B, n_points_level, 3, 1)

                # Initialize feature list for this level
                level_features = []

                for i in temporal_window:
                    neighbor_xyz = xyzs[i]  # (B, N, 3)

                    scaling = self.scale_xyz.to(device)  # (1, 1, 3)

                    # Apply EEQ (Elastic Ellipse Query) scaling
                    neighbor_xyz = neighbor_xyz * scaling
                    sampled_xyz = sampled_xyz * scaling

                    idx = point_utils.ball_query(self.r, self.k, neighbor_xyz, sampled_xyz)  # (B, n_points_level, k)
 
                    idx = idx.int().to(device)
            
                    neighbor_xyz_flipped = neighbor_xyz.transpose(1, 2).contiguous()  # (B, 3, N)
                    neighbor_xyz_grouped = point_utils.grouping_operation(neighbor_xyz_flipped, idx)  # (B, 3, n_points_level, k)

                    xyz_displacement = neighbor_xyz_grouped - anchor_xyz_expanded.transpose(1, 2).contiguous()  # (B, 3, n_points_level, k)
                    t_displacement = torch.ones(
                        (xyz_displacement.size()[0], 1, xyz_displacement.size()[2], xyz_displacement.size()[3]),
                        dtype=torch.float32,
                        device=device
                    ) * (i - t)
                    displacement = torch.cat((xyz_displacement, t_displacement), dim=1)  # (B, 4, n_points_level, k)
                    displacement = self.conv_d(displacement)  # (B, mlp_planes[0], n_points_level, k)

                    if self.in_planes != 0:
                        neighbor_feature_grouped = point_utils.grouping_operation(features[i], idx)  # (B, in_planes, n_points_level, k)
                        feature = self.conv_f(neighbor_feature_grouped)  # (B, mlp_planes[0], n_points_level, k)
                        if self.operator == '+':
                            feature = feature + displacement
                        else:
                            feature = feature * displacement
                    else:
                        feature = displacement  # (B, mlp_planes[0], n_points_level, k)

                    # MLP
                    feature = self.mlp(feature)  # (B, mlp_planes[-1], n_points_level, k)

                    # Spatial Pooling
                    if self.spatial_pooling == 'max':
                        feature = torch.max(feature, dim=-1)[0]  # (B, mlp_planes[-1], n_points_level)
                    elif self.spatial_pooling == 'sum':
                        feature = torch.sum(feature, dim=-1)  # (B, mlp_planes[-1], n_points_level)
                    else:
                        feature = torch.mean(feature, dim=-1)  # (B, mlp_planes[-1], n_points_level)

                    level_features.append(feature)

                # Stack features across temporal window
                level_features = torch.stack(level_features, dim=1)  # (B, T', mlp_planes[-1], n_points_level)

                # Temporal Pooling
                if self.temporal_pooling == 'max':
                    level_features = torch.max(level_features, dim=1)[0]  # (B, mlp_planes[-1], n_points_level)
                elif self.temporal_pooling == 'sum':
                    level_features = torch.sum(level_features, dim=1)  # (B, mlp_planes[-1], n_points_level)
                else:
                    level_features = torch.mean(level_features, dim=1)  # (B, mlp_planes[-1], n_points_level)

                new_features_levels[level].append(level_features)

        final_new_xyzs = {}
        final_new_features = {}
        for level in [1, 2, 3]:
            final_new_xyzs[level] = torch.stack(new_xyzs_levels[level], dim=1)  # (B, num_windows, n_points_level, 3)
            final_new_features[level] = torch.stack(new_features_levels[level], dim=1)  # (B, num_windows, mlp_planes[-1], n_points_level)

        return final_new_xyzs, final_new_features