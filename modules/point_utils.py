#
''' Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch '''
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
from torch.autograd import Function
import torch.nn as nn
import pytorch_utils as pt_utils
import sys

try:
    import builtins
except:
    import __builtin__ as builtins

try:
    import pointnet2._ext as _ext
except ImportError:
    if not getattr(builtins, "__POINTNET2_SETUP__", False):
        raise ImportError(
            "Could not import _ext module.\n"
            "Please see the setup instructions in the README: "
            "https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/README.rst"
        )

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *

from ipdb import set_trace as st
class RandomDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        fps_inds = _ext.furthest_point_sampling(xyz, npoint)
        ctx.mark_non_differentiable(fps_inds)
        return fps_inds

    @staticmethod
    def backward(xyz, a=None):
        return None, None


def furthest_point_sampling(xyz, npoint):
    return FurthestPointSampling.apply(xyz, npoint)


import torch
import pandas as pd
import os

file_counter = 0

def increment_file_counter():
    global file_counter
    file_counter += 1
    return file_counter

def save_point_cloud_to_file(xyz, stage_1_flags, stage_2_candidate_flags, stage_2_flags):
    file_name = f"point_cloud_stage_{increment_file_counter()}.csv"
    
    xyz_flattened = xyz.view(-1, 3).cpu().numpy() 
    x, y, z = xyz_flattened[:, 0], xyz_flattened[:, 1], xyz_flattened[:, 2]
    
    stage_1_flags_flattened = stage_1_flags.view(-1).cpu().numpy()
    stage_2_candidate_flags_flattened = stage_2_candidate_flags.view(-1).cpu().numpy()
    stage_2_flags_flattened = stage_2_flags.view(-1).cpu().numpy()
    

    assert len(x) == len(stage_1_flags_flattened) == len(stage_2_candidate_flags_flattened) == len(stage_2_flags_flattened), "Column lengths do not match."
    
    data = {
        'x': x,
        'y': y,
        'z': z,
        'is_original': [True] * len(x),  
        'is_stage_1': stage_1_flags_flattened,
        'is_stage_2_candidate': stage_2_candidate_flags_flattened,
        'is_stage_2': stage_2_flags_flattened
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)
    print(f"Point cloud saved to {file_name}")


        
class MNS(Function):
    @staticmethod
    def forward(ctx, xyz, npoints_hierarchy, k=16, radius=0.2):
        B, N, _ = xyz.shape
        sampled_indices = {}
        current_inds = None

        for level, npoints in enumerate(npoints_hierarchy, 1):
            if level == 1:
                sampled_inds = furthest_point_sampling(xyz, npoints)  # (B, npoints)
            else:
                sampled_xyz_prev = xyz.gather(1, current_inds.unsqueeze(-1).expand(-1, -1, 3))  # (B, n_prev, 3)
                neighbor_idx = ball_query(radius, k, xyz, sampled_xyz_prev)  # (B, n_prev, k)

                neighbor_idx = neighbor_idx.view(B, -1).long()  # (B, n_prev * k)

                unique_neighbor_idx = []
                max_unique = 0
                for b in range(B):
                    unique_idx = torch.unique(neighbor_idx[b], sorted=False)
                    unique_neighbor_idx.append(unique_idx)
                    if unique_idx.size(0) > max_unique:
                        max_unique = unique_idx.size(0)
                
                padded_unique_neighbor_idx = neighbor_idx.new_full((B, max_unique), neighbor_idx.min())
                for b in range(B):
                    padded_unique_neighbor_idx[b, :unique_neighbor_idx[b].size(0)] = unique_neighbor_idx[b]
                
                sampled_inds_subset = furthest_point_sampling(
                    xyz.gather(1, padded_unique_neighbor_idx.unsqueeze(-1).expand(-1, -1, 3)), 
                    npoints
                )  # (B, npoints)
                sampled_inds_subset = sampled_inds_subset.long()
                
                sampled_inds = padded_unique_neighbor_idx.gather(1, sampled_inds_subset)  # (B, npoints)

            sampled_inds = sampled_inds.long().to(xyz.device) 

            sampled_indices[f"level_{level}"] = sampled_inds  
            current_inds = sampled_inds  

        return sampled_indices



    @staticmethod
    def backward(ctx, *args):
        return None, None

def multilevel_neighborhood_sampling(xyz, npoints_hierarchy, k=16, radius=0.2):
    return MNS.apply(xyz, npoints_hierarchy, k, radius)



furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        _, C, N = features.size()

        ctx.for_backwards = (idx, C, N)

        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        dist2, idx = _ext.three_nn(unknown, known)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        B, c, m = features.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, m = ctx.three_interpolate_for_backward

        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        ctx.for_backwards = (idx, N)
        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        inds = _ext.ball_query(new_xyz, xyz, radius, nsample)
        ctx.mark_non_differentiable(inds)
        return inds

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind


        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        else:
            return new_features
