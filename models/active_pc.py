import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from multilevel_convolution import *
from transformer_v1 import *
from ipdb import set_trace as st


class ACTIVEPC(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                
                 temporal_kernel_size, temporal_stride,                               
                 dim, depth, heads, dim_head, dropout1,                            
                 mlp_dim, num_classes, dropout2):                                    
        super().__init__()

        self.tube_embedding1 = MultilevelConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=2,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=2, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        
        self.transformer1 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1)
        self.transformer2 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1)
        self.transformer3 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1)
        self.gmp1 = nn.AdaptiveMaxPool1d(1)

        self.gap2 = nn.AdaptiveAvgPool1d(1)

        self.gap3 = nn.AdaptiveAvgPool1d(1)

        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(mlp_dim, num_classes),
        )

        self.mlp_head2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(mlp_dim, num_classes),
        )

        self.mlp_head3 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(mlp_dim, num_classes),
        )




    def forward(self, input):                    
        device = input.get_device()
        
        xyzs, features = self.tube_embedding1(input)           
        
        xyzs1, features1 = xyzs[1], features[1]
        xyzs2, features2 = xyzs[2], features[2]
        xyzs3, features3 = xyzs[3], features[3]

        features1 = features1.permute(0, 1, 3, 2)   
        output1 = self.transformer1(xyzs1, features1)
        B,L,N,C = output1.shape
        output1 = output1.reshape(B, L * N,  C)
        output1 = output1.permute(0, 2, 1)

        output1 = self.gmp1(output1).squeeze(-1)
        output1 = self.mlp_head1(output1)  

        features2 = features2.permute(0, 1, 3, 2)   
        output2 = self.transformer2(xyzs2, features2)

        B,L,N,C = output2.shape
        output2 = output2.reshape(B, L * N,  C)
        output2 = output2.permute(0, 2, 1)
        output2 = self.gap2(output2).squeeze(-1) 
        output2 = self.mlp_head2(output2) 


        fusion_output = (output1 + output2) / 2

        B, L, n, _ = xyzs3.shape
        C = features3.shape[2]
        features3 = features3.permute(0, 1, 3, 2)  

        output = self.transformer3(xyzs3, features3)    

        B,L,N,C = output.shape
        output = output.reshape(B, L * N,  C)
        output = output.permute(0, 2, 1)
        output = self.gap3(output).squeeze(-1)
        output = self.mlp_head3(output)

        output = (fusion_output + output)/2

        return output
