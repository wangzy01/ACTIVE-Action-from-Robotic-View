import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from ipdb import set_trace as st
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x) + x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.spatial_op = nn.Linear(3, dim_head, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, xyzs, features):
        b, l, n, _, h = *features.shape, self.heads

        norm_features = self.norm(features)
        qkv = self.to_qkv(norm_features).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b l n (h d) -> b h (l n) d', h = h), qkv)                           

        xyzs_flatten = rearrange(xyzs, 'b l n d -> b (l n) d')                                                      # [b, m, 3]

        delta_xyzs = torch.unsqueeze(input=xyzs_flatten, dim=1) - torch.unsqueeze(input=xyzs_flatten, dim=2)        # [b, m, m, 3]

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale                                             # [b, h, m, m]
        attn = dots.softmax(dim=-1)

        v = einsum('b h i j, b h j d -> b h i d', attn, v)                                                          # [b, h, m, d]

        attn = torch.unsqueeze(input=attn, dim=4)                                                                   # [b, h, m, m, 1]
        delta_xyzs = torch.unsqueeze(input=delta_xyzs, dim=1)                                                       # [b, 1, m, m, 3]
        delta_xyzs = torch.sum(input=attn*delta_xyzs, dim=3, keepdim=False)                                         

        displacement_features = self.spatial_op(delta_xyzs)                                                         # [b, h, m, d]

        out = v + displacement_features
        out = rearrange(out, 'b h m d -> b m (h d)')
        out =  self.to_out(out)
        out = rearrange(out, 'b (l n) d -> b l n d', l=l, n=n)
        return out + features
    



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, xyzs, features):
        for attn, ff in self.layers:
            features = attn(xyzs, features)
            features = ff(features)
        return features


# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#     def forward(self, x, **kwargs):
#         return self.fn(x, **kwargs) + x

# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn
#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)
    

# class Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.GELU(),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x):
#         b, n, _, h = *x.shape, self.heads
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

#         dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

#         attn = dots.softmax(dim=-1)

#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out =  self.to_out(out)
#         return out


# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
            # self.layers.append(nn.ModuleList([
            #     Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = 0.))),
            #     Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            # ]))
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x)
#             x = ff(x)
#         return x