# --------------------------------------------------------
# TinyViT Model Architecture
# Copyright (c) 2022 Microsoft
# Adapted from LeViT and Swin Transformer
#   LeViT: (https://github.com/facebookresearch/levit)
#   Swin: (https://github.com/microsoft/swin-transformer)
# Build the TinyViT Model
# --------------------------------------------------------

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal
from timm.models.layers import DropPath as TimmDropPath,\
    to_2tuple, trunc_normal_
from timm.models.registry import register_model
from typing import Tuple
from .common import LayerNorm2d, MLPBlock, Adapter
from .common import moe_forward#, _gates_to_load, _prob_in_top_k, cv_squared

import copy

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class DropPath(TimmDropPath):
    def __init__(self, drop_prob=None):
        super().__init__(drop_prob=drop_prob)
        self.drop_prob = drop_prob

    def __repr__(self):
        msg = super().__repr__()
        msg += f'(drop_prob={self.drop_prob})'
        return msg


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, resolution, activation):
        super().__init__()
        img_size: Tuple[int, int] = to_2tuple(resolution)
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * \
            self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x):
        return self.seq(x)


class MBConv(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio,
                 activation, drop_path):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()

        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans,
                               ks=3, stride=1, pad=1, groups=self.hidden_chans)
        self.act2 = activation()

        self.conv3 = Conv2d_BN(
            self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        x = self.drop_path(x)

        x += shortcut
        x = self.act3(x)

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim, activation):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        stride_c=2
        if(out_dim==320 or out_dim==448 or out_dim==576):
            stride_c=1
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # (B, C, H, W)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ConvLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth,
                 activation,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 out_dim=None,
                 conv_expand_ratio=4.,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            MBConv(dim, dim, conv_expand_ratio, activation,
                   drop_path[i] if isinstance(drop_path, list) else drop_path,
                   )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x, labels=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=(14, 14),
                 ):
        super().__init__()
        # (h, w)
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(
            range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N),
                             persistent=False)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.register_buffer('ab',
                                 self.attention_biases[:, self.attention_bias_idxs],
                                 persistent=False)

    def forward(self, x):  # x (B,N,C)
        B, N, _ = x.shape

        # Normalization
        x = self.norm(x)

        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        q, k, v = qkv.view(B, N, self.num_heads, -
                           1).split([self.key_dim, self.key_dim, self.d], dim=3)
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x

# From https://github.com/davidmrau/mixture-of-experts/blob/master/moe.py#L178
#class SparseDispatcher(object):
#    def __init__(self, num_experts, gates):
#        self._gates = gates
#        self._num_experts = num_experts
#        # sort experts
#        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
#        # drop indices
#        _, self._expert_index = sorted_experts.split(1, dim=1)
#        # get according batch index for each expert
#        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
#        # calculate num samples that each expert gets
#        self._part_sizes = (gates > 0).sum(0).tolist()
#        # expand gates to match with self._batch_index
#        gates_exp = gates[self._batch_index.flatten()]
#        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)
#
#    def dispatch(self, inp):
#        inp_exp = inp[self._batch_index].squeeze(1)
#        return torch.split(inp_exp, self._part_sizes, dim=0)
#
#    def combine(self, expert_out, multiply_by_gates):
#        stitched = torch.cat(expert_out, 0)
#
#        if multiply_by_gates:
#            stitched = stitched.mul(self._nonzero_gates)
#        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
#        # combine samples that have been processed by the same k experts
#        combined = zeros.index_add(0, self._batch_index, stitched.float())
#        return combined
#
#    def expert_to_gates(self):
#        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class TinyViTBlock(nn.Module):
    r""" TinyViT Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int, int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        local_conv_size (int): the kernel size of the convolution between
                               Attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
    """

    def __init__(self,args, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 depth=1,
                 local_conv_size=3,
                 activation=nn.GELU,
                 moe=-1,
                 k=-1,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, 'window_size must be greater than 0'
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.depth = depth
        self.args = args

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads,
                              attn_ratio=1, resolution=window_resolution)

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=mlp_activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)
        
        if self.args.if_encoder_adapter and (self.depth in self.args.encoder_adapter_depths):
            self.scale = 0.5
            self.moe = moe

            if moe > 0:
                W, H = self.input_resolution
                #self.centroids_mlp = torch.zeros([moe, W*H, dim]).cuda()
                #self.centroids_space = torch.zeros([moe, W*H, dim]).cuda()

                #self.centroids_mlp = torch.zeros([moe, dim]).cuda()
                #self.centroids_space = torch.zeros([moe, dim]).cuda()

                #self.centroids_mlp = nn.Parameter(self.centroids_mlp)
                #self.centroids_space = nn.Parameter(self.centroids_space)

                self.k = k
                print('Use En-MOE with %s experts and %s selection' % (self.moe, self.k))
                self.MLP_Adapter = nn.ModuleList([Adapter(dim, skip_connect=False) for _ in range(moe)])  # MLP-adapter, no skip connection
                self.Space_Adapter = nn.ModuleList([Adapter(dim) for _ in range(moe)])  # with skip connection

                self.gater = nn.ModuleList([nn.Linear(dim, self.moe, bias=False) for _ in range(1)])
                self.noise = nn.Linear(dim, self.moe, bias = False)

                self.softplus = nn.Softplus()

                self.register_buffer("mean", torch.tensor([0.0]))
                self.register_buffer("std", torch.tensor([1.0]))
            else:
                self.MLP_Adapter = Adapter(dim, skip_connect=False)  # MLP-adapter, no skip connection
                self.Space_Adapter = Adapter(dim)  # with skip connection


    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        if len(clean_values.shape) == 3:
            clean_values = clean_values.view(-1,clean_values.shape[-1])
            noisy_values = noisy_values.view(-1,noisy_values.shape[-1])
            noise_stddev = noise_stddev.view(-1,noise_stddev.shape[-1])
            noisy_top_values = noisy_top_values.view(-1,noisy_top_values.shape[-1])

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def cv_squared(self, x):
        eps = 1e-10
        return x.float().var() / (x.float().mean()**2 + eps)

    #def moe_forward_centroid(self, x, expert, centroids, training, labels=None, noise_epsilon=1e-2, printing=False):
    #    B, L, dim = x.shape
    #   
    #    avg_feature = False
    #    #avg_feature = True
    #    with torch.no_grad():
    #        norm1 = torch.nn.functional.normalize(centroids)
    #        if avg_feature:
    #            norm2 = torch.nn.functional.normalize(x.mean(1))
    #            sim = torch.mm(norm2, norm1.T)
    #        else:
    #            norm2 = torch.nn.functional.normalize(x)
    #            sim = torch.einsum('nlh,blh->bln', norm1, norm2)
    #    sim = torch.softmax(sim*10, -1)

    #    predicts = []
    #    for e in expert:
    #        out = e(x)
    #        predicts.append(out.unsqueeze(-1))
    #    predicts = torch.cat(predicts, dim=-1)
    #    
    #    if avg_feature:
    #        predicts_merged = torch.einsum('blhn,bn->blh', predicts, sim)
    #    else:
    #        predicts_merged = torch.einsum('blhn,bln->blh', predicts, sim)
    #    
    #    if training:
    #    #if 0:
    #        for i in range(len(labels)):
    #            if avg_feature:
    #                centroids[labels[i]-1] = 0.99 * centroids[labels[i]-1] + 0.01 * x[i].mean(0)
    #            else:
    #                centroids[labels[i]-1] = 0.99 * centroids[labels[i]-1] + 0.01 * x[i]

    #    #predicts_merged = []
    #    #for i in range(len(labels)):
    #    #    tmp = predicts[i,:,:,labels[i]-1]
    #    #    predicts_merged.append(tmp.unsqueeze(0))
    #    #predicts_merged = torch.cat(predicts_merged, 0)

    #    if printing:
    #        print(labels[0], sim[0], x[0][0][:10])
    #        if len(centroids.shape) == 3:
    #            print(centroids[:2,:2,:5])
    #        else:
    #            print(centroids[:2,:5])

    #    return predicts_merged, 0

    #def moe_forward(self, x, expert, training, labels=None, noise_epsilon=1e-2, printing=False):
    #    B, L, dim = x.shape

    #    use_flat = False
    #    if use_flat:
    #        x_flat = x.reshape(B*L, dim)

    #    common_moe = 2
    #    specific_moe = 1
    #    if labels is None:
    #        if use_flat:
    #            clean_logits = self.gater[-1](x_flat)
    #        else:
    #            clean_logits = self.gater[-1](x.mean(1))
    #            #clean_logits = self.gater[-1](x.max(1).values)
    #    else:
    #        clean_logits = -torch.ones(B, self.moe).cuda() * 1000
    #        for i in range(len(labels)):
    #            logits_i = self.gater[labels[i]](x[i].unsqueeze(0).mean(1))
    #            clean_logits[i] = logits_i
    #            #clean_logits[i,:common_moe] = logits_i[:,:common_moe] / 10
    #            #clean_logits[i,common_moe+labels[i]*specific_moe:common_moe+specific_moe+labels[i]*specific_moe] = logits_i[:,common_moe:common_moe+specific_moe]

    #        clean_logits = -torch.ones(x_flat.shape[0], self.moe).cuda() * 1000
    #        for i in range(len(labels)):
    #            logits_i = self.gater[labels[i]](x[i])
    #            #clean_logits[i*L:(i+1)*L] = logits_i
    #            clean_logits[i*L:(i+1)*L, :common_moe] = logits_i[:,:common_moe]
    #            clean_logits[i*L:(i+1)*L, common_moe+labels[i]*specific_moe:common_moe+specific_moe+labels[i]*specific_moe] = logits_i[:,common_moe:common_moe+specific_moe]
    #    
    #    if training:
    #        if use_flat:
    #            raw_noise_stddev = self.noise(x_flat)
    #        else:
    #            raw_noise_stddev = self.noise(x.mean(1))
    #            #raw_noise_stddev = self.noise(x.max(1).values)
    #        noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
    #        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
    #        #noisy_logits = clean_logits + (torch.randn_like(clean_logits) * 0.1)
    #        logits = noisy_logits
    #    else:
    #        logits = clean_logits
    #    
    #    #if use_flat:
    #    if 0:
    #        logits = logits.view(B,L,-1)
    #        #logits = logits.max(1).values
    #        logits = logits.mean(1)

    #    logits = logits.softmax(-1)

    #    # Select top k values
    #    try:
    #        top_logits, top_indices = logits.topk(self.k + 1, dim=1)
    #    except:
    #        top_logits, top_indices = logits.topk(self.k, dim=1)
    #    top_k_logits = top_logits[:, :self.k]
    #    top_k_indices = top_indices[:, :self.k]
    #    top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  #

    #    zeros = torch.zeros_like(logits, requires_grad=True)
    #    gates = zeros.scatter(1, top_k_indices, top_k_gates)

    #    if training and self.k < top_logits.shape[1]:
    #        load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
    #    else:
    #        load = self._gates_to_load(gates)
    #    
    #    # B * (D*n_expert)
    #    #predicts = expert(x).view(B,L,dim,-1)
    #    predicts = []
    #        
    #    full_access_num = 5
    #    #if x.shape[1] > 100:
    #    if 0:
    #        partial = x.shape[1] // (len(expert) - full_access_num)
    #        for i, e in enumerate(expert):
    #            if i < full_access_num:
    #                out = e(x)
    #            else:
    #                j = i - full_access_num
    #                mask = torch.zeros(x.shape).to(x.device)
    #                mask[:, j*partial:(j+1)*partial] = 1
    #                out = e(x*mask)
    #            predicts.append(out.unsqueeze(-1))
    #    else:
    #        for e in expert:
    #            out = e(x)
    #            predicts.append(out.unsqueeze(-1))

    #    predicts = torch.cat(predicts, dim=-1)
    #    
    #    if use_flat:
    #        gates = gates.view(B,L,-1)
    #        predicts_merged = torch.einsum('blhn,bln->blh', predicts, gates)
    #    else:
    #        predicts_merged = torch.einsum('blhn,bn->blh', predicts, gates)
    #        #predicts_merged = predicts.sum(-1) / predicts.shape[-1]
    #        #predicts_merged = predicts[:,:,:,labels[0]]
    #    
    #    # Additional MOE loss
    #    importance = gates.sum(0)
    #    loss = self.cv_squared(importance) + self.cv_squared(load)
    #    loss *= 1e-2
    #    
    #    if printing:
    #        #print(full_access_num, 'receives full inputs')
    #        #print('Encoder only')
    #        print(gates[0])

    #    return predicts_merged, loss

    def forward(self, x, labels=None, training=False, moe_print=False):
        moe_total_loss = 0
        gates_total = []

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        res_x = x
        if H == self.window_size and W == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(B, H, W, C)
            pad_b = (self.window_size - H %
                     self.window_size) % self.window_size
            pad_r = (self.window_size - W %
                     self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            # window partition
            x = x.view(B, nH, self.window_size, nW, self.window_size, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_size * self.window_size, C)
            x = self.attn(x)
            # window reverse
            x = x.view(B, nH, nW, self.window_size, self.window_size,
                       C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.view(B, L, C)


        if self.args.if_encoder_adapter and (self.depth in self.args.encoder_adapter_depths): 
            #if labels is None:
            if self.moe <= 0:
                x = self.Space_Adapter(x)
            else:
                if self.depth == 0 and moe_print:
                    moe_out, moe_loss, gates = moe_forward(self, x, self.Space_Adapter, training, labels, printing=True)
                else:
                    moe_out, moe_loss, gates = moe_forward(self, x, self.Space_Adapter, training, labels)

                moe_total_loss += moe_loss
                gates_total.append(gates)

                x = moe_out

        x = res_x + self.drop_path(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)
        
        if self.args.if_encoder_adapter and (self.depth in self.args.encoder_adapter_depths):
            if self.moe <= 0:
                x = x + self.drop_path(self.mlp(x)) + self.scale * self.MLP_Adapter(x)
            else:
                moe_out, moe_loss, gates = moe_forward(self, x, self.MLP_Adapter, training, labels)
                x = x + self.drop_path(self.mlp(x)) + self.scale * moe_out

                moe_total_loss += moe_loss
                gates_total.append(gates)
        else:
            x = x + self.drop_path(self.mlp(x))

        return x, moe_total_loss, gates_total

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"


class BasicLayer(nn.Module):
    """ A basic TinyViT layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        local_conv_size: the kernel size of the depthwise convolution between attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
        out_dim: the output dimension of the layer. Default: dim
    """

    def __init__(self,args, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0.,
                 block_idx = 0,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 local_conv_size=3,
                 activation=nn.GELU,
                 out_dim=None,
                 moe=-1,
                 k=-1,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.args = args

        # build blocks
        self.blocks = nn.ModuleList([
            TinyViTBlock(args=self.args,dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(
                             drop_path, list) else drop_path,
                         depth = block_idx,
                         local_conv_size=local_conv_size,
                         activation=activation,
                         moe=moe,
                         k=k,
                         )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x, labels=None, training=False, moe_print=False):
        moe_loss = 0
        gates_total = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, curr_loss, gates = blk(x, labels=labels, training=training, moe_print=moe_print)

                moe_loss += curr_loss
                gates_total.append(gates)

        if self.downsample is not None:
            x = self.downsample(x)
        return x, moe_loss, gates_total

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class TinyViT(nn.Module):
    def __init__(self,args, img_size=224, in_chans=3, num_classes=1000,
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_sizes=[7, 7, 14, 7],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 use_checkpoint=False,
                 mbconv_expand_ratio=4.0,
                 local_conv_size=3,
                 layer_lr_decay=1.0,
                 moe=-1,
                 k=-1,
                 ):
        super().__init__()
        self.img_size=img_size
        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        self.args = args

        activation = nn.GELU

        self.patch_embed = PatchEmbed(in_chans=in_chans,
                                      embed_dim=embed_dims[0],
                                      resolution=img_size,
                                      activation=activation)

        #self.patch_embed1 = PatchEmbed(in_chans=in_chans,
        #                               embed_dim=embed_dims[0],
        #                               resolution=img_size,
        #                               activation=activation)
        #self.patch_embed2 = PatchEmbed(in_chans=in_chans,
        #                               embed_dim=embed_dims[0],
        #                               resolution=img_size,
        #                               activation=activation)
        #self.patch_embed3 = PatchEmbed(in_chans=in_chans,
        #                               embed_dim=embed_dims[0],
        #                               resolution=img_size,
        #                               activation=activation)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs = dict(dim=embed_dims[i_layer],
                        input_resolution=(patches_resolution[0] // (2 ** (i_layer-1 if i_layer == 3 else i_layer)),
                                patches_resolution[1] // (2 ** (i_layer-1 if i_layer == 3 else i_layer))),
                        #   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                        #                     patches_resolution[1] // (2 ** i_layer)),
                          depth=depths[i_layer],
                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                          downsample=PatchMerging if (
                              i_layer < self.num_layers - 1) else None,
                          use_checkpoint=use_checkpoint,
                          out_dim=embed_dims[min(
                              i_layer + 1, len(embed_dims) - 1)],
                          activation=activation,
                          )
            if i_layer == 0:
                layer = ConvLayer(
                    conv_expand_ratio=mbconv_expand_ratio,
                    **kwargs,
                )
            else:
                layer = BasicLayer(
                    args = self.args,
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    block_idx = i_layer-1,
                    local_conv_size=local_conv_size,
                    moe=moe,
                    k=k,
                    **kwargs)
            self.layers.append(layer)

        # Classifier head
        self.norm_head = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

        # init weights
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dims[-1],
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )

    def set_layer_lr_decay(self, layer_lr_decay):
        decay_rate = layer_lr_decay

        # layers -> blocks (depth)
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]
        #print("LR SCALES:", lr_scales)

        def _set_lr_scale(m, scale):
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))

        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(
                    lambda x: _set_lr_scale(x, lr_scales[i - 1]))
        assert i == depth
        for m in [self.norm_head, self.head]:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            for p in m.parameters():
                assert hasattr(p, 'lr_scale'), p.param_name

        self.apply(_check_lr_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'attention_biases'}

    def forward_features(self, x, labels=None, training=False, moe_print=False):
        x = self.patch_embed(x)

        x = self.layers[0](x)
        start_i = 1
        
        moe_loss_total = 0
        gates_total = []

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x, moe_loss, gates = layer(x, labels=labels, training=training, moe_print=moe_print)

            moe_loss_total += moe_loss
            gates_total.append(gates)

        B,_,C=x.size()
        x = x.view(B, 64, 64, C)
        x=x.permute(0, 3, 1, 2)
        x=self.neck(x)
        return x, moe_loss_total, gates_total

    def forward(self, x, labels=None, training=False, moe_print=False):
        x, moe_loss, gates = self.forward_features(x, labels=labels, training=training, moe_print=moe_print)
        #x = self.norm_head(x)
        #x = self.head(x)
        return x, moe_loss, gates


_checkpoint_url_format = \
    'https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/{}.pth'
_provided_checkpoints = {
    'tiny_vit_5m_224': 'tiny_vit_5m_22kto1k_distill',
    'tiny_vit_11m_224': 'tiny_vit_11m_22kto1k_distill',
    'tiny_vit_21m_224': 'tiny_vit_21m_22kto1k_distill',
    'tiny_vit_21m_384': 'tiny_vit_21m_22kto1k_384_distill',
    'tiny_vit_21m_512': 'tiny_vit_21m_22kto1k_512_distill',
}


def register_tiny_vit_model(fn):
    '''Register a TinyViT model
    It is a wrapper of `register_model` with loading the pretrained checkpoint.
    '''
    def fn_wrapper(pretrained=False, **kwargs):
        model = fn()
        if pretrained:
            model_name = fn.__name__
            assert model_name in _provided_checkpoints, \
                f'Sorry that the checkpoint `{model_name}` is not provided yet.'
            url = _checkpoint_url_format.format(
                _provided_checkpoints[model_name])
            checkpoint = torch.hub.load_state_dict_from_url(
                url=url,
                map_location='cpu', check_hash=False,
            )
            model.load_state_dict(checkpoint['model'])

        return model

    # rename the name of fn_wrapper
    fn_wrapper.__name__ = fn.__name__
    return register_model(fn_wrapper)


@register_tiny_vit_model
def tiny_vit_5m_224(pretrained=False, num_classes=1000, drop_path_rate=0.0):
    return TinyViT(
        num_classes=num_classes,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=drop_path_rate,
    )


@register_tiny_vit_model
def tiny_vit_11m_224(pretrained=False, num_classes=1000, drop_path_rate=0.1):
    return TinyViT(
        num_classes=num_classes,
        embed_dims=[64, 128, 256, 448],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 14],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=drop_path_rate,
    )


@register_tiny_vit_model
def tiny_vit_21m_224(pretrained=False, num_classes=1000, drop_path_rate=0.2):
    return TinyViT(
        num_classes=num_classes,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=drop_path_rate,
    )


@register_tiny_vit_model
def tiny_vit_21m_384(pretrained=False, num_classes=1000, drop_path_rate=0.1):
    return TinyViT(
        img_size=384,
        num_classes=num_classes,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[12, 12, 24, 12],
        drop_path_rate=drop_path_rate,
    )


@register_tiny_vit_model
def tiny_vit_21m_512(pretrained=False, num_classes=1000, drop_path_rate=0.1):
    return TinyViT(
        img_size=512,
        num_classes=num_classes,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[16, 16, 32, 16],
        drop_path_rate=drop_path_rate,
    )
