# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn
from torch.distributions.normal import Normal

import math
from typing import Tuple, Type

from .common import MLPBlock, Adapter
from .common import moe_forward#, _gates_to_load, _prob_in_top_k, cv_squared

import copy

class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        args,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        moe = -1,
        k = -1,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.args = args
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        for i in range(depth):
            if i<args.decoder_adapt_depth:
                if_adapter = args.if_mask_decoder_adapter
            else:
                if_adapter = False
            self.layers.append(
                TwoWayAttentionBlock(
                    args = self.args,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    if_adapter=if_adapter,
                    skip_first_layer_pe=(i == 0),
                    moe=moe,
                    k=k,
                    depth=i,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
        labels=None,
        training=False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding
        
        moe_total_loss = 0
        gates_total = []

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys, moe_loss, gates = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
                labels=labels,
                training=training,
            )
            moe_total_loss += moe_loss
            gates_total.append(gates)

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys, moe_total_loss, gates_total


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        args,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        if_adapter:bool = False,
        skip_first_layer_pe: bool = False,
        moe = -1,
        k = -1,
        depth = 0,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.args = args
        self.if_adapter = if_adapter
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        #||print(if_adapter)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        if self.if_adapter:
            self.scale = 0.5
            self.moe = moe
            
            if moe > 0:
                #self.centroids1 = torch.zeros([moe, 5, embedding_dim]).cuda()
                #self.centroids_mlp = torch.zeros([moe, 5, embedding_dim]).cuda()
                #self.centroids2 = torch.zeros([moe, 4096, embedding_dim]).cuda()

                #self.centroids1 = torch.zeros([moe, embedding_dim]).cuda()
                #self.centroids_mlp = torch.zeros([moe, embedding_dim]).cuda()
                #self.centroids2 = torch.zeros([moe, embedding_dim]).cuda()

                #self.centroids1 = nn.Parameter(self.centroids1)
                #self.centroids_mlp = nn.Parameter(self.centroids_mlp)
                #self.centroids2 = nn.Parameter(self.centroids2)

                self.k = k
                print('Use Dn-MOE with %s experts and %s selection' % (self.moe, self.k))
                self.MLP_Adapter = nn.ModuleList([Adapter(embedding_dim, skip_connect=False) for _ in range(moe)])  # MLP-adapter, no skip connection
                self.Adapter  = nn.ModuleList([Adapter(embedding_dim) for _ in range(moe)])  # with skip connection
                self.Adapter2 = nn.ModuleList([Adapter(embedding_dim) for _ in range(moe)])  # with skip connection

                self.gater = nn.ModuleList([nn.Linear(embedding_dim, self.moe, bias=False) for _ in range(1)])
                self.noise = nn.Linear(embedding_dim, self.moe, bias = False)

                self.softplus = nn.Softplus()

                self.register_buffer("mean", torch.tensor([0.0]))
                self.register_buffer("std", torch.tensor([1.0]))
            else:
                self.MLP_Adapter = Adapter(embedding_dim, skip_connect=False)  # MLP-adapter, no skip connection
                self.Adapter  = Adapter(embedding_dim)  # with skip connection
                self.Adapter2 = Adapter(embedding_dim)  # with skip connection


        self.skip_first_layer_pe = skip_first_layer_pe

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
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

    #def moe_forward_centroid(self, x, expert, centroids, training=False, labels=None, noise_epsilon=1e-2):
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

    #    return predicts_merged, 0

    #def moe_forward(self, x, expert, training=False, labels=None, noise_epsilon=1e-2):
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

    #        #clean_logits = -torch.ones(x_flat.shape[0], self.moe).cuda() * 1000
    #        #for i in range(len(labels)):
    #        #    logits_i = self.gater[labels[i]](x[i])
    #        #    #clean_logits[i*L:(i+1)*L] = logits_i
    #        #    clean_logits[i*L:(i+1)*L, :common_moe] = logits_i[:,:common_moe]
    #        #    clean_logits[i*L:(i+1)*L, common_moe+labels[i]*specific_moe:common_moe+specific_moe+labels[i]*specific_moe] = logits_i[:,common_moe:common_moe+specific_moe]
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
    #    # select based on token
    #    #if use_flat:
    #    if 0:
    #        logits = logits.view(B,L,-1)
    #        #logits = logits.max(1).values
    #        logits = logits.mean(1)
    #    
    #    logits = logits.softmax(-1)
    #    
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

    #    # B * (D*n_expert)
    #    #predicts = expert(x).view(B,L,dim,-1)
    #    predicts = []

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
    #    #if 0:
    #        gates = gates.view(B,L,-1)
    #        predicts_merged = torch.einsum('blhn,bln->blh', predicts, gates)
    #        #gates = gates.max(1).values
    #        #gates = gates.mean(1)
    #        #predicts_merged = torch.einsum('blhn,bn->blh', predicts, gates)
    #    else:
    #        predicts_merged = torch.einsum('blhn,bn->blh', predicts, gates)
    #        #predicts_merged = predicts.sum(-1) / predicts.shape[-1]
    #        #predicts_merged = predicts[:,:,:,labels[0]]
    #        

    #    # Additional MOE loss
    #    importance = gates.sum(0)
    #    loss = self.cv_squared(importance) + self.cv_squared(load)
    #    loss *= 1e-2

    #    return predicts_merged, loss


    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor, labels=None, training=False,
    ) -> Tuple[Tensor, Tensor]:
        moe_loss_total = 0
        gates_total = []

        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out

        # add adapter layer
        if self.if_adapter:
            if self.moe <= 0:
                queries = self.Adapter(queries)
            else:
                moe_out, moe_loss, gates = moe_forward(self, queries, self.Adapter, training, labels)
                queries = moe_out

                moe_loss_total += moe_loss
                gates_total.append(gates)

        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        if self.if_adapter:
            #if labels is None:
            if self.moe <= 0:
                queries = queries + mlp_out + self.scale * self.MLP_Adapter(queries)
            else:
                moe_out, moe_loss, gates = moe_forward(self, queries, self.MLP_Adapter, training, labels)
                queries = queries + mlp_out + self.scale * moe_out

                moe_loss_total += moe_loss
                gates_total.append(gates)
        else:
            queries = queries + mlp_out 
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        
        if self.if_adapter:
            if self.moe <= 0:
                keys = self.Adapter2(keys)
            else:
                moe_out, moe_loss, gates = moe_forward(self, keys, self.Adapter2, training, labels)

                moe_loss_total += moe_loss
                gates_total.append(gates)
                keys = moe_out

        keys = self.norm4(keys)

        return queries, keys, moe_loss_total, gates_total


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

