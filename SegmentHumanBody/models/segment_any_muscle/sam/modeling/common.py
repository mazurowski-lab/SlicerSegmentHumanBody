# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from typing import Type

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
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


def moe_forward(self, x, expert, training, labels=None, noise_epsilon=1e-2, \
                         use_flat=False, masking=False, printing=False):
    B, L, dim = x.shape

    if use_flat:
        x_flat = x.reshape(B*L, dim)
        clean_logits = self.gater[-1](x_flat)
    else:
        clean_logits = self.gater[-1](x.mean(1))

    if training:
    #if 0:
        if use_flat:
            raw_noise_stddev = self.noise(x_flat)
        else:
            raw_noise_stddev = self.noise(x.mean(1))
        noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        logits = noisy_logits
    else:
        logits = clean_logits
    
    logits = logits.softmax(-1)

    try:
        top_logits, top_indices = logits.topk(self.k + 1, dim=1)
    except:
        top_logits, top_indices = logits.topk(self.k, dim=1)
    top_k_logits  = top_logits[:, :self.k]
    top_k_indices = top_indices[:, :self.k]

    if labels is not None:
        label2expert = {0: [18,19],
                        1: [0,1,2,3],
                        2: [4,5,6],
                        3: [7,8],
                        4: [9,10],
                        5: [11,12],
                        6: [13,14],
                        7: [15,16],
                        8: [17]}


        for idx, label in enumerate(labels):
            activate_expert = label2expert[label]
            for i in range(self.k):
                if i not in activate_expert:
                    top_k_logits[idx, i] = 0

    top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  #

    zeros = torch.zeros_like(logits, requires_grad=True)
    gates = zeros.scatter(1, top_k_indices, top_k_gates)

    if training and self.k < top_logits.shape[1]:
        load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
    else:
        load = self._gates_to_load(gates)
    
    predicts = []
        
    if masking and x.shape[1] > 100:
        full_access_num = 5
        partial = x.shape[1] // (len(expert) - full_access_num)
        for i, e in enumerate(expert):
            if i < full_access_num:
                out = e(x)
            else:
                j = i - full_access_num
                mask = torch.zeros(x.shape).to(x.device)
                mask[:, j*partial:(j+1)*partial] = 1
                out = e(x*mask)
            predicts.append(out.unsqueeze(-1))
    else:
        for e in expert:
            out = e(x)
            predicts.append(out.unsqueeze(-1))
    
    # B * D * H * num_expert
    predicts = torch.cat(predicts, dim=-1)
    
    #predicts_by_expert = predicts.permute(0,2,3,1).contiguous().view(-1, x.shape[1])
    ##print(predicts.shape, predicts_by_expert.shape)
    #if len(predicts_by_expert) > 2000:
    #    predicts_by_expert = predicts_by_expert[torch.randperm(len(predicts_by_expert))]
    #    predicts_by_expert = predicts_by_expert[0:2000]
    #S = torch.linalg.svdvals(predicts_by_expert)
    #print(U.shape, S.shape, Vh.shape)
    #print(S[0], S[-1])


    # Convert unwanted predicts to zero if label present
    if labels is not None:
        label_mask = torch.zeros(predicts.shape).cuda()
        activate_expert = label2expert[label]
        label_mask[idx,:,:,activate_expert] = 1
        
        predicts = predicts * label_mask
    
    if use_flat:
        gates = gates.view(B,L,-1)
        predicts_merged = torch.einsum('blhn,bln->blh', predicts, gates)
    else:
        predicts_merged = torch.einsum('blhn,bn->blh', predicts, gates)
    
    # Additional MOE loss
    importance = gates.sum(0)
    loss = self.cv_squared(importance) + self.cv_squared(load)
    loss *= 1e-2
    
    if printing:
        print(gates[0])

    return predicts_merged, loss, gates
    #return predicts_merged, loss, S
