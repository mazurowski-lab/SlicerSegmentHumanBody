import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sam2_annotation_tool.sam2.modeling.sam2_utils import DropPath, get_clones, LayerNorm2d

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze: Global Average Pooling
        squeeze = self.global_avg_pool(x)  # (B, C, H, W) -> (B, C, 1, 1)
        
        # Excitation: Fully Connected layers
        excitation = self.fc1(squeeze)  # (B, C, 1, 1) -> (B, C//reduction_ratio, 1, 1)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)  # (B, C//reduction_ratio, 1, 1) -> (B, C, 1, 1)
        excitation = self.sigmoid(excitation)
        
        # Scale: Multiply the input feature map by the excitation weights
        scale = excitation * x  # (B, C, 1, 1) * (B, C, H, W) -> (B, C, H, W)
        
        return scale

class SEFusionAdd(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEFusionAdd, self).__init__()
        self.se_block1 = SEBlock(in_channels, reduction_ratio)
        self.se_block2 = SEBlock(in_channels, reduction_ratio)

    def forward(self, feature1, feature2):
        # Apply SE block to both feature maps
        modulated_feature1 = self.se_block1(feature1)
        modulated_feature2 = self.se_block2(feature2)
        
        # Fuse the feature maps by element-wise addition
        fused_features = modulated_feature1 + modulated_feature2
        
        return fused_features

class SEFusionCat(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEFusionCat, self).__init__()
        self.se_block = SEBlock(in_channels * 2, reduction_ratio)  # We double the in_channels for concatenation
        self.conv_reduce = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)  # To reduce back to in_channels

    def forward(self, feature1, feature2):
        # Concatenate along the channel dimension
        concatenated_features = torch.cat((feature1, feature2), dim=1)  # (B, 2C, H, W)
        
        # Apply SE block
        modulated_features = self.se_block(concatenated_features)  # Still (B, 2C, H, W)
        
        # Reduce the channels back to original size
        fused_features = self.conv_reduce(modulated_features)  # (B, C, H, W)
        
        return fused_features

class WeightedAdditionFusion(nn.Module):
    def __init__(self, cond_ratio):
        super(WeightedAdditionFusion, self).__init__()
        # Learnable weights initialized to 0.5 for equal contribution initially
        self.weight1 = nn.Parameter(torch.tensor(cond_ratio))
        self.weight2 = nn.Parameter(torch.tensor(1 - cond_ratio))

    def forward(self, feature_cond, feature):
        w1 = self.weight1
        w2 = self.weight2
        
        # Perform weighted addition
        fused_features = w1 * feature_cond + w2 * feature
        
        return fused_features

# Lightly adapted from ConvNext (https://github.com/facebookresearch/ConvNeXt)
class CXBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class Fuser(nn.Module):
    def __init__(self, layer, num_layers, dim=None, input_projection=False):
        super().__init__()
        self.proj = nn.Identity()
        self.layers = get_clones(layer, num_layers)

        if input_projection:
            assert dim is not None
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x_cond, x):
        # normally x: (N, C, H, W)
        x = x + x_cond
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x

class LightweightFuser(nn.Module):
    def __init__(self, input_channels, fusion_type='add'):
        super(LightweightFuser, self).__init__()
        self.fusion_type = fusion_type
        if fusion_type == 'concat':
            self.fused_channels = 2 * input_channels
        else:
            self.fused_channels = input_channels
        
        # Simple 1x1 convolution for reducing dimensions after fusion
        self.conv = nn.Conv2d(self.fused_channels, input_channels, kernel_size=1)
        self.relu = nn.ReLU()
    
    def forward(self, feat1, feat2):
        if self.fusion_type == 'add':
            fused = feat1 + feat2
        elif self.fusion_type == 'multiply':
            fused = feat1 * feat2
        elif self.fusion_type == 'concat':
            fused = torch.cat([feat1, feat2], dim=1)  # Concatenating along channel dimension
        else:
            raise ValueError("Invalid fusion type. Choose from 'add', 'multiply', or 'concat'.")
        
        # Apply a 1x1 convolution to the fused features
        fused = self.conv(fused)
        return fused
