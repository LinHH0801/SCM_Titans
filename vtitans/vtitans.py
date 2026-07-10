from typing import Sequence
import math
import collections
from itertools import repeat

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from models.vtitans.titans_block import TitansBlock

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, 2))


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The type of convolution
            to generate patch embedding. Default: "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: 16.
        padding (int | tuple | string): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only works when `dynamic_size`
            is False. Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 kernel_size=16,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None):
        super().__init__()

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adaptive_padding = None
        padding = to_2tuple(padding)

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = nn.BatchNorm2d(embed_dims)
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # e.g. when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adaptive_padding:
                pad_h, pad_w = self.adaptive_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

            - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
            - out_size (tuple[int]): Spatial shape of x, arrange as
              (out_h, out_w).
        """

        if self.adaptive_padding:
            x = self.adaptive_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class AdaptivePadding(nn.Module):
    """Applies padding adaptively to the input.

    This module can make input get fully covered by filter
    you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad
    zero around input. The "corner"  mode would pad zero
    to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel. Default: 1.
        stride (int | tuple): Stride of the filter. Default: 1.
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1.
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        super().__init__()
        assert padding in ('same', 'corner')

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        """Calculate the padding size of input.

        Args:
            input_shape (:obj:`torch.Size`): arrange as (H, W).

        Returns:
            Tuple[int]: The padding size along the
            original H and W directions
        """
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        """Add padding to `x`

        Args:
            x (Tensor): Input tensor has shape (B, C, H, W).

        Returns:
            Tensor: The tensor with adaptive padding
        """
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ])
        return x


def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=1):
    """Resize pos_embed weights.

    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (H, W).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'bicubic'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.

    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
        '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    dst_weight = F.interpolate(
        src_weight, size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)

    return torch.cat((extra_tokens, dst_weight), dim=1)

class VTitans(nn.Module):
    """
    norm w and norm u
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 embed_dims=256,
                 depth=12,
                 neural_memory_interval=3,
                 gate_attn_output=False,
                 drop_path_rate=0.,
                 chunk_size=64,
                 init_values=None,
                 final_norm=True,
                 interpolate_mode='bicubic',
                 pretrained=None  # ← 新增参数
                 ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_extra_tokens = 0
        self.num_layers = depth
        self.drop_path_rate = drop_path_rate

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True)

        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dims))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList()
        neural_memory_layers = list(range(1, depth, neural_memory_interval))
        for i in range(self.num_layers):
            self.layers.append(TitansBlock(
                dim=embed_dims,
                layer_id=i,
                neural_memory_layers=neural_memory_layers,
                chunk_size=chunk_size,
                init_values=init_values,
                drop_path=dpr[i],
                gate_attn_output=gate_attn_output
            ))

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = nn.LayerNorm(self.embed_dims)

        # ===== 自动加载预训练权重 =====
        if pretrained is not None:
            self._load_pretrained(pretrained)

    def _load_pretrained(self, pretrained_path: str):
        """内部方法：加载预训练权重"""
        # print(f"[INFO] Loading pretrained weights from: {pretrained_path}")
        pretrained_dict = torch.load(pretrained_path,weights_only=True)
        model_dict = self.state_dict()

        # 移除 'backbone.' 前缀（适配你的 checkpoint 格式）
        pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}

        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)

        model_dict.update(temp_dict)
        self.load_state_dict(model_dict)

        # # 打印加载信息
        # print(f"[INFO] Successfully loaded {len(load_key)} keys.")
        # if load_key:
        #     print("[INFO] Loaded keys:")
        #     for key in sorted(load_key):
        #         print(f"  - {key}")
        # if no_load_key:
        #     print(f"[WARNING] Skipped keys (shape mismatch or not in model):")
        #     for key in no_load_key:
        #         print(f"  - {key}")
        # else:
        #     print("[INFO] All pretrained keys matched and loaded!")

    def forward(self, x):

        # patch_resolution = (x.shape[2], x.shape[3])
        # x = x.flatten(2).transpose(1, 2)
        #
        x, patch_resolution = self.patch_embed(x)

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        outs = []
        mem_weight_residual = None
        value_residual = None
        for i, layer in enumerate(self.layers):
            x, mem_weight_residual, value_residual = layer(x, mem_weight_residual, value_residual)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                patch_token = x.reshape(B, *patch_resolution, C)
                patch_token = patch_token.permute(0, 3, 1, 2)
                out = patch_token
                outs.append(out)

        return outs


class VTitans2(nn.Module):
    """
    norm w and norm u
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 embed_dims=256,
                 depth=12,
                 neural_memory_interval=3,
                 gate_attn_output=False,
                 drop_path_rate=0.,
                 chunk_size=64,
                 init_values=None,
                 final_norm=True,
                 interpolate_mode='bicubic',
                 pretrained=None  # ← 新增参数
                 ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_extra_tokens = 0
        self.num_layers = depth
        self.drop_path_rate = drop_path_rate

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True)

        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dims))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        neural_memory_layers = list(range(1, depth, neural_memory_interval))
        for i in range(self.num_layers):
            self.layers1.append(TitansBlock(
                dim=embed_dims,
                layer_id=i,
                neural_memory_layers=neural_memory_layers,
                chunk_size=chunk_size,
                init_values=init_values,
                drop_path=dpr[i],
                gate_attn_output=gate_attn_output
            ))
            self.layers1.append(TitansBlock(
                dim=embed_dims,
                layer_id=i,
                neural_memory_layers=neural_memory_layers,
                chunk_size=chunk_size,
                init_values=init_values,
                drop_path=dpr[i],
                gate_attn_output=gate_attn_output
            ))
            self.layers2.append(TitansBlock(
                dim=embed_dims,
                layer_id=i,
                neural_memory_layers=neural_memory_layers,
                chunk_size=chunk_size,
                init_values=init_values,
                drop_path=dpr[i],
                gate_attn_output=gate_attn_output
            ))
            self.layers2.append(TitansBlock(
                dim=embed_dims,
                layer_id=i,
                neural_memory_layers=neural_memory_layers,
                chunk_size=chunk_size,
                init_values=init_values,
                drop_path=dpr[i],
                gate_attn_output=gate_attn_output
            ))
            self.layers3.append(TitansBlock(
                dim=embed_dims,
                layer_id=i,
                neural_memory_layers=neural_memory_layers,
                chunk_size=chunk_size,
                init_values=init_values,
                drop_path=dpr[i],
                gate_attn_output=gate_attn_output
            ))
            self.layers3.append(TitansBlock(
                dim=embed_dims,
                layer_id=i,
                neural_memory_layers=neural_memory_layers,
                chunk_size=chunk_size,
                init_values=init_values,
                drop_path=dpr[i],
                gate_attn_output=gate_attn_output
            ))

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = nn.LayerNorm(self.embed_dims)

        # ===== 自动加载预训练权重 =====
        if pretrained is not None:
            self._load_pretrained(pretrained)

    def _load_pretrained(self, pretrained_path: str):
        """内部方法：加载预训练权重"""
        # print(f"[INFO] Loading pretrained weights from: {pretrained_path}")
        pretrained_dict = torch.load(pretrained_path,weights_only=True)
        model_dict = self.state_dict()

        # 移除 'backbone.' 前缀（适配你的 checkpoint 格式）
        pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}

        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)

        model_dict.update(temp_dict)
        self.load_state_dict(model_dict)

        # # 打印加载信息
        # print(f"[INFO] Successfully loaded {len(load_key)} keys.")
        # if load_key:
        #     print("[INFO] Loaded keys:")
        #     for key in sorted(load_key):
        #         print(f"  - {key}")
        # if no_load_key:
        #     print(f"[WARNING] Skipped keys (shape mismatch or not in model):")
        #     for key in no_load_key:
        #         print(f"  - {key}")
        # else:
        #     print("[INFO] All pretrained keys matched and loaded!")

    def forward(self, x1,x2,x3):

        # patch_resolution = (x.shape[2], x.shape[3])
        # x = x.flatten(2).transpose(1, 2)
        #
        x1, patch_resolution1 = self.patch_embed(x1)
        x2, patch_resolution2 = self.patch_embed(x2)
        x3, patch_resolution3 = self.patch_embed(x3)

        x1 = x1 + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution1,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)

        x2 = x2 + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution2,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)

        x3 = x3 + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution3,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)


        x1 = self.drop_after_pos(x1)
        x2 = self.drop_after_pos(x2)
        x3 = self.drop_after_pos(x3)


        mem_weight_residual1 = None
        value_residual1 = None
        # mem_weight_residual2 = None
        # value_residual2 = None
        # mem_weight_residual3 = None
        # value_residual3 = None

        for i, layer in enumerate(self.layers1):
            x1, mem_weight_residual1, value_residual1 = layer(x1, mem_weight_residual1, value_residual1)

            x2, mem_weight_residual2, value_residual2 = layer(x2, mem_weight_residual1, value_residual1)

            x3, mem_weight_residual3, value_residual3 = layer(x3, mem_weight_residual2, value_residual2)

            x1, mem_weight_residual_o, value_residual_o = layer(x1, mem_weight_residual3,value_residual3)

            if i == len(self.layers1) - 1 and self.final_norm:
                x1 = self.ln1(x1)
                x2 = self.ln1(x2)
                x3 = self.ln1(x3)

            if i in self.out_indices:
                B, _, C = x1.shape
                patch_token1 = x1.reshape(B, *patch_resolution1, C)
                patch_token2 = x2.reshape(B, *patch_resolution2, C)
                patch_token3 = x3.reshape(B, *patch_resolution3, C)
                patch_token1 = patch_token1.permute(0, 3, 1, 2)
                patch_token2 = patch_token2.permute(0, 3, 1, 2)
                patch_token3 = patch_token3.permute(0, 3, 1, 2)
                out1 = patch_token1
                out2 = patch_token2
                out3 = patch_token3

        return out1,out2,out3

if __name__ == '__main__':
    # x1 = torch.randn(1, 3, 512, 512).cuda()
    x1 = torch.randn(1, 256, 64, 64).cuda()
    model = VTitans2(
        in_channels=256,
        img_size=64,
        patch_size=1,
        embed_dims=256,
        out_indices=-1,
        depth=1,
        final_norm=True,
        neural_memory_interval=3,
        init_values=1,
        drop_path_rate=0.1,
        pretrained=r"E:\Lp_10\ChangeTitans-main\vtitans_in1k.pth"  # ← 直接传入
    ).cuda()

    with torch.no_grad():
        all_change1,all_change2,all_change3 = model(x1,x1,x1)
    print("Output shape:", all_change1.shape)
    print("Output shape:", all_change2.shape)
    print("Output shape:", all_change3.shape)


