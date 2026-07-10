import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_

from vtitans.vtitans_adapter import VTitansAdapter
from FusionModules import build_fusion_module
from vtitans.titans_block import TitansBlock
from utils.func import default


class Block(nn.Module):
    def __init__(self, dim, depth=1, chunk_size=64, dpr=0., init_values=1, gate_attn_output=False):
        super().__init__()
        self.layers = nn.Sequential(*[TitansBlock(dim=dim, layer_id=i, neural_memory_layers=[0, ],
                                                  chunk_size=chunk_size, init_values=init_values, drop_path=dpr,
                                                  gate_attn_output=gate_attn_output
                                                  ) for i in range(depth)])

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        mem_weight_residual, value_residual = None, None
        for layer in self.layers:
            x, mem_weight_residual, value_residual = layer(x, mem_weight_residual, value_residual)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ChangeTitansHead(nn.Module):
    def __init__(self, dim, chunk_size, inner_dim=128, drop_path_rate=0.1, init_values=1, num_blocks=None,
                 gate_attn_output=False):
        super(ChangeTitansHead, self).__init__()
        num_blocks = default(num_blocks, [2, 4, 4, 2])
        dpr = [drop_path_rate, ] * 4

        self.decoder_block_4 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=dim, out_channels=inner_dim),
            Block(dim=inner_dim, depth=num_blocks[0], chunk_size=chunk_size // 4, dpr=dpr[0],
                  init_values=init_values, gate_attn_output=gate_attn_output),
            nn.BatchNorm2d(inner_dim), nn.ReLU()
        )

        self.decoder_block_3 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=dim, out_channels=inner_dim),
            Block(dim=inner_dim, depth=num_blocks[1], chunk_size=chunk_size, dpr=dpr[1],
                  init_values=init_values, gate_attn_output=gate_attn_output),
            nn.BatchNorm2d(inner_dim), nn.ReLU()
        )

        self.decoder_block_2 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=dim, out_channels=inner_dim),
            Block(dim=inner_dim, depth=num_blocks[2], chunk_size=chunk_size * 4, dpr=dpr[2],
                  init_values=init_values, gate_attn_output=gate_attn_output),
            nn.BatchNorm2d(inner_dim), nn.ReLU()
        )

        self.decoder_block_1 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=dim, out_channels=inner_dim),
            Block(dim=inner_dim, depth=num_blocks[3], chunk_size=chunk_size * 16, dpr=dpr[3],
                  init_values=init_values, gate_attn_output=gate_attn_output),
            nn.BatchNorm2d(inner_dim), nn.ReLU()
        )

        # Smooth layer
        self.smooth_layer_3 = ResBlock(in_channels=inner_dim, out_channels=inner_dim, stride=1)
        self.smooth_layer_2 = ResBlock(in_channels=inner_dim, out_channels=inner_dim, stride=1)
        self.smooth_layer_1 = ResBlock(in_channels=inner_dim, out_channels=inner_dim, stride=1)

    @staticmethod
    def _upsample_add(x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, features):
        feat_1, feat_2, feat_3, feat_4 = features

        p4 = self.decoder_block_4(feat_4)

        p3 = self.decoder_block_3(feat_3)
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3(p3)

        p2 = self.decoder_block_2(feat_2)
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2(p2)

        p1 = self.decoder_block_1(feat_1)
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1(p1)

        return p1
class ChangeTitans(nn.Module):
    def __init__(self, img_size=256, chunk_size=256, dim=192, decoder_dim=128, out_channels=1, fusion_type="TSCBAM",
                 num_blocks=None, fusion_blocks=3, gate_attn_output=True):
        super().__init__()
        num_blocks = default(num_blocks, [2, 4, 4, 2])
        assert np.sum(num_blocks) == 12
        interaction_indexes = [
            [0, num_blocks[0] - 1],
            [num_blocks[0], num_blocks[0] + num_blocks[1] - 1],
            [num_blocks[0] + num_blocks[1], num_blocks[0] + num_blocks[1] + num_blocks[2] - 1],
            [num_blocks[0] + num_blocks[1] + num_blocks[2],
             num_blocks[0] + num_blocks[1] + num_blocks[2] + num_blocks[3] - 1]
        ]

        self.backbone = VTitansAdapter(img_size=img_size, patch_size=16, embed_dims=192, depth=12, init_values=1,
                                       neural_memory_interval=3, chunk_size=64, gate_attn_output=False,
                                       drop_path_rate=0.1, conv_inplane=64, n_points=4, deform_num_heads=6,
                                       cffn_ratio=0.25, deform_ratio=1.0, interaction_indexes=interaction_indexes)
        self.fusion_module = build_fusion_module(module_type=fusion_type, dim=dim, blocks=fusion_blocks)
        self.head = ChangeTitansHead(dim, chunk_size, inner_dim=decoder_dim, num_blocks=num_blocks,
                                     gate_attn_output=gate_attn_output)

        # mask of convex upsampling
        self.mask_gen = nn.Sequential(
            nn.Conv2d(decoder_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 16 * 9, 1, padding=0))

        self.clf = nn.Conv2d(in_channels=decoder_dim, out_channels=out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid() if out_channels == 1 else nn.Identity()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def upsample_pred(pred, mask):
        """ Upsample change map [H/4, W/4, C] -> [H, W, C] using convex combination """
        N, C, H, W = pred.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_pred = F.unfold(pred, (3, 3), padding=1)
        up_pred = up_pred.view(N, C, 9, 1, 1, H, W)

        up_pred = torch.sum(mask * up_pred, dim=2)
        up_pred = up_pred.permute(0, 1, 4, 2, 5, 3)
        return up_pred.reshape(N, C, 4 * H, 4 * W)

    def forward(self, x1, x2):
        x1 = x1.contiguous()
        x2 = x2.contiguous()
        b, c, h, w = x1.size()
        feature_list = self.backbone(torch.concat([x1, x2], dim=0))
        print(feature_list[0].shape)
        feature_list = [
            self.fusion_module[0](feature_list[0][:b].to(x1.device), feature_list[0][b:].to(x1.device)),
            self.fusion_module[1](feature_list[1][:b].to(x1.device), feature_list[1][b:].to(x1.device)),
            self.fusion_module[2](feature_list[2][:b].to(x1.device), feature_list[2][b:].to(x1.device)),
            self.fusion_module[3](feature_list[3][:b].to(x1.device), feature_list[3][b:].to(x1.device)),
        ]
        output = self.head(feature_list)

        # scale mask to balance gradients
        mask = .25 * self.mask_gen(output)
        output = self.clf(output)

        output = self.upsample_pred(output, mask)
        output = self.sigmoid(output)
        return output


if __name__ == '__main__':
    x1 = torch.randn(1, 3, 512, 512).cuda()
    # x2 = torch.randn(1, 3, 512, 512).cuda()
    model =  ChangeTitans(img_size=512).cuda()
    all_change1 = model(x1,x1)
    print(all_change1.shape)






