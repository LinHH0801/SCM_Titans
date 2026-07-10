import torch
import torch.nn as nn
# from sympy.solvers.diophantine.diophantine import Linear
from torch.nn import functional as F
from thop import profile
from models.pvtv2 import ourpvt_v2_b1,ourpvt_v2_b2
# from timm.models.layers import DropPath
# from functools import partial
# from models.vtitans.titans_pytorch import NeuralMemory
# from models.vtitans.titans_pytorch.mac_transformer import GEGLU, SegmentedAttention, flex_attention, create_mac_block_mask
# from models.utils.func import exists, default
# from models.vtitans.titans_block import TitansBlock
import collections
from itertools import repeat
from typing import Sequence
import math
# from mamba_ssm import Mamba

from models.vtitans.vtitans import VTitans2
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class SA(nn.Module):
    def __init__(self, in_dim):
        super(SA, self).__init__()

        self.chanel_in = in_dim
        self.query = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        # self.value = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.GAP1 = nn.AdaptiveAvgPool2d(8)
        self.GAP2 = nn.AdaptiveAvgPool2d(8)

    def forward(self, x,y):
        m_batchsize, C, height, width = x.size()
        query_c = self.GAP1(self.query(y))
        key_c = self.GAP2(self.key(x))

        query_c = query_c.view(m_batchsize, -1, width//4 * height//4).permute(0, 2, 1)
        key_c = key_c.view(m_batchsize, -1, width//4 * height//4)

        energy_c = torch.bmm(query_c, key_c)
        attention_c = self.softmax(energy_c)
        # out_c = torch.bmm(value_c, attention_c.permute(0, 2, 1))
        # out = out_c.view(m_batchsize, C, height, width)
        return attention_c



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
#
#
# class MeasureFusion(nn.Module):
#     def __init__(self, in_channel,out_channel):
#         super(MeasureFusion, self).__init__()
#
#         self.conv = BasicConv2d(in_channel, out_channel, 3, padding=1)
#     def forward(self, x1,x2):
#
#         x = self.conv(torch.abs(x1-x2))
#         return x



class Backbone(nn.Module):
    def __init__(self, pretrained_path=r'E:\Lp_8\WUSU\models\pvt_v2_b2.pth'):
        super(Backbone, self).__init__()
        self.backbone = ourpvt_v2_b2()  # [64, 128, 320, 512]
        save_model = torch.load(pretrained_path, weights_only=True)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.Conv_o = nn.Conv2d(4, 3, 3, padding=1)
        self.Translayer0 = BasicConv2d(64, 64, 3, padding=1)
        self.Translayer1 = BasicConv2d(128, 64, 3, padding=1)
        self.Translayer2 = BasicConv2d(320, 64, 3, padding=1)
        self.Translayer3 = BasicConv2d(512, 64, 3, padding=1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        self.conv = self._make_layer(ResBlock, 64 * 4,64 * 4, 1, stride=1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x1):
        if x1.shape[1] == 4:
            x1 = x1[:, :3, :, :]
        # x1 = self.Conv_o(x1)  # 丢弃alpha通道
        x1 = self.backbone(x1)
        x1_1 = self.mp1(self.Translayer0(x1[0]))
        x1_2 = self.Translayer1(x1[1])
        x1_3 = self.up1(self.Translayer2(x1[2]))
        x1_4 = self.up2(self.Translayer3(x1[3]))
        x1_4 = self.conv(torch.cat([x1_1, x1_2, x1_3, x1_4], dim=1))

        return x1_4

# class MambaBlock_Temporal(nn.Module):
#     def __init__(self, hidden_dim, d_state=128, d_conv=4, expand=2, channel_first=True,downsample_ratio=2):
#         super().__init__()
#         self.channel_first = channel_first
#         self.mamba = Mamba(
#             d_model=hidden_dim,
#             d_state=d_state,
#             d_conv=d_conv,
#             expand=expand,
#         )
#         self.downsample_ratio = downsample_ratio
#         self.linear1 = nn.Linear(hidden_dim, hidden_dim//4)
#         self.linear2 = nn.Linear(hidden_dim//4, hidden_dim)
#         self.LN = nn.LayerNorm(512)
#         self.norm = nn.LayerNorm(hidden_dim * downsample_ratio ** 2)
#         self.reduction = nn.Linear(hidden_dim * downsample_ratio ** 2, hidden_dim)
#         self.GAP = nn.AdaptiveAvgPool2d(8)
#
#         self.pool = nn.MaxPool1d(4)
#         # self.conv = nn.Conv2d(512,256,kernel_size=3,padding=1)
#         self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
#
#     def forward(self, x1,x2,x3,x4,a1,a2,a3,a4):
#
#
#         x1 = self.GAP(x1)
#         x2 = self.GAP(x2)
#         x3 = self.GAP(x3)
#         x4 = self.GAP(x4)
#
#         m_batchsize,channel,width,height = x1.size()
#
#         x1_a = torch.bmm(x1.view(m_batchsize, -1, width * height), a1.permute(0, 2, 1))
#         x2_a = torch.bmm(x2.view(m_batchsize, -1, width * height), a2.permute(0, 2, 1))
#         x3_a = torch.bmm(x3.view(m_batchsize, -1, width * height), a3.permute(0, 2, 1))
#         x4_a = torch.bmm(x4.view(m_batchsize, -1, width * height), a3.permute(0, 2, 1))
#         x1 = x1_a.view(m_batchsize, channel, height, width)
#         x2 = x2_a.view(m_batchsize, channel, height, width)
#         x3 = x3_a.view(m_batchsize, channel, height, width)
#         x4 = x4_a.view(m_batchsize, channel, height, width)
#
#         x1 = x1.flatten(2).permute(0, 2, 1)
#         x2 = x2.flatten(2).permute(0, 2, 1)
#         x3 = x3.flatten(2).permute(0, 2, 1)
#         x4 = x4.flatten(2).permute(0, 2, 1)
#
#         x = torch.cat([x1,x2,x3,x4],dim=1)
#         x = self.mamba(x)
#         x = x.permute(0, 2, 1)
#         x = self.pool(x)
#         x = x.view(m_batchsize, channel, height, width)
#         x = self.up((x))
#
#         return x

class SptialCNN(nn.Module):
    def __init__(self, ):
        super(SptialCNN, self).__init__()
        self.backbone = Backbone()  # [64, 128, 320, 512]
        self.ln1 = nn.Linear(256,512)
        self.ln2 = nn.Linear(256,512)
        self.SA = SA(512)

        self.vitan = VTitans2(
        in_channels=64 * 4,
        img_size=64,
        patch_size=1,
        embed_dims=64 * 4,
        out_indices=-1,
        depth=1,
        final_norm=True,
        neural_memory_interval=3,
        init_values=1,
        drop_path_rate=0.1,
        pretrained=r"E:\Lp_10\ChangeTitans-main\vtitans_in1k.pth" ) # ← 直接传入

        # self.SSM_Temporal = MambaBlock_Temporal(hidden_dim=512, d_state=64)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.cov_diff1 = self._make_layer(ResBlock, 64 * 4 , 64*2, 1, stride=1)
        self.cov_diff = self._make_layer(ResBlock, 64 * 4, 64*2, 1, stride=1)
        self.cov_ss = self._make_layer(ResBlock, 64 * 4, 64, 1, stride=1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_list):

        features1 = self.backbone(x_list[0])
        features2 = self.backbone(x_list[1])
        features3 = self.backbone(x_list[2])
        print(features1.shape)
        features1, features2, features3 = self.vitan(features1, features2, features3)
        print(features1.shape)
        diff1 = self.cov_diff(torch.abs(features1 - features2))
        diff2 = self.cov_diff(torch.abs(features2 - features3))
        diff = self.cov_diff1(torch.cat([diff1, diff2],dim=1))

        features1 = self.cov_ss(features1)
        features2 = self.cov_ss(features2)
        features3 = self.cov_ss(features3)

        # features1 = self.cov_ss(features1)
        # features2 = self.cov_ss(features2)
        # features3 = self.cov_ss(features3)

        return features1,features2,features3,diff


class CSTMNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(CSTMNet, self).__init__()
        self.Sptial = SptialCNN()
        self.conv_cat2 = self._make_layer(ResBlock, 128 * 2, 512, 1, stride=1)
        self.SA = SA(512)

        self.classifier_ss1 = nn.Conv2d(64, 12, kernel_size=3, padding=1)
        self.classifier_ss2 = nn.Conv2d(64, 12, kernel_size=3, padding=1)
        self.classifier_ss3 = nn.Conv2d(64, 12, kernel_size=3, padding=1)

        self.classifier_bcd = nn.Conv2d(64*2, 4, kernel_size=3, padding=1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_list):
        x_size = x_list[0].size()

        features1,features2,features3,diff = self.Sptial(x_list)
        semantic_moments = []
        semantic_moments1 = F.interpolate(self.classifier_ss1(features1), x_size[2:], mode='bilinear')
        semantic_moments.append(semantic_moments1)
        semantic_moments2 = F.interpolate(self.classifier_ss1(features2), x_size[2:], mode='bilinear')
        semantic_moments.append(semantic_moments2)
        semantic_moments3 = F.interpolate(self.classifier_ss1(features3), x_size[2:], mode='bilinear')
        semantic_moments.append(semantic_moments3)

        change_binary = F.interpolate(self.classifier_bcd(diff), x_size[2:], mode='bilinear')

        return semantic_moments,change_binary


if __name__ == '__main__':

    x1 = torch.randn(1, 3, 256, 256).cuda()

    model =  CSTMNet(num_classes=2).cuda()
    change,change_diff = model([x1, x1, x1])
    print(change[0].shape)
    print(change_diff.shape)



    # def set_profiling_mode(module, mode=True):
    #     if hasattr(module, 'profiling_mode'):
    #         module.profiling_mode = mode
    #     for child in module.children():
    #         set_profiling_mode(child, mode)
    #
    #
    # # 使用前
    # set_profiling_mode(model, True)
    # flops, params = profile(model, inputs=(x1, x1, x1))
    # set_profiling_mode(model, False)
    #
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')