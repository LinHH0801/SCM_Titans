import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.func import default


def build_fusion_module(module_type: str, dim=192, blocks=3):
    """

    Args:
        module_type: TSCBAM / TSCBAMSub / TSCBAMConv / FHD
        dim: input dim
        blocks: block num of TSCBAM

    Returns: FusionModuleList (nn.ModuleList)

    """
    module_type = default(module_type, "TSCBAM")
    if module_type == "TSCBAM":
        module = nn.ModuleList([
            TSCBAM(dim, depth=blocks),
            TSCBAM(dim, depth=blocks),
            TSCBAM(dim, depth=blocks),
            TSCBAM(dim, depth=blocks),
        ])
    elif module_type == "TSCBAMSub":
        module = nn.ModuleList([
            TSCBAMSub(dim, depth=blocks),
            TSCBAMSub(dim, depth=blocks),
            TSCBAMSub(dim, depth=blocks),
            TSCBAMSub(dim, depth=blocks),
        ])
    elif module_type == "TSCBAMConv":
        module = nn.ModuleList([
            TSCBAMConv(dim, depth=blocks),
            TSCBAMConv(dim, depth=blocks),
            TSCBAMConv(dim, depth=blocks),
            TSCBAMConv(dim, depth=blocks),
        ])
    elif module_type == "FHD":
        module = nn.ModuleList([
            FHDModule(dim),
            FHDModule(dim),
            FHDModule(dim),
            FHDModule(dim),
        ])
    else:
        raise NotImplementedError(f"Unimplemented module type: {module_type}")
    return module


class TSCBAM(nn.Module):
    def __init__(self, dim, depth=1):
        super().__init__()
        self.layers = nn.Sequential(*[TSCBAMBlock(dim) for _ in range(depth)])

    def forward(self, x1, x2):
        for layer in self.layers:
            x1, x2 = layer(x1, x2)
        return x1 + x2


class TSCBAMSub(nn.Module):
    def __init__(self, dim, depth=1):
        super().__init__()
        self.layers = nn.Sequential(*[TSCBAMBlock(dim) for _ in range(depth)])

    def forward(self, x1, x2):
        for layer in self.layers:
            x1, x2 = layer(x1, x2)
        return x1 - x2


class TSCBAMConv(nn.Module):
    def __init__(self, dim, depth=1):
        super().__init__()
        self.layers = nn.Sequential(*[TSCBAMBlock(dim) for _ in range(depth)])
        self.reduce_channel = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True)

    def forward(self, x1, x2):
        for layer in self.layers:
            x1, x2 = layer(x1, x2)
        return self.reduce_channel(torch.concat([x1, x2], dim=1))


class TSCBAMBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv1 = Conv(dim, dim, 3, 1, 1)
        self.conv21 = nn.Sequential(ConvRelu(dim, dim, 1, 1, 0), Conv(dim, dim, 1, 1, 0))
        self.conv22 = nn.Sequential(ConvRelu(dim, dim, 1, 1, 0), Conv(dim, dim, 1, 1, 0))
        self.conv31 = nn.Sequential(ConvRelu(2, 16, 3, 1, 1), Conv(16, 1, 3, 1, 1))
        self.conv32 = nn.Sequential(ConvRelu(2, 16, 3, 1, 1), Conv(16, 1, 3, 1, 1))
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)
        c1 = torch.sigmoid(self.conv21(F.adaptive_avg_pool2d(x1, output_size=(1, 1))) + self.conv21(
            F.adaptive_max_pool2d(x1, output_size=(1, 1))))
        c2 = torch.sigmoid(self.conv22(F.adaptive_avg_pool2d(x2, output_size=(1, 1))) + self.conv22(
            F.adaptive_max_pool2d(x2, output_size=(1, 1))))
        x1 = x1 * c2
        x2 = x2 * c1
        s1 = torch.sigmoid(
            self.conv31(torch.cat([torch.mean(x1, dim=1, keepdim=True), torch.max(x1, dim=1, keepdim=True)[0]], dim=1)))
        s2 = torch.sigmoid(
            self.conv32(torch.cat([torch.mean(x2, dim=1, keepdim=True), torch.max(x2, dim=1, keepdim=True)[0]], dim=1)))
        x1 = self.norm1(x1 * s2)
        x2 = self.norm2(x2 * s1)
        return x1, x2


class Conv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('relu', nn.ReLU())
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class FHDModule(nn.Module):
    """
        @article{pei2022feature,
          title={Feature Hierarchical Differentiation for Remote Sensing Image Change Detection},
          author={Pei, Gensheng and Zhang, Lulu},
          journal={IEEE Geoscience and Remote Sensing Letters},
          year={2022},
          publisher={IEEE}
        }
    """

    def __init__(self, channels=64, r=4):
        super(FHDModule, self).__init__()
        inter_channels = int(channels // r)

        # -------------------------------------   HD   -------------------------------------#
        # local attention
        la_conv1 = nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        la_ln1 = nn.BatchNorm2d(inter_channels)
        la_act1 = nn.ReLU()
        la_conv2 = nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        la_ln2 = nn.BatchNorm2d(channels)
        la_layers = [la_conv1, la_ln1, la_act1, la_conv2, la_ln2]
        self.la_layers = nn.Sequential(*la_layers)
        # global attention
        aap = nn.AdaptiveAvgPool2d(1)
        ga_conv1 = nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        ga_ln1 = nn.BatchNorm2d(inter_channels)
        ga_act1 = nn.ReLU()
        ga_conv2 = nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        ga_ln2 = nn.BatchNorm2d(channels)
        ga_layers = [aap, ga_conv1, ga_ln1, ga_act1, ga_conv2, ga_ln2]
        self.ga_layers = nn.Sequential(*ga_layers)

        self.sigmoid = nn.Sigmoid()
        # ----------------------------------------------------------------------------------#
        # -------------------------------------   TSA   ------------------------------------#
        self.tsa_bm = TSA(channels=channels, inter_channels=inter_channels)
        self.tsa_fm = TSA(channels=channels, inter_channels=inter_channels)
        # ----------------------------------------------------------------------------------#
        # -------------------------------------   TSC   ------------------------------------#
        self.tsc_bm = TSC(scale=1)
        self.tsc_fm = TSC(scale=1)
        # ----------------------------------------------------------------------------------#
        # -------------------------------------   TSM   ------------------------------------#
        self.tsm_bm = nn.Conv2d(channels, 2, kernel_size=1)
        self.tsm_fm = nn.Conv2d(channels, 2, kernel_size=1)
        # ----------------------------------------------------------------------------------#

    def forward(self, bm, fm):
        bm_pred = self.tsm_bm(bm)
        fm_pred = self.tsm_fm(fm)
        bm_context = self.tsc_bm(bm, bm_pred)
        fm_context = self.tsc_fm(fm, fm_pred)
        bm_agg = self.tsa_bm(bm, bm_context)
        fm_agg = self.tsa_fm(fm, fm_context)

        agg = bm_agg + fm_agg
        agg_loc = self.la_layers(agg)
        agg_glo = self.ga_layers(agg)
        agg_lg = agg_loc + agg_glo
        w = self.sigmoid(agg_lg)

        diff = 2 * bm_agg * w + 2 * fm_agg * (1 - w)
        return diff


# (TSC) Time-Specific Context
class TSC(nn.Module):
    def __init__(self, scale):
        super(TSC, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        # b, c, h, w = feats.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        # [batch_size, height*width, num_classes]
        feats = feats.permute(0, 2, 1)
        # [batch_size, channels, height*width]
        probs = F.softmax(self.scale * probs, dim=2)
        # [batch_size, channels, num_classes]
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context


class SelfAttentionBlock(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, channels,
                 out_channels, key_query_num_convs, value_out_num_convs,
                 query_downsample=None, key_downsample=None):
        super(SelfAttentionBlock, self).__init__()
        self.key_in_channels = key_in_channels
        self.query_in_channels = query_in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.key_project = self.build_project(
            key_in_channels,
            channels,
            num_convs=key_query_num_convs)
        self.query_project = self.build_project(
            query_in_channels,
            channels,
            num_convs=key_query_num_convs)
        self.value_project = self.build_project(
            key_in_channels,
            channels,
            num_convs=value_out_num_convs)
        self.out_project = self.build_project(
            channels,
            out_channels,
            num_convs=value_out_num_convs)

        self.query_downsample = query_downsample
        self.key_downsample = key_downsample

    @staticmethod
    def build_project(in_channels, channels, num_convs):
        """Build projection layer for key/query/value/out."""
        convs = [
            nn.Conv2d(in_channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        ]
        for _ in range(num_convs - 1):
            convs = convs + [
                nn.Conv2d(channels, channels, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            ]
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query_feats, key_feats):
        """Forward function."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        if self.query_downsample is not None:
            query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1)

        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(query, key)
        sim_map = (self.channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        context = self.out_project(context)
        return context


# (TSA) Time-Specific Aggregation
class TSA(SelfAttentionBlock):
    def __init__(self, channels, inter_channels):
        super(TSA, self).__init__(
            key_in_channels=channels,
            query_in_channels=channels,
            channels=inter_channels,
            out_channels=channels,
            key_query_num_convs=2,
            value_out_num_convs=1)
        self.bottleneck = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, query_feats, key_feats):
        """Forward function."""
        context = super(TSA, self).forward(query_feats, key_feats)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        return output
