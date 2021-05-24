""" Deep Layer Aggregation Backbone
"""
import os
import math
import torch
import torch.nn as nn
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import DFConv2d
from maskrcnn_benchmark.layers import FrozenBatchNorm2d

from torchvision.models.utils import load_state_dict_from_url

from timm.models.layers import SelectAdaptivePool2d

from maskrcnn_benchmark.utils.registry import Registry
from maskrcnn_benchmark.utils.model_serialization import load_state_dict

model_urls = {
    'dla34': 'http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth',
    'dla46_c': 'http://dl.yf.io/dla/models/imagenet/dla46_c-2bfd52c3.pth',
    'dla46x_c': 'http://dl.yf.io/dla/models/imagenet/dla46x_c-d761bae7.pth',
    'dla60': 'http://dl.yf.io/dla/models/imagenet/dla60-24839fc4.pth',
    'dla102': 'http://dl.yf.io/dla/models/imagenet/dla102-d94d9790.pth',
    'dla169': 'http://dl.yf.io/dla/models/imagenet/dla169-0914e092.pth',
    'dla60_res2net':
        'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net_dla60_4s-d88db7f9.pth',
}


class DlaBasic(nn.Module):
    """DLA Basic"""
    def __init__(self, inplanes, planes, stride=1, dilation=1, batch_norm=FrozenBatchNorm2d, **_):
        super(DlaBasic, self).__init__()
        self.conv1 = Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = batch_norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = batch_norm(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class DlaBottleneck(nn.Module):
    """DLA/DLA-X Bottleneck"""
    expansion = 2

    def __init__(self, inplanes, outplanes, stride=1, dilation=1,
                 cardinality=1, base_width=64, batch_norm=FrozenBatchNorm2d,
                 with_dcn=False):
        super(DlaBottleneck, self).__init__()
        self.stride = stride
        mid_planes = int(math.floor(outplanes * (base_width / 64)) * cardinality)
        mid_planes = mid_planes // self.expansion

        self.conv1 = Conv2d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = batch_norm(mid_planes)
        if with_dcn:
            self.conv2 = DFConv2d(mid_planes, mid_planes, with_modulated_dcn=False,
                                  kernel_size=3, stride=stride, bias=False,
                                  dilation=dilation, groups=cardinality)
        else:
            self.conv2 = Conv2d(
                mid_planes, mid_planes, kernel_size=3, stride=stride, padding=dilation,
                bias=False, dilation=dilation, groups=cardinality)
        self.bn2 = batch_norm(mid_planes)
        self.conv3 = Conv2d(mid_planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = batch_norm(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class DlaBottle2neck(nn.Module):
    """ Res2Net/Res2NeXT DLA Bottleneck
    Adapted from https://github.com/gasvn/Res2Net/blob/master/dla.py
    """
    expansion = 2

    def __init__(self, inplanes, outplanes, stride=1, dilation=1, scale=4,
                 cardinality=8, base_width=4, batch_norm=FrozenBatchNorm2d):
        super(DlaBottle2neck, self).__init__()
        self.is_first = stride > 1
        self.scale = scale
        mid_planes = int(math.floor(outplanes * (base_width / 64)) * cardinality)
        mid_planes = mid_planes // self.expansion
        self.width = mid_planes

        self.conv1 = Conv2d(inplanes, mid_planes * scale, kernel_size=1, bias=False)
        self.bn1 = batch_norm(mid_planes * scale)

        num_scale_convs = max(1, scale - 1)
        convs = []
        bns = []
        for _ in range(num_scale_convs):
            convs.append(Conv2d(
                mid_planes, mid_planes, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation, groups=cardinality, bias=False))
            bns.append(batch_norm(mid_planes))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        if self.is_first:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        self.conv3 = Conv2d(mid_planes * scale, outplanes, kernel_size=1, bias=False)
        self.bn3 = batch_norm(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        spo = []
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            sp = spx[i] if i == 0 or self.is_first else sp + spx[i]
            sp = conv(sp)
            sp = bn(sp)
            sp = self.relu(sp)
            spo.append(sp)
        if self.scale > 1 :
            spo.append(self.pool(spx[-1]) if self.is_first else spx[-1])
        out = torch.cat(spo, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class DlaRoot(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual, batch_norm=FrozenBatchNorm2d):
        super(DlaRoot, self).__init__()
        self.conv = Conv2d(
            in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = batch_norm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class DlaTree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 dilation=1, cardinality=1, base_width=64,
                 level_root=False, root_dim=0, root_kernel_size=1, root_residual=False,
                 batch_norm=FrozenBatchNorm2d, with_dcn=False):
        super(DlaTree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        cargs = dict(dilation=dilation, cardinality=cardinality, base_width=base_width, batch_norm=batch_norm, with_dcn=with_dcn)
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, **cargs)
            self.tree2 = block(out_channels, out_channels, 1, **cargs)
        else:
            cargs.update(dict(root_kernel_size=root_kernel_size, root_residual=root_residual))
            self.tree1 = DlaTree(
                levels - 1, block, in_channels, out_channels, stride, root_dim=0, **cargs)
            self.tree2 = DlaTree(
                levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels, **cargs)
        if levels == 1:
            self.root = DlaRoot(root_dim, out_channels, root_kernel_size, root_residual, batch_norm=batch_norm)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = nn.MaxPool2d(stride, stride=stride) if stride > 1 else None
        self.project = None
        if in_channels != out_channels:
            self.project = nn.Sequential(
                Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                batch_norm(out_channels)
            )
        self.levels = levels

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000, in_chans=3, cardinality=1, base_width=64,
                 block=DlaBottle2neck, residual_root=False, linear_root=False, batch_norm=FrozenBatchNorm2d,
                 drop_rate=0.0, global_pool='avg', feature_only=True, dcn_config=(False,)):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.cardinality = cardinality
        self.base_width = base_width
        self.drop_rate = drop_rate

        # check whether deformable conv config is right
        if len(dcn_config) != 6:
            raise ValueError("Deformable configuration is not correct, "
                             "every level should specifcy a configuration.")

        self.base_layer = nn.Sequential(
            Conv2d(in_chans, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            batch_norm(channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0], batch_norm=batch_norm)
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2, batch_norm=batch_norm)
        cargs = dict(cardinality=cardinality, base_width=base_width, root_residual=residual_root, batch_norm=batch_norm)
        self.level2 = DlaTree(levels[2], block, channels[1], channels[2], 2, level_root=False,
                              with_dcn=dcn_config[2], **cargs)
        self.level3 = DlaTree(levels[3], block, channels[2], channels[3], 2, level_root=True,
                              with_dcn=dcn_config[3], **cargs)
        self.level4 = DlaTree(levels[4], block, channels[3], channels[4], 2, level_root=True,
                              with_dcn=dcn_config[4], **cargs)
        self.level5 = DlaTree(levels[5], block, channels[4], channels[5], 2, level_root=True,
                              with_dcn=dcn_config[5], **cargs)

        if not feature_only:
            self.num_features = channels[-1]
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.fc = nn.Conv2d(self.num_features * self.global_pool.feat_mult(), num_classes, 1, bias=True)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1, batch_norm=FrozenBatchNorm2d):
        modules = []
        for i in range(convs):
            modules.extend([
                Conv2d(inplanes, planes, kernel_size=3, stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                batch_norm(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        features = []
        x = self.base_layer(x)
        x0 = self.level0(x)
        x1 = self.level1(x0)
        x2 = self.level2(x1)
        x3 = self.level3(x2)
        x4 = self.level4(x3)
        x5 = self.level5(x4)

        features.append(x2)
        features.append(x3)
        features.append(x4)
        features.append(x5)

        return features


def dla_34(dcn_config, feature_only=True, batch_norm=FrozenBatchNorm2d):
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512],
                block=DlaBasic,
                batch_norm=batch_norm,
                feature_only=feature_only,
                dcn_config=dcn_config)
    return model


def dla_46_c(dcn_config, feature_only=True, batch_norm=FrozenBatchNorm2d):
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 64, 128, 256],
                block=DlaBottleneck,
                batch_norm=batch_norm,
                feature_only=feature_only,
                dcn_config=dcn_config)
    return model


def dla_46_xc(dcn_config, feature_only=True, batch_norm=FrozenBatchNorm2d):
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 64, 128, 256],
                block=DlaBottleneck,
                cardinality=32,
                base_width=4,
                batch_norm=batch_norm,
                feature_only=feature_only,
                dcn_config=dcn_config)
    return model


def dla_60(dcn_config, feature_only=True, batch_norm=FrozenBatchNorm2d):
    model = DLA([1, 1, 1, 2, 3, 1], [16, 32, 128, 256, 512, 1024],
                block=DlaBottleneck,
                batch_norm=batch_norm,
                feature_only=feature_only,
                dcn_config=dcn_config)
    return model


def dla60_res2net(dcn_config, feature_only=True, batch_norm=FrozenBatchNorm2d):
    model = DLA(levels=(1, 1, 1, 2, 3, 1),
                channels=(16, 32, 128, 256, 512, 1024),
                block=DlaBottle2neck,
                cardinality=1,
                base_width=28,
                batch_norm=batch_norm,
                feature_only=feature_only,
                dcn_config=dcn_config)
    return model


def dla_102(dcn_config, feature_only=True, batch_norm=FrozenBatchNorm2d):
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=DlaBottleneck,
                residual_root=True,
                batch_norm=batch_norm,
                feature_only=feature_only,
                dcn_config=dcn_config)
    return model


def dla_169(dcn_config, feature_only=True, batch_norm=FrozenBatchNorm2d):
    model = DLA([1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024],
                block=DlaBottleneck,
                residual_root=True,
                batch_norm=batch_norm,
                feature_only=feature_only,
                dcn_config=dcn_config)
    return model


BACKBONE = Registry({
    "DLA-34-FPN": dla_34,
    "DLA-46-C-FPN": dla_46_c,
    "DLA-46-XC-FPN": dla_46_xc,
    "DLA-60-FPN": dla_60,
    "DLA-60-RES2NET-FPN": dla60_res2net,
    "DLA-102-FPN": dla_102,
    "DLA-169-FPN": dla_169
})

BACKBONE_ARCH = {
    "DLA-34-FPN": "dla34",
    "DLA-46-C-FPN": "dla_46_c",
    "DLA-46-XC-FPN": "dla_46_xc",
    "DLA-60-FPN": "dla_60",
    "DLA-60-RES2NET-FPN": "dla60_res2net",
    "DLA-102-FPN": "dla_102",
    "DLA-169-FPN": "dla_169"
}


def dla(cfg):
    model = BACKBONE[cfg.MODEL.BACKBONE.CONV_BODY](cfg.MODEL.DLA.STAGE_WITH_DCN)

    # Load the ImageNet pretrained backbone if no valid pre-trained model weights are given
    if not os.path.exists(cfg.MODEL.WEIGHT):
        state_dict = load_state_dict_from_url(model_urls[BACKBONE_ARCH[cfg.MODEL.BACKBONE.CONV_BODY]],
                                              progress=True)
        load_state_dict(model, state_dict)

    return model


