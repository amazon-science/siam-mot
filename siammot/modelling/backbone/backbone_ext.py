from torch import nn
from collections import OrderedDict

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
import maskrcnn_benchmark.modeling.backbone.fpn as fpn_module

from .dla import dla


@registry.BACKBONES.register("DLA-34-FPN")
@registry.BACKBONES.register("DLA-46-C-FPN")
@registry.BACKBONES.register("DLA-60-FPN")
@registry.BACKBONES.register("DLA-102-FPN")
@registry.BACKBONES.register("DLA-169-FPN")
def build_dla_fpn_backbone(cfg):
    body = dla(cfg)
    in_channels_stage2 = cfg.MODEL.DLA.DLA_STAGE2_OUT_CHANNELS
    in_channels_stage3 = cfg.MODEL.DLA.DLA_STAGE3_OUT_CHANNELS
    in_channels_stage4 = cfg.MODEL.DLA.DLA_STAGE4_OUT_CHANNELS
    in_channels_stage5 = cfg.MODEL.DLA.DLA_STAGE5_OUT_CHANNELS
    out_channels = cfg.MODEL.DLA.BACKBONE_OUT_CHANNELS

    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage3,
            in_channels_stage4,
            in_channels_stage5
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)

