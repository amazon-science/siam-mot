from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.make_layers import make_conv3x3

from .sr_pool import SRPooler


class EMMFeatureExtractor(nn.Module):
    """
    Feature extraction for template and search region is different in this case
    """

    def __init__(self, cfg):
        super(EMMFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.TRACK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.TRACK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.TRACK_HEAD.POOLER_SAMPLING_RATIO
        r = cfg.MODEL.TRACK_HEAD.SEARCH_REGION

        pooler_z = SRPooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio)
        pooler_x = SRPooler(
            output_size=(int(resolution*r), int(resolution*r)),
            scales=scales,
            sampling_ratio=sampling_ratio)

        self.pooler_x = pooler_x
        self.pooler_z = pooler_z

    def forward(self, x, proposals, sr=None):
        if sr is not None:
            x = self.pooler_x(x, proposals, sr)
        else:
            x = self.pooler_z(x, proposals)

        return x


class EMMPredictor(nn.Module):
    def __init__(self, cfg):
        super(EMMPredictor, self).__init__()

        if cfg.MODEL.BACKBONE.CONV_BODY.startswith("DLA"):
            in_channels = cfg.MODEL.DLA.BACKBONE_OUT_CHANNELS
        elif cfg.MODEL.BACKBONE.CONV_BODY.startswith("R-"):
            in_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        else:
            in_channels = 128

        self.cls_tower = make_conv3x3(in_channels=in_channels, out_channels=in_channels,
                                      use_gn=True, use_relu=True, kaiming_init=False)
        self.reg_tower = make_conv3x3(in_channels=in_channels, out_channels=in_channels,
                                      use_gn=True, use_relu=True, kaiming_init=False)
        self.cls = make_conv3x3(in_channels=in_channels, out_channels=2, kaiming_init=False)
        self.center = make_conv3x3(in_channels=in_channels, out_channels=1, kaiming_init=False)
        self.reg = make_conv3x3(in_channels=in_channels, out_channels=4, kaiming_init=False)

    def forward(self, x):
        cls_x = self.cls_tower(x)
        reg_x = self.reg_tower(x)
        cls_logits = self.cls(cls_x)
        center_logits = self.center(cls_x)
        reg_logits = F.relu(self.reg(reg_x))

        return cls_logits, center_logits, reg_logits