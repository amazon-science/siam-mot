import torch
import numpy as np
import torch.nn.functional as F

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.utils import cat

from siammot.utils import registry
from .xcorr import xcorr_depthwise
from .feature_extractor import EMMFeatureExtractor, EMMPredictor
from .track_loss import EMMLossComputation


@registry.SIAMESE_TRACKER.register("EMM")
class EMM(torch.nn.Module):
    def __init__(self, cfg, track_utils):
        super(EMM, self).__init__()
        self.feature_extractor = EMMFeatureExtractor(cfg)
        self.predictor = EMMPredictor(cfg)
        self.loss = EMMLossComputation(cfg)

        self.track_utils = track_utils
        self.amodal = cfg.INPUT.AMODAL
        self.use_centerness = cfg.MODEL.TRACK_HEAD.EMM.USE_CENTERNESS
        self.pad_pixels = cfg.MODEL.TRACK_HEAD.PAD_PIXELS
        self.sigma = cfg.MODEL.TRACK_HEAD.EMM.COSINE_WINDOW_WEIGHT

    def forward(self, features, boxes, sr, targets=None, template_features=None):
        """
        forward functions of the tracker
        :param features: raw FPN feature maps from feature backbone
        :param boxes: template bounding boxes
        :param sr: search region bounding boxes
        :param targets:
        :param template_features: features of the template bounding boxes

        the number of track boxes should be the same as that of
        search region and template_features
        """

        # x, y shifting due to feature padding
        shift_x = self.pad_pixels
        shift_y = self.pad_pixels

        if self.training:
            template_features = self.feature_extractor(features, boxes)
            features = self.track_utils.shuffle_feature(features)

        features = self.track_utils.pad_feature(features)

        sr_features = self.feature_extractor(features, boxes, sr)

        response_map = xcorr_depthwise(sr_features, template_features)
        cls_logits, center_logits, reg_logits = self.predictor(response_map)

        if self.training:
            locations = get_locations(sr_features, template_features, sr, shift_xy=(shift_x, shift_y))
            src_bboxes = cat([b.bbox for b in boxes], dim=0)
            gt_bboxes = cat([b.bbox for b in targets], dim=0)
            cls_loss, reg_loss, centerness_loss = self.loss(
                locations, cls_logits, reg_logits, center_logits, src_bboxes, gt_bboxes)

            loss = dict(loss_tracker_class=cls_loss,
                        loss_tracker_motion=reg_loss,
                        loss_tracker_center=centerness_loss)

            return {}, {}, loss
        else:
            cls_logits = F.interpolate(cls_logits, scale_factor=16, mode='bicubic')
            center_logits = F.interpolate(center_logits, scale_factor=16, mode='bicubic')
            reg_logits = F.interpolate(reg_logits, scale_factor=16, mode='bicubic')

            locations = get_locations(sr_features, template_features, sr, shift_xy=(shift_x, shift_y), up_scale=16)

            assert len(boxes) == 1
            bb, bb_conf = decode_response(cls_logits, center_logits, reg_logits, locations, boxes[0],
                                          use_centerness=self.use_centerness, sigma= self.sigma)
            track_result = wrap_results_to_boxlist(bb, bb_conf, boxes, amodal=self.amodal)
            return {}, track_result, {}

    def extract_cache(self, features, detection):
        """
        Get the cache (state) that is necessary for tracking
        output: (features for tracking targets,
                 search region,
                 detection bounding boxes)
        """

        # get cache features for search region
        # FPN features
        detection = [detection]
        x = self.feature_extractor(features, detection)

        sr = self.track_utils.update_boxes_in_pad_images(detection)
        sr = self.track_utils.extend_bbox(sr)

        cache = (x, sr, detection)
        return cache


def decode_response(cls_logits, center_logits, reg_logits, locations, boxes,
                    use_centerness=True, sigma=0.4):
    cls_logits = F.softmax(cls_logits, dim=1)

    cls_logits = cls_logits[:, 1:2, :, :]
    if use_centerness:
        centerness = F.sigmoid(center_logits)
        obj_confidence = cls_logits * centerness
    else:
        obj_confidence = cls_logits

    num_track_objects = obj_confidence.shape[0]
    obj_confidence = obj_confidence.reshape((num_track_objects, -1))
    tlbr = reg_logits.reshape((num_track_objects, 4, -1))

    scale_penalty = get_scale_penalty(tlbr, boxes)
    cos_window = get_cosine_window_penalty(tlbr)
    p_obj_confidence = (obj_confidence * scale_penalty)*(1-sigma) + sigma*cos_window

    idxs = torch.argmax(p_obj_confidence, dim=1)

    target_ids = torch.arange(num_track_objects)
    bb_c = locations[target_ids, idxs, :]
    shift_tlbr = tlbr[target_ids, :, idxs]

    bb_tl_x = bb_c[:, 0:1] - shift_tlbr[:, 0:1]
    bb_tl_y = bb_c[:, 1:2] - shift_tlbr[:, 1:2]
    bb_br_x = bb_c[:, 0:1] + shift_tlbr[:, 2:3]
    bb_br_y = bb_c[:, 1:2] + shift_tlbr[:, 3:4]
    bb = torch.cat((bb_tl_x, bb_tl_y, bb_br_x, bb_br_y), dim=1)

    cls_logits = cls_logits.reshape((num_track_objects, -1))
    bb_conf = cls_logits[target_ids, idxs]

    return bb, bb_conf


def get_scale_penalty(tlbr: torch.Tensor, boxes: BoxList):
    box_w = boxes.bbox[:, 2] - boxes.bbox[:, 0]
    box_h = boxes.bbox[:, 3] - boxes.bbox[:, 1]

    r_w = tlbr[:, 2] + tlbr[:, 0]
    r_h = tlbr[:, 3] + tlbr[:, 1]

    scale_w = r_w / box_w[:, None]
    scale_h = r_h / box_h[:, None]
    scale_w = torch.max(scale_w, 1 / scale_w)
    scale_h = torch.max(scale_h, 1 / scale_h)

    scale_penalty = torch.exp((-scale_w * scale_h + 1) * 0.1)

    return scale_penalty


def get_cosine_window_penalty(tlbr: torch.Tensor):
    num_boxes, _, num_elements = tlbr.shape
    h_w = int(np.sqrt(num_elements))
    hanning = torch.hann_window(h_w, dtype=torch.float, device=tlbr.device)
    window = torch.ger(hanning, hanning)
    window = window.reshape(-1)

    return window[None, :]


def wrap_results_to_boxlist(bb, bb_conf, boxes: [BoxList], amodal=False):
    num_boxes_per_image = [len(box) for box in boxes]
    bb = bb.split(num_boxes_per_image, dim=0)
    bb_conf = bb_conf.split(num_boxes_per_image, dim=0)

    track_boxes = []
    for _bb, _bb_conf, _boxes in zip(bb, bb_conf, boxes):
        _bb = _bb.reshape(-1, 4)
        track_box = BoxList(_bb, _boxes.size, mode="xyxy")
        track_box.add_field("ids", _boxes.get_field('ids'))
        track_box.add_field("labels", _boxes.get_field('labels'))
        track_box.add_field("scores", _bb_conf)
        if not amodal:
            track_box.clip_to_image(remove_empty=True)
        track_boxes.append(track_box)

    return track_boxes


def get_locations(fmap: torch.Tensor, template_fmap: torch.Tensor,
                  sr_boxes: [BoxList], shift_xy, up_scale=1):
    """

    """
    h, w = fmap.size()[-2:]
    h, w = h*up_scale, w*up_scale
    concat_boxes = cat([b.bbox for b in sr_boxes], dim=0)
    box_w = concat_boxes[:, 2] - concat_boxes[:, 0]
    box_h = concat_boxes[:, 3] - concat_boxes[:, 1]
    stride_h = box_h / (h - 1)
    stride_w = box_w / (w - 1)

    device = concat_boxes.device
    delta_x = torch.arange(0, w, dtype=torch.float32, device=device)
    delta_y = torch.arange(0, h, dtype=torch.float32, device=device)

    delta_x = (concat_boxes[:, 0])[:, None] + delta_x[None, :] * stride_w[:, None]
    delta_y = (concat_boxes[:, 1])[:, None] + delta_y[None, :] * stride_h[:, None]

    h0, w0 = template_fmap.shape[-2:]
    assert (h0 == w0)
    border = np.int(np.floor(h0 / 2))
    st_end_idx = int(border * up_scale)
    delta_x = delta_x[:, st_end_idx:-st_end_idx]
    delta_y = delta_y[:, st_end_idx:-st_end_idx]

    locations = []
    num_boxes = delta_x.shape[0]
    for i in range(num_boxes):
        _y, _x = torch.meshgrid((delta_y[i, :], delta_x[i, :]))
        _y = _y.reshape(-1)
        _x = _x.reshape(-1)
        _xy = torch.stack((_x, _y), dim=1)
        locations.append(_xy)
    locations = torch.stack(locations)

    # shift the coordinates w.r.t the original image space (before padding)
    locations[:, :, 0] -= shift_xy[0]
    locations[:, :, 1] -= shift_xy[1]

    return locations