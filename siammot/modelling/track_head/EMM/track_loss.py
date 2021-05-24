import torch
from torch import nn
from torch.nn import functional as F


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def log_softmax(cls_logits):
    b, a2, h, w = cls_logits.size()
    cls_logits = cls_logits.view(b, 2, a2 // 2, h, w)
    cls_logits = cls_logits.permute(0, 2, 3, 4, 1).contiguous()
    cls_logits = F.log_softmax(cls_logits, dim=4)
    return cls_logits


class IOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        pred_l = pred[:, 0]
        pred_t = pred[:, 1]
        pred_r = pred[:, 2]
        pred_b = pred[:, 3]

        target_l = target[:, 0]
        target_t = target[:, 1]
        target_r = target[:, 2]
        target_b = target[:, 3]

        target_area = (target_l + target_r) * (target_t + target_b)
        pred_area = (pred_l + pred_r) * (pred_t + pred_b)

        w_intersect = torch.min(pred_l, target_l) + torch.min(pred_r, target_r)
        h_intersect = torch.min(pred_b, target_b) + torch.min(pred_t, target_t)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        losses = -torch.log((area_intersect + 1.) / (area_union + 1.))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            return losses.mean()


class EMMLossComputation(object):
    def __init__(self, cfg):
        self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.cfg = cfg
        self.pos_ratio = cfg.MODEL.TRACK_HEAD.EMM.CLS_POS_REGION
        self.loss_weight = cfg.MODEL.TRACK_HEAD.EMM.TRACK_LOSS_WEIGHT

    def prepare_targets(self, points, src_bbox, gt_bbox):

        cls_labels, reg_targets = self.compute_targets(points, src_bbox, gt_bbox)

        return cls_labels, reg_targets

    def compute_targets(self, locations, src_bbox, tar_bbox):
        xs, ys = locations[:, :, 0], locations[:, :, 1]

        num_boxes, num_locations, _ = locations.shape
        cls_labels = torch.zeros((num_boxes, num_locations),
                                 dtype=torch.int64, device=locations.device)

        _l = xs - tar_bbox[:, 0:1].float()
        _t = ys - tar_bbox[:, 1:2].float()
        _r = tar_bbox[:, 2:3].float() - xs
        _b = tar_bbox[:, 3:4].float() - ys

        s1 = _l > self.pos_ratio * ((tar_bbox[:, 2:3] - tar_bbox[:, 0:1]) / 2).float()
        s2 = _r > self.pos_ratio * ((tar_bbox[:, 2:3] - tar_bbox[:, 0:1]) / 2).float()
        s3 = _t > self.pos_ratio * ((tar_bbox[:, 3:4] - tar_bbox[:, 1:2]) / 2).float()
        s4 = _b > self.pos_ratio * ((tar_bbox[:, 3:4] - tar_bbox[:, 1:2]) / 2).float()

        is_in_pos_boxes = s1 * s2 * s3 * s4
        cls_labels[is_in_pos_boxes == 1] = 1

        reg_targets = torch.stack([_l, _t, _r, _b], dim=2)

        return cls_labels.contiguous(), reg_targets.contiguous()

    @staticmethod
    def compute_centerness_targets(reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    @staticmethod
    def normalize_regression_outputs(src_bbox, regression_outputs):
        # normalize the regression targets
        half_src_box_w = (src_bbox[:, 2:3] - src_bbox[:, 0:1]) / 2. + 1e-10
        half_src_box_h = (src_bbox[:, 3:4] - src_bbox[:, 1:2]) / 2. + 1e-10
        assert (all(half_src_box_w > 0))
        assert (all(half_src_box_h > 0))

        regression_outputs[:, :, 0] = (regression_outputs[:, :, 0] / half_src_box_w) * 128
        regression_outputs[:, :, 1] = (regression_outputs[:, :, 1] / half_src_box_h) * 128
        regression_outputs[:, :, 2] = (regression_outputs[:, :, 2] / half_src_box_w) * 128
        regression_outputs[:, :, 3] = (regression_outputs[:, :, 3] / half_src_box_h) * 128

        return regression_outputs

    def __call__(self, locations, box_cls, box_regression, centerness, src, targets):
        """
        """

        cls_labels, reg_targets = self.prepare_targets(locations, src, targets)

        box_regression = (box_regression.permute(0, 2, 3, 1).contiguous()).view(-1, 4)
        box_regression_flatten = box_regression.view(-1, 4)
        reg_targets_flatten = reg_targets.view(-1, 4)
        cls_labels_flatten = cls_labels.view(-1)
        centerness_flatten = centerness.view(-1)

        in_box_inds = torch.nonzero(cls_labels_flatten > 0).squeeze(1)
        box_regression_flatten = box_regression_flatten[in_box_inds]
        reg_targets_flatten = reg_targets_flatten[in_box_inds]
        centerness_flatten = centerness_flatten[in_box_inds]

        box_cls = log_softmax(box_cls)
        cls_loss = select_cross_entropy_loss(box_cls, cls_labels_flatten)

        if in_box_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:
            reg_loss = 0. * box_regression_flatten.sum()
            centerness_loss = 0. * centerness_flatten.sum()

        return self.loss_weight*cls_loss, self.loss_weight*reg_loss, self.loss_weight*centerness_loss


