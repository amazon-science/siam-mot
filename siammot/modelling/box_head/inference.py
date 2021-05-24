import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
        bbox_aug_enabled=False,
        amodal_inference=False
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_aug_enabled = bbox_aug_enabled
        self.amodal_inference = amodal_inference

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for each image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)
        device = class_logits.device

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = [self.create_empty_boxlist(device) for _ in boxes]

        for i, (prob, boxes_per_img, image_shape, _box) in enumerate(zip(
                class_prob, proposals, image_shapes, boxes)):

            # get ids for each bbox
            if _box.has_field('ids'):
                ids = _box.get_field('ids')
            else:
                # deafult id is -1
                ids = torch.zeros((len(_box),), dtype=torch.int64, device=device) - 1

            # this only happens for tracks
            if _box.has_field('labels'):
                labels = _box.get_field('labels')

                # tracks
                track_inds = torch.squeeze(torch.nonzero(ids >= 0))

                # avoid track bbs be suppressed during nms
                if track_inds.numel() > 0:
                    prob_cp = prob.clone()
                    prob[track_inds, :] = 0.
                    prob[track_inds, labels] = prob_cp[track_inds, labels] + 1.

                # # avoid track bbs be suppressed during nms
            # prob[ids >= 0] = prob[ids >= 0] + 1.

            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape, ids)
            if not self.amodal_inference:
                boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = self.filter_results(boxlist, num_classes)

            results[i] = boxlist
        return results

    @staticmethod
    def prepare_boxlist(boxes, scores, image_shape, ids):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        boxlist.add_field("ids", ids)
        return boxlist

    def create_empty_boxlist(self, device="cpu"):

        init_bbox = torch.zeros(([0, 4]), dtype=torch.float32, device=device)
        init_score = torch.zeros([0, ], dtype=torch.float32, device=device)
        init_ids = torch.zeros(([0, ]), dtype=torch.int64, device=device)
        empty_boxlist = self.prepare_boxlist(init_bbox, init_score, [0, 0], init_ids)
        return empty_boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)
        device = scores.device

        assert (boxlist.has_field('ids'))
        ids = boxlist.get_field('ids')

        result = [self.create_empty_boxlist(device=device)
                  for _ in range(1, num_classes)]

        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4: (j + 1) * 4]
            ids_j = ids[inds]

            det_idx = ids_j < 0
            det_boxlist = BoxList(boxes_j[det_idx, :], boxlist.size, mode="xyxy")
            det_boxlist.add_field("scores", scores_j[det_idx])
            det_boxlist.add_field("ids", ids_j[det_idx])
            det_boxlist = boxlist_nms(det_boxlist, self.nms)

            track_idx = ids_j >= 0
            # track_box is available
            if torch.any(track_idx > 0):
                track_boxlist = BoxList(boxes_j[track_idx, :], boxlist.size, mode="xyxy")
                track_boxlist.add_field("scores", scores_j[track_idx])
                track_boxlist.add_field("ids", ids_j[track_idx])
                det_boxlist = cat_boxlist([det_boxlist, track_boxlist])

            num_labels = len(det_boxlist)
            det_boxlist.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result[j-1] = det_boxlist

        result = cat_boxlist(result)
        return result


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    bbox_aug_enabled = cfg.TEST.BBOX_AUG.ENABLED

    amodal_inference = cfg.INPUT.AMODAL

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg,
        bbox_aug_enabled,
        amodal_inference
    )
    return postprocessor
