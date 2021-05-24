import numpy as np
from gluoncv.torch.data.gluoncv_motion_dataset.dataset import AnnoEntity


def bbs_iou(entities_1: [AnnoEntity], entities_2: [AnnoEntity]):
    """
    Compute iou matrix between two lists of Entity
    bbox in AnnoEntity is in the format of xywh

    Different from boxlist_iou in terms of not adding TO_REMOVE to wh
    """

    if not isinstance(entities_1, list):
        entities_1 = [entities_1]
    if not isinstance(entities_2, list):
        entities_2 = [entities_2]

    if len(entities_1) == 0 or len(entities_2) == 0:
        return np.zeros((len(entities_1), len(entities_2)))

    box_xywh_1 = np.array([entity.bbox for entity in entities_1])
    box_xywh_2 = np.array([entity.bbox for entity in entities_2])

    # compute the area of union regions
    area1 = box_xywh_1[:, 2] * box_xywh_1[:, 3]
    area2 = box_xywh_2[:, 2] * box_xywh_2[:, 3]

    # to xyxy
    box_xyxy_1 = np.zeros_like(box_xywh_1)
    box_xyxy_2 = np.zeros_like(box_xywh_2)
    box_xyxy_1[:, :2] = box_xywh_1[:, 0:2]
    box_xyxy_2[:, :2] = box_xywh_2[:, 0:2]
    box_xyxy_1[:, 2:] = box_xywh_1[:, :2] + box_xywh_1[:, 2:]
    box_xyxy_2[:, 2:] = box_xywh_2[:, :2] + box_xywh_2[:, 2:]

    lt = np.maximum(box_xyxy_1[:, None, :2], box_xyxy_2[:, :2])  # [N,M,2]
    rb = np.minimum(box_xyxy_1[:, None, 2:], box_xyxy_2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)

    return iou