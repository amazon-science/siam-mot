import numpy as np
from sklearn.metrics import auc

from gluoncv.torch.data.gluoncv_motion_dataset.dataset import AnnoEntity
from ..utils.entity_utils import bbs_iou


def evaluate_recall(gt: [AnnoEntity], pred: [AnnoEntity], iou_thresh=0.5):
    """
    :param gt: groundtruth entities for a frame
    :param pred: prediction entities for a frame
    :param iou_thresh:

    """
    iou_matrix = bbs_iou(pred, gt)
    pred_ious, gt_ious = greedy_matching(iou_matrix, iou_thresh=iou_thresh)

    tp = 0
    fn = len(gt)

    for pred_iou in pred_ious:
        if pred_iou == 1:
            tp += 1
            fn -= 1

    assert(tp+fn == len(gt))

    return tp, fn


def precision_recall_curve(scores, pred_ious, gt_ious, iou_threshold=0.5):
    """
    Return a list of precision/recall based on different confidence thresholds
    """
    precisions = []
    recalls = []
    sorted_ = sorted(zip(scores, pred_ious), reverse=True)

    tp = 0
    fp = 0
    fn = len(gt_ious)
    for (score, pred_iou) in sorted_:
        if pred_iou >= iou_threshold:
            tp += 1
            fn -= 1
        else:
            fp += 1
        precisions.append(float(tp)/float(tp+fp+1e-4))
        recalls.append(float(tp)/float(tp+fn+1e-4))

    return precisions, recalls


def greedy_matching(iou_matrix, iou_thresh=0.5):
    """
        Do the greedy matching across predictions and ground truth annotations
        Returns the matching ious for every predictions and ground truths
        Every row denotes a ground truth matching with all predictions
        """
    (num_pred, num_gt) = iou_matrix.shape
    gt_ious = np.zeros(num_gt)
    pred_ious = np.zeros(num_pred)
    if iou_matrix.size > 0:
        for i in range(num_pred):
            max_iou = np.amax(iou_matrix[i, :])
            if max_iou >= iou_thresh:
                _id = np.where(iou_matrix[i, :] == max_iou)[0][0]
                pred_ious[i] = 1
                gt_ious[_id] = 1
                iou_matrix[:, _id] = 0
    return pred_ious.tolist(), gt_ious.tolist()


def compute_AP(scores, pred_ious, gt_ious):
    """
    Computer  Average Precision (AP) given a list of score
    :param scores: A list of confidence scores of detections
    :param pred_ious: A list of iou of detections w.r.t the most matching ground truth bounding boxes
    :param gt_ious: A list of iou of ground truth bounding boxes w.r.t the most matching detections
    :return: Average Precision (AP)
    """
    if not isinstance(scores[0], list):
        scores = [scores]
        pred_ious = [pred_ious]
        gt_ious = [gt_ious]
    assert (len(scores) == len(pred_ious))
    assert (len(scores) == len(gt_ious))

    ap_list = np.zeros((len(scores), ))
    precisions = []
    recalls = []
    for i in range(len(scores)):
        precision, recall = precision_recall_curve(scores[i],
                                                     pred_ious[i],
                                                     gt_ious[i])

        if len(recall) >= 2:
            ap_list[i] = auc(recall, precision)
            precisions.append(precision)
            recalls.append(recall)

    return ap_list

