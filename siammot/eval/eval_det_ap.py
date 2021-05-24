import numpy as np
import copy
from tqdm import tqdm

from gluoncv.torch.data.gluoncv_motion_dataset.dataset import DataSample

from .eval_utils import greedy_matching, compute_AP, bbs_iou


def eval_det_ap(gt: list, pred: dict, class_table=None, data_filter_fn=None, iou_threshold=[0.5]):
    """
    Evaluate the detection performance (COCO-style ap) on PoseTrack dataset
    :param gt: ground truth annotations for all videos
    :type gt: dict(vid_id: DataSample)
    :param pred: predictions for all videos
    :type pred: dict(vid_id: DataSample)
    :param data_filter_fn: a callable function that filters out detections that are not considered during evaluation
    :param class_table: class table specify the class order
    :param iou_threshold:
    :return: Average Precision (AP) over different thresholds
    """
    if class_table is None:
        class_table = ["person"]
    num_classes = len(class_table)

    all_scores = [[[] for _ in range(len(iou_threshold))] for _ in range(num_classes)]
    all_pr_ious = [[[] for _ in range(len(iou_threshold))] for _ in range(num_classes)]
    all_gt_ious = [[[] for _ in range(len(iou_threshold))] for _ in range(num_classes)]

    for (vid_id, vid_gt) in tqdm(gt):
        vid_pred = pred[vid_id]

        eval_frame_idxs = vid_gt.get_non_empty_frames()

        # Loop over all classes
        for class_id in range(0, num_classes):
            gt_class_entities = vid_gt.entities
            # gt_class_entities = vid_gt.get_entities_with_label(class_table[class_id])
            pred_class_entities = vid_pred.get_entities_with_label(class_table[class_id])

            # Wrap entities to a DataSample
            vid_class_gt = DataSample(vid_id, metadata=vid_gt.metadata)
            vid_class_pred = DataSample(vid_id, metadata=vid_pred.metadata)
            for _entity in gt_class_entities:
                vid_class_gt.add_entity(_entity)
            for _entity in pred_class_entities:
                vid_class_pred.add_entity(_entity)

            # Get AP for this class and video
            vid_class_scores, vid_class_pr_ious, vid_class_gt_ious = \
                get_ap(vid_class_gt, vid_class_pred, data_filter_fn, eval_frame_idxs, iou_threshold)

            for iou_id in range(len(iou_threshold)):
                all_scores[class_id][iou_id] += vid_class_scores[iou_id]
                all_pr_ious[class_id][iou_id] += vid_class_pr_ious[iou_id]
                all_gt_ious[class_id][iou_id] += vid_class_gt_ious[iou_id]

    class_ap_matrix = np.zeros((num_classes, len(iou_threshold)))
    for class_id in range(num_classes):
        class_ap_matrix[class_id, :] = compute_AP(all_scores[class_id],
                                                  all_pr_ious[class_id],
                                                  all_gt_ious[class_id])

    return class_ap_matrix


def get_ap(vid_class_gt: DataSample, vid_class_pred: DataSample, filter_fn, eval_frame_idxs, iou_thresh=[0.5]):
    """
    :param vid_class_gt: the ground truths for a specific class, in DataSample format
    :param vid_class_pred: the predictions for a specific class, in DataSample format
    :param filter_fn: a callable function to filter out detections
    :param eval_frame_idxs: the frame indexs where evaluation happens
    :param iou_thresh: the list of iou threshod that determines whether a detection is TP
    :returns
           vid_scores: the confidence for every predicted entity (a Python list)
           vid_pr_ious: the iou between the predicted entity and its matching gt entity (a Python list)
           vid_gt_ious: the iou between the gt entity and its matching predicted entity (a Python list)
    """
    if not isinstance(iou_thresh, list):
        iou_thresh = [iou_thresh]
    vid_scores = [[] for _ in iou_thresh]
    vid_pr_ious = [[] for _ in iou_thresh]
    vid_gt_ious = [[] for _ in iou_thresh]
    for frame_idx in eval_frame_idxs:

        gt_entities = vid_class_gt.get_entities_for_frame_num(frame_idx)
        pred_entities = vid_class_pred.get_entities_for_frame_num(frame_idx)

        # Remove detections for evaluation that are within ignore regions
        if filter_fn is not None:
            # Filter out ignored gt entities
            gt_entities, ignore_gt_entities = filter_fn(gt_entities, meta_data=vid_class_gt.metadata)
            # Filter out predicted entities that overlaps with ignored gt entities
            pred_entities, ignore_pred_entities = filter_fn(pred_entities, ignore_gt_entities)

        # sort the entity based on confidence scores
        pred_entities = sorted(pred_entities, key=lambda x: x.confidence, reverse=True)
        iou_matrix = bbs_iou(pred_entities, gt_entities)
        scores = [entity.confidence for entity in pred_entities]
        for i, _iou in enumerate(iou_thresh):
            # pred_ious, gt_ious = target_matching(pred_entities, gt_entities)
            pred_ious, gt_ious = greedy_matching(copy.deepcopy(iou_matrix), _iou)
            vid_scores[i] += scores
            vid_pr_ious[i] += pred_ious
            vid_gt_ious[i] += gt_ious

    return vid_scores, vid_pr_ious, vid_gt_ious
