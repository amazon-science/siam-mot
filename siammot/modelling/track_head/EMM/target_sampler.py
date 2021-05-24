import torch
import copy

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist, boxlist_iou
from maskrcnn_benchmark.modeling.matcher import Matcher

from siammot.utils import registry


class EMMTargetSampler(object):
    """
    Sample track targets for SiamMOT.
    It samples from track proposals from RPN
    """
    def __init__(self, track_utils, matcher, propsals_per_image=256,
                 pos_ratio=0.25, hn_ratio=0.25):
        self.track_utils = track_utils
        self.proposal_iou_matcher = matcher
        self.proposals_per_image = propsals_per_image
        self.hn_ratio = hn_ratio
        self.pos_ratio = pos_ratio

    def match_targets_with_iou(self, proposal: BoxList, gt: BoxList):
        match_quality_matrix = boxlist_iou(gt, proposal)
        matched_idxs = self.proposal_iou_matcher(match_quality_matrix)

        target = gt.copy_with_fields(("ids", "labels"))
        matched_target = target[torch.clamp_min(matched_idxs, -1)]
        proposal_ids = matched_target.get_field('ids')
        proposal_labels = matched_target.get_field('labels')

        # id = -1 for background
        # id = -2 for ignore proposals
        proposal_ids[matched_idxs == -1] = -1
        proposal_ids[matched_idxs == -2] = -2
        proposal_labels[matched_idxs < 0] = 0

        return proposal_ids.type(torch.float), proposal_labels.type(torch.float)

    def assign_matched_ids_to_proposals(self, proposals: BoxList, gts: BoxList):
        """
        Assign for each proposal a matched id, if it is matched to a gt box
        Otherwise, it is assigned -1
        """
        for proposal, gt in zip(proposals, gts):
            proposal_ids, proposal_labels = self.match_targets_with_iou(proposal, gt)
            proposal.add_field('ids', proposal_ids)
            proposal.add_field('labels', proposal_labels)

    def duplicate_boxlist(self, boxlist, num_duplicates):
        """
        Duplicate samples in box list by concatenating multiple times.
        """
        if num_duplicates == 0:
            return self.get_dummy_boxlist(boxlist)
        list_to_join = []
        for _ in range(num_duplicates):
            dup = boxlist.copy_with_fields(list(boxlist.extra_fields.keys()))
            list_to_join.append(dup)

        return cat_boxlist(list_to_join)

    def get_dummy_boxlist(self, boxlist:BoxList, num_boxes=0):
        """
        Create dummy boxlist, with bbox [-1, -1, -1, -1],
        id -1, label -1
        when num_boxes = 0, it means return an empty BoxList
        """
        boxes = torch.zeros((num_boxes, 4)) - 1.
        dummy_boxlist = self.get_default_boxlist(boxlist, boxes)

        return dummy_boxlist

    @staticmethod
    def get_default_boxlist(boxlist:BoxList, bboxes, ids=None, labels=None):
        """
        Construct a boxlist with bbox as bboxes,
        all other fields to be default
        id -1, label -1
        """
        device = boxlist.bbox.device
        num_boxes = bboxes.shape[0]
        if ids is None:
            ids = torch.zeros((num_boxes,)) - 1.
        if labels is None:
            labels = torch.zeros((num_boxes,)) - 1.

        default_boxlist = BoxList(bboxes, image_size=boxlist.size, mode='xyxy')
        default_boxlist.add_field('labels', labels)
        default_boxlist.add_field('ids', ids)

        return default_boxlist.to(device)

    @staticmethod
    def sample_examples(src_box: [BoxList], pair_box: [BoxList],
                        tar_box: [BoxList], num_samples):
        """
        Sample examples
        """
        src_box = cat_boxlist(src_box)
        pair_box = cat_boxlist(pair_box)
        tar_box = cat_boxlist(tar_box)

        assert (len(src_box) == len(pair_box) and len(src_box) == len(tar_box))

        if len(src_box) <= num_samples:
            return [src_box, pair_box, tar_box]
        else:
            indices = torch.zeros((len(src_box), ), dtype=torch.bool)
            permuted_idxs = torch.randperm(len(src_box))
            sampled_idxs = permuted_idxs[: num_samples]
            indices[sampled_idxs] = 1

            sampled_src_box = src_box[indices]
            sampled_pair_box = pair_box[indices]
            sampled_tar_box = tar_box[indices]
            return [sampled_src_box, sampled_pair_box, sampled_tar_box]

    def sample_boxlist(self, boxlist: BoxList, indices, num_samples):
        assert (num_samples <= indices.numel())

        if num_samples == 0:
            sampled_boxlist = self.get_dummy_boxlist(boxlist, num_boxes=0)
        else:
            permuted_idxs = torch.randperm(indices.numel())
            sampled_idxs = indices[permuted_idxs[: num_samples], 0]
            sampled_boxes = boxlist.bbox[sampled_idxs, :]
            sampled_ids = None
            sampled_labels = None
            if 'ids' in boxlist.fields():
                sampled_ids = boxlist.get_field('ids')[sampled_idxs]
            if 'labels' in boxlist.fields():
                sampled_labels = boxlist.get_field('labels')[sampled_idxs]

            sampled_boxlist = self.get_default_boxlist(boxlist, sampled_boxes,
                                                       sampled_ids, sampled_labels)
        return sampled_boxlist

    def get_target_box(self, target_gt, indices):
        """
        Get the corresponding target box given by the 1-off indices
        if there is no matching target box, it returns a dummy box
        """
        tar_box = target_gt[indices]
        # the assertion makes sure that different boxes have different ids
        assert (len(tar_box) <= 1)
        if len(tar_box) == 0:
            # dummy bounding boxes
            tar_box = self.get_dummy_boxlist(target_gt, num_boxes=1)

        return tar_box

    def generate_hn_pair(self, src_gt: BoxList, proposal: BoxList,
                         src_h=None, proposal_h=None):
        """
        Generate hard negative pair by sampling non-negative proposals
        """
        proposal_ids = proposal.get_field('ids')
        src_id = src_gt.get_field('ids')

        scales = torch.ones_like(proposal_ids)
        if (src_h is not None) and (proposal_h is not None):
            scales = src_h / proposal_h

        # sample proposals with similar scales
        # and non-negative proposals
        hard_bb_idxs = ((proposal_ids >= 0) & (proposal_ids != src_id))
        scale_idxs = (scales >= 0.5) & (scales <= 2)
        indices = (hard_bb_idxs & scale_idxs)
        unique_ids = torch.unique(proposal_ids[indices])
        idxs = indices.nonzero()

        # avoid sampling redundant samples
        num_hn = min(idxs.numel(), unique_ids.numel())
        sampled_hn_boxes = self.sample_boxlist(proposal, idxs, num_hn)

        return sampled_hn_boxes

    def generate_pos(self, src_gt: BoxList, proposal: BoxList):
        assert (src_gt.mode == 'xyxy' and len(src_gt) == 1)
        proposal_ids = proposal.get_field('ids')
        src_id = src_gt.get_field('ids')

        pos_indices = (proposal_ids == src_id)
        pos_boxes = proposal[pos_indices]
        pos_boxes = pos_boxes.copy_with_fields(('ids', 'labels'))

        return pos_boxes

    def generate_pos_hn_example(self, proposals, gts):
        """
        Generate positive and hard negative training examples
        """
        src_gts = copy.deepcopy(gts)
        tar_gts = self.track_utils.swap_pairs(copy.deepcopy(gts))

        track_source = []
        track_target = []
        track_pair = []
        for src_gt, tar_gt, proposal in zip(src_gts, tar_gts, proposals):
            pos_src_boxlist, pos_pair_boxlist, pos_tar_boxlist = ([] for _ in range(3))
            hn_src_boxlist, hn_pair_boxlist, hn_tar_boxlist = ([] for _ in range(3))

            proposal_h = proposal.bbox[:, 3] - proposal.bbox[:, 1]
            src_h = src_gt.bbox[:, 3] - src_gt.bbox[:, 1]
            src_ids = src_gt.get_field('ids')
            tar_ids = tar_gt.get_field('ids')

            for i, src_id in enumerate(src_ids):
                _src_box = src_gt[src_ids == src_id]
                _tar_box = self.get_target_box(tar_gt, tar_ids == src_id)

                pos_src_boxes = self.generate_pos(_src_box, proposal)
                pos_pair_boxes = copy.deepcopy(pos_src_boxes)
                pos_tar_boxes = self.duplicate_boxlist(_tar_box, len(pos_src_boxes))

                hn_pair_boxes = self.generate_hn_pair(_src_box, proposal, src_h[i], proposal_h)
                hn_src_boxes = self.duplicate_boxlist(_src_box, len(hn_pair_boxes))
                hn_tar_boxes = self.duplicate_boxlist(_tar_box, len(hn_pair_boxes))

                pos_src_boxlist.append(pos_src_boxes)
                pos_pair_boxlist.append(pos_pair_boxes)
                pos_tar_boxlist.append(pos_tar_boxes)

                hn_src_boxlist.append(hn_src_boxes)
                hn_pair_boxlist.append(hn_pair_boxes)
                hn_tar_boxlist.append(hn_tar_boxes)

            num_pos = int(self.proposals_per_image * self.pos_ratio)
            num_hn = int(self.proposals_per_image * self.hn_ratio)
            sampled_pos = self.sample_examples(pos_src_boxlist, pos_pair_boxlist,
                                               pos_tar_boxlist, num_pos)
            sampled_hn = self.sample_examples(hn_src_boxlist, hn_pair_boxlist,
                                              hn_tar_boxlist, num_hn)
            track_source.append(cat_boxlist([sampled_pos[0], sampled_hn[0]]))
            track_pair.append(cat_boxlist([sampled_pos[1], sampled_hn[1]]))
            track_target.append(cat_boxlist([sampled_pos[2], sampled_hn[2]]))

        return track_source, track_pair, track_target

    def generate_neg_examples(self, proposals: [BoxList], gts: [BoxList], pos_hn_boxes: [BoxList]):
        """
        Generate negative training examples
        """
        track_source = []
        track_pair = []
        track_target = []
        for proposal, gt, pos_hn_box in zip(proposals, gts, pos_hn_boxes):
            proposal_ids = proposal.get_field('ids')
            objectness = proposal.get_field('objectness')

            proposal_h = proposal.bbox[:, 3] - proposal.bbox[:, 1]
            proposal_w = proposal.bbox[:, 2] - proposal.bbox[:, 0]

            neg_indices = ((proposal_ids == -1) & (objectness >= 0.3) &
                           (proposal_h >= 5) & (proposal_w >= 5))
            idxs = neg_indices.nonzero()

            neg_samples = min(idxs.numel(), self.proposals_per_image - len(pos_hn_box))
            neg_samples = max(0, neg_samples)

            sampled_neg_boxes = self.sample_boxlist(proposal, idxs, neg_samples)
            # for target box
            sampled_tar_boxes = self.get_dummy_boxlist(proposal, neg_samples)

            track_source.append(sampled_neg_boxes)
            track_pair.append(sampled_neg_boxes)
            track_target.append(sampled_tar_boxes)
        return track_source, track_pair, track_target

    def __call__(self, proposals: [BoxList], gts: [BoxList]):

        self.assign_matched_ids_to_proposals(proposals, gts)

        pos_hn_src, pos_hn_pair, pos_hn_tar = self.generate_pos_hn_example(proposals, gts)
        neg_src, neg_pair, neg_tar = self.generate_neg_examples(proposals, gts, pos_hn_src)

        track_source = [cat_boxlist([pos_hn, neg]) for (pos_hn, neg) in zip(pos_hn_src, neg_src)]
        track_pair = [cat_boxlist([pos_hn, neg]) for (pos_hn, neg) in zip(pos_hn_pair, neg_pair)]
        track_target = [cat_boxlist([pos_hn, neg]) for (pos_hn, neg) in zip(pos_hn_tar, neg_tar)]

        sr = self.track_utils.update_boxes_in_pad_images(track_pair)
        sr = self.track_utils.extend_bbox(sr)

        return track_source, sr, track_target


@registry.TRACKER_SAMPLER.register("EMM")
def make_emm_target_sampler(cfg,
                            track_utils
                            ):
    matcher = Matcher(
        cfg.MODEL.TRACK_HEAD.FG_IOU_THRESHOLD,
        cfg.MODEL.TRACK_HEAD.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    track_sampler = EMMTargetSampler(track_utils, matcher,
                                     propsals_per_image=cfg.MODEL.TRACK_HEAD.PROPOSAL_PER_IMAGE,
                                     pos_ratio=cfg.MODEL.TRACK_HEAD.EMM.POS_RATIO,
                                     hn_ratio=cfg.MODEL.TRACK_HEAD.EMM.HN_RATIO,
                                     )
    return track_sampler
