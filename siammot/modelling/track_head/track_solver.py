import torch

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms


class TrackSolver(torch.nn.Module):
    def __init__(self,
                 track_pool,
                 track_thresh=0.3,
                 start_track_thresh=0.5,
                 resume_track_thresh=0.4,
                 ):
        super(TrackSolver, self).__init__()

        self.track_pool = track_pool
        self.track_thresh = track_thresh
        self.start_thresh = start_track_thresh
        self.resume_track_thresh = resume_track_thresh

    def get_nms_boxes(self, detection):
        detection = boxlist_nms(detection, nms_thresh=0.5)

        _ids = detection.get_field('ids')
        _scores = detection.get_field('scores')

        # adjust the scores to the right range
        # _scores -= torch.floor(_scores) * (_ids >= 0) * (torch.floor(_scores) != _scores)
        # _scores[_scores >= 1.] = 1.

        _scores[_scores >= 2.] = _scores[_scores >= 2.] - 2.
        _scores[_scores >= 1.] = _scores[_scores >= 1.] - 1.

        return detection, _ids, _scores

    def forward(self, detection: [BoxList]):
        """
        The solver is to merge predictions from detection branch as well as from track branch.
        The goal is to assign an unique track id to bounding boxes that are deemed tracked
        :param detection: it includes three set of distinctive prediction:
        prediction propagated from active tracks, (2 >= score > 1, id >= 0),
        prediction propagated from dormant tracks, (2 >= score > 1, id >= 0),
        prediction from detection (1 > score > 0, id = -1).
        :return:
        """

        # only process one frame at a time
        assert len(detection) == 1
        detection = detection[0]

        if len(detection) == 0:
            return [detection]

        track_pool = self.track_pool

        all_ids = detection.get_field('ids')
        all_scores = detection.get_field('scores')
        active_ids = track_pool.get_active_ids()
        dormant_ids = track_pool.get_dormant_ids()
        device = all_ids.device

        active_mask = torch.tensor([int(x) in active_ids for x in all_ids], device=device)

        # differentiate active tracks from dormant tracks with scores
        # active tracks, (3 >= score > 2, id >= 0),
        # dormant tracks, (2 >= score > 1, id >= 0),
        # By doing this, dormant tracks will be merged to active tracks during nms,
        # if they highly overlap
        all_scores[active_mask] += 1.

        nms_detection, nms_ids, nms_scores = self.get_nms_boxes(detection)

        combined_detection = nms_detection
        _ids = combined_detection.get_field('ids')
        _scores = combined_detection.get_field('scores')

        # start track ids
        start_idxs = ((_ids < 0) & (_scores >= self.start_thresh)).nonzero()

        # inactive track ids
        inactive_idxs = ((_ids >= 0) & (_scores < self.track_thresh))
        nms_track_ids = set(_ids[_ids >= 0].tolist())
        all_track_ids = set(all_ids[all_ids >= 0].tolist())
        # active tracks that are removed by nms
        nms_removed_ids = all_track_ids - nms_track_ids
        inactive_ids = set(_ids[inactive_idxs].tolist()) | nms_removed_ids

        # resume dormant mask, if needed
        dormant_mask = torch.tensor([int(x) in dormant_ids for x in _ids], device=device)
        resume_ids = _ids[dormant_mask & (_scores >= self.resume_track_thresh)]
        for _id in resume_ids.tolist():
            track_pool.resume_track(_id)

        for _idx in start_idxs:
            _ids[_idx] = track_pool.start_track()

        active_ids = track_pool.get_active_ids()
        for _id in inactive_ids:
            if _id in active_ids:
                track_pool.suspend_track(_id)

        # make sure that the ids for inactive tracks in current frame are meaningless (< 0)
        _ids[inactive_idxs] = -1

        track_pool.expire_tracks()
        track_pool.increment_frame()

        return [combined_detection]


def builder_tracker_solver(cfg, track_pool):
    return TrackSolver(track_pool,
                       cfg.MODEL.TRACK_HEAD.TRACK_THRESH,
                       cfg.MODEL.TRACK_HEAD.START_TRACK_THRESH,
                       cfg.MODEL.TRACK_HEAD.RESUME_TRACK_THRESH)