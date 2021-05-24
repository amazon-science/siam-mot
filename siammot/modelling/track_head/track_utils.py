import torch
import numpy as np
import torch.nn.functional as F

from maskrcnn_benchmark.structures.bounding_box import BoxList


class TrackUtils(object):
    """
    A class that includes utility functions unique to track branch
    """
    def __init__(self, search_expansion=1.0, min_search_wh=128, pad_pixels=256):
        """
        :param search_expansion: expansion ratio (of the search region)
        w.r.t the size of tracking targets
        :param min_search_wh: minimal size (width and height) of the search region
        :param pad_pixels: the padding pixels that are neccessary to keep the
        feature map pf search region and that of template target in the same scale
        """
        self.search_expansion = search_expansion
        self.min_search_wh = min_search_wh
        self.pad_pixels = pad_pixels

    @staticmethod
    def swap_pairs(entity_list):
        assert len(entity_list) % 2 == 0
        # Take the targets of the other frame (in a tracking pair) as input during training, thus swap order
        for xx in range(0, len(entity_list), 2):
            entity_list[xx], entity_list[xx + 1] = entity_list[xx + 1], entity_list[xx]
        return entity_list

    @staticmethod
    def shuffle_feature(f):
        """
        odd-even order swap of the feature tensor in the batch dimension
        """

        def shuffle_feature_tensor(x):
            batch_size = x.shape[0]
            assert batch_size % 2 == 0

            # get channel swap order [1, 0, 3, 2, ...]
            odd_idx = range(1, batch_size, 2)
            even_idx = range(0, batch_size, 2)
            idxs = np.arange(0, batch_size)
            idxs[even_idx] = idxs[even_idx] + 1
            idxs[odd_idx] = idxs[odd_idx] - 1
            idxs = torch.tensor(idxs)

            return x[idxs]

        if isinstance(f, tuple):
            shuffle_f = []
            for i, _f in enumerate(f):
                shuffle_f.append(shuffle_feature_tensor(_f))
            shuffle_f = tuple(shuffle_f)
        else:
            shuffle_f = shuffle_feature_tensor(f)

        return shuffle_f

    def extend_bbox(self, in_box: [BoxList]):
        """
        Extend the bounding box to define the search region
        :param in_box: a set of bounding boxes in previous frame
        :param min_wh: the miniumun width/height of the search region
        """
        for i, _track in enumerate(in_box):
            bbox_w = _track.bbox[:, 2] - _track.bbox[:, 0] + 1
            bbox_h = _track.bbox[:, 3] - _track.bbox[:, 1] + 1
            w_ext = bbox_w * (self.search_expansion / 2.)
            h_ext = bbox_h * (self.search_expansion / 2.)

            # todo: need to check the equation later
            min_w_ext = (self.min_search_wh - bbox_w) / (self.search_expansion * 2.)
            min_h_ext = (self.min_search_wh - bbox_h) / (self.search_expansion * 2.)

            w_ext = torch.max(min_w_ext, w_ext)
            h_ext = torch.max(min_h_ext, h_ext)
            in_box[i].bbox[:, 0] -= w_ext
            in_box[i].bbox[:, 1] -= h_ext
            in_box[i].bbox[:, 2] += w_ext
            in_box[i].bbox[:, 3] += h_ext
            # in_box[i].clip_to_image()
        return in_box

    def pad_feature(self, f):
        """
        Pad the feature maps with 0
        :param f: [torch.tensor] or torch.tensor
        """

        if isinstance(f, (list, tuple)):
            pad_f = []
            for i, _f in enumerate(f):
                # todo fix this hack, should read from cfg file
                pad_pixels = int(self.pad_pixels / ((2 ** i) * 4))
                x = F.pad(_f, [pad_pixels, pad_pixels, pad_pixels, pad_pixels],
                          mode='constant', value=0)
                pad_f.append(x)
            pad_f = tuple(pad_f)
        else:
            pad_f = F.pad(f, [self.pad_pixels, self.pad_pixels,
                              self.pad_pixels, self.pad_pixels],
                          mode='constant', value=0)

        return pad_f

    def update_boxes_in_pad_images(self, boxlists:[BoxList]):
        """
        Update the coordinates of bounding boxes in the padded image
        """

        pad_width = self.pad_pixels
        pad_height = self.pad_pixels

        pad_boxes = []
        for _boxlist in boxlists:
            im_width, im_height = _boxlist.size
            new_width = int(im_width + pad_width*2)
            new_height = int(im_height + pad_height*2)

            assert (_boxlist.mode == 'xyxy')
            xmin, ymin, xmax, ymax = _boxlist.bbox.split(1, dim=-1)
            new_xmin = xmin + pad_width
            new_ymin = ymin + pad_height
            new_xmax = xmax + pad_width
            new_ymax = ymax + pad_height
            bbox = torch.cat((new_xmin, new_ymin, new_xmax, new_ymax), dim=-1)
            bbox = BoxList(bbox, [new_width, new_height], mode='xyxy')
            for _field in _boxlist.fields():
                bbox.add_field(_field, _boxlist.get_field(_field))
            pad_boxes.append(bbox)

        return pad_boxes


class TrackPool(object):
    """
    A class to manage the track id distribution (initiate/kill a track)
    """
    def __init__(self, active_ids=None, max_entangle_length=10, max_dormant_frames=1):
        if active_ids is None:
            self._active_ids = set()
            # track ids that are killed up to previous frames
            self._dormant_ids = {}
            # track ids that are killed in current frame
            self._kill_ids = set()
            self._max_id = -1
        self._embedding = None
        self._cache = {}
        self._frame_idx = 0
        self._max_dormant_frames = max_dormant_frames
        self._max_entangle_length = max_entangle_length

    def suspend_track(self, track_id):
        """
        Suspend an active track, and add it to dormant track pools
        """
        if track_id not in self._active_ids:
            raise ValueError

        self._active_ids.remove(track_id)
        self._dormant_ids[track_id] = self._frame_idx - 1

    def expire_tracks(self):
        """
        Expire the suspended tracks after they are inactive
        for a consecutive self._max_dormant_frames frames
        """
        for track_id, last_active in list(self._dormant_ids.items()):
            if self._frame_idx - last_active >= self._max_dormant_frames:
                self._dormant_ids.pop(track_id)
                self._kill_ids.add(track_id)
                self._cache.pop(track_id, None)

    def increment_frame(self, value=1):
        self._frame_idx += value

    def update_cache(self, cache):
        """
        Update the latest position (bbox) / search region / template feature
        for each track in the cache
        """
        template_features, sr, template_boxes = cache
        sr = sr[0]
        template_boxes = template_boxes[0]
        for idx in range(len(template_boxes)):
            if len(template_features) > 0:
                assert len(template_features) == len(sr)
                features = template_features[idx]
            else:
                features = template_features
            search_region = sr[idx: idx+1]
            box = template_boxes[idx: idx+1]
            track_id = box.get_field("ids").item()
            self._cache[track_id] = (features, search_region, box)

    def resume_track(self, track_id):
        """
        Resume a dormant track
        """
        if track_id not in self._dormant_ids or \
                track_id in self._active_ids:
            raise ValueError

        self._active_ids.add(track_id)
        self._dormant_ids.pop(track_id)

    def kill_track(self, track_id):
        """
        Kill a track
        """
        if track_id not in self._active_ids:
            raise ValueError

        self._active_ids.remove(track_id)
        self._kill_ids.add(track_id)
        self._cache.pop(track_id, None)

    def start_track(self):
        """
        Return a new track id, when starting a new track
        """
        new_id = self._max_id + 1
        self._max_id = new_id
        self._active_ids.add(new_id)

        return new_id

    def get_active_ids(self):
        return self._active_ids

    def get_dormant_ids(self):
        return set(self._dormant_ids.keys())

    def get_cache(self):
        return self._cache

    def activate_tracks(self, track_id):
        if track_id in self._active_ids or \
           track_id not in self._dormant_ids:
            raise ValueError

        self._active_ids.add(track_id)
        self._dormant_ids.pop(track_id)

    def reset(self):
        self._active_ids = set()
        self._kill_ids = set()
        self._dormant_ids = {}
        self._embedding = None
        self._cache = {}
        self._max_id = -1
        self._frame_idx = 0


def build_track_utils(cfg):

    search_expansion = cfg.MODEL.TRACK_HEAD.SEARCH_REGION - 1.
    pad_pixels = cfg.MODEL.TRACK_HEAD.PAD_PIXELS
    min_search_wh = cfg.MODEL.TRACK_HEAD.MINIMUM_SREACH_REGION

    track_utils = TrackUtils(search_expansion=search_expansion,
                             min_search_wh=min_search_wh,
                             pad_pixels=pad_pixels)
    track_pool = TrackPool(max_dormant_frames=cfg.MODEL.TRACK_HEAD.MAX_DORMANT_FRAMES)

    return track_utils, track_pool



