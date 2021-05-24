import numpy as np

from gluoncv.torch.data.gluoncv_motion_dataset.dataset import AnnoEntity

from siammot.utils.entity_utils import bbs_iou


def build_data_filter_fn(dataset_key: str, *args, **kwargs):
    """
    Get dataset specific filter function list, if there is any
    """
    filter_fn = None
    if dataset_key == 'CRP':
        filter_fn = CRPFilter(*args, **kwargs)
    elif dataset_key.startswith('MOT'):
        filter_fn = MOTFilter(*args, **kwargs)
    elif dataset_key == 'AOT':
        filter_fn = AOTFilter(*args, **kwargs)
    return filter_fn


class BaseFilter:
    def __init__(self):
        pass

    # the default filter does not filter any entity, which is technically doing nothing
    def _filter(self, entity: AnnoEntity, ignored_gt_entities=None):
        raise False

    def filter(self, entity:AnnoEntity, ignored_gt_entities=None):
        return self._filter(entity, ignored_gt_entities)

    def __call__(self, entities: [AnnoEntity], ignored_entities=None, meta_data=None):
        """
            Check each entity whether it is valid or should be filtered (ignored).
            :param entities: A list of entities (for a single frame) to be evaluated
            :param ignored_entities: A list of ignored entities or a binary mask indicating ignored regions
            :param meta_data: The meta data for the frame (or video)
            :return: A list of valid entities and a list of filtered (ignored) entities
            """
        valid_entities = []
        filtered_entities = []

        for entity in entities:
            if self._filter(entity, ignored_entities):
                filtered_entities.append(entity)
            else:
                valid_entities.append(entity)

        return valid_entities, filtered_entities


class CRPFilter(BaseFilter):
    """
        A class for filtering JTA dataset entities during evaluation
        A gt entity will be filtered (ignored) if its id is -1 (negative)
        A predicted entity will be filtered (ignored) if it is matched to a ignored ground truth entity
        """
    def __init__(self, iou_thresh=0.2, is_train=False):
        """
        :param iou_thresh: a predicted entity which overlaps with any ignored gt entity with at least
         iou_thresh would be filtered
        """
        self.iou_thresh = iou_thresh

    def _filter(self, entity: AnnoEntity, ignored_gt_entities=None):
        if ignored_gt_entities is None:
            if entity.id < 0:
                return True
        else:
            for entity_ in ignored_gt_entities:
                if bbs_iou(entity, entity_) >= self.iou_thresh:
                    return True
        return False


class MOTFilter(BaseFilter):
    """
    A class for filtering MOT dataset entities
    A gt entity will be filtered (ignored) if its visibility ratio is very low
    A predicted entity will be filtered (ignored) if it is matched to a ignored ground truth entity
    """
    def __init__(self, visibility_thresh=0.1, iou_thresh=0.5, is_train=False):
        self.visibility_thresh = visibility_thresh
        self.iou_thresh = iou_thresh
        self.is_train = is_train

    def _filter(self, entity: AnnoEntity, ignored_gt_entities=None):
        if ignored_gt_entities is None:
            if self.is_train:
                # any entity whose visibility is below the pre-defined
                # threshold should be filtered out
                # meanwhile, any entity whose class does not have label
                # needs to be filtered
                if entity.blob['visibility'] < self.visibility_thresh or \
                        not any(k in ('person', '2', '7') for k in entity.labels):
                    return True
            else:
                if 'person' not in entity.labels or int(entity.id) < 0:
                    return True
        else:
            for entity_ in ignored_gt_entities:
                if bbs_iou(entity, entity_) >= self.iou_thresh:
                    return True
            return False


class AOTFilter(BaseFilter):
    """
    A class for filtering AOT entities
    A gt entity will be filtered if it falls into one the following criterion
      1. tracking id is not Helicopter1 or Airplane1
      2. range distance is larger than 1200
    """

    def __init__(self, range_distance_thresh=1200, iou_thresh=0.2, is_train=False):
        self.range_distance_thresh = range_distance_thresh
        self.iou_thresh = iou_thresh
        self.is_train = is_train

    def _filter(self, entity: AnnoEntity, ignored_gt_entities=None):
        if ignored_gt_entities is None:
            range_distance_m = np.inf
            if 'range_distance_m' in entity.blob:
                range_distance_m = entity.blob['range_distance_m']

            labels = []
            if entity.labels is not None:
                labels = entity.labels

            if ('intruder' not in labels) or \
                    (range_distance_m >= self.range_distance_thresh):
                return True
        else:
            for entity_ in ignored_gt_entities:
                if entity_.bbox is not None:
                    if bbs_iou(entity, entity_) >= self.iou_thresh:
                        return True
        return False

