from maskrcnn_benchmark.structures.bounding_box import BoxList
from gluoncv.torch.data.gluoncv_motion_dataset.dataset import AnnoEntity


def boxlists_to_entities(boxlists, firstframe_idx, timestamps, class_table=None):
    """
    Convert a list of boxlist to entities
    :return:
    """

    if isinstance(boxlists, BoxList):
        boxlists = [boxlists]

    # default class is person only
    if class_table is None:
        class_table = ["person"]

    assert isinstance(boxlists, list), "The input has to be a list"

    entities = []
    for i, boxlist in enumerate(boxlists):
        for j in range(len(boxlist)):
            entity = AnnoEntity()
            entity.bbox = boxlist.bbox[j].tolist()
            entity.confidence = boxlist.get_field('scores')[j].item()
            _label = boxlist.get_field('labels')[j].item()
            entity.labels = {class_table[_label - 1]: entity.confidence}
            # the default id is -1
            entity.id = -1
            if boxlist.has_field('ids'):
                entity.id = boxlist.get_field('ids')[j].item()
            entity.frame_num = firstframe_idx + i
            entity.time = timestamps[i]
            entities.append(entity)

    return entities
