import os

from gluoncv.torch.data.gluoncv_motion_dataset.dataset import GluonCVMotionDataset
from pycocotools.coco import COCO

from .dataset_info import dataset_maps


def load_motion_anno(dataset_folder,
                     anno_file,
                     split_file,
                     set=None,
                     ):
    """
    Load GluonCVMotionDataset format annotations for downstream training / testing
    """

    dataset = GluonCVMotionDataset(anno_file,
                                   root_path=dataset_folder,
                                   split_file=split_file
                                   )

    if set == 'train':
        dataset = list(dataset.train_samples)
    elif set == 'val':
        dataset = list(dataset.val_samples)
    elif set == 'test':
        dataset = list(dataset.test_samples)

    return dataset


def load_coco_anno(dataset_folder,
                   anno_file):

    dataset_anno_path = os.path.join(dataset_folder, anno_file)
    dataset = COCO(dataset_anno_path)
    return dataset


def load_dataset_anno(cfg, dataset_key, set=None):
    dataset_folder, anno_file, split_file, modality = dataset_maps[dataset_key]

    dataset_info = dict()
    dataset_info['modality'] = modality

    dataset_folder = os.path.join(cfg.DATASETS.ROOT_DIR, dataset_folder)
    if modality == 'video':
        dataset = load_motion_anno(dataset_folder,
                                   anno_file,
                                   split_file,
                                   set)
    elif modality == 'image':
        dataset = load_coco_anno(dataset_folder,
                                 anno_file)
        image_folder = os.path.join(dataset_folder, split_file)
        dataset_info['image_folder'] = image_folder
    else:
        raise ValueError("dataset has to be video or image.")

    return dataset, dataset_info


def load_public_detection(cfg, dataset_key):
    dataset_folder, _, split_file, _ = dataset_maps[dataset_key]

    dataset_folder = os.path.join(cfg.DATASETS.ROOT_DIR, dataset_folder)

    try:
        public_detection = load_motion_anno(dataset_folder,
                                            'anno_pub_detection.json',
                                            split_file)
    except:
        print("The public detection is not ingested or provided in {}, skip public detection".
              format(os.path.join(dataset_folder, 'annotation/anno_pub_detection.json')))

        return None

    return public_detection

