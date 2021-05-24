import torch
import os
from tqdm import tqdm
from PIL import Image

import torch.utils.data as data
from pycocotools.coco import COCO
from gluoncv.utils.bbox import bbox_xywh_to_xyxy, bbox_clip_xyxy

from maskrcnn_benchmark.structures.bounding_box import BoxList


class ImageDataset(data.Dataset):
    def __init__(self,
                 dataset: COCO,
                 image_dir,
                 transforms=None,
                 frames_per_image=1,
                 amodal=False,
                 skip_empty=True,
                 min_object_area=0,
                 use_crowd=False,
                 include_bg=False,
                 ):
        """
        :param dataset: the ingested dataset with COCO-format
        :param transforms: image transformation
        :param frames_per_image: how many image copies are generated from a single image
        :param amodal: whether to use amodal ground truth (no image boundary clipping)
        :param include_bg: whether to include the full background images during training
        """

        self.dataset = dataset
        self.image_dir = image_dir
        self.transforms = transforms
        self.frames_per_image = frames_per_image

        self._skip_empty = skip_empty
        self._min_object_area = min_object_area
        self._use_crowd = use_crowd
        self._amodal = amodal
        self._include_bg = include_bg
        self._det_classes = [c['name'] for c in self.dataset.loadCats(self.dataset.getCatIds())]

        # These are tha mapping table of COCO labels
        self.json_category_id_to_contiguous_id = {
            v: i+1 for i, v in enumerate(self.dataset.getCatIds())
        }

        self._labels, self._im_aspect_ratios, self._items, self._ids \
            = self._dataset_preprocess()

        self.id_to_img_map = {k: v for k, v in enumerate(self._ids)}

    def __getitem__(self, index):
        img_name = self._items[index]
        img_path = os.path.join(self.image_dir, img_name)

        img = Image.open(img_path).convert('RGB')
        target = self._get_target(img, index)

        # for tracking purposes, two frames are needed
        # the pairs would go into random augmentation to generate fake motion
        video_clip = [img for _ in range(self.frames_per_image)]
        video_target = [target for _ in range(self.frames_per_image)]

        if self.transforms is not None:
            video_clip, video_target = self.transforms(video_clip, video_target)

        return video_clip, video_target, img_name

    def _get_target(self, img, index):

        # a list of label (x1, y1, x2, y2, class_id, instance_id)
        labels = self._labels[index]
        if len(labels) == 0:
            assert self._include_bg is True, "The image does not has ground truth"
            bbox = torch.as_tensor(labels).reshape(-1, 4)
            class_ids = torch.as_tensor(labels)
            instance_ids = torch.as_tensor(labels)
            empty_boxlist = BoxList(bbox, img.size, mode="xyxy")
            empty_boxlist.add_field("labels", class_ids)
            empty_boxlist.add_field("ids", instance_ids)
            return empty_boxlist

        labels = torch.as_tensor(labels).reshape(-1, 6)
        boxes = labels[:, :4]
        target = BoxList(boxes, img.size, mode="xyxy")

        class_ids = labels[:, 4].clone().to(torch.int64)
        target.add_field("labels", class_ids)

        instance_ids = labels[:, -1].clone().to(torch.int64)
        target.add_field("ids", instance_ids)

        if not self._amodal:
            target = target.clip_to_image(remove_empty=True)

        return target

    def _dataset_preprocess(self):
        items = []
        labels = []
        ids = []
        im_aspect_ratios = []
        image_ids = sorted(self.dataset.getImgIds())
        instance_id = 0
        rm_redundant = 0
        all_amodal = 0

        for entry in tqdm(self.dataset.loadImgs(image_ids)):
            label, num_instances, num_redundant, num_amodal\
                = self._check_load_bbox(entry, instance_id)
            if not label and not self._include_bg:
                continue
            instance_id += num_instances
            rm_redundant += num_redundant
            all_amodal += num_amodal
            labels.append(label)
            ids.append(entry['id'])
            items.append(entry['file_name'])
            im_aspect_ratios.append(float(entry['width']) / entry['height'])

        print('{} / {} valid images...'.format(len(labels), len(image_ids)))
        print('{} instances...'.format(instance_id))
        print('{} redundant instances are removed...'.format(rm_redundant))
        print('{} amodal instances...'.format(all_amodal))
        return labels, im_aspect_ratios, items, ids

    def _check_load_bbox(self, entry, instance_id):
        """
        Check and load ground-truth labels
        """
        entry_id = entry['id']
        entry_id = [entry_id] if not isinstance(entry_id, (list, tuple)) else entry_id
        ann_ids = self.dataset.getAnnIds(imgIds=entry_id, iscrowd=None)
        objs = self.dataset.loadAnns(ann_ids)

        # check valid bboxes
        valid_objs = []
        width = entry['width']
        height = entry['height']
        _instance_count = 0
        _redudant_count = 0
        _amodal_count = 0
        unique_bbs = set()
        for obj in objs:
            if obj.get('ignore', 0) == 1:
                continue
            if not self._use_crowd and obj.get('iscrowd', 0):
                continue
            if self._amodal:
                xmin, ymin, xmax, ymax = bbox_xywh_to_xyxy(obj['bbox'])
                if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                    _amodal_count += 1
            else:
                xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)

            if (xmin, ymin, xmax, ymax) in unique_bbs:
                _redudant_count += 1
                continue

            box_w = (xmax - xmin)
            box_h = (ymax - ymin)
            area = box_w * box_h
            if area <= self._min_object_area:
                continue

            # require non-zero box area
            if xmax > xmin and ymax > ymin:
                unique_bbs.add((xmin, ymin, xmax, ymax))
                contiguous_cid = self.json_category_id_to_contiguous_id[obj['category_id']]
                valid_objs.append([xmin, ymin, xmax, ymax, contiguous_cid,
                                   instance_id+_instance_count])
                _instance_count += 1
        if not valid_objs:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append([-1, -1, -1, -1, -1, -1])
        return valid_objs, _instance_count, _redudant_count, _amodal_count

    def __len__(self):
        return len(self._items)

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.dataset.imgs[img_id]
        return img_data

    @property
    def classes(self):
        return self._det_classes

    def get_im_aspect_ratio(self):
        return self._im_aspect_ratios


if __name__ == "__main__":

    from siammot.configs.defaults import cfg
    from siammot.data.video_dataset import VideoDatasetBatchCollator
    from siammot.data.adapters.utils.data_utils import load_dataset_anno
    from siammot.data.adapters.augmentation.build_augmentation import build_siam_augmentation

    torch.manual_seed(0)

    dataset_anno, dataset_info = load_dataset_anno('COCO17_train')
    collator = VideoDatasetBatchCollator()
    transforms = build_siam_augmentation(cfg, modality=dataset_info['modality'])

    dataset = ImageDataset(dataset_anno,
                           dataset_info['image_folder'],
                           frames_per_image=2,
                           transforms=transforms,
                           amodal=True)

    batch_size = 16
    sampler = torch.utils.data.sampler.RandomSampler(dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=False)
    dataloader = data.DataLoader(dataset,
                                 num_workers=4,
                                 batch_sampler=batch_sampler,
                                 collate_fn=collator
                                 )
    import time
    tic = time.time()
    for iteration, (image, target, image_ids) in enumerate(dataloader):
        data_time = time.time() - tic
        print("Data loading time: {}".format(data_time))
        tic = time.time()
        print(image_ids)