import random
import torch
import itertools
import torch.utils.data as data
from tqdm import tqdm
from collections import defaultdict
from PIL.Image import Image

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList

from gluoncv.torch.data.gluoncv_motion_dataset.dataset import GluonCVMotionDataset, AnnoEntity


class VideoDataset(data.Dataset):

    def __init__(self, dataset: GluonCVMotionDataset, sampling_interval=250, clip_len=1000,
                 is_train=True, frames_in_clip=2, transforms=None, filter_fn=None,
                 amodal=False):
        """
        :param dataset: the ingested dataset with GluonCVMotionDataset
        :param sampling_interval: the temporal stride (in ms) of sliding window
        :param clip_len: the temporal length (in ms) of video clips
        :param is_train: a boolean flag indicating whether it is training
        :param frames_in_clip: the number of frames sampled in a video clip (for a training example)
        :param transforms: frame-level transformation before they are fed into neural networks
        :param filter_fn: a callable function to filter entities
        :param amodal: whether to clip the bounding box beyond image boundary
        """

        if dataset is None:
            raise Exception('dataset should not be None. Call GluonCVMotionDataset to construct dataset first.')

        assert is_train is True, "The dataset class only supports training"
        assert (2 >= frames_in_clip > 0), "frames_in_clip has to be 1 or 2"

        self.data = dict(dataset.train_samples)

        self.clip_len = clip_len
        self.transforms = transforms
        self.filter_fn = filter_fn
        self.frames_in_clip = min(clip_len, frames_in_clip)

        # Process dataset to get all valid video clips
        self.clips = self.get_video_clips(sampling_interval_ms=sampling_interval)
        self.amodal = amodal

    def __getitem__(self, item_id):

        video = []
        target = []

        (sample_id, clip_frame_ids) = self.clips[item_id]
        video_info = self.data[sample_id]
        video_reader = video_info.get_data_reader()

        # Randomly sampling self.frames_in_clip frames
        # And keep their relative temporal order
        rand_idxs = sorted(random.sample(clip_frame_ids, self.frames_in_clip))
        for frame_idx in rand_idxs:
            im = video_reader[frame_idx][0]
            entities = video_info.get_entities_for_frame_num(frame_idx)
            if self.filter_fn is not None:
                entities, _ = self.filter_fn(entities, meta_data=video_info.metadata)
            boxes = self.entity2target(im, entities)

            video.append(im)
            target.append(boxes)

        # Video clip-level augmentation
        if self.transforms is not None:
            video, target = self.transforms(video, target)

        return video, target, sample_id

    def __len__(self):
        return len(self.clips)

    def get_video_clips(self, sampling_interval_ms=250):
        """
        Process the long videos to a small video chunk (with self.clip_len seconds)
        Video clips are generated in a temporal sliding window fashion
        """
        video_clips = []
        for (sample_id, sample) in tqdm(self.data.items()):
            frame_idxs_with_anno = sample.get_non_empty_frames(self.filter_fn)
            if len(frame_idxs_with_anno) == 0:
                continue
            # The video clip may not be temporally continuous
            start_frame = min(frame_idxs_with_anno)
            end_frame = max(frame_idxs_with_anno)
            # make sure that the video clip has at least two frames
            clip_len_in_frames = max(self.frames_in_clip, int(self.clip_len / 1000. * sample.fps))
            sampling_interval = int(sampling_interval_ms / 1000. * sample.fps)
            for idx in range(start_frame, end_frame, sampling_interval):
                clip_frame_ids = []
                # only include frames with annotation within the video clip
                for frame_idx in range(idx, idx + clip_len_in_frames):
                    if frame_idx in frame_idxs_with_anno:
                        clip_frame_ids.append(frame_idx)
                # Only include video clips that have at least self.frames_in_clip annotating frames
                if len(clip_frame_ids) >= self.frames_in_clip:
                    video_clips.append((sample_id, clip_frame_ids))

        return video_clips

    def entity2target(self, im: Image, entities: [AnnoEntity]):
        """
        Wrap up the entity to maskrcnn-benchmark compatible format - BoxList
        """
        boxes = [entity.bbox for entity in entities]
        ids = [int(entity.id) for entity in entities]
        # we only consider person tracking for now,
        # thus all the labels are 1,
        # reserve category 0 for background during training
        int_labels = [1 for _ in entities]

        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        boxes = BoxList(boxes, im.size, mode='xywh').convert('xyxy')
        if not self.amodal:
            boxes = boxes.clip_to_image(remove_empty=False)
        boxes.add_field('labels', torch.as_tensor(int_labels, dtype=torch.int64))
        boxes.add_field('ids', torch.as_tensor(ids, dtype=torch.int64))

        return boxes


class VideoDatasetBatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        image_batch = list(itertools.chain(*transposed_batch[0]))
        image_batch = to_image_list(image_batch, self.size_divisible)

        # to make sure that the id of each instance
        # are unique across the whole batch
        targets = transposed_batch[1]
        video_ids = transposed_batch[2]
        uid = 0
        video_id_map = defaultdict(dict)
        for targets_per_video, video_id in zip(targets, video_ids):
            for targets_per_video_frame in targets_per_video:
                if targets_per_video_frame.has_field('ids'):
                    _ids = targets_per_video_frame.get_field('ids')
                    _uids = _ids.clone()
                    for i in range(len(_ids)):
                        _id = _ids[i].item()
                        if _id not in video_id_map[video_id]:
                            video_id_map[video_id][_id] = uid
                            uid += 1
                        _uids[i] = video_id_map[video_id][_id]
                    targets_per_video_frame.extra_fields['ids'] = _uids

        targets = list(itertools.chain(*targets))

        return image_batch, targets, video_ids


if __name__ == "__main__":

    from siammot.data.adapters.utils.data_utils import load_dataset_anno

    torch.manual_seed(0)

    dataset_anno, dataset_info = load_dataset_anno('MOT17')
    collator = VideoDatasetBatchCollator()

    dataset = VideoDataset(dataset_anno,
                           frames_in_clip=2,
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
