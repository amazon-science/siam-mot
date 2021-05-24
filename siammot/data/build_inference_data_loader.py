import torch
import torch.utils.data as data

from gluoncv.torch.data.gluoncv_motion_dataset.dataset import DataSample
from maskrcnn_benchmark.structures.bounding_box import BoxList


class InferenceVideoData(data.Dataset):
    """
    Split the video into small chunks (in an non-overlapping fashion) for inference
    """

    def __init__(self, video: DataSample, clip_len=1, transforms=None):
        """
        Construct a data loader for inference
        :param video: a video stream in DataSample format
        :param clip_len: the length of video clips
        :param transforms: transform function for video pre-processing
        """
        self.video = video
        self.video_reader = video.get_data_reader()
        self.clip_len = clip_len
        self.transforms = transforms
        self.clip_idxs = list(range(0, len(self.video), self.clip_len))

    def __getitem__(self, id):
        video_clip = []
        # this is needed for transformation
        dummy_boxes = []
        timestamps = []
        start_idx = self.clip_idxs[id]
        end_idx = min(len(self.video), start_idx + self.clip_len)
        for frame_idx in range(start_idx, end_idx):
            (im, timestamp, _) = self.video_reader[frame_idx]
            dummy_bbox = torch.tensor([[0, 0, 1, 1]])
            dummy_boxlist = BoxList(dummy_bbox, im.size, mode='xywh')

            video_clip.append(im)
            timestamps.append(torch.tensor(timestamp))
            dummy_boxes.append(dummy_boxlist)

        if self.transforms is not None:
            video_clip, _ = self.transforms(video_clip, dummy_boxes)

        return torch.stack(video_clip), start_idx, torch.stack(timestamps)

    def __len__(self):
        return len(self.clip_idxs)


def build_video_loader(cfg, video: DataSample, transforms):
    clip_len = cfg.INFERENCE.CLIP_LEN
    videodata = InferenceVideoData(video, clip_len=clip_len, transforms=transforms)
    videoloader = data.DataLoader(videodata, num_workers=4, batch_size=1, shuffle=False)

    return videoloader
