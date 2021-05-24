import torch.utils.data

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.data.build import make_data_sampler, make_batch_data_sampler
from maskrcnn_benchmark.data.datasets.concat_dataset import ConcatDataset

from .video_dataset import VideoDataset, VideoDatasetBatchCollator
from .image_dataset import ImageDataset
from .adapters.utils.data_utils import load_dataset_anno
from .adapters.augmentation.build_augmentation import build_siam_augmentation
from .adapters.handler.data_filtering import build_data_filter_fn


def build_dataset(cfg):
    """

    """

    dataset_list = cfg.DATASETS.TRAIN
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )

    datasets = []
    for dataset_key in dataset_list:
        dataset_anno, dataset_info = load_dataset_anno(cfg, dataset_key)
        modality = dataset_info['modality']
        transforms = build_siam_augmentation(cfg, is_train=True, modality=modality)
        data_filter_fn = build_data_filter_fn(dataset_key, is_train=True)

        if modality == 'image':
            assert 'image_folder' in dataset_info
            _dataset = ImageDataset(dataset_anno,
                                    dataset_info['image_folder'],
                                    transforms=transforms,
                                    frames_per_image=cfg.VIDEO.RANDOM_FRAMES_PER_CLIP,
                                    amodal=cfg.INPUT.AMODAL)
        else:
            _dataset = VideoDataset(dataset_anno,
                                    sampling_interval=cfg.VIDEO.TEMPORAL_SAMPLING,
                                    clip_len=cfg.VIDEO.TEMPORAL_WINDOW,
                                    transforms=transforms,
                                    filter_fn=data_filter_fn,
                                    frames_in_clip=cfg.VIDEO.RANDOM_FRAMES_PER_CLIP,
                                    amodal=cfg.INPUT.AMODAL)
        datasets.append(_dataset)

    dataset = ConcatDataset(datasets)

    return dataset


def build_train_data_loader(cfg, is_distributed=False, start_iter=0):

    num_gpus = get_world_size()

    video_clips_per_batch = cfg.SOLVER.VIDEO_CLIPS_PER_BATCH
    assert (
        video_clips_per_batch % num_gpus == 0
    ), "SOLVER.VIDEO_CLIPS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
        video_clips_per_batch, num_gpus)

    video_clips_per_gpu = video_clips_per_batch // num_gpus

    dataset = build_dataset(cfg)
    num_iters = cfg.SOLVER.MAX_ITER
    sampler = make_data_sampler(dataset, True, is_distributed)
    batch_sampler = make_batch_data_sampler(
        dataset, sampler, [], video_clips_per_gpu, num_iters, start_iter
    )

    num_workers = cfg.DATALOADER.NUM_WORKERS
    collator = VideoDatasetBatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
    data_loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers,
                                              batch_sampler=batch_sampler, collate_fn=collator)
    return data_loader
