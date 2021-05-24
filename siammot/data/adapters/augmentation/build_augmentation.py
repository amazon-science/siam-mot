from .video_augmentation import SiamVideoResize, \
    SiamVideoColorJitter, SiamVideoCompressionAugment, SiamVideoMotionAugment, \
    SiamVideoMotionBlurAugment, SiamVideoRandomHorizontalFlip, VideoTransformer
from .image_augmentation import ToTensor, ToBGR255

import maskrcnn_benchmark.data.transforms as T


def build_siam_augmentation(cfg, is_train=True, modality='video'):

    motion_limit = 0.0
    motion_blur_prob = 0.0
    compression_limit = 0.0
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_horizontal_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE

        if modality == 'image':
            motion_limit = cfg.INPUT.MOTION_LIMIT
            motion_blur_prob = cfg.INPUT.MOTION_BLUR_PROB
            compression_limit = cfg.INPUT.COMPRESSION_LIMIT

    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_horizontal_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    amodal = cfg.INPUT.AMODAL
    SIZE_DIVISIBILITY = cfg.DATALOADER.SIZE_DIVISIBILITY
    to_bgr255 = cfg.INPUT.TO_BGR255

    video_color_jitter = SiamVideoColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    transform = Compose(
        [
            video_color_jitter,
            SiamVideoMotionBlurAugment(motion_blur_prob),
            SiamVideoCompressionAugment(compression_limit),
            SiamVideoMotionAugment(motion_limit, amodal),
            SiamVideoResize(min_size, max_size, SIZE_DIVISIBILITY),
            SiamVideoRandomHorizontalFlip(prob=flip_horizontal_prob),
            # PIL image
            VideoTransformer(ToTensor()),
            # Torch tensor, CHW (RGB format), and range from [0, 1]
            # VideoTransformer(ToBGR255(to_bgr255=to_bgr255))
            VideoTransformer(normalize_transform),
        ]
    )
    return transform


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string