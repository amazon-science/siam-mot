import torch
import random
from torchvision.transforms import functional as F
from torchvision.transforms import ColorJitter as ImageColorJitter

from .image_augmentation import ImageResize, ImageCropResize, \
    ImageMotionBlur, ImageCompression


class VideoTransformer(object):
    def __init__(self, transform_fn=None):
        if transform_fn is None:
            raise KeyError('Transform function should not be None.')
        self.transform_fn = transform_fn

    def __call__(self, video, target=None):
        """
        A data transformation wrapper for video
        :param video: a list of images
        :param target: a list of BoxList (per image)
        """
        if not isinstance(video, (list, tuple)):
            return self.transform_fn(video, target)

        new_video = []
        new_target = []
        for (image, image_target) in zip(video, target):
            (image, image_target) = self.transform_fn(image, image_target)
            new_video.append(image)
            new_target.append(image_target)

        return new_video, new_target


class SiamVideoResize(ImageResize):
    def __init__(self, min_size, max_size, size_divisibility):
        super(SiamVideoResize, self).__init__(min_size, max_size, size_divisibility)

    def __call__(self, video, target=None):

        if not isinstance(video, (list, tuple)):
            return super(SiamVideoResize, self).__call__(video, target)

        assert len(video) >= 1
        new_size = self.get_size(video[0].size)

        new_video = []
        new_target = []
        for (image, image_target) in zip(video, target):
            (image, image_target) = self._resize(image, new_size, image_target)
            new_video.append(image)
            new_target.append(image_target)

        return new_video, new_target

    def _resize(self, image, size, target=None):
        image = F.resize(image, size)
        target = target.resize(image.size)
        return image, target


class SiamVideoRandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, video, target=None):

        if not isinstance(video, (list, tuple)):
            return video, target

        new_video = []
        new_target = []
        # All frames should have the same flipping operation
        if random.random() < self.prob:
            for (image, image_target) in zip(video, target):
                new_video.append(F.hflip(image))
                new_target.append(image_target.transpose(0))
        else:
            new_video = video
            new_target = target
        return new_video, new_target


class SiamVideoColorJitter(ImageColorJitter):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None):
        super(SiamVideoColorJitter, self).__init__(brightness, contrast, saturation, hue)

    def __call__(self, video, target=None):
        # Color jitter only applies for Siamese Training
        if not isinstance(video, (list, tuple)):
            return video, target

        idx = random.choice((0, 1))
        # all frames in the video should go through the same transformation
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        new_video = []
        new_target = []
        for i, (image, image_target) in enumerate(zip(video, target)):
            if i == idx:
                image = transform(image)
            new_video.append(image)
            new_target.append(image_target)

        return new_video, new_target


class SiamVideoMotionAugment(object):
    def __init__(self, motion_limit=None, amodal=False):
        # maximum motion augmentation
        self.motion_limit = min(0.1, motion_limit)
        if motion_limit is None:
            self.motion_limit = 0
        self.motion_augment = ImageCropResize(self.motion_limit, amodal)

    def __call__(self, video, target=None):

        # Motion augmentation only applies for Siamese Training
        if not isinstance(video, (list, tuple)) or self.motion_limit == 0:
            return video, target

        new_video = []
        new_target = []
        # Only 1 frame go through the motion augmentation,
        # the other unchanged
        idx = random.choice((0, 1))
        for i, (image, image_target) in enumerate(zip(video, target)):
            if i == idx:
                (image, image_target) = self.motion_augment(image, image_target)
            new_video.append(image)
            new_target.append(image_target)

        return new_video, new_target


class SiamVideoMotionBlurAugment(object):
    def __init__(self, motion_blur_prob=None):
        self.motion_blur_prob = motion_blur_prob
        if motion_blur_prob is None:
            self.motion_blur_prob = 0.0
        self.motion_blur_func = ImageMotionBlur()

    def __call__(self, video, target):
        # Blur augmentation only applies for Siamese Training
        if not isinstance(video, (list, tuple)) or self.motion_blur_prob == 0.0:
            return video, target

        new_video = []
        new_target = []
        idx = random.choice((0, 1))
        for i, (image, image_target) in enumerate(zip(video, target)):
            if i == idx:
                random_prob = random.uniform(0, 1)
                if random_prob < self.motion_blur_prob:
                    image = self.motion_blur_func(image)
            new_video.append(image)
            new_target.append(image_target)

        return new_video, new_target


class SiamVideoCompressionAugment(object):
    def __init__(self, max_compression=None):
        self.max_compression = max_compression
        if max_compression is None:
            self.max_compression = 0.0
        self.compression_func = ImageCompression(self.max_compression)

    def __call__(self, video, target):
        # Compression augmentation only applies for Siamese Training
        if not isinstance(video, (list, tuple)) or self.max_compression == 0.0:
            return video, target

        idx = random.choice((0, 1))
        new_video = []
        new_target = []
        for i, (image, image_target) in enumerate(zip(video, target)):
            if i == idx:
                image = self.compression_func(image)
            new_video.append(image)
            new_target.append(image_target)

        return new_video, new_target