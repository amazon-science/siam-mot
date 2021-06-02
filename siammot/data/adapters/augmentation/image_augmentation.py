import torch
import random
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F

import imgaug.augmenters as iaa

from maskrcnn_benchmark.structures.bounding_box import BoxList


class ImageResize(object):
    def __init__(self, min_size, max_size, size_divisibility):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.size_divisibility = size_divisibility

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        if self.size_divisibility > 0:
            oh = (int(oh / self.size_divisibility) * self.size_divisibility)
            ow = (int(ow / self.size_divisibility) * self.size_divisibility)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image, target
        target = target.resize(image.size)
        return image, target


class ImageCropResize(object):
    """
    Crop a patch from the image and resize to its original size
    """
    def __init__(self, crop_limit=None, amodal=False):
        self.crop_limit = crop_limit
        self.amodal = amodal

    def remove_invisible_box(self, box: BoxList):
        """
        Remove boxes that are not visible (out of image boundary) after motion augmentation
        """
        bbox = box.bbox.clone()
        xmin_clip = bbox[:, 0].clamp(min=0, max=box.size[0] - 1)
        ymin_clip = bbox[:, 1].clamp(min=0, max=box.size[1] - 1)
        xmax_clip = bbox[:, 2].clamp(min=0, max=box.size[0] - 1)
        ymax_clip = bbox[:, 3].clamp(min=0, max=box.size[1] - 1)
        keep = (xmax_clip > xmin_clip) & (ymax_clip > ymin_clip)

        return box[keep]

    def boxlist_crop(self, box: BoxList, x1, y1, x2, y2):
        """
         Adjust the coordinate of the bounding box within
         image crop specified by (x1, y1, x2, y2)
        """

        w, h = (x2 - x1), (y2 - y1)
        xmin, ymin, xmax, ymax = box._split_into_xyxy()
        cropped_xmin = (xmin - x1)
        cropped_ymin = (ymin - y1)
        cropped_xmax = (xmax - x1)
        cropped_ymax = (ymax - y1)
        cropped_bbox = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
        )
        cropped_box = BoxList(cropped_bbox, (w, h), mode="xyxy")
        for k, v in box.extra_fields.items():
            cropped_box.add_field(k, v)

        if self.amodal:
            # amodal allows the corners of bbox go beyond image boundary
            cropped_box = self.remove_invisible_box(cropped_box)
        else:
            # the corners of bbox need to be within image boundary for non-amodal training
            cropped_box = cropped_box.clip_to_image(remove_empty=True)
        return cropped_box.convert(box.mode)

    def __call__(self, image, target):
        w, h = image.size

        tl_x = int(w * (random.random() * self.crop_limit))
        tl_y = int(h * (random.random() * self.crop_limit))
        br_x = int(w - w * (random.random() * self.crop_limit))
        # keep aspect ratio
        br_y = int((h / w) * (br_x - tl_x) + tl_y)

        if len(target) > 0:
            box = target.bbox.clone()
            # get the visible part of the objects
            box_w = box[:, 2].clamp(min=0, max=target.size[0] - 1) - \
                    box[:, 0].clamp(min=0, max=target.size[0] - 1)
            box_h = box[:, 3].clamp(min=0, max=target.size[1] - 1) - \
                    box[:, 1].clamp(min=0, max=target.size[1] - 1)
            box_area = box_h * box_w
            max_area_idx = torch.argmax(box_area, dim=0)
            max_motion_limit_w = int(box_w[max_area_idx] * 0.25)
            max_motion_limit_h = int(box_h[max_area_idx] * 0.25)

            # make sure at least one bounding box is preserved
            # after motion augmentation
            tl_x = min(tl_x, max_motion_limit_w)
            tl_y = min(tl_y, max_motion_limit_h)
            br_x = max(br_x, w-max_motion_limit_w)
            br_y = max(br_y, h-max_motion_limit_h)

        assert (tl_x < br_x) and (tl_y < br_y)

        crop = F.crop(image, tl_y, tl_x, (br_y-tl_y), (br_x-tl_x))
        crop = F.resize(crop, (h, w))
        if len(target) > 0:
            target = self.boxlist_crop(target, tl_x, tl_y, br_x, br_y)
        target = target.resize(image.size)

        return crop, target


class ImageMotionBlur(object):
    """
    Perform motion augmentation to an image
    """
    def __init__(self):
        motion_blur = iaa.MotionBlur(k=10, angle=[-30, 30])
        gaussian_blur = iaa.GaussianBlur(sigma=(0.0, 2.0))

        self.blur_func_pool = [motion_blur, gaussian_blur]

        pass

    def __call__(self, image):
        blur_id = random.choice(list(range(0, len(self.blur_func_pool))))
        blur_func = self.blur_func_pool[blur_id]
        np_image = np.asarray(image)
        blurred_image = blur_func.augment_image(np_image)
        pil_image = Image.fromarray(np.uint8(blurred_image))
        return pil_image


class ImageCompression(object):
    """
    Perform JPEG compression augmentation to an image
    """
    def __init__(self, max_compression):
        self.max_compression = max_compression

    def __call__(self, image):
        ratio = random.uniform(0, 1)
        compression = min(100, int(ratio * self.max_compression))
        np_image = np.asarray(image)
        compressed_image = iaa.arithmetic.compress_jpeg(np_image, compression)
        pil_image = Image.fromarray(np.uint8(compressed_image))
        return pil_image


class ToTensor(object):
    def __call__(self, image, target=None):
        return F.to_tensor(image), target


class ToBGR255(object):
    def __init__(self, to_bgr255=True):
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        return image, target

