import torch
import itertools
import numpy as np
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from maskrcnn_benchmark.utils.comm import get_world_size


class TensorboardWriter(SummaryWriter):
    def __init__(self, cfg, train_dir):
        if get_world_size() < 2 or dist.get_rank() == 0:
            super(TensorboardWriter, self).__init__(log_dir=train_dir)

        device = torch.device(cfg.MODEL.DEVICE)
        self.model_mean = torch.as_tensor(cfg.INPUT.PIXEL_MEAN, device=device)
        self.model_std = torch.as_tensor(cfg.INPUT.PIXEL_STD, device=device)

        self.image_to_bgr255 = cfg.INPUT.TO_BGR255

        # number of images per row during visualization
        self.num_col = cfg.VIDEO.RANDOM_FRAMES_PER_CLIP

    def __call__(self, iter, loss, loss_dict, images, targets):
        """

        :param iter:
        :param loss_dict:
        :param images:
        :return:
        """
        if get_world_size() < 2 or dist.get_rank() == 0:
            self.add_scalar('loss', loss.detach().cpu().numpy(), iter)
            for (_loss_key, _val) in loss_dict.items():
                self.add_scalar(_loss_key, _val.detach().cpu().numpy(), iter)

            # write down images / ground truths every 500 images
            if iter == 1 or iter % 500 == 0:
                show_images = images.tensors
                show_images = show_images.mul_(self.model_std[None, :, None, None]).\
                    add_(self.model_mean[None, :, None, None])

                # From RGB255 to BGR255
                if self.image_to_bgr255:
                    show_images = show_images[:, [2, 1, 0], :, :] / 255.

                # Detection ground truth
                merged_image, bbox_in_merged_image = self.images_with_boxes(show_images, targets)
                self.add_image_with_boxes('ground truth', merged_image, bbox_in_merged_image, iter)

    def images_with_boxes(self, images, boxes):
        """
        Get images inpainted with bounding boxes
        :param images: A batch of images are packed in a torch tensor BxCxHxW
        :param boxes:  A list of bounding boxes for the corresponding images
        :param ncols:
        """
        # To numpy array
        images = images.detach().cpu().numpy()
        # new stitched image
        batch, channels, height, width = images.shape
        assert batch % self.num_col == 0
        nrows = batch // self.num_col

        new_height = height * nrows
        new_width = width * self.num_col

        merged_image = np.zeros([channels, new_height, new_width])
        bbox_in_merged_image = []

        for img_idx in range(batch):
            row = img_idx // self.num_col
            col = img_idx % self.num_col
            merged_image[:, row * height:(row + 1) * height, col * width:(col + 1) * width] = \
                images[img_idx, :, :, :]
            box = boxes[img_idx].bbox.detach().cpu().numpy()
            if box.size > 0:
                box[:, 0] += col * width
                box[:, 1] += row * height
                box[:, 2] += col * width
                box[:, 3] += row * height
                bbox_in_merged_image.append(box)

        bbox_in_merged_image = np.array(list(itertools.chain(*bbox_in_merged_image)))

        return merged_image, bbox_in_merged_image
