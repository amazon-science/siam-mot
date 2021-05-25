import cv2
import numpy as np
from matplotlib import cm

from maskrcnn_benchmark.structures.bounding_box import BoxList


class VisGenerator:
    """
    Generate a video for visualization
    """
    def __init__(self, vis_height=None):
        """
        vis_height is the resolution of output frame
        """
        self._vis_height = vis_height
        # by default, 50 colors
        self.num_colors = 50
        self.colors = self.get_n_colors(self.num_colors)
        # use coco class name order
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']

    @staticmethod
    def get_n_colors(n, colormap="gist_ncar"):
        # Get n color samples from the colormap, derived from: https://stackoverflow.com/a/25730396/583620
        # gist_ncar is the default colormap as it appears to have the highest number of color transitions.
        # tab20 also seems like it would be a good option but it can only show a max of 20 distinct colors.
        # For more options see:
        # https://matplotlib.org/examples/color/colormaps_reference.html
        # and https://matplotlib.org/users/colormaps.html

        colors = cm.get_cmap(colormap)(np.linspace(0, 1, n))
        # Randomly shuffle the colors
        np.random.shuffle(colors)
        # Opencv expects bgr while cm returns rgb, so we swap to match the colormap (though it also works fine without)
        # Also multiply by 255 since cm returns values in the range [0, 1]
        colors = colors[:, (2, 1, 0)] * 255
        return colors

    def normalize_output(self, frame, results: BoxList):
        if self._vis_height is not None:
            boxlist_height = results.size[1]
            frame_height, frame_width = frame.shape[:2]
            assert (boxlist_height == frame_height)

            rescale_ratio = float(self._vis_height) / float(frame_height)
            new_height = int(round(frame_height * rescale_ratio))
            new_width = int(round(frame_width * rescale_ratio))

            frame = cv2.resize(frame, (new_width, new_height))
            results = results.resize((new_width, new_height))

        return frame, results

    def frame_vis_generator(self, frame, results: BoxList):
        frame, results = self.normalize_output(frame, results)
        ids = results.get_field('ids')
        results = results[ids >= 0]
        results = results.convert('xyxy')
        bbox = results.bbox.detach().cpu().numpy()
        ids = results.get_field('ids').tolist()
        labels = results.get_field('labels').tolist()

        for i, entity_id in enumerate(ids):
            color = self.colors[entity_id % self.num_colors]
            class_name = self.class_names[labels[i] - 1]
            text_width = len(class_name) * 20
            x1, y1, x2, y2 = (np.round(bbox[i, :])).astype(np.int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=3)
            cv2.putText(frame, str(entity_id), (x1 + 5, y1 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, thickness=3)
            # Draw black background rectangle for test
            cv2.rectangle(frame, (x1-5, y1-25), (x1+text_width, y1), color, -1)
            cv2.putText(frame, '{}'.format(class_name), (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
        return frame
