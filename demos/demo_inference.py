import os
import logging
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import urllib
import zipfile

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

from siammot.configs.defaults import cfg
from siammot.modelling.rcnn import build_siammot
from siammot.data.adapters.augmentation.build_augmentation import build_siam_augmentation


class DemoInference:
    """
    Implement a wrapper to call tracker
    """

    def __init__(self,
                 gpu_id=0,
                 track_class=None,
                 vis_generator=None,
                 vis_writer=None):

        self.device = torch.device("cuda:{}".format(gpu_id))
        self.track_class = track_class

        cfg_file, model_path = self._get_artifacts()
        cfg.merge_from_file(cfg_file)
        self.cfg = cfg
        self.model_path = model_path

        self.transform = build_siam_augmentation(cfg, is_train=False)
        self.tracker = self._build_and_load_tracker()
        self.tracker.eval()

        self.vis_generator = vis_generator
        self.vis_writer = vis_writer

    def _get_artifacts(self):
        file_path = Path(os.path.abspath(__file__))
        workplace_dir = str(file_path.parent.absolute())

        # create demo model folder
        model_dir = os.path.join(workplace_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)

        # check the artifacts exist or not
        artifact_path = os.path.join(model_dir, '{}.zip'.format(self.track_class))
        if not os.path.exists(artifact_path):

            model_url = 'https://aws-cv-sci-motion-public.s3-us-west-2.amazonaws.com/SiamMOT/demo_models/{}.zip'\
                .format(self.track_class)

            logging.info('Downloading model and configurations...')
            urllib.request.urlretrieve(model_url, artifact_path)

            with zipfile.ZipFile(artifact_path, 'r') as zip_ref:
                zip_ref.extractall(model_dir)

        cfg_file = os.path.join(model_dir, self.track_class, 'DLA34_emm.yaml')
        if self.track_class == 'person':
            model_name = 'DLA34_emm_coco_crowdhuman.pth'
        else:
            model_name = 'DLA34_emm_coco_voc.pth'
        model_path = os.path.join(model_dir, self.track_class, model_name)

        return cfg_file, model_path

    def _preprocess(self, frame):

        # frame is RGB-Channel
        frame = Image.fromarray(frame, 'RGB')
        dummy_bbox = torch.tensor([[0, 0, 1, 1]])
        dummy_boxlist = BoxList(dummy_bbox, frame.size, mode='xywh')
        frame, _ = self.transform(frame, dummy_boxlist)

        return frame

    def _build_and_load_tracker(self):
        tracker = build_siammot(self.cfg)
        tracker.to(self.device)
        checkpointer = DetectronCheckpointer(cfg, tracker,
                                              save_dir=self.model_path)
        if os.path.isfile(self.model_path):
            _ = checkpointer.load(self.model_path)
        elif os.path.isdir(self.model_path):
            _ = checkpointer.load(use_latest=True)
        else:
            raise ValueError("No model parameters are loaded.")

        return tracker

    def process(self, frame):
        orig_h, orig_w, _ = frame.shape
        # frame should be RGB image
        frame = self._preprocess(frame)

        with torch.no_grad():
            results = self.tracker(frame.to(self.device))

        assert (len(results) == 1)
        results = results[0].to('cpu')
        results = results.resize([orig_w, orig_h]).convert('xywh')

        return results

    def process_frame_sequence(self, frame_iterator):
        self.tracker.reset_siammot_status()
        for frame_id, frame in tqdm(frame_iterator):
            orig_frame = frame[:, :, ::-1]
            results = self.process(frame)

            if self.vis_generator and self.vis_writer:
                vis_frame = self.vis_generator.frame_vis_generator(orig_frame, results)
                self.vis_writer.dump_artifacts(vis_frame, frame_id)

            yield frame_id, results
