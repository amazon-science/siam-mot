import logging
import argparse
import os

from demos.demo_inference import DemoInference
from demos.utils.vis_generator import VisGenerator
from demos.utils.vis_writer import VisWriter
from demos.video_iterator import build_video_iterator

parser = argparse.ArgumentParser(" SiamMOT Inference Demo")
parser.add_argument('--demo-video', metavar="FILE", type=str,
                    required=True)
parser.add_argument('--track-class', type=str, choices=('person', 'person_vehicle'),
                    default='person',
                    help='Tracking person or person/vehicle jointly')
parser.add_argument("--dump-video", type=bool, default=False,
                    help="Dump the videos as results")
parser.add_argument("--vis-resolution", type=int, default=1080)
parser.add_argument("--output-path", type=str, default=None,
                    help='The path of dumped videos')


if __name__ == '__main__':
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.INFO)

    # Build visulization generator and writer
    vis_generator = VisGenerator(vis_height=args.vis_resolution)
    vis_writer = VisWriter(dump_video=args.dump_video,
                           out_path=args.output_path,
                           file_name=os.path.basename(args.demo_video))

    # Build demo inference
    tracker = DemoInference(track_class=args.track_class,
                            vis_generator=vis_generator,
                            vis_writer=vis_writer)

    # Build video iterator for inference
    video_reader = build_video_iterator(args.demo_video)

    results = list(tracker.process_frame_sequence(video_reader()))

    if args.dump_video:
        vis_writer.close_video_writer()


