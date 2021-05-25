import os
import cv2
import subprocess
from pathlib import Path


class VisWriter:
    """
    Write the artifact of tracking results
    """

    def __init__(self, fps=None, dump_video=False, out_path=None, file_name=None):
        if fps is None:
            self._fps = 30
        else:
            self._fps = fps

        self._dump_video = dump_video
        self._video_writer = None
        self._video_path = None

        if out_path is None:
            file_path = Path(os.path.abspath(__file__))
            workplace_dir = str(file_path.parent.parent.absolute())
            self._output_path = os.path.join(workplace_dir, 'demo_vis')
        else:
            self._output_path = out_path

        if file_name is None:
            self._file_name = 'debug.mp4'
        else:
            self._file_name = file_name

        os.makedirs(self._output_path, exist_ok=True)

    def _init_video_writer(self, frame_width, frame_height):
        self._video_path = os.path.join(self._output_path, self._file_name)
        self._video_writer = cv2.VideoWriter(str(self._video_path), cv2.VideoWriter_fourcc(*'avc1'), self._fps,
                                            (int(frame_width), int(frame_height)))

    def close_video_writer(self):
        assert (self._video_writer is not None)
        self._video_writer.release()

        # compress the videos
        comp_video_path = self._video_path.replace(".mp4", "_comp.mp4")
        subprocess.run(["ffmpeg", "-i",
                        str(self._video_path),
                        "-vcodec", "libx264",
                        "-crf", "26",
                        str(comp_video_path)])
        subprocess.run(["rm", str(self._video_path)])
        subprocess.run(["mv", str(comp_video_path), self._video_path])

    def dump_artifacts(self, frame, frame_id):
        height, width, _ = frame.shape

        if self._dump_video:
            if self._video_writer is None:
                self._init_video_writer(width, height)
            self._video_writer.write(frame)
        else:
            output_path = os.path.join(self._output_path,
                                       "res_{:06d}.jpg".format(frame_id))
            cv2.imwrite(output_path, frame)

