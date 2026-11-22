from copy import deepcopy

import torch

from genpa.datasets.lerobot.transforms.image import VideoAugmentation
from genpa.robotwin.multiview.spatial import PairedAugmentation, prepare_views
from genpa.robotwin.depth.contract import DEPTH_PROMPT
from genpa.robotwin.depth.data import DepthDataset
from genpa.robotwin.depth.spatial import CAMERAS, unpack_depth
from .contract import AUGMENTATION, MODE


class RobotwinAugmentation(PairedAugmentation):
    def __init__(self):
        VideoAugmentation.__init__(self, **deepcopy(AUGMENTATION))
        self.geometry = []


def prepare_depth_views(rgb, depth_mm, augmentation=None):
    return prepare_views(rgb, depth_mm, augmentation)


class RobotwinDataset(DepthDataset):
    def __init__(self, *, mode=MODE, **kwargs):
        if mode != MODE:
            raise ValueError(mode)
        super().__init__(mode="absolute_depth", height=192, **kwargs)
        self.variant = mode
        self.video_augmentation = RobotwinAugmentation() if self.video_augmentation is not None else None

    def __getitem__(self, index):
        # The action pass reads the composite current RGB frame; the optional
        # perception pass reads the aligned composite depth target. Each
        # 128-token Qwen state is retrieved by its exact prompt string.
        self._worker()
        sample = self.numeric_sample(index)
        episode, frame = sample["episode_index"], sample["frame_index"]
        rgb = torch.stack([self._frame(episode, frame, camera) for camera in CAMERAS])
        depth = torch.stack([unpack_depth(self._frame(episode, frame, camera, True)) for camera in CAMERAS])
        prepared = prepare_depth_views(rgb, depth, self.video_augmentation)
        prompt = self._prompt(sample["task_index"])
        context, mask = self._context(prompt)
        if self._depth_context is None:
            self._depth_context = self._context(DEPTH_PROMPT)
        sample.update(prompt=prompt, text_hidden_states=context, text_attention_mask=mask,
                      video=torch.stack([prepared["reference"], prepared["target"]], dim=1),
                      visual_text_hidden_states=self._depth_context[0],
                      visual_text_attention_mask=self._depth_context[1])
        return sample
