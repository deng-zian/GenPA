import math

import torch
from torchvision.transforms import functional as F

from genpa.datasets.lerobot.transforms.image import VideoAugmentation
from genpa.robotwin.depth.data import AUGMENTATION

SIZES = ((192, 256), (96, 128), (96, 128))


def depth_rgb01(depth_mm):
    depth = depth_mm.float() / 1000.0
    gray = 1.0 - torch.log(depth.clamp(0.01, 3.5) / 0.01) / math.log(3.5 / 0.01)
    gray = torch.where(depth_mm > 0, gray, 0.0)
    return gray.unsqueeze(-3).expand(*gray.shape[:-2], 3, *gray.shape[-2:])


def resize(value, size, interpolation):
    return F.resize(value, list(size), interpolation=interpolation,
                    antialias=interpolation == F.InterpolationMode.BILINEAR)


def composite(views):
    return torch.cat([views[0], torch.cat([views[1], views[2]], dim=-1)], dim=-2)


class PairedAugmentation(VideoAugmentation):
    """Record official RGB geometry and replay it on numerical targets without jitter."""

    def __init__(self):
        super().__init__(**AUGMENTATION)
        self.geometry = []

    def _apply_random_resized_crop(self, frames):
        params = self._sample_crop_params(frames)
        self.geometry.append([params, None])
        return self._apply_crop_params(frames, params, F.InterpolationMode.BILINEAR)

    def _apply_rotate(self, frames):
        angle = self._sample_rotation_angle(frames.device)
        self.geometry[-1][1] = angle
        return self._rotate_with_angle(frames, angle, F.InterpolationMode.BILINEAR,
                                       float(frames.mean().item()))

    def paired(self, rgb, target, interpolation, fill):
        self.geometry = []
        rgb = self(rgb.unsqueeze(1)).squeeze(1)
        if not self.geometry:
            return rgb, target
        targets = []
        for image, (crop, angle) in zip(target, self.geometry, strict=True):
            image = self._apply_crop_params(image.unsqueeze(0), crop, interpolation)
            image = self._rotate_with_angle(image, angle, interpolation, fill)
            targets.append(image.squeeze(0).clamp(0.0, 1.0))
        return rgb, torch.stack(targets)


def prepare_views(rgb, depth_mm, augmentation=None):
    if tuple(rgb.shape) != (3, 3, 240, 320) or tuple(depth_mm.shape) != (3, 240, 320):
        raise ValueError(f"Unexpected native RGB-D shape: {rgb.shape}, {depth_mm.shape}")
    interpolation = F.InterpolationMode.BILINEAR
    target = depth_rgb01(depth_mm)
    rgb = resize(rgb.float() / 255.0, (240, 320), F.InterpolationMode.BILINEAR)
    target = resize(target, (240, 320), interpolation)
    if augmentation is not None:
        rgb, target = augmentation.paired(rgb, target, interpolation, 0.0)
    reference = composite([resize(view, size, F.InterpolationMode.BILINEAR) for view, size in zip(rgb, SIZES)])
    target = composite([resize(view, size, interpolation) for view, size in zip(target, SIZES)])
    return {"reference": reference * 2.0 - 1.0, "target": target * 2.0 - 1.0}
