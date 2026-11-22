from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode, functional as TF

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DEPTH_MIN_M = 0.04
DEPTH_MAX_M = 3.5


@dataclass(frozen=True)
class Resolution:
    content_height: int
    content_width: int
    model_height: int
    camera_width: int

    @classmethod
    def from_height(cls, height):
        height = int(height)
        if height not in (192, 196, 224, 240):
            raise ValueError("Supported content heights: 192, 196, 224, 240")
        width = round(height * 4 / 3)
        return cls(height, width, math.ceil(height / 16) * 16, math.ceil(width / 16) * 16)

    @property
    def padding(self):
        dy = self.model_height - self.content_height
        dx = self.camera_width - self.content_width
        return (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2)

    @property
    def width(self):
        return self.camera_width * 3

    @property
    def tokens(self):
        return (self.model_height // 16) * (self.width // 16)

    def metadata(self):
        return {**asdict(self), "padding_lrtb": self.padding, "model_width": self.width, "reference_tokens": self.tokens}


def unpack_depth(packed_rgb):
    if packed_rgb.dtype != torch.uint8 or packed_rgb.shape[-3] != 3:
        raise ValueError("Depth packing requires uint8 RGB [...,3,H,W]")
    if torch.any(packed_rgb[..., 2, :, :] != 0):
        raise ValueError("Depth packing blue byte must be zero; check decoder/color order")
    return (packed_rgb[..., 0, :, :].to(torch.int32) << 8) | packed_rgb[..., 1, :, :].to(torch.int32)


def encode_depth(depth_mm):
    valid = depth_mm > 0
    depth_m = depth_mm.float().mul(0.001).clamp(DEPTH_MIN_M, DEPTH_MAX_M)
    gray = (math.log(DEPTH_MAX_M) - depth_m.log()) / math.log(DEPTH_MAX_M / DEPTH_MIN_M)
    gray = torch.where(valid, gray.clamp(0, 1), torch.zeros_like(gray))
    return gray.unsqueeze(-3).expand(*gray.shape[:-2], 3, *gray.shape[-2:]), valid


def resize_content(tensor, resolution, nearest=False):
    size = [resolution.content_height, resolution.content_width]
    if list(tensor.shape[-2:]) == size:
        return tensor
    return TF.resize(tensor, size, interpolation=InterpolationMode.NEAREST if nearest else InterpolationMode.BILINEAR, antialias=not nearest)


def pad_camera(tensor, resolution, validity=False):
    if validity:
        return F.pad(tensor, resolution.padding, mode="constant", value=0)
    return F.pad(tensor, resolution.padding, mode="replicate")


def prepare_views(rgb, depth_mm, height, augmentation=None):
    if tuple(rgb.shape) != (3, 3, 240, 320):
        raise ValueError(f"Expected three RGB cameras [3,3,240,320], got {rgb.shape}")
    resolution = Resolution.from_height(height)
    rgb_content = resize_content(rgb.float() / 255.0, resolution)
    target, validity = None, None
    if depth_mm is not None:
        if tuple(depth_mm.shape) != (3, 240, 320):
            raise ValueError(f"Depth/RGB mismatch: {depth_mm.shape}")
        target, native_valid = encode_depth(depth_mm)
        target = resize_content(target, resolution)
        validity = resize_content(native_valid[:, None].float(), resolution, nearest=True)
        target = target * validity
    if augmentation is not None:
        if target is None:
            rgb_content = augmentation.forward_structured(rgb_content[:, None], target_type=())[:, 0]
        else:
            frames = torch.stack([rgb_content, target, validity.expand(-1, 3, -1, -1)], dim=1)
            frames = augmentation.forward_structured(frames, target_type=("depth", "role_mask"))
            rgb_content, target, validity = frames[:, 0], frames[:, 1], frames[:, 2, :1]
            target = target * validity
    rgb_model = pad_camera(rgb_content, resolution)
    result = {"rgb_content": rgb_content, "rgb_model": rgb_model,
              "reference": torch.cat(list(rgb_model), dim=-1).mul(2).sub(1),
              "resolution": resolution}
    if target is not None:
        target_model = pad_camera(target, resolution)
        valid_model = pad_camera(validity, resolution, validity=True)
        result.update({"depth_content": target, "depth_model": target_model,
                       "target": torch.cat(list(target_model), dim=-1).mul(2).sub(1),
                       "visual_pixel_valid": torch.cat(list(valid_model), dim=-1)})
    return result
