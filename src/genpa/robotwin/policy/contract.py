from copy import deepcopy

from genpa.robotwin.depth.contract import DEPTH_PROMPT, checkpoint_contract as base_contract
from genpa.robotwin.depth.data import AUGMENTATION as BASE_AUGMENTATION

MODE = "genpa-depth"
AUGMENTATION = deepcopy(BASE_AUGMENTATION)


def checkpoint_contract(stats_path):
    contract = base_contract(stats_path, 192, "absolute_depth")
    contract.update(
        version="genpa_robotwin_depth_v1", mode=MODE,
        resolution={"camera_layout": "compact_288x256", "processor_hw": [240, 320],
                    "camera_hw": [[192, 256], [96, 128], [96, 128]], "final_hw": [288, 256]},
        preprocessing_order="resize_240x320_then_augment_then_resize_views_then_composite",
        augmentation=deepcopy(AUGMENTATION), visual_prompt=DEPTH_PROMPT,
        visual_loss_reduction="mean_all_tokens_and_channels_then_batch",
        visual_loss_validity_mask=False, camera_area_ratio=[4, 1, 1],
        visual_encoding={"type": "log_inverse_depth", "range_m": [0.01, 3.5],
                         "invalid_raw_mm": 0, "invalid_image_rgb01": [0, 0, 0],
                         "resize": "bilinear_antialias"},
    )
    return contract
