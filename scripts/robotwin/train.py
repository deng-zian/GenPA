import argparse
import logging
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from genpa.robotwin.runtime import _resolve_train_device
from genpa.utils.logging_config import setup_logging
from genpa.utils.misc import register_work_dir
from genpa.utils.pytorch_utils import set_global_seed
from genpa.robotwin.depth.trainer import RGBDTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    register_work_dir(cfg.output_dir)
    setup_logging(log_level=logging.INFO)
    set_global_seed(cfg.seed)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=_resolve_train_device())
    trainer = RGBDTrainer(model=model, train_dataset=instantiate(cfg.data.train),
                         val_dataset=instantiate(cfg.data.val), cfg=cfg)
    try:
        trainer.train()
    finally:
        trainer._finish_wandb()


if __name__ == "__main__":
    main()
