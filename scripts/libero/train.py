import hydra
from omegaconf import DictConfig

from genpa.runtime import run_training
from genpa.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../../configs/libero", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
