import hydra
from omegaconf import OmegaConf

from tracka.event_probe import evaluate_event_encoder


@hydra.main(version_base="1.1", config_path="conf", config_name="tracka_event")
def main(cfg):
    evaluate_event_encoder(OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    main()
