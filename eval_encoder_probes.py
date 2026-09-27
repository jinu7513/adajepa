import hydra
from omegaconf import OmegaConf
from tracka.probes import evaluate


@hydra.main(version_base="1.1", config_path="conf", config_name="tracka")
def main(cfg):
    evaluate(OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    main()
