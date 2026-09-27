import hydra
from omegaconf import OmegaConf
from tracka.data import generate


@hydra.main(version_base="1.1", config_path="conf", config_name="tracka")
def main(cfg):
    result = generate(OmegaConf.to_container(cfg, resolve=True))
    print({k: result[k] for k in ("preview_path", "accepted", "sample_count") if k in result})


if __name__ == "__main__":
    main()
