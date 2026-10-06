import hydra
from omegaconf import OmegaConf

from tracka.event_data import generate_event_windows


@hydra.main(version_base="1.1", config_path="conf", config_name="tracka_event")
def main(cfg):
    result = generate_event_windows(OmegaConf.to_container(cfg, resolve=True))
    print({k: result[k] for k in ("preview_path", "accepted", "sample_count") if k in result})


if __name__ == "__main__":
    main()
