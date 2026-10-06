"""W&B is a mirror: locally flushed JSONL is the durable scalar record."""
import json
import os
import traceback
import warnings
from pathlib import Path

from PIL import Image

from .common import write_json


class RunLogger:
    def __init__(self, folder, cfg, metadata, resume_id=None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.cfg, self.run, self.error, self.resume_id = cfg, None, None, resume_id
        self.records = open(self.folder / "metrics.jsonl", "a", encoding="utf-8")
        write_json(self.folder / "run_metadata.json", metadata)
        self.wandb = None

    @property
    def run_id(self):
        return self.run.id if self.run is not None else self.resume_id

    def __enter__(self):
        write_json(self.folder / "run_status.json", {"status": "running", "wandb_mode": self.cfg["mode"]})
        if self.cfg["mode"] == "disabled":
            return self
        if self.cfg["mode"] not in ("online", "offline"):
            self.records.close()
            raise ValueError("logging.mode must be online, offline, or disabled")
        try:
            # Artifact/table staging must be writable on shared cluster nodes too.
            # Respect explicit user paths; otherwise keep runtime caches with outputs.
            for variable, subdir in (("WANDB_CACHE_DIR", "cache"), ("WANDB_DATA_DIR", "staging")):
                path = self.folder.resolve() / "wandb_runtime" / subdir
                os.environ.setdefault(variable, str(path))
                Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)
            import wandb
            self.wandb = wandb
            metadata = json.loads((self.folder / "run_metadata.json").read_text(encoding="utf-8"))
            kwargs = dict(project=self.cfg["project"], entity=self.cfg["entity"],
                          name=self.cfg["name"], group=self.cfg["group"],
                          mode=self.cfg["mode"], dir=str(self.folder), config=metadata,
                          settings=wandb.Settings(init_timeout=30))
            if self.resume_id and self.cfg["mode"] == "online":
                kwargs.update(id=self.resume_id, resume="must")
            self.run = wandb.init(**kwargs)
            self.run.define_metric("train/global_step")
            for namespace in ("clean/*", "render/*", "corruption/*", "latent/*", "loss/*", "mask/*", "val/*", "train/*", "images/*",
                              "event/*", "gate/*", "temporal/*", "probe/*", "collapse/*", "teacher/*"):
                self.run.define_metric(namespace, step_metric="train/global_step")
            write_json(self.folder / "wandb_identity.json", {"id": self.run.id,
                       "project": self.cfg["project"], "entity": self.run.entity,
                       "mode": self.cfg["mode"]})
        except Exception as exc:
            self.error = str(exc)
            if not self.cfg["fallback_to_local"]:
                self.records.close()
                write_json(self.folder / "run_status.json", {"status": "failed", "wandb_error": self.error})
                raise
            warnings.warn(f"W&B unavailable; metrics continue locally: {exc}")
        return self

    def log(self, metrics):
        record = {k: float(v) for k, v in metrics.items()}
        self.records.write(json.dumps(record, allow_nan=False) + "\n")
        self.records.flush()
        if self.run is not None:
            try:
                self.run.log(record)
            except Exception as exc:
                self.error = str(exc)
                warnings.warn(f"W&B log failed; local metrics retained: {exc}")

    def image(self, image, step):
        path = self.folder / f"reconstruction_{step:08d}.png"
        Image.fromarray(image).save(path)
        if self.run is not None:
            try:
                self.run.log({"train/global_step": step,
                              "images/reconstruction": self.wandb.Image(str(path), caption="Rows clean/A/B/corrupt; columns input/masked/prediction")})
            except Exception as exc:
                self.error = str(exc)
                warnings.warn(f"W&B image upload failed; local image retained: {exc}")

    def table(self, name, rows):
        write_json(self.folder / (name + ".json"), rows)
        if self.run is not None and rows:
            columns = sorted(set().union(*(r.keys() for r in rows)))
            try:
                self.run.log({name: self.wandb.Table(columns=columns,
                              data=[[r.get(c) for c in columns] for r in rows])})
            except Exception as exc:
                self.error = str(exc)
                warnings.warn(f"W&B table upload failed; local table retained: {exc}")

    def robustness_plots(self, rows, figure):
        """Tables retain exact numbers; custom charts expose usable condition axes."""
        if self.run is None:
            return
        try:
            payload = {"plots/robustness": self.wandb.Image(str(figure))}
            full = [r for r in rows if r["segment"] == "full"]
            render = [r for r in full if r["corruption_type"] is None]
            table = self.wandb.Table(columns=["condition", "standardized_mse"],
                                     data=[[r["condition"], r["standardized_mse"]] for r in render])
            payload["plots/render_probe"] = self.wandb.plot.bar(table, "condition", "standardized_mse",
                                                                title="Clean-trained probe under color shift")
            for kind in ("gaussian", "salt_pepper", "blur"):
                table = self.wandb.Table(columns=["strength", "standardized_mse"],
                    data=[[r["strength"], r["standardized_mse"]] for r in full if r["corruption_type"] == kind])
                payload["plots/" + kind] = self.wandb.plot.line(table, "strength", "standardized_mse",
                                                               title=kind + " robustness")
            self.run.log(payload)
        except Exception as exc:
            self.error = str(exc)
            warnings.warn(f"W&B chart upload failed; local figure/tables retained: {exc}")

    def __exit__(self, kind, value, tb):
        self.records.close()
        if kind is not None:
            (self.folder / "exception.txt").write_text("".join(traceback.format_exception(kind, value, tb)), encoding="utf-8")
        try:
            if self.run is not None:
                self.run.finish(exit_code=0 if kind is None else 1)
        except Exception as exc:
            self.error = str(exc)
            warnings.warn(f"W&B finalization failed: {exc}")
        write_json(self.folder / "run_status.json", {"status": "finished" if kind is None else "failed",
                   "wandb_error": self.error, "wandb_run_id": self.run_id})
        return False
