"""Guardrails for the one-command Event dataset check and smoke run."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import run_event_smoke


class EventSmokeScriptTests(unittest.TestCase):
    def test_incomplete_generation_stops_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Dataset generation is not finished"):
                run_event_smoke.inspect_dataset(Path(folder))

    def test_valid_dataset_summary_and_exactly_two_update_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / "images"
            images.mkdir()
            for i in range(18):
                (images / f"{i}.png").touch()
            manifest = {"schema_version": "tracka-event-window-v1",
                        "image_count": 18, "rejected_count": 1}
            report = {"accepted": 3, "rejected": 1,
                      "accepted_by_split": {"train": 1, "validation": 1, "test": 1},
                      "accepted_by_shift": {"blur": 3}}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "generation_report.json").write_text(json.dumps(report), encoding="utf-8")
            (root / "samples.jsonl").write_text("{}\n", encoding="utf-8")
            rows = [{"split": split, "shift_kind": "blur"}
                    for split in ("train", "validation", "test")]
            fake_data = types.ModuleType("tracka.event_data")
            fake_data.validate_event_windows = lambda path: (manifest, rows)
            with mock.patch.dict(sys.modules, {"tracka.event_data": fake_data}):
                self.assertEqual(run_event_smoke.inspect_dataset(root), manifest)

            def fake_train(command, cwd, check):
                self.assertTrue(check)
                self.assertEqual(cwd, root)
                self.assertIn("training.total_optimizer_updates=2", command)
                self.assertIn("logging.fallback_to_local=false", command)
                output = Path(next(arg.split("=", 1)[1] for arg in command
                                   if arg.startswith("training.output_dir=")))
                output.mkdir(parents=True)
                (output / "checkpoint_latest.pt").touch()
                (output / "run_status.json").write_text(json.dumps({
                    "status": "finished", "wandb_error": None}), encoding="utf-8")
                (output / "metrics.jsonl").write_text(json.dumps({
                    "train/global_step": 2, "loss/total": 1.0}) + "\n", encoding="utf-8")
                (output / "wandb_identity.json").write_text(json.dumps({
                    "mode": "online", "entity": "test", "project": "event", "id": "abc"}),
                    encoding="utf-8")

            with mock.patch.object(run_event_smoke, "REPO", root), \
                    mock.patch.object(run_event_smoke.subprocess, "run", side_effect=fake_train):
                run_event_smoke.run_smoke(root)


if __name__ == "__main__":
    unittest.main()
