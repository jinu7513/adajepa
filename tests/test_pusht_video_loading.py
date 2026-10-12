"""Video-loading optimizations must preserve every requested PushT frame."""

import os
from pathlib import Path

import pytest
import torch

import datasets.pusht_dset as pusht
from datasets.traj_dset import TrajSlicerDataset


class CountingReader:
    opens = []
    batches = []

    def __init__(self, path, num_threads):
        self.opens.append(path)

    def get_batch(self, frames):
        frames = list(frames)
        self.batches.append(frames)
        return torch.stack(
            [torch.full((4, 4, 3), frame, dtype=torch.uint8) for frame in frames]
        )


def make_dataset(tmp_path, *, merge, cache_size, frame_cache_size=0):
    data = pusht.PushTDataset.__new__(pusht.PushTDataset)
    data.data_path = Path(tmp_path)
    data.actions = torch.zeros(2, 20, 2)
    data.states = torch.zeros(2, 20, 7)
    data.proprios = torch.zeros(2, 20, 4)
    data.shapes = ["T", "T"]
    data.transform = None
    data.include_previous_frames = True
    data.merge_video_reads = merge
    data.video_reader_cache_size = cache_size
    data.video_frame_cache_size = frame_cache_size
    data._video_reader_pid = None
    return data


def test_merged_reads_match_original_and_use_one_batch(monkeypatch):
    monkeypatch.setattr(pusht, "VideoReader", CountingReader)
    CountingReader.opens = []
    CountingReader.batches = []
    baseline = make_dataset("unused", merge=False, cache_size=0)
    original, _, _, _ = baseline.get_frames(0, [0, 5, 10])
    assert CountingReader.batches == [[0, 5, 10], [0, 4, 9]]

    CountingReader.batches = []
    merged = make_dataset("unused", merge=True, cache_size=0)
    optimized, _, _, _ = merged.get_frames(0, [0, 5, 10])
    assert CountingReader.batches == [[0, 4, 5, 9, 10]]
    for key in original:
        torch.testing.assert_close(optimized[key], original[key])


def test_reader_cache_is_bounded_and_process_local(monkeypatch):
    monkeypatch.setattr(pusht, "VideoReader", CountingReader)
    CountingReader.opens = []
    pid = [10]
    monkeypatch.setattr(pusht.os, "getpid", lambda: pid[0])
    data = make_dataset("unused", merge=True, cache_size=1)
    data.get_frames(0, [1])
    data.get_frames(0, [2])
    assert len(CountingReader.opens) == 1
    data.get_frames(1, [1])
    data.get_frames(0, [3])
    assert len(CountingReader.opens) == 3
    assert len(data._video_reader_cache) == 1
    pid[0] = 11
    data.get_frames(0, [4])
    assert len(CountingReader.opens) == 4
    assert not data.__getstate__()["_video_reader_cache"]


def test_decoded_frame_cache_reuses_adjacent_windows_without_changing_values(monkeypatch):
    monkeypatch.setattr(pusht, "VideoReader", CountingReader)
    CountingReader.opens = []
    CountingReader.batches = []
    pid = [10]
    monkeypatch.setattr(pusht.os, "getpid", lambda: pid[0])
    baseline = make_dataset("unused", merge=True, cache_size=1)
    first, _, _, _ = baseline.get_frames(0, [0, 5, 10, 15])
    second, _, _, _ = baseline.get_frames(0, [1, 6, 11, 16])
    assert CountingReader.batches[-1] == [0, 1, 5, 6, 10, 11, 15, 16]

    CountingReader.batches = []
    cached = make_dataset("unused", merge=True, cache_size=1, frame_cache_size=32)
    cached_first, _, _, _ = cached.get_frames(0, [0, 5, 10, 15])
    cached_second, _, _, _ = cached.get_frames(0, [1, 6, 11, 16])
    assert CountingReader.batches == [
        [0, 4, 5, 9, 10, 14, 15], [1, 6, 11, 16]
    ]
    for expected, actual in ((first, cached_first), (second, cached_second)):
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
    assert len(cached._video_frame_cache) == 11
    assert not cached.__getstate__()["_video_frame_cache"]
    pid[0] = 11
    cached.get_frames(0, [0, 5, 10, 15])
    assert CountingReader.batches[-1] == [0, 4, 5, 9, 10, 14, 15]

    # A cache smaller than one request must still return every requested frame.
    tiny = make_dataset("unused", merge=True, cache_size=0, frame_cache_size=2)
    tiny_obs, _, _, _ = tiny.get_frames(0, [0, 5, 10, 15])
    for key in first:
        torch.testing.assert_close(tiny_obs[key], first[key])
    assert len(tiny._video_frame_cache) == 2


def test_window_blocks_cover_exact_same_windows():
    class Trajectories:
        proprio_dim = 4
        action_dim = 2
        state_dim = 7

        def __len__(self):
            return 2

        def get_seq_length(self, idx):
            return 21

    original = TrajSlicerDataset(Trajectories(), 3, 2)
    grouped = TrajSlicerDataset(Trajectories(), 3, 2, window_block_size=4)
    assert sorted(map(tuple, original.slices)) == sorted(map(tuple, grouped.slices))
    for offset in range(0, len(grouped), 4):
        chunk = grouped.slices[offset:offset + 4]
        assert len({item[0] for item in chunk}) == 1
        assert [item[1] for item in chunk] == list(range(chunk[0][1], chunk[0][1] + 4))
    with pytest.raises(ValueError, match="positive"):
        TrajSlicerDataset(Trajectories(), 3, 2, window_block_size=0)


def test_merged_reads_match_real_video_if_dataset_is_available():
    root = Path(os.environ.get("DATASET_DIR", "")) / "pusht_noise" / "train"
    if not (root / "obses" / "episode_000.mp4").is_file():
        pytest.skip("Set DATASET_DIR to run the real PushT video equivalence check")
    data = pusht.PushTDataset(
        data_path=str(root), n_rollout=1, include_previous_frames=True,
        video_reader_cache_size=0, merge_video_reads=False,
    )
    baseline, _, _, _ = data.get_frames(0, [0, 5, 10, 15])
    data.merge_video_reads = True
    merged, _, _, _ = data.get_frames(0, [0, 5, 10, 15])
    for key in baseline:
        torch.testing.assert_close(merged[key], baseline[key])
    next_baseline, _, _, _ = data.get_frames(0, [1, 6, 11, 16])
    data.video_frame_cache_size = 32
    cached_first, _, _, _ = data.get_frames(0, [0, 5, 10, 15])
    cached_second, _, _, _ = data.get_frames(0, [1, 6, 11, 16])
    for expected, actual in ((baseline, cached_first), (next_baseline, cached_second)):
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
