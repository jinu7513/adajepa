import torch
import decord
import pickle
import os
import numpy as np
from collections import OrderedDict
from pathlib import Path
from einops import rearrange
from decord import VideoReader
from typing import Callable, Optional
from .traj_dset import TrajDataset, TrajSlicerDataset
from typing import Optional, Callable, Any
decord.bridge.set_bridge("torch")

# precomputed dataset stats
ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
ACTION_STD = torch.tensor([0.2019, 0.2002])
STATE_MEAN = torch.tensor([236.6155, 264.5674, 255.1307, 266.3721, 1.9584, -2.93032027,  2.54307914])
STATE_STD = torch.tensor([101.1202, 87.0112, 52.7054, 57.4971, 1.7556, 74.84556075, 74.14009094])
PROPRIO_MEAN = torch.tensor([236.6155, 264.5674, -2.93032027,  2.54307914])
PROPRIO_STD = torch.tensor([101.1202, 87.0112, 74.84556075, 74.14009094])

class PushTDataset(TrajDataset):
    def __init__(
        self,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        data_path: str = "data/pusht_dataset",
        normalize_action: bool = True,
        relative=True,
        action_scale=100.0,
        with_velocity: bool = True, # agent's velocity
        include_previous_frames: bool = False,
        video_reader_cache_size: int = 0,
        merge_video_reads: bool = False,
    ):  
        self.data_path = Path(data_path)
        self.transform = transform
        self.relative = relative
        self.normalize_action = normalize_action
        self.states = torch.load(self.data_path / "states.pth")
        self.states = self.states.float()
        if relative:
            self.actions = torch.load(self.data_path / "rel_actions.pth")
        else:
            self.actions = torch.load(self.data_path / "abs_actions.pth")
        self.actions = self.actions.float()
        self.actions = self.actions / action_scale  # scaled back up in env

        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)
        
        # load shapes, assume all shapes are 'T' if file not found
        shapes_file = self.data_path / "shapes.pkl"
        if shapes_file.exists():
            with open(shapes_file, 'rb') as f:
                shapes = pickle.load(f)
                self.shapes = shapes
        else:
            self.shapes = ['T'] * len(self.states)

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.proprios = self.states[..., :2].clone()  # For pusht, first 2 dim of states is proprio
        # load velocities and update states and proprios
        self.with_velocity = with_velocity
        self.include_previous_frames = include_previous_frames
        if video_reader_cache_size < 0:
            raise ValueError("video_reader_cache_size must be nonnegative")
        self.video_reader_cache_size = video_reader_cache_size
        self.merge_video_reads = merge_video_reads
        # DataLoader workers must never share an open decoder across processes.
        self._video_reader_pid = os.getpid()
        self._video_reader_cache = OrderedDict()
        if with_velocity:
            self.velocities = torch.load(self.data_path / "velocities.pth")
            self.velocities = self.velocities[:n].float()
            self.states = torch.cat([self.states, self.velocities], dim=-1)
            self.proprios = torch.cat([self.proprios, self.velocities], dim=-1)
        print(f"Loaded {n} rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean = ACTION_MEAN
            self.action_std = ACTION_STD
            self.state_mean = STATE_MEAN[:self.state_dim]
            self.state_std = STATE_STD[:self.state_dim]
            self.proprio_mean = PROPRIO_MEAN[:self.proprio_dim]
            self.proprio_std = PROPRIO_STD[:self.proprio_dim]
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_video_reader_cache"] = OrderedDict()
        state["_video_reader_pid"] = None
        return state

    def _get_video_reader(self, idx):
        cache_size = getattr(self, "video_reader_cache_size", 0)
        path = self.data_path / "obses" / f"episode_{idx:03d}.mp4"
        if not cache_size:
            return VideoReader(str(path), num_threads=1)
        pid = os.getpid()
        if getattr(self, "_video_reader_pid", None) != pid:
            self._video_reader_cache = OrderedDict()
            self._video_reader_pid = pid
        cache = self._video_reader_cache
        if idx in cache:
            cache.move_to_end(idx)
            return cache[idx]
        reader = VideoReader(str(path), num_threads=1)
        cache[idx] = reader
        if len(cache) > cache_size:
            cache.popitem(last=False)
        return reader

    def get_frames(self, idx, frames):
        frames = list(frames)
        reader = self._get_video_reader(idx)
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        shape = self.shapes[idx]

        previous_frames = [max(0, frame - 1) for frame in frames]
        if self.include_previous_frames and getattr(self, "merge_video_reads", False):
            # Decode each requested frame at most once, in temporal order.
            requested = sorted(set(frames + previous_frames))
            positions = {frame: position for position, frame in enumerate(requested)}
            decoded = reader.get_batch(requested)
            image = decoded[[positions[frame] for frame in frames]]
            previous_image = decoded[[positions[frame] for frame in previous_frames]]
        else:
            image = reader.get_batch(frames)  # THWC
            previous_image = None
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        if self.include_previous_frames:
            if previous_image is None:
                previous_image = reader.get_batch(previous_frames)
            previous = previous_image / 255.0
            previous = rearrange(previous, "T H W C -> T C H W")
            if self.transform:
                previous = self.transform(previous)
            obs["visual_previous"] = previous
        return obs, act, state, {'shape': shape}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0


def load_pusht_slice_train_val(
    transform,
    n_rollout=50,
    data_path="data/pusht_dataset",
    normalize_action=True,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    with_velocity=True,
    include_previous_frames=False,
    video_reader_cache_size=0,
    merge_video_reads=False,
    window_block_size=1,
):
    train_dset = PushTDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/train",
        normalize_action=normalize_action,
        with_velocity=with_velocity,
        include_previous_frames=include_previous_frames,
        video_reader_cache_size=video_reader_cache_size,
        merge_video_reads=merge_video_reads,
    )
    val_dset = PushTDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/val",
        normalize_action=normalize_action,
        with_velocity=with_velocity,
        include_previous_frames=include_previous_frames,
        video_reader_cache_size=video_reader_cache_size,
        merge_video_reads=merge_video_reads,
    )

    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip, window_block_size=window_block_size)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip, window_block_size=window_block_size)

    datasets = {}
    datasets["train"] = train_slices
    datasets["valid"] = val_slices
    traj_dset = {}
    traj_dset["train"] = train_dset
    traj_dset["valid"] = val_dset
    return datasets, traj_dset
