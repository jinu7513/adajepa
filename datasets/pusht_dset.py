import torch
import decord
import pickle
import numpy as np
import hashlib
import multiprocessing
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

    def get_frames(self, idx, frames):
        reader = VideoReader(str(self._video_path(idx, frames)), num_threads=1)
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        shape = self.shapes[idx]

        image = reader.get_batch(frames)  # THWC
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {'shape': shape}

    def _video_path(self, idx, frames):
        return self.data_path / "obses" / f"episode_{idx:03d}.mp4"

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0


class ColorPushTDataset(PushTDataset):
    """Same physical episodes/actions; deterministic window-level color draws.

    Coloring videos are generated separately from saved states. The window
    color does not depend on DataLoader worker order, so B and C see exactly
    the same physical windows and color choices under a common seed.
    """

    def __init__(self, *, color_domains, **kwargs):
        super().__init__(**kwargs)
        self.color_domains = tuple(color_domains)
        if len(self.color_domains) != 3 or len(set(self.color_domains)) != 3:
            raise ValueError("ColorPushTDataset requires exactly three distinct domains")

    def _video_path(self, idx, frames):
        start = frames.start if isinstance(frames, range) else int(frames[0])
        # Stable arithmetic instead of Python's process-randomized hash().
        domain = self.color_domains[(idx * 1009 + start) % len(self.color_domains)]
        path = self.data_path / "colors" / domain / "obses" / f"episode_{idx:03d}.mp4"
        if not path.is_file():
            raise FileNotFoundError(f"Paired-color video missing: {path}")
        return path


class FixedColorPushTDataset(PushTDataset):
    """Evaluation-only fixed appearance; never selected by the train loader."""

    def __init__(self, *, color_domain, **kwargs):
        super().__init__(**kwargs)
        self.color_domain = str(color_domain)

    def _video_path(self, idx, frames):
        path = self.data_path / "colors" / self.color_domain / "obses" / f"episode_{idx:03d}.mp4"
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation color video missing: {path}")
        return path


class ColorPairPushTDataset(PushTDataset):
    """Two paired renders of one physical window, deterministic across workers."""

    def __init__(self, *, color_domains, color_pair_mode, color_seed=0, **kwargs):
        super().__init__(**kwargs)
        self.color_domains = tuple(color_domains)
        self.color_pair_mode = str(color_pair_mode)
        self.color_seed = int(color_seed)
        self.epoch_shared = multiprocessing.Value("i", 0)
        if len(self.color_domains) != 3 or len(set(self.color_domains)) != 3:
            raise ValueError("Exactly three distinct training colors are required")
        if self.color_pair_mode not in ("same_fixed", "same_window_random", "cross_fixed", "cross_window_random"):
            raise ValueError("Unknown color pair mode: {}".format(self.color_pair_mode))

    def set_epoch(self, epoch):
        self.epoch_shared.value = int(epoch)

    def pair_for(self, episode, start):
        episode, start = int(episode), int(start)
        colors = self.color_domains
        if self.color_pair_mode == "same_fixed":
            color = colors[(episode * 1009 + start) % 3]
            return color, color
        pairs = [(a, b) for a in colors for b in colors if a != b]
        if self.color_pair_mode == "cross_fixed":
            return pairs[(episode * 1009 + start) % len(pairs)]
        key = "{}:{}:{}:{}".format(self.color_seed, self.epoch_shared.value, episode, start)
        draw = int.from_bytes(hashlib.blake2b(key.encode("ascii"), digest_size=8).digest(), "big")
        if self.color_pair_mode == "same_window_random":
            color = colors[draw % 3]
            return color, color
        return pairs[draw % len(pairs)]

    def _paired_path(self, episode, color):
        path = self.data_path / "colors" / color / "obses" / "episode_{:03d}.mp4".format(episode)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return path

    def _video_path(self, idx, frames):
        start = frames.start if isinstance(frames, range) else int(frames[0])
        return self._paired_path(idx, self.pair_for(idx, start)[0])

    def get_frames(self, idx, frames):
        obs, act, state, info = super().get_frames(idx, frames)
        start = frames.start if isinstance(frames, range) else int(frames[0])
        input_color, target_color = self.pair_for(idx, start)
        if input_color == target_color:
            obs["visual_target"] = obs["visual"].clone()
        else:
            reader = VideoReader(str(self._paired_path(idx, target_color)), num_threads=1)
            target = reader.get_batch(frames) / 255.0
            target = rearrange(target, "T H W C -> T C H W")
            obs["visual_target"] = self.transform(target) if self.transform else target
        return obs, act, state, info


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
    color_domains=None,
    validation_color_domain=None,
    color_pair_mode=None,
    color_seed=0,
):
    if color_pair_mode is not None and color_domains is None:
        raise ValueError("Color pair mode requires training colors")
    if validation_color_domain is not None and color_domains is None:
        raise ValueError("Held-out validation color requires color_domains for training")
    dataset_type = (ColorPairPushTDataset if color_pair_mode is not None else
                    ColorPushTDataset if color_domains is not None else PushTDataset)
    color_kwargs = ({"color_domains": color_domains, "color_pair_mode": color_pair_mode,
                     "color_seed": color_seed} if color_pair_mode is not None else
                    {"color_domains": color_domains} if color_domains is not None else {})
    train_dset = dataset_type(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/train",
        normalize_action=normalize_action,
        with_velocity=with_velocity,
        **color_kwargs,
    )
    val_type = FixedColorPushTDataset if validation_color_domain is not None else dataset_type
    val_kwargs = {"color_domain": validation_color_domain} if validation_color_domain is not None else color_kwargs
    val_dset = val_type(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/val",
        normalize_action=normalize_action,
        with_velocity=with_velocity,
        **val_kwargs,
    )

    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)

    datasets = {}
    datasets["train"] = train_slices
    datasets["valid"] = val_slices
    traj_dset = {}
    traj_dset["train"] = train_dset
    traj_dset["valid"] = val_dset
    return datasets, traj_dset
