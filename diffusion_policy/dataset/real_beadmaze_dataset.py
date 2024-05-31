import copy
import hashlib
import io
import json
import os
import shutil
from typing import Dict, List

import cv2
import numpy as np
import torch
import zarr
from filelock import FileLock
from omegaconf import OmegaConf
from PIL import Image
from threadpoolctl import threadpool_limits
from torchvision import transforms
import pytorch_kinematics as pk

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer


def compute_diff(img1, img2, offset=0.0):
    img1 = np.int32(img1)
    img2 = np.int32(img2)
    diff = img1 - img2
    diff = diff / 255.0 + offset
    diff = np.clip(diff, 0.0, 1.0)
    diff = np.uint8(diff * 255.0)
    return diff


def load_sample_from_buf(img, img_bg=None):
    if img_bg is not None:
        img = compute_diff(img, img_bg, offset=0.5)
    img = Image.fromarray(img)
    return img


def load_bin_image(io_buf):
    img = Image.open(io.BytesIO(io_buf))
    img = np.array(img)
    return img


def get_resize_transform(img_size=(224, 224)):
    t = transforms.Compose(
        [
            transforms.Resize((img_size[0], img_size[1]), antialias=True),
            transforms.ToTensor(),  # converts to [0 - 1]
        ]
    )
    return t


class RealBeadMazeImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        n_latency_steps=0,
        use_cache=False,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        delta_action=False,
        tactile_input: str = {},
    ):
        assert os.path.isdir(dataset_path)
        replay_buffer = None
        if use_cache:
            # fingerprint shape_meta
            shape_meta_json = json.dumps(
                OmegaConf.to_container(shape_meta), sort_keys=True
            )
            shape_meta_hash = hashlib.md5(shape_meta_json.encode("utf-8")).hexdigest()
            cache_zarr_path = os.path.join(dataset_path, shape_meta_hash + ".zarr.zip")
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print("Cache does not exist. Creating!")
                        replay_buffer = _get_replay_buffer(
                            dataset_path=dataset_path,
                            shape_meta=shape_meta,
                            store=zarr.MemoryStore(),
                        )
                        print("Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print("Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                    print("Loaded!")
        else:
            replay_buffer = _get_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=zarr.MemoryStore(),
            )

        if delta_action:
            # replace action as relative to previous frame
            actions = replay_buffer["action"][:]
            # support positions only at this time
            assert actions.shape[1] <= 3
            actions_diff = np.zeros_like(actions)
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i - 1]
                end = episode_ends[i]
                # delta action is the difference between previous desired position and the current
                # it should be scheduled at the previous timestep for the current timestep
                # to ensure consistency with positional mode
                actions_diff[start + 1 : end] = np.diff(actions[start:end], axis=0)
            replay_buffer["action"][:] = actions_diff

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed
        )

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )
        data = sampler.sample_sequence(0)
        self.thumb_bg = load_sample_from_buf(data["digit_thumb"][0])
        self.index_bg = load_sample_from_buf(data["digit_index"][0])

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = ["digit_thumb", "digit_index"]  # rgb_keys
        # self.rgb_keys = rgb_keys
        self.lowdim_keys = ["robot_joint", "allegro_joint"]  # lowdim_keys
        # self.lowdim_keys = ["robot_eef_pose"]  # lowdim_keys
        # self.lowdim_keys = []
        # self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.img_loader = load_sample_from_buf
        self.transform_resize = get_resize_transform()

        self.out_format = tactile_input["out_format"]  # if output video
        assert self.out_format in [
            "video",
            "concat_ch_img",
            "single_image",
        ], ValueError(
            "out_format should be 'video' or 'concat_ch_img' or 'single_image'"
        )
        frame_stride = tactile_input["frame_stride"]
        self.num_frames = (
            1 if self.out_format == "single_image" else tactile_input["num_frames"]
        )
        self.frames_concat_idx = np.arange(
            0, self.num_frames * frame_stride, frame_stride
        )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action

        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["action"]
        )
        normalizer["allegro_action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["allegro_action"]
        )

        # obs
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key]
            )

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_identity_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def _get_tactile_images(self, data, T_slice, bg=None):
        data_slice = []
        for i in range(T_slice.stop):
            idx_start = i + 5
            sample_images = []
            for i in self.frames_concat_idx:
                idx_sample = idx_start - i
                # print(f"idx_sample: {idx_sample}")
                image = self.img_loader(data[idx_sample], bg)
                image = self.transform_resize(image)
                sample_images.append(image)

            if self.out_format == "single_image":
                output = sample_images[0]
            elif self.out_format == "video":
                output = torch.stack(sample_images, dim=0)
                output = output.permute(1, 0, 2, 3)
            elif self.out_format == "concat_ch_img":
                output = torch.cat(sample_images, dim=0)

            data_slice.append(output)

        output = torch.stack(data_slice, dim=0)
        return output

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            threadpool_limits(1)
            data = self.sampler.sample_sequence(idx)

            T_slice = slice(self.n_obs_steps)

            # rgb_keys = ['digit_thumb', 'digit_index']
            # lowdim_keys = ['robot_joint', 'allegro_joint']
            obs_dict = dict()
            for key in self.rgb_keys:
                bg = None
                if key == "digit_thumb":
                    bg = self.thumb_bg
                elif key == "digit_index":
                    bg = self.index_bg
                obs_dict[key] = self._get_tactile_images(data[key], T_slice, bg=bg)
                del data[key]
            for key in self.lowdim_keys:
                obs_dict[key] = data[key][T_slice].astype(np.float32)
                del data[key]

            action = data["action"].astype(np.float32)
            allegro_action = data["allegro_action"].astype(np.float32)
            # handle latency by dropping first n_latency_steps action
            # observations are already taken care of by T_slice
            if self.n_latency_steps > 0:
                action = action[self.n_latency_steps :]
                allegro_action = allegro_action[self.n_latency_steps :]

            torch_data = {
                "obs": obs_dict,
                "action": torch.from_numpy(action),
                "allegro_action": torch.from_numpy(allegro_action),
            }
            return torch_data
        except Exception as e:
            print(f"Error: {e}")
            return self.__getitem__(np.clip(idx + 1, 0, len(self) - 1))


def zarr_resize_index_last_dim(zarr_arr, idxs):
    actions = zarr_arr[:]
    actions = actions[..., idxs]
    zarr_arr.resize(zarr_arr.shape[:-1] + (len(idxs),))
    zarr_arr[:] = actions
    return zarr_arr


def _get_replay_buffer(dataset_path, shape_meta, store):
    # parse shape meta
    rgb_keys = list()
    lowdim_keys = list()
    out_resolutions = dict()
    lowdim_shapes = dict()
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        type = attr.get("type", "low_dim")
        shape = tuple(attr.get("shape"))
        if type == "rgb":
            rgb_keys.append(key)
            c, h, w = shape
            out_resolutions[key] = (w, h)
        elif type == "low_dim":
            lowdim_keys.append(key)
            lowdim_shapes[key] = tuple(shape)
            if "pose" in key:
                assert tuple(shape) in [(2,), (3,), (6,), (7,), (16,)]

    action_shape = tuple(shape_meta["action"]["shape"])
    assert action_shape in [(2,), (3,), (6,), (7,), (16,)]

    # load data
    cv2.setNumThreads(1)
    with threadpool_limits(1):
        replay_buffer = real_data_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,
            out_resolutions=out_resolutions,
            lowdim_keys=lowdim_keys + ["action"] + ["allegro_action"],
            image_keys=rgb_keys,
        )

    # transform lowdim dimensions
    if action_shape == (2,):
        # 2D action space, only controls X and Y
        zarr_arr = replay_buffer["action"]
        zarr_resize_index_last_dim(zarr_arr, idxs=[0, 1])

    for key, shape in lowdim_shapes.items():
        if "pose" in key and shape == (2,):
            # only take X and Y
            zarr_arr = replay_buffer[key]
            zarr_resize_index_last_dim(zarr_arr, idxs=[0, 1])

    return replay_buffer


def test():
    import hydra
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with hydra.initialize("../config"):
        cfg = hydra.compose("train_diffusion_unet_beadmaze_image_workspace")
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)

    from matplotlib import pyplot as plt

    normalizer = dataset.get_normalizer()
    nactions = normalizer["action"].normalize(dataset.replay_buffer["action"][:])
    dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
    _ = plt.hist(dists, bins=100)
    plt.title("real action velocity")
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    for i in range(len(dataset)):
        ax[0].cla()
        ax[1].cla()

        torch_data = dataset[i]
        img = torch_data["obs"]["digit_index"][0].permute(1, 2, 0).numpy()[:, :, 0:3]
        ax[0].imshow(img)
        ax[0].set_title(f"Index idx: {i}")

        img = torch_data["obs"]["digit_thumb"][0].permute(1, 2, 0).numpy()[:, :, 0:3]
        ax[1].imshow(img)
        ax[1].set_title(f"Thumb idx: {i}")

        plt.pause(0.001)
    plt.show()
    print("done")


if __name__ == "__main__":
    test()
