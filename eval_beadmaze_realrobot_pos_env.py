from typing import List
import dill
import time

import hydra
import numpy as np
from data_clients.data_client import DataClient
from data_clients.env import Env, ActionCmd
from omegaconf import OmegaConf
import einops
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as R

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)

index_no_contact_image = cv2.imread("./digit_index_no_contact.png")
index_no_contact_image = cv2.cvtColor(index_no_contact_image, cv2.COLOR_BGR2RGB)

thumb_no_contact_image = cv2.imread("./digit_thumb_no_contact.png")
thumb_no_contact_image = cv2.cvtColor(thumb_no_contact_image, cv2.COLOR_BGR2RGB)


def stack_image_for_vis(images: List[np.ndarray]):
    new_image = np.concatenate(images, axis=1)
    new_image = einops.rearrange(new_image, "c w h -> h w c")
    new_image = (new_image).astype(np.uint8)
    return new_image


target_metahand_position = np.array(
    [
        -0.2,
        1.4,  # 0.9,
        0.8,
        -0.2,
        0.2,
        0.0,  # 0.9,
        0.25,
        0.1,
        0.2,
        0.0,  # 1.0,
        0.25,
        0.1,
        1.5,
        0.0,
        0.5,  # 0.8,
        0.0,  # 0.35,
    ]
)


def compute_diff(img1, img2, offset=0.0):
    img1 = np.int32(img1)
    img2 = np.int32(img2)
    diff = img1 - img2
    diff = diff / 255.0 + offset
    diff = np.clip(diff, 0.0, 1.0)
    diff = np.uint8(diff * 255.0)
    return diff


def get_single_observation(env, franka_urdf_chain):
    _, obs_history = env.get_data()
    env.obs_history.append(obs_history)
    obs_history = DataClient.listdicts_to_dictlists(env.obs_history[-2:])

    obs_dict = {}
    for key, val in obs_history.items():
        if key == "digit_index/compressed":
            val = [compute_diff(v, index_no_contact_image, offset=0.5) for v in val]
            val = [
                np.array(Image.fromarray(v).resize((224, 224), Image.BILINEAR))
                for v in val
            ]
            val = np.concatenate([val[1], val[0]], axis=-1)
            val = val.astype(np.float32) / 255.0
            obs_dict["digit_index"] = val
        if key == "digit_thumb/compressed":
            val = [compute_diff(v, thumb_no_contact_image, offset=0.5) for v in val]
            val = [
                np.array(Image.fromarray(v).resize((224, 224), Image.BILINEAR))
                for v in val
            ]
            val = np.concatenate([val[1], val[0]], axis=-1)
            val = val.astype(np.float32) / 255.0
            obs_dict["digit_thumb"] = val
        if key == "joint_states/position":
            val = torch.from_numpy(np.array(val))
            franka_pose = franka_urdf_chain.forward_kinematics(
                val[:, :7], end_only=True
            )
            franka_pose = franka_pose.get_matrix().detach().numpy()
            franka_pos = franka_pose[:, :3, 3]
            franka_quat = R.from_matrix(franka_pose[:, :3, :3])
            franka_quat = R.as_quat(franka_quat).astype(np.float64)
            pose = np.concatenate([franka_pos, franka_quat], axis=-1)
            print(pose.shape)
            obs_dict["robot_eef_pose"] = pose
    return obs_dict


def get_obs(env, n_obs_steps, franka_urdf_chain):
    new_obs_dict = []
    for i in range(n_obs_steps):
        obs_dict = get_single_observation(env, franka_urdf_chain)
        env.step_sync()
        new_obs_dict.append(obs_dict)
    new_obs_dict.reverse()
    new_obs_dict = DataClient.listdicts_to_dictlists(new_obs_dict)
    obs = {}
    for key, val in new_obs_dict.items():
        if key == "digit_index" or key == "digit_thumb":
            val = np.stack(val)
            obs[key] = einops.rearrange(val, "t h w c -> t c w h")
        else:
            new_val = np.stack([val[0][0], val[1][1]], axis=0)
            obs[key] = new_val
    return obs


@hydra.main(
    config_path="./env_cfg", config_name="bead_maze_pos.yaml", version_base="1.2"
)
def main(cfg):
    device = torch.device(cfg.rl_device)

    payload = torch.load(open(cfg.ckpt_path, "rb"), pickle_module=dill)
    diffusion_cfg = payload["cfg"]
    cls = hydra.utils.get_class(diffusion_cfg._target_)
    workspace: BaseWorkspace = cls(diffusion_cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if diffusion_cfg.training.use_ema:
        policy = workspace.ema_model

    policy.eval().to(device)
    policy.reset()

    n_obs_steps = diffusion_cfg.n_obs_steps
    print(f"n_obs_steps: {n_obs_steps}")
    policy.num_inference_steps = 16  # DDIM iterations
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    franka_urdf_chain = pk.build_serial_chain_from_urdf(
        open(cfg.urdf_path).read(),
        end_link_name="panda_link8",
        root_link_name="base_link",
    )
    franka_urdf_chain = franka_urdf_chain.to(device="cpu")

    # Create env
    env = Env(cfg.env, cfg.logging_dir)
    env.start()

    print("...ros started")

    # Warm up to get first two observations observation
    env.reset()
    with torch.no_grad():
        policy.reset()
        obs_dict_np = get_obs(env, n_obs_steps, franka_urdf_chain)
        obs_dict = dict_apply(
            obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
        )
        obs_dict["robot_eef_pose"] = obs_dict["robot_eef_pose"][..., :3]
        for key, val in obs_dict.items():
            print(f"key: {key}, val: {val.shape}")
        result = policy.predict_action(obs_dict)
        action = result["action"][0].detach().to("cpu").numpy()
        del result
    try:
        policy.reset()
        prev_target_pose = None
        pred_target_quat = None
        predicted_poses = []
        done = False
        while not done:
            if prev_target_pose is None:
                prev_target_pose = obs_dict_np["robot_eef_pose"][-1][:3]
                prev_target_quat = obs_dict_np["robot_eef_pose"][-1][3:]

                # this_target_pose = prev_target_pose.copy() + action[-1]
                # env_action_cmd = ActionCmd()
                # env_action_cmd.set_franka_cart_ctrl(this_target_pose, prev_target_quat)
                # env.action_exec.execute_action(env_action_cmd)
                # prev_target_joint_tensor = torch.from_numpy(prev_target_joint)
                # prev_target_pose = franka_urdf_chain.forward_kinematics(
                #     prev_target_joint_tensor, end_only=True
                # )
                # prev_target = prev_target_pose.get_matrix()[0].detach().numpy()
                # prev_target_pose = prev_target[:3, 3]
                # prev_target_quat = R.from_matrix(prev_target[:3, :3])
                # prev_target_quat = R.as_quat(prev_target_quat).astype(np.float64)
            print(f"action: {action}")
            print(f"action[-1]: {action[-1]}")
            print(f"prev_target_pose: {prev_target_pose}")
            this_target_pose = prev_target_pose.copy() + action[-1]
            print(f"this_target_pose: {this_target_pose} {this_target_pose.shape}")

            # this_target_pose_tensor = torch.from_numpy(this_target_pose)
            # pred_cart_pose = franka_urdf_chain.forward_kinematics(
            #     this_target_pose_tensor, end_only=True
            # )
            # print(pred_cart_pose.shape)
            # pred_cart_pose = pred_cart_pose.get_matrix()[0].detach().numpy()
            # print(pred_cart_pose.shape)
            # pred_cart_pos = pred_cart_pose[:3, 3].astype(np.float64)
            # pred_cart_rot = R.from_matrix(pred_cart_pose[:3, :3])
            # pred_cart_quat = R.as_quat(pred_cart_rot).astype(np.float64)
            # print(pred_cart_pos, pred_cart_quat)
            # pred_cart_pos = np.array([0.470, -0.022, 0.5])
            # pred_cart_quat = np.array([-0.652, -0.225, -0.283, 0.666])
            env_action_cmd = ActionCmd()
            # print(pred_cart_pos.dtype, pred_cart_quat.dtype)
            # env_action_cmd.set_franka_jnt_ctrl(this_target_pose)
            env_action_cmd.set_franka_cart_ctrl(this_target_pose, prev_target_quat)
            # env_action_cmd.set_metahand_pos_ctrl(target_metahand_position)
            env.action_exec.execute_action(env_action_cmd)
            # env.obs_history[-1].update(env_action_cmd.to_dict())
            predicted_poses.append(this_target_pose)
            prev_target_pose = this_target_pose

            with torch.no_grad():
                s = time.time()
                obs_dict_np = get_obs(env, n_obs_steps, franka_urdf_chain)
                obs_dict = dict_apply(
                    obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
                )
                obs_dict["robot_eef_pose"] = obs_dict["robot_eef_pose"][..., :3]
                for key, val in obs_dict.items():
                    print(f"key: {key}, val: {val.shape}, val type: {val.dtype}")
                result = policy.predict_action(obs_dict)
                action = result["action"][0].detach().to("cpu").numpy()
                print(f"Inference latency: {time.time() - s}")

            env.cur_step += 1
            done = env.cur_step >= env.episode_length
            # digit_thumb = obs_dict_np["digit_thumb"]
            # digit_index = obs_dict_np["digit_index"]
            # digit_thumb = [
            #     digit_thumb[0, :3],
            #     digit_thumb[0, 3:],
            #     digit_thumb[1, :3],
            #     digit_thumb[1, 3:],
            # ]
            # digit_index = [
            #     digit_index[0, :3],
            #     digit_index[0, 3:],
            #     digit_index[1, :3],
            #     digit_index[1, 3:],
            # ]
            # digit_thumb_vis = stack_image_for_vis(digit_thumb)
            # digit_index_vis = stack_image_for_vis(digit_index)
            # cv2.imshow("thumb", digit_thumb_vis[..., ::-1])
            # cv2.imshow("index", digit_index_vis[..., ::-1])
            # cv2.waitKey(0)
    except KeyboardInterrupt:
        print("Interrupted!")
        # stop robot.
        env.shutdown()


if __name__ == "__main__":
    main()
