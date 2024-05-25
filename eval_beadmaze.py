from typing import Callable, Dict, Optional, List
import cv2
import click
import dill
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_kinematics as pk
import torch
from omegaconf import OmegaConf
import einops

from diffusion_policy.workspace.base_workspace import BaseWorkspace


def stack_image_for_vis(images: List[np.ndarray]):
    num_images = len(images)
    c, h, w = images[0].shape
    new_image = np.zeros((c, h, w * num_images))
    for k, image in enumerate(images):
        new_image[:, :, int(k * w) : int((k + 1) * w)] = image

    new_image = einops.rearrange(new_image, "c h w -> h w c")
    new_image = (new_image * 255).astype(np.uint8)
    return new_image


def draw_3d_axes(
    ax,
    world_T_camera: Optional[np.ndarray] = None,
    axis_length: float = 1.0,
    traj_linestyle: str = "-",
    traj_color: str = "b",
    traj_label: str = "",
):
    """
    Draw 3D axes in 3D matplotlib plot.
    Args:
        ax: matplotlib 3D axis
        world_T_camera: n x 4 x 4 transformation matrix from camera to world frame
    """
    # Define the origin and axes endpoints
    origin = np.array([0, 0, 0])
    x_axis = np.array([axis_length, 0, 0])
    y_axis = np.array([0, axis_length, 0])
    z_axis = np.array([0, 0, axis_length])

    if world_T_camera is not None:
        origin = world_T_camera[:, :3, 3]
        x_axis = world_T_camera[:, :3, 0]
        y_axis = world_T_camera[:, :3, 1]
        z_axis = world_T_camera[:, :3, 2]

    # Plot the axes
    ax.quiver(*origin.T, *x_axis.T, color="red", length=axis_length, normalize=True)
    ax.quiver(*origin.T, *y_axis.T, color="green", length=axis_length, normalize=True)
    ax.quiver(*origin.T, *z_axis.T, color="blue", length=axis_length, normalize=True)

    ax.plot3D(*origin.T, linestyle=traj_linestyle, color=traj_color, label=traj_label)


def set_equal_aspect_ratio_3D(ax, xs, ys, zs, alpha=1.5, delta=0.0):
    mn = np.array([xs.min(), ys.min(), zs.min()])
    mx = np.array([xs.max(), ys.max(), zs.max()])
    d = 0.5 * (mx - mn)
    c = mn + d
    d = alpha * np.max(d) + delta

    ax.set_xlim(c[0] - d, c[0] + d)
    ax.set_ylim(c[1] - d, c[1] + d)
    ax.set_zlim(c[2] - d, c[2] + d)


def dict_apply(
    x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


@click.command()
@click.option("ckpt_path", "-i", required=True, help="Path to checkpoint")
@click.option("urdf_path", "-u", required=True, help="Path to URDF file")
def main(ckpt_path: str, urdf_path: str):
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    franka_urdf_chain = pk.build_serial_chain_from_urdf(
        open(urdf_path).read(),
        end_link_name="meta_hand_base_frame",
        root_link_name="base_link",
    )
    franka_urdf_chain = franka_urdf_chain.to(device="cpu")
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    dataset = None
    with hydra.initialize("./diffusion_policy/config"):
        cfg = hydra.compose("train_diffusion_unet_beadmaze_image_workspace")
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)
    assert dataset is not None

    normalizer = dataset.get_normalizer()
    val_dataset = dataset.get_validation_dataset()
    val_sampler = val_dataset.sampler
    # replay_buffer = val_dataset.replay_buffer
    # last_episode = replay_buffer.get_episode(replay_buffer.n_episodes - 1)
    policy.set_normalizer(normalizer)
    policy.eval().to(device)
    policy.reset()

    # obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)

    prev_target_joint = None
    predicted_joints = []
    gt_joints = []
    print(f"len(val_dataset): {len(val_dataset)}")
    print(val_dataset.val_mask)
    # Enumerating over val_dataset calls __next__ on the dataset which is problematic because it iterates over the full dataset
    for i in range(len(val_dataset)):
        data = val_dataset[i]
        obs_dict_np = data["obs"]
        obs_dict = dict_apply(
            obs_dict_np,
            lambda x: (
                x.unsqueeze(0)
                if not isinstance(x, np.ndarray)
                else torch.from_numpy(x).unsqueeze(0).to(device)
            ),
        )
        result = policy.predict_action(obs_dict)
        action = result["action"][0].detach().to("cpu").numpy()

        executable_action = action[-1]
        # We are assuming delta_action = True
        gt_joint = obs_dict["robot_joint"].squeeze().detach().to("cpu").numpy()
        gt_joint = gt_joint[-1]
        if prev_target_joint is None:
            prev_target_joint = gt_joint.copy()

        this_target_joint = prev_target_joint.copy() + executable_action
        prev_target_joint = this_target_joint
        predicted_joints.append(this_target_joint)
        gt_joints.append(gt_joint)

        print(f"{i}")
        print(f"gt_joint         : {gt_joint}")
        print(f"this_target_joint: {this_target_joint}")

        digit_thumb = obs_dict_np["digit_thumb"].numpy()
        digit_index = obs_dict_np["digit_index"].numpy()
        print(f"digit_thumb: {digit_thumb.shape}")
        print(f"digit_index: {digit_index.shape}")
        digit_thumb = [
            digit_thumb[0, :3],
            digit_thumb[0, 3:],
            digit_thumb[1, :3],
            digit_thumb[1, 3:],
        ]
        digit_index = [
            digit_index[0, :3],
            digit_index[0, 3:],
            digit_index[1, :3],
            digit_index[1, 3:],
        ]
        digit_thumb_vis = stack_image_for_vis(digit_thumb)
        digit_index_vis = stack_image_for_vis(digit_index)
        cv2.imshow("thumb", digit_thumb_vis[..., ::-1])
        cv2.imshow("index", digit_index_vis[..., ::-1])
        cv2.waitKey(0)

    gt_joints = np.array(gt_joints)
    predicted_joints = np.array(predicted_joints)

    # Compute trajectory using forward kinematics
    gt_joints_tensor = torch.from_numpy(gt_joints)
    predicted_joints_tensor = torch.from_numpy(predicted_joints)
    gt_traj = franka_urdf_chain.forward_kinematics(gt_joints_tensor, end_only=True)
    pred_traj = franka_urdf_chain.forward_kinematics(
        predicted_joints_tensor, end_only=True
    )
    gt_traj = gt_traj.get_matrix().detach().numpy()
    pred_traj = pred_traj.get_matrix().detach().numpy()

    pos_gt = gt_traj[:, :3, 3]
    pos_pred = pred_traj[:, :3, 3]
    print(f"pos_gt: {pos_gt}")
    print(f"pos_pred: {pos_pred}")
    pos_error = np.abs(pos_gt - pos_pred)

    error = np.abs(gt_joints - predicted_joints)
    print(f"Mean error: {error.mean()}")
    print(f"Position error: {pos_error.mean()}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # for pose_gt, pose_pred in zip(gt_traj, pred_traj):
    draw_3d_axes(
        ax,
        gt_traj[::5],
        axis_length=2e-3,
        traj_color="b",
    )
    draw_3d_axes(
        ax,
        pred_traj[::5],
        axis_length=2e-3,
        traj_color="r",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("End effector trajectory")

    set_equal_aspect_ratio_3D(ax, pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], alpha=1.5)

    plt.show()


if __name__ == "__main__":
    main()
