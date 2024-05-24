from typing import Callable, Dict, Optional

import click
import dill
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_kinematics as pk
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import JointState, CompressedImage
import cv2


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

class DiffusionPolicyNode(Node):
    def __init__(self, node_name='DiffusionPolicyNode', policy, config, device) -> None:
        super().__init__(node_name=node_name):
        self.device = device
        self.policy = policy

        urdf_path = config.urdf_path
        self.fk_chain = pk.build_serial_chain_from_urdf(
            open(urdf_path).read(),
            end_link_name='meta_hand_base_frame',
            root_link_name='base_link'
        )
        self.fk_chain = self.fk_chain.to(device='cpu')

        self.bridge = CvBridge()
        self.create_subscription(
            CompressedImage,
            self.config.index_topic,
            self.index_callback,
            1
        )
        self.create_subscription(
            CompressedImage,
            self.config.thumb_topic,
            self.thumb_callback,
            1
        )
        self.create_subscription(
            JointState,
            self.config.joint_state_topic,
            self.franka_joint_state_callback,
            1,
        )
        self.joint_state_publisher = self.create_publisher(JointState, self.config.joint_state_topic, 1)

        self.timer = self.create_timer(0.01, self.on_timer)
        self.thumb_image = None
        self.index_image = None
        self.curr_joint_angles = None

    def index_callback(self, msg: CompressedImage):
        image = self.bridge.compressed_imgmsg_to_cv2(msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (240, 320))
        self.index_image = image

    def thumb_callback(self, msg: CompressedImage):
        image = self.bridge.compressed_imgmsg_to_cv2(msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (240, 320))
        self.thumb_image = image

    def franka_joint_state_callback(self, msg: JointState):
        self.curr_joint_angles = torch.tensor(msg.position).to(self.device)

    def on_timer(self):
        if self.index_image is None or self.thumb_image is None:
            return
        if self.curr_joint_angles is None:
            return

        obs_dict = {
            'digit_index': self.index_image,
            'digit_thumb': self.thumb_image,
            'robot_joint': self.curr_joint_angles
        }
        obs_dict = dict_apply(
            obs_dict,
            lambda x: (
                x.unsqueeze(0)
                if not isinstance(x, np.ndarray)
                else torch.from_numpy(x).unsqueeze(0).to(self.device)
            ),
        )
        result = self.policy.predict_action(obs_dict)
        action = result["action"][0].detach().to("cpu").numpy()
        executable_action = action[0]
        # We are assuming delta_action = True
        curr_joint_angle = self.curr_joint_angles.squeeze().detach().to("cpu").numpy()
        desired_joint_angle = curr_joint_angle + executable_action

        header = Header(stamp=self.get_clock().now().to_msg())
        msg = JointState(
            header=header,potition=desired_joint_angle, velocity=[], effort=[]
        )
        self.publisher.publish(msg)







@click.command()
@click.option("ckpt_path", "-i", required=True, help="Path to checkpoint")
@click.option("urdf_path", "-u", required=True, help="Path to URDF file")
def main(ckpt_path: str, urdf_path: str):
    rclpy.init()
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

    config = OmegaConf.create({
        'ckpt_path': 'ckpt_path',
        'urdf_path': 'urdf_path',
        'index_topic': '/digit_index/compressed',
        'thumb_topic': '/digit_thumb/compressed',
        'joint_state_topic': '/joint_pos_cmd',
    })
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
    # val_dataset = dataset.get_validation_dataset()

    policy.set_normalizer(normalizer)
    policy.eval().to(device)
    policy.reset()

    node = DiffusionPolicyNode(policy=policy, config=config, device=device)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()





    # obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)

    # prev_target_joint = None
    # predicted_joints = []
    # gt_joints = []
    # print(f"len(val_dataset): {len(val_dataset)}")
    # for i, data in enumerate(dataset):
    #     if i > 500:
    #         break
    #
    #     obs_dict = data["obs"]
    #     obs_dict = dict_apply(
    #         obs_dict,
    #         lambda x: (
    #             x.unsqueeze(0)
    #             if not isinstance(x, np.ndarray)
    #             else torch.from_numpy(x).unsqueeze(0).to(device)
    #         ),
    #     )
    #
    #     result = policy.predict_action(obs_dict)
    #     action = result["action"][0].detach().to("cpu").numpy()
    #
    #     executable_action = action[0]
    #     # We are assuming delta_action = True
    #     gt_joint = obs_dict["robot_joint"].squeeze().detach().to("cpu").numpy()
    #     gt_joint = gt_joint[-1]
    #     if prev_target_joint is None:
    #         prev_target_joint = gt_joint.copy()
    #
    #     this_target_joint = prev_target_joint.copy() + executable_action
    #     prev_target_joint = this_target_joint
    #     predicted_joints.append(this_target_joint)
    #     gt_joints.append(gt_joint)
    #     print(f"{i}")
    #     print(f"gt_joint         : {gt_joint}")
    #     print(f"this_target_joint: {this_target_joint}")
    #
    # gt_joints = np.array(gt_joints)
    # predicted_joints = np.array(predicted_joints)
    #
    # # Compute trajectory using forward kinematics
    # gt_joints_tensor = torch.from_numpy(gt_joints)
    # predicted_joints_tensor = torch.from_numpy(predicted_joints)
    # gt_traj = franka_urdf_chain.forward_kinematics(gt_joints_tensor, end_only=True)
    # pred_traj = franka_urdf_chain.forward_kinematics(
    #     predicted_joints_tensor, end_only=True
    # )
    # gt_traj = gt_traj.get_matrix().detach().numpy()
    # pred_traj = pred_traj.get_matrix().detach().numpy()
    #
    # pos_gt = gt_traj[:, :3, 3]
    # pos_pred = pred_traj[:, :3, 3]
    # print(f"pos_gt: {pos_gt}")
    # print(f"pos_pred: {pos_pred}")
    # pos_error = np.abs(pos_gt - pos_pred)
    #
    # error = np.abs(gt_joints - predicted_joints)
    # print(f"Mean error: {error.mean()}")
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    #
    # # for pose_gt, pose_pred in zip(gt_traj, pred_traj):
    # draw_3d_axes(ax, gt_traj[::5], axis_length=2e-3, traj_color="b")
    # draw_3d_axes(ax, pred_traj[::5], axis_length=2e-3, traj_color="r")
    #
    # set_equal_aspect_ratio_3D(ax, pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], alpha=1.5)
    #
    # plt.show()


if __name__ == "__main__":
    main()


