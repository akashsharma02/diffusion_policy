from typing import Callable, Dict, Optional
import collections
import math
import time

import click
import dill
import hydra
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytorch_kinematics as pk
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import JointState, CompressedImage
from std_msgs.msg import Header
import cv2


def compute_diff(img1, img2, offset=0.0):
    img1 = np.int32(img1)
    img2 = np.int32(img2)
    diff = img1 - img2
    diff = diff / 255.0 + offset
    diff = np.clip(diff, 0.0, 1.0)
    diff = np.uint8(diff * 255.0)
    return diff


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
    def __init__(self, policy, config, device) -> None:
        super().__init__(node_name="DiffusionPolicyNode")
        self.device = device
        self.policy = policy
        self.config = config

        self.n_obs_steps = self.policy.n_obs_steps
        self.policy.num_inference_steps = 16

        urdf_path = config.urdf_path
        self.fk_chain = pk.build_serial_chain_from_urdf(
            open(urdf_path).read(),
            end_link_name="meta_hand_base_frame",
            root_link_name="base_link",
        )
        self.fk_chain = self.fk_chain.to(device="cpu")

        self.bridge = CvBridge()
        self.create_subscription(
            CompressedImage, self.config.index_topic, self.index_callback, 1
        )
        self.create_subscription(
            CompressedImage, self.config.thumb_topic, self.thumb_callback, 1
        )
        self.create_subscription(
            JointState,
            self.config.joint_state_topic,
            self.franka_joint_state_callback,
            1,
        )
        self.joint_state_publisher = self.create_publisher(
            JointState, self.config.joint_cmd_topic, 1
        )

        index_nocontact_image = cv2.imread("./digit_index_no_contact.png")
        self.index_nocontact_image = cv2.cvtColor(
            index_nocontact_image, cv2.COLOR_BGR2RGB
        )

        thumb_nocontact_image = cv2.imread("./digit_thumb_no_contact.png")
        self.thumb_nocontact_image = cv2.cvtColor(
            thumb_nocontact_image, cv2.COLOR_BGR2RGB
        )

        self.frequency = 10
        self.timer = self.create_timer(1 / self.frequency, self.on_timer)
        self.thumb_image = None
        self.index_image = None
        self.prev_target_joint = None
        num_required_frames = self.config.digit_frame_stride + (
            self.policy.n_obs_steps - 1
        )

        k = math.ceil(
            (self.config.robot_state_fps / self.frequency) * self.policy.n_obs_steps
        )
        self.index_digit_queue = collections.deque(maxlen=num_required_frames)
        self.thumb_digit_queue = collections.deque(maxlen=num_required_frames)
        self.joint_state_queue = collections.deque(maxlen=k)

    def digit_callback(self, msg: CompressedImage):
        timestamp = None
        if hasattr(msg, "header"):
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        else:
            timestamp = self.get_clock().now().nanoseconds * 1e-9

        image = self.bridge.compressed_imgmsg_to_cv2(msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("uint8")
        return timestamp, image

    def index_callback(self, msg: CompressedImage):
        timestamp, image = self.digit_callback(msg)
        image = compute_diff(image, self.index_nocontact_image, offset=0.5)
        pil_image = Image.fromarray(image)
        image = np.array(pil_image.resize((224, 224), Image.BILINEAR))
        image = image.astype(np.float32) / 255.0
        self.get_logger().debug(f"Received index image: {image.shape}")
        self.index_digit_queue.append((timestamp, image))

    def thumb_callback(self, msg: CompressedImage):
        timestamp, image = self.digit_callback(msg)
        image = compute_diff(image, self.thumb_nocontact_image, offset=0.5)
        pil_image = Image.fromarray(image)
        image = np.array(pil_image.resize((224, 224), Image.BILINEAR))
        image = image.astype(np.float32) / 255.0
        self.get_logger().debug(f"Received thumb image: {image.shape}")
        self.thumb_digit_queue.append((timestamp, image))

    def franka_joint_state_callback(self, msg: JointState):
        timestamp = None
        if hasattr(msg, "header"):
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        else:
            timestamp = self.get_clock().now().nanoseconds * 1e-9

        self.joint_state_queue.append((timestamp, msg.position))

    def get_obs(self):
        num_images_per_step = self.config.digit_fps / self.frequency
        num_required_frames = self.config.digit_frame_stride + (
            self.policy.n_obs_steps - 1
        )
        if len(self.index_digit_queue) < num_required_frames:
            return False, None
        if len(self.thumb_digit_queue) < num_required_frames:
            return False, None

        index_digits = [d for d in self.index_digit_queue]
        thumb_digits = [d for d in self.thumb_digit_queue]
        # index_digits = self.index_digit_queue.index(-num_required_frames,
        # thumb_digits = self.thumb_digit_queue[-num_required_frames:]

        # latest_index_timestamp = index_digits[-1][0]
        # latest_thumb_timestamp = thumb_digits[-1][0]

        index_observations = []
        thumb_observations = []
        index_timestamps = []
        thumb_timestamps = []
        for obs_step in np.arange(self.policy.n_obs_steps - 1, -1, -1):
            curr_idx = -1 - obs_step
            curr_stride_idx = -self.config.digit_frame_stride - obs_step
            index_obs = [index_digits[curr_idx][1], index_digits[curr_stride_idx][1]]
            index_obs = np.concatenate(index_obs, axis=-1)
            thumb_obs = [thumb_digits[curr_idx][1], thumb_digits[curr_stride_idx][1]]
            thumb_obs = np.concatenate(thumb_obs, axis=-1)
            index_observations.append(index_obs)
            thumb_observations.append(thumb_obs)
            index_timestamps.append(index_digits[curr_idx][0])
            thumb_timestamps.append(thumb_digits[curr_idx][0])

        k = math.ceil(
            (self.config.robot_state_fps / self.frequency) * self.policy.n_obs_steps
        )

        joint_timestamps, joint_states = zip(*[k for k in self.joint_state_queue])
        joint_timestamps = np.array(list(joint_timestamps))
        joint_states = np.array(list(joint_states))

        this_idxs = list()
        for t in index_timestamps:
            is_before_idxs = np.nonzero(joint_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)

        index_observations = np.stack(index_observations, axis=0)
        thumb_observations = np.stack(thumb_observations, axis=0)
        index_observations = np.moveaxis(index_observations, -1, 1)
        thumb_observations = np.moveaxis(thumb_observations, -1, 1)
        joint_observations = joint_states[this_idxs]
        # print(
        #     index_observations.shape, thumb_observations.shape, joint_observations.shape
        # )

        obs_dict = {}
        obs_dict["digit_index"] = index_observations
        obs_dict["digit_thumb"] = thumb_observations
        obs_dict["robot_joint"] = joint_observations[:, :7]
        obs_dict["allegro_joint"] = joint_observations[:, 7:]
        # obs_dict["timestamp"] = np.array(index_timestamps)
        return True, obs_dict

    def on_timer(self):
        ok, obs_dict_np = self.get_obs()
        if not ok:
            return

        assert obs_dict_np is not None

        with torch.no_grad():
            s = time.time()
            obs_dict = dict_apply(
                obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
            )
            result = self.policy.predict_action(obs_dict)
            action = result["action"][0].detach().to("cpu").numpy()
            print(f"Inference latency: {time.time() - s}")

        if self.prev_target_joint is None:
            self.prev_target_joint = obs_dict_np["robot_joint"][-1]
        executable_action = self.prev_target_joint.copy() + action[0]
        # executable_action = obs_dict_np["robot_joint"][-1] + action[-1]

        header = Header(stamp=self.get_clock().now().to_msg())
        msg = JointState(
            header=header, position=executable_action, velocity=[], effort=[]
        )
        self.joint_state_publisher.publish(msg)

        self.prev_target_joint = executable_action
        input()


def main():
    rclpy.init()
    config = OmegaConf.create(
        {
            "ckpt_path": "./checkpoints/dino_unet_bugfix/latest-006.ckpt",
            "urdf_path": "../GUM/gum/devices/metahand/ros/meta_hand_description/urdf/meta_hand_franka.urdf",
            "index_topic": "/digit_index/compressed",
            "thumb_topic": "/digit_thumb/compressed",
            "joint_state_topic": "/joint_states",
            "joint_cmd_topic": "/joint_pos_cmd",
            "digit_fps": 30,
            "digit_frame_stride": 5,
            "robot_state_fps": 30,
        }
    )
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    # config = OmegaConf.resolve(config)
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    franka_urdf_chain = pk.build_serial_chain_from_urdf(
        open(config.urdf_path).read(),
        end_link_name="meta_hand_base_frame",
        root_link_name="base_link",
    )
    franka_urdf_chain = franka_urdf_chain.to(device="cpu")

    payload = torch.load(open(config.ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    # dataset = None
    # with hydra.initialize("./diffusion_policy/config"):
    #     cfg = hydra.compose("train_diffusion_unet_beadmaze_image_workspace")
    #     OmegaConf.resolve(cfg)
    #     dataset = hydra.utils.instantiate(cfg.task.dataset)
    # assert dataset is not None

    # normalizer = dataset.get_normalizer()
    # val_dataset = dataset.get_validation_dataset()

    # policy.set_normalizer(normalizer)
    policy.eval().to(device)
    policy.reset()

    node = DiffusionPolicyNode(policy=policy, config=config, device=device)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
