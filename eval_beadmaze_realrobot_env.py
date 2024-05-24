import dill

import hydra
import numpy as np
from data_clients.data_client import DataClient
from data_clients.env import Env
from omegaconf import OmegaConf
import einops
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from gum.labs.grasping.wandb_util import WandbLogger
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)

index_no_contact_image = cv2.imread("./digit_index_no_contact.png")
index_no_contact_image = cv2.cvtColor(index_no_contact_image, cv2.COLOR_BGR2RGB)

thumb_no_contact_image = cv2.imread("./digit_thumb_no_contact.png")
thumb_no_contact_image = cv2.cvtColor(thumb_no_contact_image, cv2.COLOR_BGR2RGB)


def step_without_action(env):
    env.step_sync()
    _, obs_history = env.get_data()
    env.obs_history.append(obs_history)


def get_single_observation(env):
    step_without_action(env)
    obs_history = DataClient.listdicts_to_dictlists(env.obs_history[-2:])
    obs_dict = {}
    for key, val in obs_history.items():
        if key == "digit_index/compressed":
            val = [compute_diff(v, index_no_contact_image, offset=0.5) for v in val]
            val = [
                np.array(Image.fromarray(v).resize((224, 224), Image.BILINEAR))
                for v in val
            ]
            val = np.concatenate(val, axis=-1)
            obs_dict["digit_index"] = val
        if key == "digit_thumb/compressed":
            val = [compute_diff(v, thumb_no_contact_image, offset=0.5) for v in val]
            val = [
                np.array(Image.fromarray(v).resize((224, 224), Image.BILINEAR))
                for v in val
            ]
            val = np.concatenate(val, axis=-1)
            obs_dict["digit_thumb"] = val
        if key == "joint_states/position":
            val = np.array(val)
            obs_dict["robot_joint"] = val[:, :7]
            obs_dict["allegro_joint"] = val[:, 7:]
    return obs_dict


def get_obs(env, n_obs_steps):
    new_obs_dict = []
    for i in range(n_obs_steps):
        obs_dict = get_single_observation(env)
        new_obs_dict.append(obs_dict)
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


@hydra.main(config_path="./env_cfg", config_name="bead_maze.yaml", version_base="1.2")
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

    # Create env
    env = Env(cfg.env, cfg.logging_dir)
    env.start()

    print("...ros started")

    # Warm up to get first two observations observation
    env.reset()
    with torch.no_grad():
        policy.reset()
        obs_dict_np = get_obs(env, n_obs_steps)
        obs_dict = dict_apply(
            obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
        )
        for key, val in obs_dict.items():
            print(f"key: {key}, val: {val.shape}")
        result = policy.predict_action(obs_dict)
        action = result["action"][0].detach().to("cpu").numpy()
        del result
    try:
        #     # setup_policy.reset(obs)
        #
        #     # print(f"[Trial {trial_count}] Setup start")
        #     # while not setup_policy.done():
        #     a = setup_policy.act(obs)
        #     obs, rew, _ = env.step(a)
        #
        #     obs = env.reset()  # Clear history (i.e. dont record setup)
        #     eval_policy.reset(obs)
        #
        #     if not env.reward.reset_success:
        #         print(f"[Trial {trial_count}] Object not grasped, try again")
        #         continue
        #
        #     done = False
        #
        #     print(f"[Trial {trial_count}] Eval start")
        #     while not done:
        #         a = eval_policy.act(obs)
        #         obs, rew, done = env.step(a)
        #
        #     obs_history = DataClient.listdicts_to_dictlists(env.obs_history)
        #     hold_reward = np.sum(obs_history["hold_reward"])
        #     reward = np.sum(obs_history["reward"])
        #     print(
        #         f"[Trial {trial_count}] Hold Reward: {hold_reward}, Overall Reward: {reward}"
        #     )
        #
        #     log_dir = env.log_history()
        #     wandb_logger.log_trial(log_dir)
        #
        #     trial_count += 1
        #
        # env.shutdown()
        # print("...eval done")

        # start episode
        policy.reset()
        start_delay = 1.0
        eval_t_start = time.time() + start_delay
        t_start = time.monotonic() + start_delay
        env.start_episode(eval_t_start)
        # wait for 1/30 sec to get the closest frame actually
        # reduces overall latency
        frame_latency = 1 / 30
        precise_wait(eval_t_start - frame_latency, time_func=time.time)
        print("Started!")
        iter_idx = 0
        term_area_start_timestamp = float("inf")
        perv_target_pose = None
        while True:
            # calculate timing
            t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

            # get obs
            print("get_obs")
            obs = env.get_obs()
            obs_timestamps = obs["timestamp"]
            print(f"Obs latency {time.time() - obs_timestamps[-1]}")

            # run inference
            with torch.no_grad():
                s = time.time()
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta
                )
                obs_dict = dict_apply(
                    obs_dict_np,
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device),
                )
                result = policy.predict_action(obs_dict)
                # this action starts from the first obs step
                action = result["action"][0].detach().to("cpu").numpy()
                print("Inference latency:", time.time() - s)

            # convert policy action to env actions
            if delta_action:
                assert len(action) == 1
                if perv_target_pose is None:
                    perv_target_pose = obs["robot_eef_pose"][-1]
                this_target_pose = perv_target_pose.copy()
                this_target_pose[[0, 1]] += action[-1]
                perv_target_pose = this_target_pose
                this_target_poses = np.expand_dims(this_target_pose, axis=0)
            else:
                this_target_poses = np.zeros(
                    (len(action), len(target_pose)), dtype=np.float64
                )
                this_target_poses[:] = target_pose
                this_target_poses[:, [0, 1]] = action

            # deal with timing
            # the same step actions are always the target for
            action_timestamps = (
                np.arange(len(action), dtype=np.float64) + action_offset
            ) * dt + obs_timestamps[-1]
            action_exec_latency = 0.01
            curr_time = time.time()
            is_new = action_timestamps > (curr_time + action_exec_latency)
            if np.sum(is_new) == 0:
                # exceeded time budget, still do something
                this_target_poses = this_target_poses[[-1]]
                # schedule on next available step
                next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                action_timestamp = eval_t_start + (next_step_idx) * dt
                print("Over budget", action_timestamp - curr_time)
                action_timestamps = np.array([action_timestamp])
            else:
                this_target_poses = this_target_poses[is_new]
                action_timestamps = action_timestamps[is_new]

            # clip actions
            this_target_poses[:, :2] = np.clip(
                this_target_poses[:, :2], [0.25, -0.45], [0.77, 0.40]
            )

            # execute actions
            env.exec_actions(actions=this_target_poses, timestamps=action_timestamps)
            print(f"Submitted {len(this_target_poses)} steps of actions.")

            # visualize
            episode_id = env.replay_buffer.n_episodes
            vis_img = obs[f"camera_{vis_camera_idx}"][-1]
            text = "Episode: {}, Time: {:.1f}".format(
                episode_id, time.monotonic() - t_start
            )
            cv2.putText(
                vis_img,
                text,
                (10, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                thickness=1,
                color=(255, 255, 255),
            )
            cv2.imshow("default", vis_img[..., ::-1])

            key_stroke = cv2.pollKey()
            if key_stroke == ord("s"):
                # Stop episode
                # Hand control back to human
                env.end_episode()
                print("Stopped.")
                break

            # auto termination
            terminate = False
            if time.monotonic() - t_start > max_duration:
                terminate = True
                print("Terminated by the timeout!")

            term_pose = np.array(
                [
                    3.40948500e-01,
                    2.17721816e-01,
                    4.59076878e-02,
                    2.22014183e00,
                    -2.22184883e00,
                    -4.07186655e-04,
                ]
            )
            curr_pose = obs["robot_eef_pose"][-1]
            dist = np.linalg.norm((curr_pose - term_pose)[:2], axis=-1)
            if dist < 0.03:
                # in termination area
                curr_timestamp = obs["timestamp"][-1]
                if term_area_start_timestamp > curr_timestamp:
                    term_area_start_timestamp = curr_timestamp
                else:
                    term_area_time = curr_timestamp - term_area_start_timestamp
                    if term_area_time > 0.5:
                        terminate = True
                        print("Terminated by the policy!")
            else:
                # out of the area
                term_area_start_timestamp = float("inf")

            if terminate:
                env.end_episode()
                break

            # wait for execution
            precise_wait(t_cycle_end - frame_latency)
            iter_idx += steps_per_inference

    except KeyboardInterrupt:
        print("Interrupted!")
        # stop robot.
        env.end_episode()


if __name__ == "__main__":
    main()
