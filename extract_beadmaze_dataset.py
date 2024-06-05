import os
from pathlib import Path
from rosbags.highlevel import AnyReader
from cv_bridge import CvBridge
from PIL import Image
import numpy as np
import pickle
import cv2
from tqdm import tqdm
import io
import matplotlib.pyplot as plt
from diffusion_policy.common.replay_buffer import ReplayBuffer
import pytorch_kinematics as pk
import torch

TOPICS_EXTRACT = [
    # "/digit_middle/compressed",
    "/digit_index/compressed",
    "/digit_thumb/compressed",
    "/realsense/color/image_raw/compressed",
    "/franka/joint_states",
    "/allegroHand/joint_states",
]

STEP_SIZE = 1
FPS = 30.0

PATH_ROSBAGS = f"/home/akashsharma/workspace/datasets/diffusion_policy/dp_rosbags/"
PATH_OUTPUT = f"./dataset_pickle_new2/"
PATH_OUPUT_ZARR = f"./data/bead_maze_new2/"

old_thumb_bg = cv2.imread("./digit_thumb.jpg", cv2.IMREAD_COLOR)
old_index_bg = cv2.imread("./digit_index.jpg", cv2.IMREAD_COLOR)
new_thumb_bg = cv2.imread("./digit_thumb_no_contact.png", cv2.IMREAD_COLOR)
new_thumb_bg = cv2.cvtColor(new_thumb_bg, cv2.COLOR_BGR2RGB)
new_index_bg = cv2.imread("./digit_index_no_contact.png", cv2.IMREAD_COLOR)
new_index_bg = cv2.cvtColor(new_index_bg, cv2.COLOR_BGR2RGB)


# plt.imshow(old_thumb_bg)
# plt.show()
# plt.imshow(new_thumb_bg)
# plt.show()
# exit()
# ==================================================================================================


def read_messages(bag_path: str):
    with AnyReader([Path(bag_path)]) as reader:
        topic_names = [x.topic for x in reader.connections]
        connections = [x for x in reader.connections if x.topic in TOPICS_EXTRACT]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            yield connection.topic, msg, timestamp


def img_msg_to_array(msg):
    bridge = CvBridge()
    cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
    return np.asarray(cv_image)


def numpy_to_binary(arr):
    is_success, buffer = cv2.imencode(".jpg", arr)
    io_buf = io.BytesIO(buffer)
    return io_buf.read()


# ==================================================================================================
def bag2dataset(bag_id):
    path_bag = f"{PATH_ROSBAGS}/{bag_id}"
    print(f"Reading {bag_id}.bag ...")
    msg_data = {k: [] for k in TOPICS_EXTRACT}

    read_messages(path_bag)
    print(f"Reading {bag_id}.bag ...")
    for topic, msg, timestamp in read_messages(path_bag):
        if topic in msg_data.keys():
            if msg.header.stamp.sec > 0 or msg.header.stamp.nanosec > 0:
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            else:
                if len(msg_data[topic]) <= 0:
                    print(
                        "WARN: {} has no time stamps in header, using bag stamps".format(
                            topic
                        )
                    )
                stamp = timestamp / 1e9
            msg_data[topic].append([stamp, msg])

    print("Keys found in bag: ")
    for topic in msg_data.keys():
        print(f"{topic}: {len(msg_data[topic])}")

    t0 = -np.inf
    for topic in msg_data.keys():
        if len(msg_data[topic]) <= 0:
            print("Missing topic: {}".format(topic))
            if "digit" in topic:
                continue
            else:
                assert False
        if msg_data[topic][0][0] > t0:
            t0 = msg_data[topic][0][0]
        for i in range(len(msg_data[topic]) - 1):
            # Check that messages are in order based on time
            assert msg_data[topic][i][0] < msg_data[topic][i + 1][0]

    # Convert timestamps to be w.r.t t0
    for topic in msg_data.keys():
        for i in range(len(msg_data[topic])):
            msg_data[topic][i][0] = msg_data[topic][i][0] - t0

    sync_msg_data = {}
    min_count = np.inf
    max_count = -np.inf
    for topic in msg_data.keys():
        t = 0.0
        idx = 0
        sync_msg_data[topic] = []

        while idx < len(msg_data[topic]):
            if msg_data[topic][idx][0] < t:
                if t - msg_data[topic][idx][0] > 3.0:  # DEL_T:
                    print("Found OOB msg {} {}".format(msg_data[topic][idx][0], topic))
                idx += 1
            else:
                sync_msg_data[topic].append(msg_data[topic][idx][1])
                t += 1 / FPS

        count = len(sync_msg_data[topic])
        if count < min_count:
            min_count = count
        elif count > max_count:
            max_count = count
    print(
        "Msg count - min: {}, max: {}, sync len: {}".format(
            min_count, max_count, max_count * (1 / FPS)
        )
    )

    dataset = {
        "digit_thumb": [],
        "digit_index": [],
        "realsense": [],
        "action": [],
        "allegro_action": [],
        "robot_eef_pose": [],
        "robot_joint": [],
        "robot_joint_vel": [],
        "allegro_joint": [],
        "allegro_joint_vel": [],
        "timestamp": [],
        "bg_index": [],
        "bg_thumb": [],
        "step": [],
    }

    dataset["action"] = [np.zeros(7)]
    dataset["allegro_action"] = [np.zeros(16)]

    # extract allegro joints
    for i in range(min_count):
        msg = sync_msg_data["/allegroHand/joint_states"][i]
        dataset["allegro_joint"].append(msg.position)
        dataset["allegro_joint_vel"].append(msg.velocity)

        # get timestamp of the message
        dataset["timestamp"].append(
            msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        )
        dataset["step"].append(i)

        if i > 0:
            action = dataset["allegro_joint"][-1] - dataset["allegro_joint"][-2]
            dataset["allegro_action"].append(action)

    # extract franka joints
    for i in range(min_count):
        msg = sync_msg_data["/franka/joint_states"][i]
        dataset["robot_joint"].append(msg.position)
        dataset["robot_joint_vel"].append(msg.velocity)

        if i > 0:
            action = dataset["robot_joint"][-1] - dataset["robot_joint"][-2]
            dataset["action"].append(action)

    urdf_path = "/home/akashsharma/workspace/projects/gum_ws/src/GUM/gum/devices/metahand/ros/meta_hand_description/urdf/meta_hand_franka.urdf"
    franka_urdf_chain = pk.build_serial_chain_from_urdf(
        open(urdf_path).read(),
        end_link_name="panda_link8",
        root_link_name="base_link",
    )
    franka_urdf_chain = franka_urdf_chain.to(device="cpu")
    robot_joints = np.array(dataset["robot_joint"])
    robot_joint_tensor = torch.from_numpy(robot_joints).to(device="cpu")
    franka_eef_pose = franka_urdf_chain.forward_kinematics(
        robot_joint_tensor, end_only=True
    )
    franka_eef_pose = franka_eef_pose.get_matrix().numpy()
    franka_eef_position = franka_eef_pose[:, :3, 3]
    dataset["robot_eef_pose"] = franka_eef_position
    # dataset["action"] = franka_eef_position.copy()

    # extract digit images (images are compressed)
    print("Extracting digit images ...")
    for i in range(min_count):
        msg = sync_msg_data["/digit_thumb/compressed"][i]
        img_thumb = img_msg_to_array(msg)
        img_thumb = cv2.cvtColor(img_thumb, cv2.COLOR_BGR2RGB).astype("uint8")
        dataset["digit_thumb"].append(numpy_to_binary(img_thumb))

        msg = sync_msg_data["/digit_index/compressed"][i]
        img_index = img_msg_to_array(msg)
        img_index = cv2.cvtColor(img_index, cv2.COLOR_BGR2RGB).astype("uint8")
        dataset["digit_index"].append(numpy_to_binary(img_index))
        if "new" in bag_id:
            dataset["bg_index"].append(numpy_to_binary(new_index_bg))
            dataset["bg_thumb"].append(numpy_to_binary(new_thumb_bg))
        else:
            dataset["bg_index"].append(numpy_to_binary(old_index_bg))
            dataset["bg_thumb"].append(numpy_to_binary(old_thumb_bg))

    # extract realsense images (images are compressed)
    print("Extracting realsense images ...")
    for i in range(min_count):
        msg = sync_msg_data["/realsense/color/image_raw/compressed"][i]
        img = img_msg_to_array(msg)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("uint8")
        img = img.astype("uint8")
        dataset["realsense"].append(numpy_to_binary(img))

    # save dataset
    print("Saving dataset ... \n")
    with open(f"{PATH_OUTPUT}/{bag_id}.pkl", "wb") as f:
        pickle.dump(dataset, f)


def pickle2zarr(bag_id, buffer):
    # load dataset
    try:
        with open(f"{PATH_OUTPUT}/{bag_id}", "rb") as f:
            dataset = pickle.load(f)
    except Exception as e:
        print(f"Error in {bag_id}: {e}")
        return

    print(f"Adding the following keys from {bag_id}")
    for key in dataset.keys():
        dataset[key] = np.array(dataset[key])
        print(f"{key}: {dataset[key].shape}")
    print("---")
    dataset["ep_id"] = np.array([bag_id] * len(dataset["step"]))

    try:
        buffer.extend(dataset)  # , compressors="disk")
    except Exception as e:
        print(f"Error in creating episode: {repr(e)}")


def check_zarr(buffer, episode=0):
    ep = buffer.get_episode(episode)
    digit_index = ep["digit_index"]
    fig, ax = plt.subplots(1, 1)
    for i in range(len(digit_index)):
        ax.clear()
        img = digit_index[i]
        img_buf = np.frombytes(img, np.uint8)
        img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Frame {i}")
        plt.pause(0.01)
    plt.close()


def convert2pickle():
    bags = os.listdir(PATH_ROSBAGS)
    for bag in bags:
        try:
            bag2dataset(bag)
        except Exception as e:
            print(f"{e} Error in {bag}")
            continue


def convert2zarr():
    zarr_path = os.path.join(PATH_OUPUT_ZARR, "replay_buffer.zarr")
    buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode="a")
    bags = os.listdir(PATH_OUTPUT)
    for bag in bags:
        try:
            pickle2zarr(bag, buffer)
        except Exception as e:
            print(f"Error exception {e} in {bag}")
            continue
    return buffer


def main():
    os.makedirs(PATH_OUTPUT, exist_ok=True)
    os.makedirs(PATH_OUPUT_ZARR, exist_ok=True)

    # extract dataset in pickle format
    convert2pickle()

    # convert pickle to zarr
    buffer = convert2zarr()

    # check zarr file. This should visualize the digit_index images for the episode
    check_zarr(buffer, episode=0)

    print("*********")


if __name__ == "__main__":
    main()
