import time

import cv2
import numpy as np
from phosphobot.api.client import PhosphoApi
from phosphobot.camera import AllCameras

from gr00t.eval.robot import RobotInferenceClient

host = "20.199.85.87"
port = 5555

# Change this by your task description
TASK_DESCRIPTION = "Pick up the green lego brick from table and place it into the black container."

# Connect to the phosphobot server
client = PhosphoApi(base_url="http://localhost:80")

# Get a camera frame
allcameras = AllCameras()

# Need to wait for the cameras to initialize
time.sleep(1)

while True:
    images = [
        allcameras.get_rgb_frame(camera_id=0, resize=(320, 240)),
        allcameras.get_rgb_frame(camera_id=1, resize=(320, 240)),
    ]

    for i in range(0, len(images)):
        image = images[i]

        # Convert to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Add a batch dimension (from (240, 320, 3) to (1, 240, 320, 3))
        converted_array = np.expand_dims(image, axis=0)

        # Ensure dtype is uint8 (if it isn’t already)
        converted_array = converted_array.astype(np.uint8)

        images[i] = converted_array

    # Create a policy wrapper
    policy_client = RobotInferenceClient(host=host)

    state = np.array(client.control.read_joints().angles_rad)
    obs = {
        "video.cam_context": images[0],
        "video.cam_wrist": images[1],
        "state.single_arm": state[0:5].reshape(1, 5),
        "state.gripper": np.array([state[5]]).reshape(1, 1),
        "annotation.human.action.task_description": [TASK_DESCRIPTION],
    }

    # print("-> obs keys")
    # for key in obs.keys():
    #     print(obs[key].shape)

    response = policy_client.get_action(obs)

    for i in range(0, response["action.single_arm"].shape[0]):
        arm_action = response["action.single_arm"][i]
        gripper_action = response["action.gripper"][i]

        # action = np.concatenate((arm_action, gripper_action))
        action = np.append(arm_action, gripper_action)

        # Add a condition to force the gripper to close
        if action[-1] < 0.35:
            print(f"Overide to close gripper for: {action[-1]}")
            action[-1] = 0.0

        client.control.write_joints(angles=action.tolist())
        # Wait to respect frequency control (30 Hz)
        time.sleep(1 / 30)
