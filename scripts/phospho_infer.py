from phosphobot.camera import AllCameras
from phosphobot.api.client import PhosphoApi
from gr00t.eval.robot import RobotInferenceClient

import time
import numpy as np

host = "20.199.85.87"
port = 5555

# Connect to the phosphobot server
client = PhosphoApi(base_url="http://localhost:80")

# Get a camera frame
allcameras = AllCameras()

# Need to wait for the cameras to initialize
time.sleep(1)

while True:
    images = [
        allcameras.get_rgb_frame(camera_id=0, resize=(240, 320)),
        allcameras.get_rgb_frame(camera_id=1, resize=(240, 320)),
    ]

    for i in range(0, len(images)):
        image = images[i]
        # Step 1: Transpose width and height (from (320, 240, 3) to (240, 320, 3))
        transposed_array = image.swapaxes(0, 1)

        # Step 2: Add a batch dimension (from (240, 320, 3) to (1, 240, 320, 3))
        converted_array = np.expand_dims(transposed_array, axis=0)

        # Step 3: Ensure dtype is uint8 (if it isn’t already)
        converted_array = converted_array.astype(np.uint8)

        images[i] = converted_array

    # Create a policy wrapper
    policy_client = RobotInferenceClient(host=host)

    state = np.array(client.control.read_joints().angles_rad)
    print(state[0:5].shape)
    obs = {
        "video.cam_context": images[0],
        "video.cam_wrist": images[1],
        "state.single_arm": state[0:5].reshape(1, 5),
        "state.gripper": np.array([state[5]]).reshape(1, 1),
        "annotation.human.action.task_description": ["Put the green brick in the box"],
    }

    # print("-> obs keys")
    # for key in obs.keys():
    #     print(obs[key].shape)

    response = policy_client.get_action(obs)

    for i in range(0, 5):
        arm_action = response["action.single_arm"][i]
        gripper_action = response["action.gripper"][i]

        # action = np.concatenate((arm_action, gripper_action))
        action = np.append(arm_action, gripper_action)

        state = np.array(client.control.read_joints().angles_rad)
        target_state = state + action

        print(f"action {action}")
        print(f"state: {state}")
        print(f"target_state: {target_state}")
        print("---")

        client.control.write_joints(angles=target_state.tolist())
        # Wait to respect frequency control (30 Hz)
        time.sleep(1 / 30)
