from gr00t.eval.robot import RobotInferenceClient
import numpy as np
import time

# Server details
host = "r18.modal.host"
port = 34187


def test_server_connection(timeout=10.0):
    print(f"Testing connection to inference server at {host}:{port} with {timeout}s timeout...")

    # Create a policy client to test connection (without timeout parameter)
    policy_client = RobotInferenceClient(host=host, port=port)

    # Create minimal dummy data
    obs = {
        "video.cam_context": np.random.randint(0, 256, (1, 240, 320, 3), dtype=np.uint8),
        "video.cam_wrist": np.random.randint(0, 256, (1, 240, 320, 3), dtype=np.uint8),
        "state.single_arm": np.random.rand(1, 5),
        "state.gripper": np.random.rand(1, 1),
        "annotation.human.action.task_description": ["do your thing!"],
    }

    # Implement timeout using a simple approach
    start_time = time.time()

    action = policy_client.get_action(obs)

    for key, value in action.items():
        print(f"Action: {key}: {value.shape}")

    elapsed_time = time.time() - start_time

    # Add a print statement to check if the server is reachable
    print(f"✅ Server is reachable! Response received in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    test_server_connection()
