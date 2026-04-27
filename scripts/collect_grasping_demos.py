"""
Collect grasping demonstrations and save them as a LeRobot v3.0 dataset.

Each episode contains the full 4-stage sequence:
  Stage 1  IL approach + initial finger closure   (arm joint actions)
  Stage 3  Micro-lift and F/T recording           (stored in observation.state)
  Stage 4  PINN grip-force command                (stored as action channel)

Observation state vector layout (example, 25-dim)
---------------------------------------------------
  [0 : 7]   arm joint positions
  [7]       gripper aperture
  [8 :14]   wrist F/T sensor  [Fx Fy Fz Tx Ty Tz]
  [14:26]   tactile sensor array (12 taxels)

Action vector layout (8-dim)
-----------------------------
  [0:7]    target arm joint positions
  [7]      target grip force command (N)

Usage
-----
python scripts/collect_grasping_demos.py \\
    --repo_id myorg/grasp_demos \\
    --root   ./data/grasp_demos \\
    --n_episodes 50 \\
    --fps 30

After collection, push to HuggingFace Hub:
    python scripts/collect_grasping_demos.py --push_to_hub --repo_id myorg/grasp_demos
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset feature schema
# ---------------------------------------------------------------------------

STATE_DIM   = 26   # joints (7) + gripper (1) + FT (6) + tactile (12)
ACTION_DIM  = 8    # joint targets (7) + grip force (1)
IMAGE_H     = 480
IMAGE_W     = 640
FPS         = 30

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (STATE_DIM,),
        "names": (
            ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
            + ["gripper_aperture"]
            + ["ft_fx", "ft_fy", "ft_fz", "ft_tx", "ft_ty", "ft_tz"]
            + [f"tactile_{i}" for i in range(12)]
        ),
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": (IMAGE_H, IMAGE_W, 3),
        "names": ["height", "width", "channel"],
        "info": {"video.fps": FPS, "video.codec": "av1", "video.pix_fmt": "yuv420p",
                 "video.is_depth_map": False, "has_audio": False},
    },
    "action": {
        "dtype": "float32",
        "shape": (ACTION_DIM,),
        "names": (
            ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
            + ["grip_force_n"]
        ),
    },
}


# ---------------------------------------------------------------------------
# Stub robot interface — replace with your actual hardware driver
# ---------------------------------------------------------------------------

class RobotInterface:
    """
    Replace this with your real robot driver.

    Methods
    -------
    get_state()     → np.ndarray (STATE_DIM,)
    get_image()     → np.ndarray (H, W, 3) uint8
    send_action(a)  → None
    is_episode_done() → bool
    reset()         → None
    """

    def get_state(self) -> np.ndarray:
        raise NotImplementedError("Connect your robot hardware here")

    def get_image(self) -> np.ndarray:
        raise NotImplementedError("Connect your camera here")

    def send_action(self, action: np.ndarray) -> None:
        raise NotImplementedError

    def is_episode_done(self) -> bool:
        return False

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Teleop / policy action source — replace with your teleop device
# ---------------------------------------------------------------------------

class TeleopController:
    """Replace with gamepad / VR glove / leader arm."""

    def get_action(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Connect your teleoperation device here")

    def is_ready(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Data collection loop
# ---------------------------------------------------------------------------

def collect(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    root = Path(args.root)

    # Create or resume dataset
    if (root / "meta" / "info.json").exists():
        logger.info("Resuming existing dataset at %s", root)
        dataset = LeRobotDataset.resume(repo_id=args.repo_id, root=root)
    else:
        logger.info("Creating new dataset at %s", root)
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            features=FEATURES,
            root=root,
            robot_type=args.robot_type,
            use_videos=True,
        )

    robot   = RobotInterface()
    teleop  = TeleopController()

    for ep in range(args.n_episodes):
        logger.info("Episode %d / %d", ep + 1, args.n_episodes)
        robot.reset()

        ep_frames = 0
        while not robot.is_episode_done():
            state  = robot.get_state()        # (STATE_DIM,)
            image  = robot.get_image()        # (H, W, 3) uint8
            action = teleop.get_action(state) # (ACTION_DIM,)

            dataset.add_frame({
                "observation.state":         state,
                "observation.images.wrist":  image,
                "action":                    action,
            })
            robot.send_action(action)
            ep_frames += 1

        dataset.save_episode(episode_data={"task": args.task_description})
        logger.info("  Saved episode with %d frames", ep_frames)

    dataset.finalize()
    logger.info("Dataset saved to %s  (%d total episodes)", root, dataset.num_episodes)

    if args.push_to_hub:
        logger.info("Pushing to HuggingFace Hub → %s …", args.repo_id)
        dataset.push_to_hub()
        logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Collect grasping demonstrations → LeRobot dataset")
    parser.add_argument("--repo_id",      required=True, help="HF repo id, e.g. myorg/grasp_demos")
    parser.add_argument("--root",         default="./data/grasp_demos")
    parser.add_argument("--n_episodes",   type=int, default=50)
    parser.add_argument("--fps",          type=int, default=FPS)
    parser.add_argument("--robot_type",   default="custom")
    parser.add_argument("--task_description", default="Grasp object and hold stably")
    parser.add_argument("--push_to_hub",  action="store_true")
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
