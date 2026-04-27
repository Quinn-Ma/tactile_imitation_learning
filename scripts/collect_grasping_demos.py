"""
Collect grasping demonstrations in GraspEnv simulation → LeRobot dataset.

Uses a scripted OSC controller that executes a 4-phase grasp:
  Phase 1  Move EE above object   (open gripper)
  Phase 2  Descend to grasp height (open gripper)
  Phase 3  Close gripper
  Phase 4  Lift object

Recorded per frame:
  observation.state          float32 (25,)   joint + gripper + F/T + tactile
  observation.images.wrist   uint8   (H,W,3) wrist camera
  action                     float32 (7,)    arm joint positions + gripper aperture

Usage
-----
python scripts/collect_grasping_demos.py \\
    --repo_id myorg/grasp_demos \\
    --root   ./data/grasp_demos \\
    --n_episodes 50

After collection, push to HuggingFace Hub:
    python scripts/collect_grasping_demos.py --push_to_hub --repo_id myorg/grasp_demos
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── dataset schema ──────────────────────────────────────────────────────────
STATE_DIM  = 25    # 6 joints + 1 gripper + 6 F/T + 12 tactile
ACTION_DIM = 7     # 6 joint targets + 1 gripper aperture
IMAGE_H    = 120
IMAGE_W    = 160
FPS        = 30

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (STATE_DIM,),
        "names": (
            ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
            + ["gripper_aperture"]
            + ["ft_fx", "ft_fy", "ft_fz", "ft_tx", "ft_ty", "ft_tz"]
            + [f"tactile_{i}" for i in range(12)]
        ),
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": (IMAGE_H, IMAGE_W, 3),
        "names": ["height", "width", "channel"],
        "info": {
            "video.fps": FPS,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "action": {
        "dtype": "float32",
        "shape": (ACTION_DIM,),
        "names": (
            ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
            + ["gripper_aperture"]
        ),
    },
}


# ── scripted grasp controller ────────────────────────────────────────────────

class ScriptedGraspController:
    """
    State-machine OSC controller that executes one scripted grasp episode.

    Phase 0  ABOVE   — move EE 15 cm above the object (gripper open)
    Phase 1  DESCEND — lower EE to grasp height       (gripper open)
    Phase 2  CLOSE   — hold position, close gripper
    Phase 3  LIFT    — raise EE 20 cm                 (gripper closed)
    """

    ABOVE, DESCEND, CLOSE, LIFT, DONE = range(5)
    PHASE_STEPS = {0: 80, 1: 80, 2: 60, 3: 80}   # steps per phase

    def __init__(self, env, gripper_joint_max: float = 0.8):
        self.env = env
        self.gripper_joint_max = gripper_joint_max
        self._phase = self.ABOVE
        self._phase_step = 0

    def reset(self):
        self._phase = self.ABOVE
        self._phase_step = 0

    @property
    def done(self) -> bool:
        return self._phase == self.DONE

    def step(self):
        """
        Execute one control step.

        Returns
        -------
        action : np.ndarray (7,)  arm joint positions + gripper aperture
        """
        from irl_control.utils.target import Target

        env = self.env

        # Current object position in world frame
        obj_qadr = env.model.jnt_qposadr[env._active_obj_joint_id]
        obj_pos  = env.data.qpos[obj_qadr:obj_qadr + 3].copy()

        target = Target()
        target.set_abg(np.array([0.0, -np.pi / 2, 0.0]))   # gripper pointing down

        if self._phase == self.ABOVE:
            ee_goal = obj_pos.copy(); ee_goal[2] += 0.18
            gripper_cmd = 0.0
        elif self._phase == self.DESCEND:
            ee_goal = obj_pos.copy(); ee_goal[2] += 0.06
            gripper_cmd = 0.0
        elif self._phase == self.CLOSE:
            ee_goal = obj_pos.copy(); ee_goal[2] += 0.06
            gripper_cmd = 1.0
        else:  # LIFT
            ee_goal = obj_pos.copy(); ee_goal[2] += 0.22
            gripper_cmd = 1.0

        target.set_xyz(ee_goal)

        # OSC torques for arm
        ctrlr_output = env.controller.generate({"ur5right": target})
        ctrl = np.zeros(env.model.nu)
        for force_idx, force in zip(*ctrlr_output):
            ctrl[force_idx] = force

        # PD torque for gripper
        g_target = gripper_cmd * self.gripper_joint_max
        g_qadr   = env.model.jnt_qposadr[env._gripper_joint_id]
        g_vadr   = env.model.jnt_dofadr[env._gripper_joint_id]
        ctrl[6]  = 50.0 * (g_target - env.data.qpos[g_qadr]) \
                 - 5.0  * env.data.qvel[g_vadr]

        env.do_simulation(ctrl, env.frame_skip)

        # Record action = achieved joint positions (IL target for next step)
        arm_q = np.array([
            env.data.qpos[env.model.jnt_qposadr[jid]]
            for jid in env._arm_joint_ids
        ], dtype=np.float32)
        gripper_aper = float(env.data.qpos[g_qadr]) / self.gripper_joint_max
        action = np.concatenate([arm_q, [gripper_aper]]).astype(np.float32)

        # Advance phase
        self._phase_step += 1
        if self._phase_step >= self.PHASE_STEPS.get(self._phase, 80):
            self._phase += 1
            self._phase_step = 0

        return action


# ── collection loop ──────────────────────────────────────────────────────────

def collect(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from grasp_env import GraspEnv

    root = Path(args.root)

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
            robot_type="sim_ur5_robotiq85",
            use_videos=True,
        )

    env = GraspEnv(
        object_type=args.object_type,
        render_mode="human" if args.render else "rgb_array",
        width=IMAGE_W,
        height=IMAGE_H,
    )
    controller = ScriptedGraspController(env)

    success_count = 0
    ep = 0
    while ep < args.n_episodes:
        logger.info("Episode %d / %d", ep + 1, args.n_episodes)
        obs, _ = env.reset(seed=ep)
        controller.reset()

        frames = []
        while not controller.done:
            action = controller.step()
            obs    = env._get_obs()          # observation after the step
            frames.append((obs, action))

        # Check if object was successfully lifted
        obj_qadr = env.model.jnt_qposadr[env._active_obj_joint_id]
        obj_z    = float(env.data.qpos[obj_qadr + 2])
        lifted   = obj_z >= 0.76 + 0.08      # TABLE_Z + LIFT_SUCCESS_M

        if args.only_success and not lifted:
            logger.info("  Episode failed (obj_z=%.3f) — discarding", obj_z)
            continue

        # Save frames to dataset
        for obs_t, action_t in frames:
            dataset.add_frame({
                "observation.state":        obs_t["observation.state"],
                "observation.images.wrist": obs_t["observation.images.wrist"],
                "action":                   action_t,
            })

        task_tag = f"Grasp {env._active_obj_name} and lift"
        dataset.save_episode(episode_data={"task": task_tag})
        logger.info("  Saved %d frames  lifted=%s  obj_z=%.3f",
                    len(frames), lifted, obj_z)

        success_count += lifted
        ep += 1

    dataset.finalize()
    logger.info(
        "Done. %d / %d episodes successful.  Dataset → %s",
        success_count, args.n_episodes, root,
    )

    if args.push_to_hub:
        logger.info("Pushing to HuggingFace Hub → %s …", args.repo_id)
        dataset.push_to_hub()
        logger.info("Pushed.")

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Collect grasping demos in GraspEnv simulation → LeRobot dataset"
    )
    parser.add_argument("--repo_id",      required=True,
                        help="HuggingFace repo-id, e.g. myorg/grasp_demos")
    parser.add_argument("--root",         default="./data/grasp_demos")
    parser.add_argument("--n_episodes",   type=int, default=50)
    parser.add_argument("--fps",          type=int, default=FPS)
    parser.add_argument("--object_type",  choices=["random", "steel", "foam"],
                        default="random",
                        help="Object type for each episode")
    parser.add_argument("--only_success", action="store_true",
                        help="Discard episodes where object was not lifted")
    parser.add_argument("--render",       action="store_true",
                        help="Render simulation in a window (slower)")
    parser.add_argument("--push_to_hub",  action="store_true")
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
