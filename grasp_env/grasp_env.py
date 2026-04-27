"""
GraspEnv: single-arm grasping Gymnasium environment.

Observation
-----------
Dict with two keys matching the LeRobot dataset format:

    observation.state   float32 (STATE_DIM,)   [joint0..5 | gripper_aperture |
                                                  Fx Fy Fz Tx Ty Tz |
                                                  tactile_L(6) tactile_R(6)]
    observation.images.wrist  uint8 (H, W, 3)  wrist RGB camera

Action
------
    float32 (ACTION_DIM,)   [joint0..5 target positions | gripper_aperture]
    gripper_aperture ∈ [0, 1] where 0 = open, 1 = fully closed

Reward
------
    +10  object lifted ≥ LIFT_SUCCESS_M above table
    −0.1 per step (time penalty)
    −5   object dropped or left table without success

Episode terminates on success or max_episode_steps.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import mujoco
import numpy as np
from gymnasium.spaces import Box, Dict as GymDict

from irl_control.device import DeviceState
from irl_control.mujoco_gym_app import MujocoGymApp
from irl_control.utils.target import Target

# ── constants ──────────────────────────────────────────────────────────────
STATE_DIM = 25      # 6 arm joints + 1 gripper + 6 F/T + 12 tactile
ACTION_DIM = 7      # 6 joint targets + 1 gripper aperture

IMG_H, IMG_W = 120, 160   # wrist camera resolution (rows, cols)

GRIPPER_JOINT_MAX = 0.8   # right_outer_knuckle_joint upper limit (rad)
GRIPPER_ACTUATOR_IDX = 6  # ctrl index of the gripper motor (0-indexed after 6 arm joints)

# Home configuration: UR5 "elbow-up" pose pointing forward and down
HOME_JOINTS = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])

# Table surface height (z) and object initial z (surface + ball radius + small gap)
TABLE_Z = 0.76
OBJ_Z   = TABLE_Z + 0.04   # 0.03 m radius + 0.01 m clearance

# Reward thresholds
LIFT_SUCCESS_M = 0.08     # object must rise ≥ 8 cm above table to count as success

# Tactile: max contact force per pad used for normalisation
TACTILE_FMAX = 50.0


class GraspEnv(MujocoGymApp):
    """Gymnasium environment for tabletop grasping with a single UR5 + Robotiq 85."""

    # timestep=0.003 × frameskip=3 → dt=0.009 → render_fps must equal round(1/dt)=111
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 111}

    def __init__(
        self,
        object_type: str = "random",
        render_mode: str = "rgb_array",
        width: int = IMG_W,
        height: int = IMG_H,
    ):
        self._object_type = object_type

        obs_space = GymDict({
            "observation.state": Box(
                low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
            ),
            "observation.images.wrist": Box(
                low=0, high=255, shape=(IMG_H, IMG_W, 3), dtype=np.uint8
            ),
        })
        act_space = Box(
            low=np.array([-2 * np.pi] * 6 + [0.0], dtype=np.float32),
            high=np.array([2 * np.pi] * 6 + [1.0], dtype=np.float32),
            dtype=np.float32,
        )

        super().__init__(
            robot_config_file="grasp.yaml",
            scene_file="grasp_scene.xml",
            observation_space=obs_space,
            action_space=act_space,
            osc_device_pairs=[("ur5right", "arm_osc")],
            osc_use_admittance=True,
            robot_name="SingleUR5",
            render_mode=render_mode,
            width=width,
            height=height,
            hide_mjpy_warnings=True,
        )

        # Cache MuJoCo IDs looked up once
        self._arm_joint_ids: np.ndarray = self._get_arm_joint_ids()
        self._gripper_joint_id: int = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_outer_knuckle_joint_ur5right"
        )
        self._steel_ball_joint_id: int = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "steel_ball_joint"
        )
        self._foam_ball_joint_id: int = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "foam_ball_joint"
        )
        self._left_pad_geom_id: int = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_fingerpad"
        )
        self._right_pad_geom_id: int = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_fingerpad"
        )
        self._wrist_cam_id: int = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam"
        )

        self._active_obj_joint_id: int = self._steel_ball_joint_id
        self._active_obj_name: str = "steel_ball"
        self._initial_obj_z: float = OBJ_Z

        # Dedicated offscreen renderer for the wrist camera (bypasses gymnasium's
        # render() API whose signature changed between versions)
        self._wrist_renderer = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)

    # ── MujocoGymApp abstract property ─────────────────────────────────────
    @property
    def default_start_pt(self):
        return None

    def close(self):
        if hasattr(self, '_wrist_renderer'):
            self._wrist_renderer.close()
        super().close()

    # ── Gymnasium interface ─────────────────────────────────────────────────
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict, Dict]:
        # MujocoEnv.reset() seeds np_random, calls _reset_simulation(),
        # then calls reset_model() which we implement below.
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def reset_model(self) -> Dict:
        """Required by MujocoEnv; called after mj_resetData to set initial state."""
        # Reset arm to home configuration
        for i, jid in enumerate(self._arm_joint_ids):
            qadr = self.model.jnt_qposadr[jid]
            self.data.qpos[qadr] = HOME_JOINTS[i]

        # Choose object type using np_random (seeded by MujocoEnv.reset())
        if self._object_type == "random":
            use_steel = bool(self.np_random.integers(0, 2))
        else:
            use_steel = self._object_type == "steel"

        if use_steel:
            self._active_obj_joint_id = self._steel_ball_joint_id
            self._active_obj_name = "steel_ball"
            hide_joint_id = self._foam_ball_joint_id
        else:
            self._active_obj_joint_id = self._foam_ball_joint_id
            self._active_obj_name = "foam_ball"
            hide_joint_id = self._steel_ball_joint_id

        # Place active object randomly on table within arm reach
        xy = self.np_random.uniform(-0.12, 0.12, size=2)
        obj_pos = np.array([0.65 + xy[0], xy[1], OBJ_Z])
        self._place_free_body(self._active_obj_joint_id, pos=obj_pos)
        self._initial_obj_z = OBJ_Z

        # Hide inactive object below ground
        self._place_free_body(hide_joint_id, pos=np.array([0.0, 0.0, -1.0]))

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)

        q_target = action[:6]                        # desired joint positions
        gripper_cmd = float(action[6])               # [0, 1]

        self._apply_pd_control(q_target, gripper_cmd)

        obs = self._get_obs()
        reward, terminated = self._compute_reward()
        truncated = False
        info: Dict = {"object": self._active_obj_name}

        return obs, reward, terminated, truncated, info

    # ── OSC helper for micro-lift (called by GraspPipeline) ────────────────
    def micro_lift(self, delta_z: float = 0.008, steps: int = 80) -> None:
        """
        Move the EE upward by delta_z meters using OSC.
        Called by GraspPipeline before reading post-lift F/T.
        """
        target = Target()
        ee_xyz = self._get_device("ur5right").get_state(DeviceState.EE_XYZ).copy()
        target.set_xyz(ee_xyz)
        target.set_abg(np.array([0.0, -np.pi / 2, 0.0]))

        goal_z = ee_xyz[2] + delta_z
        for _ in range(steps):
            # Ramp smoothly toward goal
            current_z = self._get_device("ur5right").get_state(DeviceState.EE_XYZ)[2]
            target.set_xyz(np.array([ee_xyz[0], ee_xyz[1],
                                     current_z + (goal_z - current_z) * 0.15]))
            ctrlr_output = self.controller.generate({"ur5right": target})
            ctrl = np.zeros(self.model.nu)
            for force_idx, force in zip(*ctrlr_output):
                ctrl[force_idx] = force
            self.do_simulation(ctrl, self.frame_skip)

    # ── Sensor readings ────────────────────────────────────────────────────
    def get_ft_reading(self) -> np.ndarray:
        """Return 6-dim F/T vector [Fx,Fy,Fz,Tx,Ty,Tz] from wrist sensor."""
        dev = self._get_device("ur5right")
        force  = dev.get_state(DeviceState.FORCE)
        torque = dev.get_state(DeviceState.TORQUE)
        return np.concatenate([force, torque])

    def get_tactile_reading(self) -> np.ndarray:
        """
        Return 12-dim tactile vector.

        Layout:
          [0:3]  left pad net contact force  (Fx, Fy, Fz) in world frame
          [3]    left pad contact count / 4  (normalised coverage proxy)
          [4:5]  left pad shear magnitude, normal force
          [6:9]  right pad net contact force (Fx, Fy, Fz)
          [9]    right pad contact count / 4
          [10:11] right pad shear magnitude, normal force
        """
        return np.concatenate([
            self._pad_tactile(self._left_pad_geom_id),
            self._pad_tactile(self._right_pad_geom_id),
        ])

    # ── Internal helpers ───────────────────────────────────────────────────
    def _get_obs(self) -> Dict:
        q_arm = np.array([
            self.data.qpos[self.model.jnt_qposadr[jid]]
            for jid in self._arm_joint_ids
        ], dtype=np.float32)

        gripper_pos = float(
            self.data.qpos[self.model.jnt_qposadr[self._gripper_joint_id]]
        )
        gripper_aperture = np.array([gripper_pos / GRIPPER_JOINT_MAX], dtype=np.float32)

        ft = self.get_ft_reading().astype(np.float32)
        tactile = self.get_tactile_reading().astype(np.float32)

        state = np.concatenate([q_arm, gripper_aperture, ft, tactile])

        # Wrist camera image via mujoco.Renderer (stable API, version-independent)
        self._wrist_renderer.update_scene(self.data, camera="wrist_cam")
        img = self._wrist_renderer.render().copy()

        return {
            "observation.state": state,
            "observation.images.wrist": img,
        }

    def _apply_pd_control(self, q_target: np.ndarray, gripper_cmd: float) -> None:
        """PD torque controller for joint-space action."""
        KP_ARM = 300.0
        KD_ARM = 30.0
        # Gripper inertia ≈ 0.00022 kg⋅m², dt=0.009 s → max stable kp ≈ 2.7
        KP_GRIP = 2.0
        KD_GRIP = 0.1

        ctrl = np.zeros(self.model.nu)

        for i, jid in enumerate(self._arm_joint_ids):
            qadr = self.model.jnt_qposadr[jid]
            vadr = self.model.jnt_dofadr[jid]
            q_err  = q_target[i] - self.data.qpos[qadr]
            dq     = self.data.qvel[vadr]
            ctrl[i] = KP_ARM * q_err - KD_ARM * dq

        # Gripper: map [0,1] → [0, GRIPPER_JOINT_MAX]
        g_target = gripper_cmd * GRIPPER_JOINT_MAX
        g_qadr = self.model.jnt_qposadr[self._gripper_joint_id]
        g_vadr = self.model.jnt_dofadr[self._gripper_joint_id]
        g_err = g_target - self.data.qpos[g_qadr]
        ctrl[GRIPPER_ACTUATOR_IDX] = KP_GRIP * g_err - KD_GRIP * self.data.qvel[g_vadr]

        self.do_simulation(ctrl, self.frame_skip)

    def _compute_reward(self) -> Tuple[float, bool]:
        obj_qadr = self.model.jnt_qposadr[self._active_obj_joint_id]
        obj_z = float(self.data.qpos[obj_qadr + 2])

        if obj_z >= TABLE_Z + LIFT_SUCCESS_M:
            return 10.0, True   # success

        reward = -0.1           # time penalty
        if obj_z < TABLE_Z - 0.05:
            # Object fell off table
            return -5.0, True

        return reward, False

    def _pad_tactile(self, geom_id: int) -> np.ndarray:
        """Aggregate contact forces on one fingerpad geom → 6-dim vector."""
        fx, fy, fz = 0.0, 0.0, 0.0
        count = 0
        contact_force_buf = np.zeros(6)  # [fx,fy,fz,tx,ty,tz] in contact frame
        for c_idx in range(self.data.ncon):
            con = self.data.contact[c_idx]
            if con.geom1 != geom_id and con.geom2 != geom_id:
                continue
            # Use mj_contactForce for accurate per-contact forces
            mujoco.mj_contactForce(self.model, self.data, c_idx, contact_force_buf)
            # contact_force_buf[0] = normal force (along contact frame z)
            # contact_force_buf[1:3] = tangential (friction) forces
            # Rotate from contact frame to world frame using con.frame (3×3 row-major)
            frame = con.frame.reshape(3, 3)   # rows = contact x, y, z axes in world
            f_world = frame.T @ contact_force_buf[:3]
            # Flip sign if this geom is geom2 (force is reported on geom1's body)
            if con.geom2 == geom_id:
                f_world = -f_world
            fx += float(f_world[0])
            fy += float(f_world[1])
            fz += float(f_world[2])
            count += 1

        net = np.array([fx, fy, fz])
        shear = float(np.linalg.norm(net[:2]))
        normal_f = abs(fz)
        coverage = min(count / 4.0, 1.0)

        return np.array([fx / TACTILE_FMAX, fy / TACTILE_FMAX, fz / TACTILE_FMAX,
                         coverage, shear / TACTILE_FMAX, normal_f / TACTILE_FMAX],
                        dtype=np.float32)

    def _get_arm_joint_ids(self) -> np.ndarray:
        """Return MuJoCo joint IDs for the 6 UR5 arm joints (joint0..5)."""
        names = [f"joint{i}_ur5right" for i in range(6)]
        return np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in names
        ], dtype=np.int32)

    def _get_device(self, name: str):
        """Retrieve an irl_control Device by name.

        With a single-arm setup, MujocoGymApp places 'ur5right' as a sub-device
        inside Robot('SingleUR5'), so _irl_devices only contains the Robot wrapper.
        Look up through self.robot first, then fall back to top-level _irl_devices.
        """
        if hasattr(self, 'robot') and self.robot is not None:
            try:
                return self.robot.get_device(name)
            except KeyError:
                pass
        for dev in self._irl_devices:
            if dev.name == name:
                return dev
        raise KeyError(f"Device '{name}' not found")

    def _place_free_body(self, joint_id: int, pos: np.ndarray) -> None:
        """Set the (x,y,z) of a freejoint body."""
        qadr = self.model.jnt_qposadr[joint_id]
        self.data.qpos[qadr:qadr + 3] = pos
        # Identity quaternion
        self.data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        # Zero velocity
        vadr = self.model.jnt_dofadr[joint_id]
        self.data.qvel[vadr:vadr + 6] = 0.0
