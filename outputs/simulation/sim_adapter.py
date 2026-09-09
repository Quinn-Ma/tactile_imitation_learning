"""MuJoCo/robosuite Lift extension; executed with Python 3.12.

The model remains a rigid cuboid. Contact forces are solver outputs, not
calibrated tactile images, deformation, or physical damage measurements.
"""
from __future__ import annotations
import os, sys, json, pathlib, tempfile
ROOT = pathlib.Path(__file__).resolve().parents[2]
DEPS = ROOT / 'work' / 'sim_deps'
sys.path.insert(0, str(DEPS))
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'work' / 'sim_numba')
(ROOT / 'work' / 'sim_tmp').mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(ROOT / 'work' / 'sim_tmp')
import numpy as np
import mujoco
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.grippers.panda_gripper import PandaGripper
from scipy.spatial.transform import Rotation


class ContinuousPandaGripper(PandaGripper):
    """Explicit benchmark extension: continuous instead of sign-only rate."""
    def format_action(self, action):
        self.current_action = np.clip(
            self.current_action + np.array([-1., 1.]) * self.speed * np.clip(action, -1, 1), -1., 1.)
        return self.current_action


GRIPPER_MAPPING['ContinuousPandaGripper'] = ContinuousPandaGripper


def parameters(seed: int, ood: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    return dict(mass=float(rng.choice([.65, .9]) if ood else rng.choice([.06, .12, .25, .4])),
                friction=float(rng.choice([.12, .22]) if ood else rng.choice([.35, .6, .9])),
                actuator_cap=float(rng.choice([3., 6., 12., 20.])),
                force_budget=8., grip_rate=float(rng.choice([.15, .35, .65, 1.])),
                probe_amplitude=float(rng.uniform(.003,.012)),
                lift_height=float(rng.uniform(.12,.18)),
                grasp_offset=rng.normal(0,.0025 if not ood else .005,2).tolist(),
                behavior=int(rng.integers(0,4)))


class SimAdapter:
    dt = .05
    action_dim = 7
    horizon = 150
    takeover_step = 62
    def __init__(self, render=True):
        # robosuite samples its cuboid dimensions when compiling the model.
        # Fix that construction seed so all processes compile identical geometry.
        np.random.seed(20270915)
        cfg = load_composite_controller_config(controller='BASIC')
        cfg['body_parts'] = {'right': cfg['body_parts']['right']}
        cfg['body_parts']['right']['type'] = 'OSC_POSE'
        self.env = suite.make('Lift', robots='Panda', gripper_types='ContinuousPandaGripper',
            controller_configs=cfg, initialization_noise=None, hard_reset=False,
            has_renderer=False, has_offscreen_renderer=render, use_camera_obs=render,
            camera_names='agentview', camera_heights=64, camera_widths=64,
            reward_shaping=True, control_freq=20, horizon=self.horizon)
        self.render = render
        self.model = self.env.sim.model._model
        self.data = self.env.sim.data._data
        self.cube_geom = self.env.sim.model.geom_name2id('cube_g0')
        self.cube_vis = self.env.sim.model.geom_name2id('cube_g0_vis')
        self.cube_body = self.env.cube_body_id
        self.compiled_half_size = self.model.geom_size[self.cube_geom,:3].copy()
        self.table_geom = self.env.sim.model.geom_name2id('table_collision')
        self.finger_geoms = [set(), set()]
        for gid in range(self.model.ngeom):
            name = self.env.sim.model.geom_id2name(gid) or ''
            for f in range(2):
                if f'finger{f+1}_' in name and 'collision' in name:
                    self.finger_geoms[f].add(gid)
        self.pad_ids = [self.env.sim.model.geom_name2id(f'gripper0_right_finger{f}_pad_collision') for f in [1,2]]
        self.grip_actuators = [i for i in range(self.model.nu)
            if 'gripper' in (self.env.sim.model.actuator_id2name(i) or '')]
        self.substep_peak = np.zeros(2)
        self.substep_count = 0
        old_step2 = self.env.sim.step2
        def measured_step2():
            old_step2()
            tactile, _, _, _ = self._contact_signals()
            self.substep_peak = np.maximum(self.substep_peak, tactile[[0,3]])
            self.substep_count += 1
        self.env.sim.step2 = measured_step2
        self._configure_camera()

    def _configure_camera(self):
        cam = self.env.sim.model.camera_name2id('agentview')
        eye = np.array([.48,-.48,1.24]); target=np.array([0,0,.90])
        back=(eye-target)/np.linalg.norm(eye-target)
        right=np.cross([0,0,1],back); right/=np.linalg.norm(right)
        up=np.cross(back,right)
        q=Rotation.from_matrix(np.column_stack([right,up,back])).as_quat()
        self.model.cam_pos[cam]=eye
        self.model.cam_quat[cam]=q[[3,0,1,2]]
        self.model.cam_fovy[cam]=45

    def reset(self, seed=0, params=None):
        np.random.seed(seed)
        self.seed=int(seed); self.rng=np.random.default_rng(seed)
        self.params=parameters(seed) if params is None else dict(params)
        self.env.rng=np.random.default_rng(seed)
        self.env.placement_initializer.rng=np.random.default_rng(seed)
        self.raw=self.env.reset()
        # Geometry stays fixed at its compiled values: editing geom_size after
        # compilation invalidates MuJoCo collision structures.
        h=self.compiled_half_size
        self.params['half_size']=h.tolist()
        mass=self.params['mass']; self.model.body_mass[self.cube_body]=mass
        self.model.body_inertia[self.cube_body]=mass/3*np.array([h[1]**2+h[2]**2,h[0]**2+h[2]**2,h[0]**2+h[1]**2])
        for gid in {self.cube_geom}|self.finger_geoms[0]|self.finger_geoms[1]:
            self.model.geom_friction[gid,0]=self.params['friction']
        for aid in self.grip_actuators:
            self.model.actuator_forcerange[aid]=[-self.params['actuator_cap'],self.params['actuator_cap']]
        # Mass / inertia changes require propagating derived model constants.
        # mj_setConst computes at qpos0, so preserve the robot reset state.
        reset_qpos=self.data.qpos.copy(); reset_qvel=self.data.qvel.copy()
        mujoco.mj_setConst(self.model,self.data)
        self.data.qpos[:]=reset_qpos; self.data.qvel[:]=reset_qvel
        # Reset geometry is known to the common scripted setup, never to WM input.
        pose=np.r_[self.rng.uniform(-.018,.018,2),.8+h[2]+.001,1.,0.,0.,0.]
        self.env.sim.data.set_joint_qpos(self.env.cube.joints[0],pose)
        self.env.sim.forward()
        self._configure_camera()
        self.t=0; self.initial_z=float(pose[2]); self.initial_xy=pose[:2].copy()
        self.grasp_target=pose[:3].copy()+np.r_[self.params['grasp_offset'],.002]
        self.previous_relative_z=None
        self.substep_peak=np.zeros(2); self.substep_count=0
        self.raw=self.env._get_observations(force_update=True)
        return self.observe()

    def _contact_signals(self):
        values=np.zeros((2,3)); contact=np.zeros(2,dtype=np.float32)
        support_force=0.; vertical=np.zeros(2)
        # Common tangent basis: gravity-up projected perpendicular to pad-to-pad
        # closing direction, and its right-handed orthogonal complement.
        n=self.data.geom_xpos[self.pad_ids[1]]-self.data.geom_xpos[self.pad_ids[0]]
        n=n/max(np.linalg.norm(n),1e-9)
        tangent1=np.array([0.,0.,1.])-n*n[2]
        if np.linalg.norm(tangent1)<1e-5: tangent1=np.array([1.,0.,0.])-n*n[0]
        tangent1/=max(np.linalg.norm(tangent1),1e-9)
        tangent2=np.cross(n,tangent1)
        ft=np.zeros(6)
        for i in range(self.data.ncon):
            c=self.data.contact[i]
            if self.cube_geom not in (c.geom1,c.geom2): continue
            other=c.geom2 if c.geom1==self.cube_geom else c.geom1
            mujoco.mj_contactForce(self.model,self.data,i,ft)
            sign=1. if c.geom2==self.cube_geom else -1.
            frame=c.frame.reshape(3,3)
            world=sign*(frame.T@ft[:3])
            shear=sign*(frame[1]*ft[1]+frame[2]*ft[2])
            if other==self.table_geom: support_force+=max(0.,world[2])
            for f in range(2):
                if other in self.finger_geoms[f]:
                    values[f]+= [abs(ft[0]),float(shear@tangent1),float(shear@tangent2)]
                    vertical[f]+=world[2]; contact[f]=1.
        return values.reshape(6).astype(np.float32), contact, float(support_force), vertical

    def phase(self,t=None):
        t=self.t if t is None else t
        return 0 if t<44 else 1 if t<62 else 2 if t<78 else 3 if t<112 else 4 if t<135 else 5

    def observe(self):
        tactile,contact,support_force,vertical=self._contact_signals()
        r=self.raw
        cube=np.asarray(r['cube_pos']); eef=np.asarray(r['robot0_eef_pos'])
        relz=float(cube[2]-eef[2])
        slip=float(contact.any() and self.previous_relative_z is not None and
                   (relz-self.previous_relative_z)/self.dt < -.015)
        # Sampled macro-step slip proxy; no claim of incipient-slip ground truth.
        obs=dict(rgb=np.asarray(r['agentview_image'][::-1] if self.render else np.zeros((64,64,3)),dtype=np.uint8),
            tactile=tactile,proprio=np.r_[eef,r['robot0_eef_quat'],r['robot0_gripper_qpos']].astype(np.float32),
            height=np.array([cube[2]-self.initial_z],np.float32),
            support=np.array([support_force>.02],np.float32),contact=contact,
            cube_pose=np.r_[cube,r['cube_quat']].astype(np.float32),
            cube_velocity=self.data.cvel[self.cube_body].copy().astype(np.float32),
            support_force=np.array([support_force],np.float32),
            vertical_contact_force=np.asarray(vertical,np.float32),
            slip=np.array([slip],np.float32),success=np.array([self.env._check_success()],np.float32),
            phase=np.array([self.phase()],np.int16),
            substep_normal_peak=self.substep_peak.astype(np.float32),
            time=np.array([self.t*self.dt],np.float32))
        self.last_obs=obs
        return obs

    def script_action(self,t=None):
        t=self.t if t is None else t
        target=self.grasp_target.copy(); grip=-1.
        p=self.params; behavior=p['behavior']
        if t<22: target[2]+=.065
        elif t<44: pass
        elif t<62: grip=p['grip_rate']
        elif t<78:
            target[2]+=p['probe_amplitude']*.5*(1-np.cos((t-62)/16*np.pi))
            grip=p['grip_rate']*.25
            if behavior==1 and 68<=t<72: grip=-.18
        elif t<112:
            target[2]+=p['probe_amplitude']+(t-78)/34*p['lift_height']
            grip= .03 if behavior==2 else p['grip_rate']*.25
            if behavior==3 and 92<=t<97: grip=-.20
        elif t<135:
            target[2]+=p['lift_height']; target[0]+=.008*np.sin((t-112)*.4)
            grip=.02 if behavior in [1,2] else p['grip_rate']*.15
        else:
            target[2]+=max(0.,1-(t-135)/15)*p['lift_height']; grip=p['grip_rate']*.1
        a=np.zeros(7,np.float32)
        a[:3]=np.clip((target-self.raw['robot0_eef_pos'])*12,-.6,.6)
        if 62<=t<135:
            a[:2]+=self.rng.normal(0,.004,2)
            a[2]+=self.rng.normal(0,.002)
        a[6]=grip
        return np.clip(a,-1,1)

    def step(self,action):
        a=np.asarray(action,dtype=np.float32).reshape(7)
        if not np.isfinite(a).all(): raise ValueError('Non-finite action')
        a=np.clip(a,-1,1)
        self.previous_relative_z=float(self.raw['cube_pos'][2]-self.raw['robot0_eef_pos'][2])
        self.substep_peak=np.zeros(2); self.substep_count=0
        self.raw,reward,done,info=self.env.step(a)
        self.t+=1
        obs=self.observe()
        info.update(success=bool(obs['success'][0]),substeps=self.substep_count,
                    normal_peak=float(self.substep_peak.max()),force_budget=self.params['force_budget'])
        return obs,float(reward),bool(done),info

    def get_state(self):
        return dict(qpos=self.data.qpos.copy(),qvel=self.data.qvel.copy(),act=self.data.act.copy(),
                    ctrl=self.data.ctrl.copy(),time=float(self.data.time),t=self.t,
                    gripper=self.env.robots[0].gripper['right'].current_action.copy())

    def set_state(self,state):
        for key in ['qpos','qvel','act','ctrl']: getattr(self.data,key)[:]=state[key]
        self.data.time=state['time']; self.t=state['t'];self.env.timestep=self.t
        self.env.cur_time=self.t*self.dt;self.env.done=False
        self.env.robots[0].gripper['right'].current_action=state['gripper'].copy()
        self.env.sim.forward()
        self.env.robots[0].composite_controller.reset()
        self.raw=self.env._get_observations(force_update=True)
        return self.observe()

    def close(self): self.env.close()
