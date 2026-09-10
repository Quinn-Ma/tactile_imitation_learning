"""Fixed-geometry extension of the original rigid-box Lift benchmark.

Use ``parameters(seed, domain, geometry=...)`` then
``sim.reset(seed, params=params)``. Domains are ``id``, ``geometry_ood``,
``physics_ood`` (mass/friction only), and ``combined_ood``. Geometry may be
specified to balance a manifest, but must belong to the requested domain.
Calling ``SimAdapterV2().reset(seed)`` retains the original legacy reset.

All boxes are robosuite BoxObjects built BEFORE MuJoCo compilation. The
original adapter supplies observations, tactile extraction, actions, reward,
reset placement, state access and stepping, without changing their semantics.
No geometry size is written after compilation. The original file is unedited.

The analytic screen assumes a static, vertical, antipodal side grasp:
mg <= 2*mu*min(force_budget, actuator_cap), with capacity/mg >= 1.5.
Actuator force limits are used as a nominal per-finger normal-force proxy.
This is a necessary capacity screen under that model, NOT a sufficient
condition for a dynamically stable or successful grasp. Actual contact normals,
finger joint friction, acceleration, torque balance, contact loss and controller
behavior can reduce available support. Force budget is an evaluation threshold,
not an enforced contact-force clamp. No physical damage or deformation model.

Source construction follows installed robosuite 1.5.1 Lift._load_model and
BoxObject(size=...), preserving the original table, robot, material and naming.
The distribution and geometry supports below are fixed without outcome tuning.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
ORIGINAL_PATH = ROOT / 'outputs' / 'simulation' / 'sim_adapter.py'
# Packaging-only import/provenance guard; both predecessor bodies were audited.
EXECUTED_ORIGINAL_SHA256 = '6d8189ac2c7fd27d0c67074e6ca9a06fc042f8be88993411df85ed0d8f8a5847'
AUDITED_ORIGINAL_SHA256 = (EXECUTED_ORIGINAL_SHA256, '407a7ba79fd2b4ec29219740e185a34b497ddf643b6565ab290b823e9591c321')
ORIGINAL_SHA256 = hashlib.sha256(ORIGINAL_PATH.read_bytes()).hexdigest()
if ORIGINAL_SHA256 not in AUDITED_ORIGINAL_SHA256:
    raise RuntimeError('Unrecognized predecessor adapter; refusing unaudited import')

# Load the owned, unchanged predecessor by absolute path, regardless of cwd.
_MODULE_NAME = '_revision_v2_original_sim_adapter'
if _MODULE_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_MODULE_NAME, ORIGINAL_PATH)
    _original = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE_NAME] = _original
    _spec.loader.exec_module(_original)
else:
    _original = sys.modules[_MODULE_NAME]

np = _original.np
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import UniformRandomSampler

PROTOCOL_VERSION = 'revision-v2-fixed-boxes-feasible-v1'
GEOMETRIES = {
    'legacy_cube': None,  # Original Lift's seeded uniform half sizes [.020,.022].
    'squat_box': (.026, .020, .014),
    'tall_box': (.018, .018, .030),
    'wide_box': (.030, .015, .018),
    'narrow_tall_box': (.014, .020, .038),
}
ID_GEOMETRIES = ('legacy_cube', 'squat_box', 'tall_box')
OOD_GEOMETRIES = ('wide_box', 'narrow_tall_box')
DOMAINS = ('id', 'geometry_ood', 'physics_ood', 'combined_ood')
ID_MASSES = (.06, .12, .25)
ID_FRICTIONS = (.60, .75, .90)
OOD_MASSES = (.30, .40, .50)
OOD_FRICTIONS = (.30, .40, .50)
ACTUATOR_CAPS = (6., 12., 20.)
GRAVITY = 9.81
FORCE_BUDGET = 8.
MIN_SAFETY_RATIO = 1.5
MAX_GRASP_OFFSET_NORM = .0035
APERTURE_CLEARANCE = .002
# panda_gripper.xml: finger joints +/- .04; inner pad faces add -.001 m:
# .08 + 2*(.0085-.005-.004) = .079 m. This is geometric pad clearance,
# not a claim that dynamics reaches the joint limits during every approach.
MAX_PAD_APERTURE = .079
LEGACY_HALF_SIZE_UPPER_BOUND = (.022, .022, .022)


def geometry_support(domain: str) -> tuple[str, ...]:
    if domain == 'legacy':
        return ('legacy_cube',)
    if domain not in DOMAINS:
        raise ValueError(f'Unknown domain {domain!r}; expected {DOMAINS} or legacy')
    return OOD_GEOMETRIES if domain in ('geometry_ood', 'combined_ood') else ID_GEOMETRIES


def _geometry_key(params: dict, fallback: str = 'legacy_cube') -> str:
    key = params.get('geometry', params.get('geometry_key', fallback))
    if 'geometry' in params and 'geometry_key' in params and params['geometry'] != params['geometry_key']:
        raise ValueError('geometry and geometry_key disagree')
    if key not in GEOMETRIES:
        raise ValueError(f'Unknown geometry {key!r}; expected {tuple(GEOMETRIES)}')
    return key


def feasibility(params: dict, half_size=None) -> dict:
    """Return a JSON-serializable, conservative geometric/capacity screen.

    Horizontal box diagonal bounds every yaw's projected grasp width. Account
    for both sides of the requested XY grasp offset plus 2 mm clearance. The
    legacy upper size bound is used until its actual compiled size is known.
    """
    key = _geometry_key(params)
    h = np.asarray(half_size if half_size is not None else
                   GEOMETRIES[key] or LEGACY_HALF_SIZE_UPPER_BOUND, dtype=float)
    offset = np.asarray(params['grasp_offset'], dtype=float)
    if h.shape != (3,) or offset.shape != (2,):
        raise ValueError('half_size must have 3 entries; grasp_offset must have 2')
    mass, mu, cap = (float(params[k]) for k in ('mass', 'friction', 'actuator_cap'))
    budget = float(params['force_budget'])
    if not np.isfinite(np.r_[h, offset, mass, mu, cap, budget]).all() or min(*h, mass, mu, cap, budget) <= 0:
        raise ValueError('Geometry and physical parameters must be finite and positive')
    normal_cap = min(budget, cap)
    weight = mass * GRAVITY
    capacity = 2 * mu * normal_cap
    safety_ratio = capacity / weight
    width_bound = 2 * float(np.linalg.norm(h[:2]))
    offset_norm = float(np.linalg.norm(offset))
    required_aperture = width_bound + 2 * offset_norm + APERTURE_CLEARANCE
    return dict(
        model='static_antipodal_coulomb_nominal_actuator_proxy',
        necessary_not_sufficient=True,
        gravity_m_s2=GRAVITY, weight_n=weight, per_finger_normal_cap_proxy_n=normal_cap,
        vertical_capacity_n=capacity, safety_ratio=safety_ratio,
        required_safety_ratio=MIN_SAFETY_RATIO,
        force_feasible=bool(safety_ratio >= MIN_SAFETY_RATIO - 1e-12),
        horizontal_width_bound_m=width_bound, grasp_offset_norm_m=offset_norm,
        required_aperture_m=required_aperture, available_pad_aperture_m=MAX_PAD_APERTURE,
        width_feasible=bool(required_aperture <= MAX_PAD_APERTURE + 1e-12),
        feasible=bool(safety_ratio >= MIN_SAFETY_RATIO - 1e-12 and
                      required_aperture <= MAX_PAD_APERTURE + 1e-12),
    )


def parameters(seed: int, domain: str = 'id', geometry: str | None = None) -> dict:
    """Sample deterministic episode parameters without consulting outcomes.

    SeedSequence streams: 0 geometry, 1 mass/friction, 2 actuator, 3 nuisance.
    Same-seed ID / geometry-OOD share all physical and nuisance parameters.
    Same-seed ID / physics-OOD share geometry, actuator and nuisance parameters.
    Actuator cap is sampled first and uniformly across the same support in every
    domain; then a uniformly selected feasible mass/friction pair is selected
    conditional on that cap. Thus the screen does not shift the cap marginal.
    This is NOT independent mass/friction sampling after the feasibility screen.
    """
    seed = int(seed)
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    allowed = geometry_support(domain)
    if geometry is not None and geometry not in allowed:
        raise ValueError(f'{geometry!r} is not in domain {domain!r} support {allowed}')
    if domain == 'legacy':
        p = _original.parameters(seed)
        p.update(geometry='legacy_cube', geometry_key='legacy_cube', domain='legacy',
                 protocol_version=PROTOCOL_VERSION)
        p['feasibility'] = feasibility(p)
        return p
    streams = [np.random.default_rng(np.random.SeedSequence([seed, i])) for i in range(4)]
    grng, prng, arng, nrng = streams
    selected_geometry = str(grng.choice(allowed))
    selected_geometry = geometry if geometry is not None else selected_geometry
    cap = float(arng.choice(ACTUATOR_CAPS))
    physical_ood = domain in ('physics_ood', 'combined_ood')
    masses = OOD_MASSES if physical_ood else ID_MASSES
    frictions = OOD_FRICTIONS if physical_ood else ID_FRICTIONS
    pairs = [(m, mu) for m, mu in itertools.product(masses, frictions)
             if 2 * mu * min(FORCE_BUDGET, cap) >= MIN_SAFETY_RATIO * m * GRAVITY]
    if not pairs:
        raise ValueError('The prespecified support has no feasible mass/friction pair')
    mass, friction = pairs[int(prng.integers(len(pairs)))]
    # Same legacy nuisance supports except a declared radial offset bound, which
    # guarantees the geometric aperture screen for every listed fixed box.
    grip_rate = float(nrng.choice([.15, .35, .65, 1.]))
    probe_amplitude = float(nrng.uniform(.003, .012))
    lift_height = float(nrng.uniform(.12, .18))
    offset = nrng.normal(0, .0025, 2)
    norm = float(np.linalg.norm(offset))
    if norm > MAX_GRASP_OFFSET_NORM:
        offset *= MAX_GRASP_OFFSET_NORM / norm
    p = dict(mass=float(mass), friction=float(friction), actuator_cap=cap,
             force_budget=FORCE_BUDGET, grip_rate=grip_rate,
             probe_amplitude=probe_amplitude, lift_height=lift_height,
             grasp_offset=offset.tolist(), behavior=int(nrng.integers(0, 4)),
             geometry=selected_geometry, geometry_key=selected_geometry, domain=domain,
             protocol_version=PROTOCOL_VERSION)
    p['feasibility'] = feasibility(p)
    if not p['feasibility']['feasible']:
        raise AssertionError('Prespecified generator produced an infeasible parameter set')
    return p


parameters_v2 = parameters


class FixedBoxLiftV2(Lift):
    """Lift with an explicitly sized BoxObject inserted before compilation.

    _load_model mirrors robosuite 1.5.1 Lift._load_model, changing only the
    BoxObject size argument. Original cube names keep inherited references,
    reward, success criterion, observables and contact extraction compatible.
    """
    def __init__(self, half_size, **kwargs):
        self.fixed_half_size_v2 = tuple(float(x) for x in half_size)
        super().__init__(**kwargs)

    def _load_model(self):
        ManipulationEnv._load_model(self)
        xpos = self.robots[0].robot_model.base_xpos_offset['table'](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(table_full_size=self.table_full_size,
                           table_friction=self.table_friction, table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        material = CustomMaterial(texture='WoodRed', tex_name='redwood', mat_name='redwood_mat',
                                  tex_attrib={'type': 'cube'},
                                  mat_attrib={'texrepeat': '1 1', 'specular': '0.4', 'shininess': '0.1'})
        self.cube = BoxObject(name='cube', size=self.fixed_half_size_v2,
                              rgba=[1, 0, 0, 1], material=material)
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name='ObjectSampler', mujoco_objects=self.cube, x_range=[-.03, .03],
                y_range=[-.03, .03], rotation=None, ensure_object_boundary_in_range=False,
                ensure_valid_placement=True, reference_pos=self.table_offset, z_offset=.01)
        self.model = ManipulationTask(mujoco_arena=arena,
                                      mujoco_robots=[robot.robot_model for robot in self.robots],
                                      mujoco_objects=self.cube)


class SimAdapterV2(_original.SimAdapter):
    """Compatible adapter, compiling a fresh model when geometry changes.

    ``reset(seed)`` on the default legacy geometry delegates unchanged. Pass
    revision ``parameters(...)`` explicitly, or use ``reset(seed, domain='id')``.
    On a nonlegacy constructor geometry, reset without params samples feasible
    parameters from that geometry's native domain. ``last_provenance`` holds
    metadata separately from policy observations. Policies still use a 7D action
    with learned components at indices 2 and 6 and the original 64x64 RGB,
    tactile(6), proprioception(9) inputs. This class does not itself override the
    evaluator's common action constraints or add privileged policy inputs.
    """
    def __init__(self, render=True, geometry='legacy_cube'):
        if geometry not in GEOMETRIES:
            raise ValueError(f'Unknown geometry {geometry!r}')
        self.geometry_key = geometry
        self._compile(render, geometry)

    def _compile(self, render, geometry):
        if geometry == 'legacy_cube':
            super().__init__(render=render)
        else:
            np.random.seed(20270915)
            cfg = _original.load_composite_controller_config(controller='BASIC')
            cfg['body_parts'] = {'right': cfg['body_parts']['right']}
            cfg['body_parts']['right']['type'] = 'OSC_POSE'
            self.env = FixedBoxLiftV2(
                half_size=GEOMETRIES[geometry], robots='Panda',
                gripper_types='ContinuousPandaGripper', controller_configs=cfg,
                initialization_noise=None, hard_reset=False, has_renderer=False,
                has_offscreen_renderer=render, use_camera_obs=render,
                camera_names='agentview', camera_heights=64, camera_widths=64,
                reward_shaping=True, control_freq=20, horizon=self.horizon)
            # The following attachment mirrors the original constructor. All
            # mechanics below are inherited, including per-substep force peaks.
            self.render = render
            self.model = self.env.sim.model._model
            self.data = self.env.sim.data._data
            self.cube_geom = self.env.sim.model.geom_name2id('cube_g0')
            self.cube_vis = self.env.sim.model.geom_name2id('cube_g0_vis')
            self.cube_body = self.env.cube_body_id
            self.compiled_half_size = self.model.geom_size[self.cube_geom, :3].copy()
            self.table_geom = self.env.sim.model.geom_name2id('table_collision')
            self.finger_geoms = [set(), set()]
            for gid in range(self.model.ngeom):
                name = self.env.sim.model.geom_id2name(gid) or ''
                for f in range(2):
                    if f'finger{f+1}_' in name and 'collision' in name:
                        self.finger_geoms[f].add(gid)
            self.pad_ids = [self.env.sim.model.geom_name2id(f'gripper0_right_finger{f}_pad_collision')
                            for f in (1, 2)]
            self.grip_actuators = [i for i in range(self.model.nu)
                if 'gripper' in (self.env.sim.model.actuator_id2name(i) or '')]
            self.substep_peak = np.zeros(2)
            self.substep_count = 0
            old_step2 = self.env.sim.step2

            def measured_step2():
                old_step2()
                tactile, _, _, _ = self._contact_signals()
                self.substep_peak = np.maximum(self.substep_peak, tactile[[0, 3]])
                self.substep_count += 1

            self.env.sim.step2 = measured_step2
            self._configure_camera()
        self.geometry_key = geometry
        # Save collision AND visualization values for a post-reset integrity check.
        self._geometry_snapshot = self.model.geom_size[[self.cube_geom, self.cube_vis]].copy()
        expected = GEOMETRIES[geometry]
        if expected is not None and not np.array_equal(self.compiled_half_size, np.asarray(expected)):
            raise AssertionError('Compiled collision dimensions differ from constructor dimensions')

    def reset(self, seed=0, params=None, *, domain=None):
        if params is not None and domain is not None:
            raise ValueError('Specify params or domain, not both')
        if params is None and domain is not None:
            params = parameters(seed, domain)
        elif params is None and self.geometry_key != 'legacy_cube':
            native_domain = 'id' if self.geometry_key in ID_GEOMETRIES else 'geometry_ood'
            params = parameters(seed, native_domain, geometry=self.geometry_key)
        selected = _geometry_key(params or {}, self.geometry_key)
        if params is not None and params.get('domain') in DOMAINS:
            if selected not in geometry_support(params['domain']):
                raise ValueError('Requested geometry does not belong to the declared domain')
            if not feasibility(params)['feasible']:
                raise ValueError('Revision parameters violate the analytic feasibility screen')
        elif params is not None and selected != 'legacy_cube' and not feasibility(params)['feasible']:
            raise ValueError('Nonlegacy geometry requires analytically feasible parameters')
        if selected != self.geometry_key:
            render = self.render
            self.close()
            self._compile(render, selected)
        obs = super().reset(seed=seed, params=params)
        if not np.array_equal(self.model.geom_size[[self.cube_geom, self.cube_vis]], self._geometry_snapshot):
            raise AssertionError('Geometry changed after compilation')
        actual_feasibility = feasibility(dict(self.params, geometry=selected, geometry_key=selected),
                                         half_size=self.compiled_half_size)
        if params is not None and params.get('domain') in DOMAINS and not actual_feasibility['feasible']:
            raise AssertionError('Compiled geometry failed the analytic feasibility screen')
        self.last_provenance = dict(
            protocol_version=PROTOCOL_VERSION, geometry=selected,
            domain=self.params.get('domain', 'legacy' if selected == 'legacy_cube' else 'custom'),
            object_class='robosuite.models.objects.BoxObject',
            environment_class='Lift' if selected == 'legacy_cube' else 'FixedBoxLiftV2',
            compiled_half_size_m=self.compiled_half_size.tolist(),
            full_size_m=(2 * self.compiled_half_size).tolist(),
            geometry_construction='original seeded Lift' if selected == 'legacy_cube' else 'BoxObject(size=...) before compilation',
            geometry_modified_after_compilation=False, feasibility=actual_feasibility,
            original_adapter_sha256=hashlib.sha256(ORIGINAL_PATH.read_bytes()).hexdigest(),
            expected_original_adapter_sha256=ORIGINAL_SHA256,
            mujoco_version=_original.mujoco.__version__, robosuite_version=_original.suite.__version__,
            action_dim=self.action_dim, policy_action_indices=[2, 6],
            state_semantics='inherited unchanged from original SimAdapter',
        )
        if params is not None and 'protocol_version' in params:
            self.params['feasibility'] = actual_feasibility
        return obs


# Drop-in import spelling for existing scripts, while retaining an explicit name.
SimAdapter = SimAdapterV2
