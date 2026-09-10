# Revision-v2 executed results and reproduction appendix

This guide describes the completed revised experiment. The [original guide](README_reproduction.md) describes v1, whose 680 executions on 40 environments remain a separate historical record. Start with [INSTALL.md](INSTALL.md) and [README_REVISION_V2.md](README_REVISION_V2.md) for the portable, staged runner.

## Distinct denominators

| Component | Executed size | Independent environment unit |
| --- | --- | --- |
| New collection | 336 episodes: 240 train / 48 validation / 48 calibration | Whole episodes, disjoint from tests |
| Primary control | 34 configurations × 120 environments = **4,080 executions**; 14 groups | 120 environments, 30 each in four domains |
| Forecast accuracy | All predictors use the same 120 script trajectories; 3 model seeds | 120 environments; windows/horizons are repeated observations |
| Counterfactual action audit | 12 environments × 3 prefixes × 9 five-step branches = **324 branches** | 12 environments, not 324 independent tasks |
| Sensor-stress appendix | 11 configurations × the same 30 main ID environments = **330 additional executions** | Paired 30-environment clean/stress comparison |
| Historical v1 | **680 executions / 40 environment seeds** in two exploratory cohorts | Kept separate from revision v2 |

The forecast trajectories are already part of the main script evaluations. Sensor stress reuses ID environment parameters. Neither adds new independent environments to the main 120-case denominator. Counterfactual five-step branches are not full 150-step control trials.

## Portable execution and CPU diagnostic inference

Use user-supplied simulator and CUDA Python interpreters, with no automatic dependency installation:

```text
python outputs/revision_v2/run_experiments.py --sim-python PATH_TO_SIM_PYTHON --gpu-python PATH_TO_TRAIN_PYTHON --workers 2 --dry-run
python outputs/revision_v2/run_experiments.py --sim-python PATH_TO_SIM_PYTHON --gpu-python PATH_TO_TRAIN_PYTHON --workers 2
```

Inspect the dry-run before execution. The six world models and policy-training/control jobs use the GPU. The executed counterfactual diagnostic used **CPU inference** for its six frozen forecasting models after concurrent CUDA contexts exceeded available memory; the failure occurred before any branch outcome. The model interpreter still comes from `--gpu-python`, but the explicit device controls inference:

```text
PATH_TO_SIM_PYTHON outputs/revision_v2/counterfactual_audit.py --gpu-python PATH_TO_TRAIN_PYTHON --device cpu
```

Add `--resume` only when continuing a compatible existing diagnostic. Runtime/source/checkpoint provenance is recorded, and a changed run is rejected. A validation-only CPU/GPU comparison preserved exact simulated branches, with small floating-point forecast differences; no weights, candidate actions, or thresholds were changed. The exported runner must use `--device cpu` for this stage to match the recorded device. The [infrastructure amendment](recorded/outputs/revision_v2/counterfactual_infrastructure_amendment.json) preserves the failed initialization and repair rationale.

The training budgets are fixed: six 25-epoch compact world models; 1,000 BC plus 1,500 imagined actor-critic updates per frozen-model policy; and separate reactive 2,000-update BC / 10,000-update IQL pipelines. Training stride is 3 for the world model. The new dataset is balanced between script and force-feedback collection. Checkpoint selection uses the specified validation criterion or final fixed update; test outcomes are not used to fit or select checkpoints.

## Three focal paired control comparisons

Every table entry is **first minus second in percentage points [95% CI]**. Higher joint/strict-lift differences are favorable; lower force-violation differences are favorable. Intervals use 2,000 whole-environment bootstrap replicates with shared draws, preserving pairing and four-domain counts for pooled rows. They are pointwise, with no multiplicity adjustment, and conditional on the three fitted seeds; separate seed variation is archived.

| Comparison | Domain | Joint success Δ | Strict lift Δ | Force violations Δ |
| --- | --- | --- | --- | --- |
| Model + guard minus Reactive + guard | ID | +20.0 [6.7, 33.4] | +0.0 [0.0, 0.0] | -20.0 [-33.4, -6.7] |
| Model + guard minus Reactive + guard | Geometry OOD | +16.7 [3.3, 30.0] | +0.0 [0.0, 0.0] | -16.7 [-30.0, -3.3] |
| Model + guard minus Reactive + guard | Physics OOD | -4.4 [-18.9, 8.9] | -28.9 [-45.6, -14.4] | -1.1 [-16.7, 14.4] |
| Model + guard minus Reactive + guard | Combined OOD | -16.7 [-36.7, 3.3] | -36.7 [-53.3, -20.0] | +12.2 [-8.9, 33.3] |
| Model + guard minus Reactive + guard | All 120 | +3.9 [-4.5, 11.7] | -16.4 [-22.2, -10.8] | -6.4 [-14.4, 2.2] |
| WM-RL / vision + touch minus IQL / vision + touch | ID | -7.8 [-22.2, 5.6] | +1.1 [-8.9, 11.1] | +1.1 [-15.6, 17.8] |
| WM-RL / vision + touch minus IQL / vision + touch | Geometry OOD | -14.4 [-24.4, -4.4] | -2.2 [-16.7, 13.3] | +25.6 [12.2, 38.9] |
| WM-RL / vision + touch minus IQL / vision + touch | Physics OOD | -16.7 [-27.8, -7.8] | -11.1 [-24.4, 2.2] | +22.2 [7.8, 36.7] |
| WM-RL / vision + touch minus IQL / vision + touch | Combined OOD | -13.3 [-23.3, -4.4] | -15.6 [-30.0, 1.1] | +41.1 [27.8, 54.4] |
| WM-RL / vision + touch minus IQL / vision + touch | All 120 | -13.1 [-18.9, -7.8] | -6.9 [-13.3, -0.0] | +22.5 [15.3, 30.0] |
| WM-RL / vision + touch minus WM-BC / vision + touch | ID | +4.4 [-13.3, 20.0] | +37.8 [24.4, 52.2] | +22.2 [1.1, 42.2] |
| WM-RL / vision + touch minus WM-BC / vision + touch | Geometry OOD | +11.1 [2.2, 21.1] | +38.9 [31.1, 46.7] | +34.4 [16.7, 52.2] |
| WM-RL / vision + touch minus WM-BC / vision + touch | Physics OOD | -5.6 [-10.0, -1.1] | +24.4 [12.2, 36.7] | +45.6 [27.8, 63.3] |
| WM-RL / vision + touch minus WM-BC / vision + touch | Combined OOD | -4.4 [-12.2, 0.0] | +30.0 [15.6, 42.2] | +61.1 [42.2, 77.8] |
| WM-RL / vision + touch minus WM-BC / vision + touch | All 120 | +1.4 [-3.9, 6.4] | +32.8 [26.4, 38.9] | +40.8 [31.9, 49.7] |

The [independent recorded phase/guard audit](recorded/outputs/revision_v2/guard_details_audit.json) found 100% strict lifting and no active-phase 8 N exceedance for either controller in ID or geometry OOD; all favorable model-minus-reactive joint-success differences in those domains came from fewer exceedances during shared scripted lowering. The guard fired 23 times across 360 model-controller executions and 2 times across 120 reactive-controller executions, but all 480 paired guard-on/off comparisons retained identical strict-lift, full-episode force-violation and joint-success outcomes, so these differences do not demonstrate active-phase guard protection; this is post hoc descriptive phase accounting, not a causal mediation analysis.

These three highlighted comparisons do not replace the [complete predeclared contrast set](recorded/outputs/revision_v2/analysis/control_contrasts.csv). The model/guard factorial shares feedback and guard logic. WM-RL/IQL is a pipeline comparison. WM-RL/WM-BC shares the frozen encoder and BC initialization but includes additional imagination, critic fitting, and optimization; it does not isolate learned dynamics from all other changes.

## Forecast accuracy and persistence

MAE in N on common recorded-action script trajectories:

| Domain | Endpoint: vision | Endpoint: VT | Endpoint: persistence | Peak: vision | Peak: VT | Peak: persistence |
| --- | --- | --- | --- | --- | --- | --- |
| ID | 0.939 | 0.189 | 0.106 | 2.567 | 0.448 | 0.444 |
| Geometry OOD | 1.159 | 0.631 | 0.293 | 2.508 | 1.134 | 1.195 |
| Physics OOD | 1.230 | 0.402 | 0.133 | 2.977 | 0.789 | 0.788 |
| Combined OOD | 0.772 | 0.361 | 0.132 | 1.690 | 0.668 | 0.704 |

The forecast persistence implementation holds the observed tactile endpoint and the decoded current peak/height constant; it performs no transition rollout. Its decoded-peak persistence is distinct from the two measured-force persistence baselines below. Shuffled-touch history changes only tactile observations while preserving RGB/proprioception, actions and targets. Raw vision/VT comparisons isolate the modality input; calibration radii remain frozen from 48 mixed script/feedback calibration episodes. Empirical coverage under a different trajectory distribution is not a closed-loop guarantee. See [forecast definitions and complete intervals](recorded/outputs/revision_v2/forecast_analysis/analysis_metadata.json).

## Counterfactual executed branches

Every branch independently resets and replays its exact common prefix. Six frozen models forecast the same nine constant-action candidates at prefix steps 44, 78, and 112, then MuJoCo executes all branches. Maximum prefix replay difference is zero. The table averages model seeds within environment before averaging environments.

| Predictor | Absolute peak MAE (N) | Peak action-effect MAE (N) | Height action-effect MAE (mm) | Local selection regret | Selected violations |
| --- | --- | --- | --- | --- | --- |
| persistence | 0.371 | 0.413 | 0.864 | 0.010805 | 0.0% |
| normal_force_persistence | 0.371 | 0.413 | 0.864 | 0.010805 | 0.0% |
| wm_vision_raw | 1.628 | 1.408 | 1.451 | 0.000005 | 0.0% |
| wm_visuotactile_raw | 0.371 | 0.338 | 0.396 | 0.000007 | 0.0% |
| wm_visuotactile_anchored | 0.364 | 0.336 | 0.396 | 0.000007 | 0.0% |

Here `persistence` holds the previous measured interval peak constant; `normal_force_persistence` holds the current endpoint normal force constant. Both hold observed height/touch constant. Action effects are candidate-minus-center differences across all five horizons. Anchoring adds a common measured-force correction to predictions; it is not new learned dynamics. Anchored vision also uses measured touch through this correction and is therefore not a pure vision-only forecast.

Raw VT minus current-normal persistence peak action-effect MAE is **-0.074 [-0.087, -0.062] N**. Local regret uses clipped terminal height with utility zero for a branch force violation. It can be uninformative when candidate utilities are all zero or nearly tied. Constant-predictor rank correlations are undefined and reported with a defined fraction, rather than treated as evidence of accurate ranking. This local five-step audit does not establish full-episode lift or safety improvements. [All paired diagnostics](recorded/outputs/revision_v2/counterfactual/summary.json) are retained.

## Synthetic sensor-error appendix

Clean/stress comparisons use identical fixed policies and environment conditions. All metrics use true physics, not the perturbed observation signal. Differences below are stress minus clean, in percentage points [95% CI].

| Group | Clean joint success | Stress joint success | Joint-success Δ | Force-violation Δ |
| --- | --- | --- | --- | --- |
| WM-RL / vision + touch | 27.8% | 26.7% | -1.1 [-3.3, 0.0] | +1.1 [0.0, 3.3] |
| IQL / vision + touch | 35.6% | 36.7% | +1.1 [-2.2, 5.6] | -1.1 [-3.3, 0.0] |
| Model + guard | 93.3% | 93.3% | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] |
| Reactive + guard | 73.3% | 73.3% | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] |
| Force feedback | 73.3% | 73.3% | +0.0 [0.0, 0.0] | +0.0 [0.0, 0.0] |

Residual blocks come from held-out public GelSight force-regression errors. Their source normal-force labels span roughly 0.073–1.888 N, below the 8 N simulation criterion. Two fingers use independently drawn blocks, source samples are mapped to 20 Hz indices, and additive errors are applied even at zero simulated contact. These are explicit synthetic assumptions, not an identified sensor model. The result is an observation-sensitivity check, not optical tactile deployment, physical-robot control, hardware safety, or validated sim-to-real transfer. The embedded data retain **CC BY-NC 4.0** terms: see [SENSOR_STRESS_DATA_LICENSE.md](SENSOR_STRESS_DATA_LICENSE.md).

## Files, manuscript, and video

The new public manuscript is [`paper/revision_v2/preprint.pdf`](paper/revision_v2/preprint.pdf). The anonymous-review simulation video is [`paper/revision_v2/simulation_video.mp4`](paper/revision_v2/simulation_video.mp4). The anonymous review PDF is kept out of the public repository. A public arXiv identifier remains pending; an internal submission identifier is not an article URL.

The compact archive distributes executable scientific source and text/numeric summaries, **not raw tactile images, simulator trajectories, prediction arrays, or trained weights**. `SOURCE_MANIFEST.json` / `REVISION_V2_FILES_SHA256.json` distinguish executed source and packaged-file provenance. `python verify_package.py` checks the preserved v1 package and `python verify_revision_v2.py` checks the revision. Verifiers validate records and hashes, not independently regenerated experimental outcomes. Recreate the prescribed complete cohorts when testing reproducibility; cross-platform bitwise equivalence is not promised.
