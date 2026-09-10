# Compact Visuotactile World Models for Lifting

**[Project website](https://compact-visuotactile-world-models.thumbnodirosving.chatgpt.site/) · [Revised paper PDF](paper/revision_v2/preprint.pdf) · [Simulation video](paper/revision_v2/simulation_video.mp4) · [Reproduction guide](README_REPRODUCTION_REVISION_V2_APPENDIX.md)**

**Revision v2 contains 4,080 actual MuJoCo control executions, 324 separately executed five-step counterfactual branches, and 330 additional sensor-stress executions.** The main matrix covers **14 method groups / 34 fixed policy configurations on 120 shared environments**. The world model has **652,157 parameters**, starts from **random weights**, and uses RGB, tactile force observations, and proprioception; vision-only variants omit touch. No pretrained foundation model is downloaded.

Code and recorded results for **Compact Visuotactile World Models for Lifting: Prediction, Reward Alignment, and Force Constraints**.

**Qinzhen Ma** (Rice University) and **Sida Peng** (Zhejiang University). Correspondence: [qm18@rice.edu](mailto:qm18@rice.edu).

**arXiv status:** Submitted; a public identifier is still pending. The PDF above is the revised local manuscript. No public arXiv URL is inferred from an internal submission number, and no conference acceptance is claimed.

## What revision v2 tests

The revised study separates prediction accuracy, action-sensitive forecasts, imagined policy learning, and measured-force feedback. It uses a new 336-episode collection (240 training / 48 validation / 48 calibration), three fitted seeds, three training box geometries, two held-out box geometries, and separate geometry, physics, and combined shifts. All control policies share the same environment seeds, common scripted approach, and vertical/gripper action space.

The primary outcome is **joint success**: height at least 10 cm for ten consecutive observations before lowering **and** normal force at most 8 N per finger over every physics substep of the full episode, including approach and lowering. Failed grasps stay in the denominator.

## Complete main control results

Joint-success rates, with the same 30 environments in each domain and all 120 pooled. Learned groups average their three fixed policies within each environment; fixed baselines have one policy.

| Method group | ID | Geometry OOD | Physics OOD | Combined OOD | All 120 |
| --- | --- | --- | --- | --- | --- |
| WM-RL / vision | 20.0% | 23.3% | 1.1% | 1.1% | 11.4% |
| WM-RL / vision + touch | 27.8% | 18.9% | 1.1% | 0.0% | 11.9% |
| WM-BC / vision | 27.8% | 5.6% | 14.4% | 4.4% | 13.1% |
| WM-BC / vision + touch | 23.3% | 7.8% | 6.7% | 4.4% | 10.6% |
| BC / vision | 15.6% | 5.6% | 2.2% | 1.1% | 6.1% |
| BC / vision + touch | 15.6% | 11.1% | 6.7% | 4.4% | 9.4% |
| IQL / vision | 27.8% | 15.6% | 15.6% | 5.6% | 16.1% |
| IQL / vision + touch | 35.6% | 33.3% | 17.8% | 13.3% | 25.0% |
| Model + guard | 93.3% | 73.3% | 55.6% | 53.3% | 68.9% |
| Model / no guard | 93.3% | 73.3% | 55.6% | 53.3% | 68.9% |
| Reactive + guard | 73.3% | 56.7% | 60.0% | 70.0% | 65.0% |
| Reactive / no guard | 73.3% | 56.7% | 60.0% | 70.0% | 65.0% |
| Script | 16.7% | 23.3% | 6.7% | 3.3% | 12.5% |
| Force feedback | 73.3% | 56.7% | 60.0% | 56.7% | 61.7% |

Three focal paired contrasts on all 120 environments are shown below. Values are **first minus second**, in percentage points, with 95% paired environment-bootstrap intervals. The [complete analysis](recorded/outputs/revision_v2/analysis/control_analysis.json) retains every prespecified contrast, strict-lift rates, force violations, domain results, and training-seed variation.

| Joint-success comparison | Difference [95% CI] |
| --- | --- |
| Model + guard minus Reactive + guard | +3.9 [-4.5, 11.7] pp |
| WM-RL / vision + touch minus IQL / vision + touch | -13.1 [-18.9, -7.8] pp |
| WM-RL / vision + touch minus WM-BC / vision + touch | +1.4 [-3.9, 6.4] pp |

In ID and geometry OOD, both controllers achieved 100% strict lifting and neither exceeded 8 N during active control; their favorable joint-success differences were entirely accounted for by fewer violations during the later shared scripted lowering, not demonstrated active-phase guard protection. The guard fired 23 times across 360 model-controller executions and 2 times across 120 reactive-controller executions, yet enabling it changed none of strict lift, full-episode force violation, or joint success in any of the 480 matched guard-on/off pairs ([post hoc phase audit](recorded/outputs/revision_v2/guard_details_audit.json)).

The model-versus-reactive guard comparison shares the measured-force guard and reactive base, and tests the added short-horizon model-assisted action selection. It does not establish that the guard alone is a safety guarantee. WM-RL versus IQL compares complete pipelines with different representations, objectives, and optimization budgets. WM-RL versus its own WM-BC checkpoint uses the same frozen representation and BC start; it measures subsequent imagined policy improvement plus its additional optimization, not an isolated proof that learned dynamics are necessary.

## Prediction and diagnostic evidence

On the common ID script trajectories, endpoint-force MAE is 0.939 N for vision and 0.189 N for the visuotactile model; endpoint persistence gives 0.106 N. Forecasts use the recorded future actions, so this is an accuracy diagnostic, not an online control-return result. [All four domains, persistence, shuffled touch, and calibration](recorded/outputs/revision_v2/forecast_analysis/forecast_analysis.json) remain available.

The separate 12-environment counterfactual audit executes 3 prefixes × 9 fixed action candidates × 5 steps. Raw visuotactile force action-effect MAE is 0.338 N; the paired difference from current-normal-force persistence is -0.074 [-0.087, -0.062] N. This checks local action dependence. It is distinct from sustained lifting and full-episode force compliance. The [complete branch audit](recorded/outputs/revision_v2/counterfactual/summary.json) includes both persistence definitions, vision/visuotactile predictions, and local selection regret, including weak or uninformative comparisons.

The **330 sensor-stress executions** reuse the 30 main ID environments for 11 prespecified policies. They add held-out public tactile-CNN residual blocks to simulated normal-force observations and measure outcomes using unperturbed physics. This is a synthetic observation-sensitivity appendix, not validated optical-sensor transfer or sim-to-real evaluation. [All paired clean/stress results](recorded/outputs/revision_v2/sensor_stress_analysis/analysis.json) are retained; [dataset terms and limitations](SENSOR_STRESS_DATA_LICENSE.md) apply to the residual blocks.

## Reproduce and inspect

```text
python verify_package.py
python verify_revision_v2.py
```

These standard-library checks validate the archived text records, counts, pairing, and file provenance. They do not rerun a simulator or establish the numerical correctness of unavailable trajectories. **Raw images, trajectory/prediction arrays, and trained checkpoints are not included in the compact release.** The [revision-v2 reproduction guide](README_REPRODUCTION_REVISION_V2_APPENDIX.md), [pipeline guide](README_REVISION_V2.md), and [installation instructions](INSTALL.md) describe regeneration. Keep `recorded/` unchanged; fresh runs write to `outputs/` and `work/`.

| Path | Contents |
| --- | --- |
| [`outputs/revision_v2/`](outputs/revision_v2/) | Revised protocols, simulator extension, reactive BC/IQL, model-assisted controller, counterfactual/forecast/stress analysis |
| [`outputs/world_model/`](outputs/world_model/) | Shared compact world-model and imagined actor-critic implementation |
| [`recorded/outputs/revision_v2/`](recorded/outputs/revision_v2/) | Complete revised manifests, per-episode records, training logs, aggregate statistics, and paired intervals |
| [`paper/revision_v2/`](paper/revision_v2/) | Revised public manuscript and anonymous-review simulation video |
| [`outputs/research/`](outputs/research/) | Separate public GelSight force-estimation and calibration study |

The video is the **anonymous review version of a simulation supplement**. Its 20 replays were selected before reading their outcomes, every selected failure is retained, and replaying the recorded actions exactly reproduced all five checked observation arrays. Full simulator `qpos` was not archived, so full-state equivalence is not claimed. The anonymous review PDF is not included in the public repository.

## Scope and preserved history

This is a rigid-box simulation study after a common privileged scripted approach; only vertical motion and gripper rate are learned or selected during the intervention. Five box aspect/size families do not establish arbitrary-object generalization. The force-capacity screen is not a guarantee of dynamic grasp feasibility. Confidence intervals resample whole environments and are conditional on the fitted policies; they do not treat model seeds, windows, fingers, or branches as independent environments. No physical-robot control, validated sim-to-real transfer, or closed-loop safety guarantee is reported.

The earlier **v1 exploratory study remains separate: 680 control executions across 40 distinct environment seeds**, from two cohorts. Its [paper](paper/preprint.pdf), [first-cohort analysis](recorded/outputs/world_model/analysis/), [reward follow-up](recorded/outputs/world_model/analysis_goal_aligned/), and [original reproduction guide](README_reproduction.md) are retained. These numbers are **not pooled into any revision-v2 denominator**. The separate public GelSight sensing records are likewise not new physical-robot control trials.

## Sources and licensing

See [LICENSE](LICENSE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and the adapted [TD-MPC2 MIT notice](outputs/world_model/LICENSE_TDMPC2.txt). The compact implementation adapts selected TD-MPC2 building blocks; it is not a reproduction of the full official TD-MPC2 algorithm. Public GelSight data and derived embedded residual blocks retain **CC BY-NC 4.0** terms, separate from the software and manuscript licenses. [Publication metadata](publication.json) records the public links.
