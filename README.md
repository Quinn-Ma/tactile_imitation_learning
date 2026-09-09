# Compact Visuotactile World Models for Lifting

Code and recorded results for **Compact Visuotactile World Models for Lifting: Prediction, Reward Alignment, and Force Constraints**.

**Qinzhen Ma** (Rice University) and **Sida Peng** (Zhejiang University). Correspondence: [qm18@rice.edu](mailto:qm18@rice.edu).

This study examines how a compact action-conditioned world model combines vision, touch, and proprioception to predict lifting dynamics, and how those predictions support reinforcement learning in imagined trajectories. It measures forecast accuracy, task completion, and contact-force constraints separately. A public GelSight force-estimation study provides complementary evidence about tactile prediction and uncertainty; the control experiments use MuJoCo simulation.

- [Read the manuscript](paper/preprint.pdf).
- [Set up the environments](INSTALL.md) and follow the [complete reproduction instructions](README_reproduction.md).
- Browse the [scientific source code](outputs/) and [recorded numerical results](recorded/outputs/).
- Find [LaTeX source and submission materials](paper/README.md) in `paper/`.

## Start with the recorded results

From the repository root, run:

```text
python verify_package.py
```

This check uses only Python's standard library. It checks scientific-file integrity and Python syntax, the 160-episode model-data split, and both control-evaluation cohorts: 34 controller runs and 680 executions across 40 distinct environment seeds. It does not train a model or run a simulator.

The repository contains manifests, training logs, per-seed metrics, per-episode control records, and computed confidence intervals. **Raw tactile images, simulator trajectory arrays, prediction arrays, and trained checkpoints are not included.** The reproduction instructions describe how to download the public data and regenerate arrays and checkpoints. Per-episode control statistics can be inspected independently using the included records; recomputing forecast bootstrap intervals requires regenerated prediction arrays.

## Research components

| Component | Implementation |
|---|---|
| Compact world model | 652,157 parameters; three observations of RGB, tactile force, and proprioception; five action-conditioned future steps with latent, image, force, peak-force, height, contact, support, and reward predictions |
| Modality comparison | Matched vision-only and visuotactile models, three training seeds, and persistence and shuffled-touch diagnostics |
| Imagined reinforcement learning | Behavior-cloned actor initialization followed by actor-critic updates through a frozen world model; native-reward and height-aligned reward variants |
| Force constraints | Per-finger normal-force peaks recorded at every MuJoCo physics substep; fixed trajectory-calibration margins evaluated as additional force penalties |
| Simulation | Modified robosuite Panda Lift with continuous gripper-rate control; a common approach followed by vertical-motion and gripper-rate control |
| Public tactile sensing | Fixed-revision GelSight/ATI force records, trajectory-level data splits, force regression, uncertainty calibration, and shape-shift evaluation |

The world model starts from random weights and adapts selected TD-MPC2 building blocks. The implementation is a compact world-model and imagined actor-critic study, rather than a reproduction of the full official TD-MPC2 algorithm.

## Findings and evaluation scope

Touch improves forecast accuracy relative to vision alone, while persistence remains stronger on several force metrics. Improved prediction does not by itself produce better constrained control. After observing failures with the native reward, an exploratory follow-up changes the task reward and re-evaluates both reward variants on the same new environments.

On that follow-up's in-distribution conditions, height-aligned RL achieves 93.3% strict lift success and 33.3% success within the force budget, averaged over three training seeds on ten shared environments. Force feedback achieves 70.0% within-budget success on those environments. Calibration-margin penalties reduce force violations but can prevent task completion. The [first-cohort results](recorded/outputs/world_model/analysis/) and [matched reward follow-up](recorded/outputs/world_model/analysis_goal_aligned/) retain both task and force outcomes, including unsuccessful runs.

Strict lifting requires relative object height of at least 0.10 m at ten consecutive 20 Hz observations. Joint force-budget success additionally requires every per-finger normal-force peak in the complete episode, including setup and lowering, to remain at or below 8 N. The out-of-distribution mass/friction group stresses compatibility between task completion and this force budget; its zero strict-success results are not alone evidence of a general prediction failure. See the manuscript for the conditional friction-only feasibility analysis and its geometric limitations.

The public tactile recordings form a separate sensing experiment. Simulator contact-force aggregates are the control model's tactile inputs; the public-data CNN is not transferred into that policy. No physical-robot control evaluation or sim-to-real transfer is reported. Calibration results are empirical and do not establish a closed-loop safety guarantee.

## Code and result map

| Path | Contents |
|---|---|
| [`outputs/research/`](outputs/research/) | Public-data downloader, data conversion, force-model training, and calibration analysis |
| [`outputs/world_model/`](outputs/world_model/) | World model, training, imagined RL, controller process, evaluation matrices, and result analysis |
| [`outputs/simulation/`](outputs/simulation/) | Simulator adapter, dataset generation, control evaluation, and pairing audits |
| [`recorded/outputs/research/`](recorded/outputs/research/) | Public tactile-study metrics, splits, and provenance |
| [`recorded/outputs/world_model/`](recorded/outputs/world_model/) | Model and actor logs, calibration results, aggregate statistics, and executed-action forecast diagnostics |
| [`recorded/outputs/simulation/`](recorded/outputs/simulation/) | Dataset manifest and both cohorts' per-episode outcomes and protocol audits |
| [`paper/`](paper/) | Manuscript PDF, LaTeX sources, figures, and prepared arXiv source archive |

Keep `recorded/` unchanged when running new experiments. New runs write to `outputs/` and `work/`. Follow [README_reproduction.md](README_reproduction.md) in order: public sensing, simulator data, world-model training, first control cohort, matched reward follow-up, and analysis. The reported world-model configuration includes `--train-stride 3`; omitting it changes the training configuration.

The [installation guide](INSTALL.md) separates simulation, data analysis, and CUDA training. [ENVIRONMENT.json](ENVIRONMENT.json) records the executed versions; a new installation and cross-platform bitwise equivalence are not established by the recorded-result integrity check.

## Sources and licensing

See [LICENSE](LICENSE) for software terms and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution. The adapted TD-MPC2 layers retain their [MIT notice](outputs/world_model/LICENSE_TDMPC2.txt). Public GelSight data are downloaded from their original host under CC BY-NC 4.0; the software license does not replace those dataset terms. The manuscript's publication license is separate.

The repository includes a prepared preprint and submission materials. No arXiv identifier or conference acceptance is claimed.
