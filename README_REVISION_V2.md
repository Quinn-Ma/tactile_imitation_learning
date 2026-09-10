# Revision-v2 reproduction

This extension preserves the predecessor package and its `FILES_SHA256.json`.
New code is in `outputs/revision_v2`; sanitized archived text and numerical
records, when complete, are in `recorded/outputs/revision_v2`.
`REVISION_V2_FILES_SHA256.json` states whether this is a code-only review package.
A code-only package contains no revised experimental result archive.

## Verify without simulation or training

```sh
python verify_package.py
python verify_revision_v2.py
```

The second verifier checks source and archive hashes, the 336-episode collection,
all 34 policies on the same 120 test environments (4,080 executions), 1,440
forecast metric rows, the 12-environment / 324-branch diagnostic, and all 11
sensor-stress policies on the same 30 main ID environments (330 executions). In a
code-only package it checks source and the frozen protocol, and explicitly
reports that completed result records are absent. No raw trajectories or model
weights are distributed. Text verification cannot independently establish that
the missing trajectories produce the archived numbers.

## Recreate the revised experiment

Follow `INSTALL.md` for separate simulation and GPU environments. Supply their
Python executables yourself; this runner installs nothing and includes no
machine-specific interpreter paths. Use a CUDA-capable PyTorch environment.
Simulation requires the documented MuJoCo/robosuite stack and rendering support.

```sh
python outputs/revision_v2/run_experiments.py --sim-python /path/to/sim/python --gpu-python /path/to/gpu/python --workers 2 --dry-run
python outputs/revision_v2/run_experiments.py --sim-python /path/to/sim/python --gpu-python /path/to/gpu/python --workers 2
```

Replace the placeholder executables with paths for your machine. On Windows,
quote executable paths containing spaces. The launcher itself uses only the
standard library. It resolves executables once, invokes argument lists without
a shell, and writes logs under `work/revision_v2_runner_logs`.

Training, main control evaluation, and common-script forecasts use GPU model
execution. The counterfactual diagnostic explicitly uses **CPU** inference
through the same supplied model Python executable. This matches the recorded
infrastructure amendment after the sixth concurrent CUDA forecasting bridge
failed during initialization, before any diagnostic branch was collected.
The full archive preserves that amendment, the failed initialization manifest
and complete logs with machine paths sanitized, and the successful runtime
provenance. No model weight, action grid, environment, or evaluation threshold
was changed by that amendment.

Recorded simulation runtimes both used Python 3.12.14, MuJoCo 3.3.7 and
robosuite 1.5.1. Main simulation resolved NumPy 2.5.3; the counterfactual entry
point imported NumPy 2.3.5 before the adapter's optional dependency path was
added. This existing import-order difference is preserved in the source and
recorded in `runtime_main.json`, `runtime_branch.json`, and the counterfactual
runtime manifest. The separate training/model environment used Torch
2.14+cu130 and NumPy 2.5.2. A fresh environment can resolve a different NumPy
version; inspect and report the actual runtime rather than assuming one version
for all stages. Cross-platform or cross-version bitwise equality is not promised.

The dependency order is protocol, collection and audit, six world models, six
imagined-actor runs, six independent reactive runs, the complete control matrix,
counterfactual diagnostic, common-script forecasting, control analysis, the
sensor-stress diagnostic, and its paired clean/stress analysis.
`--workers` controls collection and control-evaluation concurrency; reactive
training runs at most two GPU jobs, and other training runs sequentially.
`--from-stage` / `--through-stage` expose the same fixed stages for recovery.
Existing complete training from this runner is retained only when its stored
command/source/data receipt still matches; partial training is not overwritten.
Use a fresh checkout for a full retraining attempt. Evaluation and diagnostic
resume only when their own source/input checks permit it.

Fixed budgets are 240 training, 48 validation, and 48 calibration episodes,
with equal script/feedback collection. The three training geometries and two
held-out geometries are compiled before stepping. Model seeds are 0, 1, and 2.
World models use 25 epochs, batch 64, history 3, forecast horizon 5, training
stride 3, and validation stride 5. The frozen-model actor has 1,000 BC and 1,500
imagined RL updates. Independent reactive actors use 2,000 BC and 10,000 IQL
updates at batch 256; exact optimizer settings are explicit in the runner.
BC chooses minimum validation action MSE, IQL uses its final fixed update, and
the world model chooses validation loss. Test outcomes select none of these.

The fresh control matrix has 30 environments each in ID, geometry shift,
mass/friction shift, and their combination. Methods share seeds, parameters,
the scripted approach, and the intervention interval [44, 135), with only
vertical motion and gripper action controlled. The primary metric requires
height at least 10 cm for ten consecutive observations before lowering and
full-episode per-finger peaks at most 8 N. All approaches and failures count.

Forecasting uses each method's identical script-generated test trajectories,
actual subsequent actions, and fixed 48-episode calibration radii. The separate
counterfactual diagnostic replays 12 environments, three prefixes, and nine
candidate actions for five steps. Neither analysis fits or adapts models.
Confidence intervals resample whole environments, retaining policy pairing;
they are conditional on the three trained models. They do not treat windows,
actions, or model seeds as independent environments.

The runner analyzes the revised matrix with `--skip-legacy`. A separately
archived exploratory legacy vision ablation, if included, uses the earlier
cohort and is not pooled with this revision. Recreating it additionally needs
the predecessor experiment and its independently specified legacy commands.

## Interpretation and provenance

These are rigid-box simulation experiments after a common scripted approach.
Analytic force-capacity screening does not guarantee successful grasping or
physical safety. Geometry generalization covers five box aspect/size families,
not arbitrary object categories. Different training objectives and optimization
budgets mean the reactive/world-model comparison is a pipeline comparison.
Paired frozen-world-model BC controls isolate subsequent policy improvement
plus its extra optimization, not the value of dynamics alone. Measured-force
guards and feedback use tactile measurements even with visual predictors.
Timing includes process communication and concurrent workloads.

See `PACKAGING_CHANGES_REVISION_V2.md` for the audited adapter wrapper and
metadata-only provenance adjustments. Executed-source and original-record
hashes remain archived; exported file hashes are separately indexed. Runtime
regeneration creates new provenance and does not rewrite the archived records.
Bitwise equality across hardware, Python, CUDA, MuJoCo, and robosuite versions
is not promised. Report your environment and compare the complete prescribed
cohorts rather than selecting favorable runs.

## Appendix sensor-error diagnostic and data terms

`sensor_stress_protocol.json` embeds the 60 fixed finger residual blocks used
by the 30 ID cases. The packaged runner checks the protocol and each block's
float32 digest; it does not require the original prediction NPZ. If that optional
source archive is present, its original hash is still checked. The separate
`sensor-stress` and `sensor-analyze` stages rerun all 11 prespecified policies and
compare them with their matched main clean ID records, using true physics for
all outcome and force-budget metrics.

These held-out CNN normal-force residuals derive from the TacBench / Sparsh
GelSight force dataset. **The embedded residual data retain CC BY-NC 4.0 terms;
they are not licensed under the repository's software MIT license.** See
`SENSOR_STRESS_DATA_LICENSE.md` for source, attribution, and transformations.
Original raw prediction arrays are excluded. Source labels cover roughly
0.073–1.888 N, below the simulator's 8 N threshold. Independent finger draws,
the imposed 20 Hz index mapping, additive transport, and errors added at zero
contact are synthetic assumptions. This appendix probes observation sensitivity;
it is not a validated sensor model, physical-safety result, or sim-to-real test.
