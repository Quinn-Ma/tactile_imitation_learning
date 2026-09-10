# Independent review of completed revision results

Only analyses whose completion markers exist are inspected. No evaluation code, policy, threshold, checkpoint, or protocol is changed.

| Analysis | Audit status |
| --- | --- |
| control | PASS |
| counterfactual | PASS |
| sensor_stress | PASS |

## Complete main control matrix

PASS: all 4080 episode records cover the prescribed 120 environments and 34 policies without an intersection or success filter. Method identity, exact paired environment parameters, model/source hashes, true-force peak partitions and the joint-success definition were checked. Independently recomputed 840 group/contrast point estimates and intervals, plus per-policy means and training-seed SDs, using Python MT19937 environment draws and independent NumPy aggregation.

The primary endpoint is **joint_success**: height at least 10 cm for ten consecutive observations before lowering, and per-finger peak at most 8 N over the complete episode. Strict lift and force violation are separate components. Intervals below are paired environment bootstrap intervals, conditional on the trained policies. Pooled rows preserve the 30:30:30:30 domain mix; 4,080 executions are not 4,080 independent environments.

| Group | Pooled joint success %, 95% CI | Strict lift % | Force violation % | Joint source index |
| --- | --- | ---: | ---: | --- |
| WM_vision | 11.388889 [7.500000, 15.833333] | 64.722222 | 75.000000 | `datasets.main.group_stats[9]` |
| WM_BC_vision | 13.055556 [9.166667, 17.222222] | 31.944444 | 22.500000 | `datasets.main.group_stats[44]` |
| BC_vision | 6.111111 [3.333333, 9.444444] | 22.500000 | 47.777778 | `datasets.main.group_stats[79]` |
| IQL_vision | 16.111111 [11.666667, 21.111111] | 45.555556 | 50.000000 | `datasets.main.group_stats[114]` |
| WM_visuotactile | 11.944444 [7.777778, 16.388889] | 62.222222 | 75.833333 | `datasets.main.group_stats[149]` |
| WM_BC_visuotactile | 10.555556 [7.222222, 14.444444] | 29.444444 | 35.000000 | `datasets.main.group_stats[184]` |
| BC_visuotactile | 9.444444 [6.388889, 12.500000] | 33.888889 | 35.277778 | `datasets.main.group_stats[219]` |
| IQL_visuotactile | 25.000000 [19.166667, 30.833333] | 69.166667 | 53.333333 | `datasets.main.group_stats[254]` |
| script | 12.500000 [6.666667, 18.333333] | 73.333333 | 60.833333 | `datasets.main.group_stats[289]` |
| force_feedback | 61.666667 [53.333333, 70.000000] | 95.000000 | 37.500000 | `datasets.main.group_stats[324]` |
| model_guard | 68.888889 [60.555556, 76.388889] | 81.111111 | 27.777778 | `datasets.main.group_stats[359]` |
| model_no_guard | 68.888889 [60.555556, 76.388889] | 81.111111 | 27.777778 | `datasets.main.group_stats[394]` |
| reactive_guard | 65.000000 [55.833333, 73.333333] | 97.500000 | 34.166667 | `datasets.main.group_stats[429]` |
| reactive_no_guard | 65.000000 [55.833333, 73.333333] | 97.500000 | 34.166667 | `datasets.main.group_stats[464]` |

Values and intervals in this table are the `.estimate`, `.ci_low`, and `.ci_high` fields of `analysis/control_analysis.json`; multiply rate fields by 100 for percentages.

| Prespecified comparison (first minus second) | Pooled joint change pp, paired 95% CI | Strict lift change pp, paired 95% CI | Force violation change pp, paired 95% CI | Joint source index |
| --- | --- | --- | --- | --- |
| WM_visuotactile - WM_vision | 0.555556 [-2.500000, 3.611111] | -2.500000 [-9.722222, 4.729167] | 0.833333 [-3.888889, 5.833333] | `datasets.main.contrasts[29]` |
| IQL_visuotactile - IQL_vision | 8.888889 [4.722222, 13.333333] | 23.611111 [16.944444, 30.000000] | 3.333333 [-1.666667, 8.333333] | `datasets.main.contrasts[64]` |
| WM_visuotactile - IQL_visuotactile | -13.055556 [-18.888889, -7.777778] | -6.944444 [-13.333333, -0.000000] | 22.500000 [15.277778, 30.000000] | `datasets.main.contrasts[99]` |
| model_guard - reactive_guard | 3.888889 [-4.451389, 11.666667] | -16.388889 [-22.229167, -10.833333] | -6.388889 [-14.444444, 2.222222] | `datasets.main.contrasts[134]` |
| model_guard - model_no_guard | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | `datasets.main.contrasts[169]` |
| reactive_guard - reactive_no_guard | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | `datasets.main.contrasts[204]` |
| model_guard - force_feedback | 7.222222 [-0.277778, 15.000000] | -13.888889 [-19.722222, -8.611111] | -9.722222 [-17.500000, -1.666667] | `datasets.main.contrasts[239]` |
| WM_vision - WM_BC_vision | -1.666667 [-6.388889, 3.333333] | 32.777778 [25.000000, 40.555556] | 52.500000 [45.555556, 59.451389] | `datasets.main.contrasts[274]` |
| WM_visuotactile - WM_BC_visuotactile | 1.388889 [-3.888889, 6.388889] | 32.777778 [26.388889, 38.888889] | 40.833333 [31.944444, 49.729167] | `datasets.main.contrasts[309]` |
| WM_BC_visuotactile - BC_visuotactile | 1.111111 [-3.055556, 5.555556] | -4.444444 [-10.277778, 1.944444] | -0.277778 [-6.951389, 6.944444] | `datasets.main.contrasts[344]` |

Difference keys are `.estimate_difference`, `.ci_low`, and `.ci_high`; percentage-point copies are also explicitly stored. Negative force-violation changes favor the first group, whereas positive lift/joint changes favor it.

| Comparison | ID joint pp, paired 95% CI | Geometry OOD joint pp, paired 95% CI | Physics OOD joint pp, paired 95% CI | Combined OOD joint pp, paired 95% CI |
| --- | --- | --- | --- | --- |
| WM_visuotactile - WM_vision | 7.777778 [1.111111, 14.444444] | -4.444444 [-13.333333, 4.444444] | 0.000000 [-3.333333, 3.333333] | -1.111111 [-3.333333, 0.000000] |
| IQL_visuotactile - IQL_vision | 7.777778 [-2.222222, 17.805556] | 17.777778 [10.000000, 25.583333] | 2.222222 [-4.444444, 10.000000] | 7.777778 [1.111111, 15.555556] |
| WM_visuotactile - IQL_visuotactile | -7.777778 [-22.222222, 5.555556] | -14.444444 [-24.444444, -4.444444] | -16.666667 [-27.777778, -7.777778] | -13.333333 [-23.333333, -4.444444] |
| model_guard - reactive_guard | 20.000000 [6.666667, 33.416667] | 16.666667 [3.333333, 30.000000] | -4.444444 [-18.888889, 8.888889] | -16.666667 [-36.666667, 3.333333] |
| model_guard - model_no_guard | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| reactive_guard - reactive_no_guard | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| model_guard - force_feedback | 20.000000 [6.666667, 33.416667] | 16.666667 [3.333333, 30.000000] | -4.444444 [-18.888889, 8.888889] | -3.333333 [-23.333333, 16.666667] |
| WM_vision - WM_BC_vision | -7.777778 [-20.000000, 5.555556] | 17.777778 [6.666667, 30.000000] | -13.333333 [-23.333333, -5.555556] | -3.333333 [-7.777778, 0.000000] |
| WM_visuotactile - WM_BC_visuotactile | 4.444444 [-13.333333, 20.000000] | 11.111111 [2.222222, 21.111111] | -5.555556 [-10.000000, -1.111111] | -4.444444 [-12.222222, 0.000000] |
| WM_BC_visuotactile - BC_visuotactile | 7.777778 [-4.444444, 22.222222] | -3.333333 [-10.000000, 3.333333] | 0.000000 [-7.777778, 6.666667] | 0.000000 [-7.777778, 8.888889] |

**Evidence that should drive the main text:** report the paired VT-versus-vision result, WM-versus-IQL result, model-versus-matched-reactive-guard result, guard-removal contrasts, and frozen-WM BC contrasts together. A lower force-violation rate does not imply better manipulation if lift/joint success deteriorates. A model-assisted advantage over one baseline is not evidence that it beats the 5 N feedback baseline unless that paired contrast supports it.

WM-versus-WM-BC compares subsequent policy improvement plus its additional update budget on the same representation. WM-BC versus independently trained reactive BC changes both representation/pipeline and training budget. These contrasts do not isolate a novel dynamics algorithm. Control after the common privileged scripted approach remains a two-action lifting task. Intervals are pointwise without multiplicity correction, and zero-width intervals from constant observed outcomes do not imply zero population uncertainty.

## Counterfactual branch diagnostic

Independently reconstructed 476 summary metrics and their 95% whole-environment bootstrap intervals from all 12 saved environment arrays. All 324 branches and frozen checkpoint/input hashes were checked. Saved prefix replay difference is zero. Fixed action clipping leaves 288 distinct candidate sequences across the 324 executed branches (6,9,9 at the three prefixes in every environment); duplicates remain included by protocol.

Runtime provenance records **cpu** model inference. The installed CUDA build version is not evidence that this diagnostic ran on a GPU. No inference-time comparison with the main GPU jobs is made.

The main-text baseline is **current endpoint-normal persistence** (`normal_force_persistence`): current tactile vector and height held constant, peak forecast set to the current endpoint normal measurement. Its current object height is privileged simulator information, not a learned-policy observation; this is an offline diagnostic reference. The alternative `persistence` holds the previous measured interval peak. Neither matches the common-script forecast analysis's learned decoded-peak/height persistence baseline. Keep these labels distinct.

| Predictor | Interval-peak MAE (N), 95% CI | Height action-effect MAE (m), 95% CI | Peak action-effect MAE (N), 95% CI | Selection regret, 95% CI |
| --- | --- | --- | --- | --- |
| normal_force_persistence | 0.371094917 [0.339675652, 0.403736852] | 0.000864265453 [0.000848628923, 0.000874276988] | 0.412639136 [0.380405491, 0.444788467] | 0.0108046064 [0.0106751449, 0.0109018709] |
| persistence | 0.371158826 [0.339701304, 0.403783068] | 0.000864265453 [0.000848628923, 0.000874276988] | 0.412639136 [0.380405491, 0.444788467] | 0.0108046064 [0.0106751449, 0.0109018709] |
| wm_vision_raw | 1.62841113 [1.35798521, 1.88552264] | 0.00145128858 [0.00128073429, 0.0016244496] | 1.40791059 [1.13706026, 1.67669583] | 4.560152e-06 [1.28790572e-06, 9.63229686e-06] |
| wm_visuotactile_raw | 0.370594345 [0.343460424, 0.397973664] | 0.000396473231 [0.000370739619, 0.000422640669] | 0.338476838 [0.312194724, 0.367361929] | 6.68511967e-06 [2.18947858e-06, 1.40670959e-05] |
| wm_vision_anchored | 1.44165918 [1.19389664, 1.69799998] | 0.00145128858 [0.00128073429, 0.0016244496] | 1.38228196 [1.11644399, 1.6470553] | 4.49575277e-06 [1.38634436e-06, 9.49732925e-06] |
| wm_visuotactile_anchored | 0.364046574 [0.333071706, 0.394247821] | 0.000396473231 [0.000370739619, 0.000422640669] | 0.336204719 [0.310119676, 0.364968135] | 6.68511967e-06 [2.18947858e-06, 1.40670959e-05] |

Exact source keys for the table: `counterfactual/summary.json: metrics.<predictor>.<metric>.mean` and `.ci95_environment_bootstrap`.

**Prespecified baseline comparison (raw VT minus current-normal persistence):**

| Metric key | Difference, paired 95% CI |
| --- | --- |
| `interval_peak_MAE_N` | -0.000500572083 [-0.0132028739, 0.0110757285] |
| `tactile_MAE_N` | 0.0105058619 [-0.00116486481, 0.019933486] |
| `height_MAE_m` | -0.00195924361 [-0.00233004383, -0.00160030566] |
| `action_effect_height_MAE_m` | -0.000467792222 [-0.00049131213, -0.00044542698] |
| `action_effect_interval_peak_MAE_N` | -0.0741622979 [-0.0870649917, -0.061893935] |
| `constraint_aware_selection_regret` | -0.0107979213 [-0.0108973053, -0.0106687071] |

These differences are at `metrics.paired_wm_visuotactile_raw_minus_normal_force_persistence.<metric>`. Smaller action-effect error is meaningful positive evidence that VT predicts candidate-minus-center responses locally. Nearly tied absolute peak error does not contradict that relative-action result. This is a five-step local diagnostic, not sustained lifting or a trained control-policy comparison.

| Raw predictor | Height rank correlation, 95% CI | Height ranking defined fraction | Peak rank correlation, 95% CI | Peak ranking defined fraction |
| --- | --- | --- | --- | --- |
| normal_force_persistence | undefined | 0 [0, 0] | undefined | 0 [0, 0] |
| wm_vision_raw | 0.459409682 [0.372856863, 0.539574189] | 0.777777778 [0.694444444, 0.861111111] | 0.524305556 [0.401383102, 0.630561343] | 0.666666667 [0.666666667, 0.666666667] |
| wm_visuotactile_raw | 0.768340352 [0.698982216, 0.831193895] | 0.777777778 [0.694444444, 0.861111111] | 0.893981481 [0.882864583, 0.905324074] | 0.666666667 [0.666666667, 0.666666667] |

Rank correlations average only defined prefix rankings, then models within environment; constant persistence rankings are undefined, not zero. Raw/anchored height predictions are identical. Force anchoring uses current measured tactile normal offsets, so “anchored vision” is not vision-only sensing. An additive offset shared across candidates cancels in action differences except where zero clipping changes values; anchoring gains do not constitute new learned action information.

**Safety discrimination limitation:** independently counted true force-budget violations among all candidates = **0/324**. All summarized selectors have zero selected violations. The oracle utility is 0.357786985 [0.352373115, 0.363173244], with prefix means [0.0, 0.07336095627397299, 1.0]. Regret is oracle minus selected utility over these nine candidates, with branch-only peaks; prefix forces are excluded. Tiny regret therefore does not establish sustained success, damage avoidance, broad safety, or superiority of VT over vision. A zero-width violation interval on this cohort is not zero population risk.

The strong positive evidence belongs in the action-effect/ranking claim; the limited absolute-peak and force-constraint discrimination belongs beside it. The two persistence baseline definitions and the observed ranking-defined fractions must remain visible.

## Matched sensor-stress appendix

PASS: 330 stress runs paired with 330 clean runs of the same 11 fixed policies on 30 main ID environments. Exact physical parameters and policy/simulator input hashes match across clean/stress. All 330 stress NPZs were independently checked: prescribed residual assignments and clipping, unchanged shear and out-of-interval observations, two-action intervention, and outcome metrics from true height/substep force arrays. Independently recomputed 105 group/contrast rows and every paired policy change.

| Group | Clean joint success %, 95% CI | Stress joint success %, 95% CI | Stress minus clean joint pp, paired 95% CI | Stress minus clean strict lift pp, paired 95% CI | Stress minus clean force violation pp, paired 95% CI |
| --- | --- | --- | --- | --- | --- |
| WM_visuotactile | 27.777778 [14.444444, 41.111111] | 26.666667 [13.333333, 41.111111] | -1.111111 [-3.333333, 0.000000] | 0.000000 [0.000000, 0.000000] | 1.111111 [0.000000, 3.333333] |
| IQL_visuotactile | 35.555556 [23.333333, 48.888889] | 36.666667 [24.444444, 51.111111] | 1.111111 [-2.222222, 5.555556] | 0.000000 [-4.444444, 4.444444] | -1.111111 [-3.333333, 0.000000] |
| model_guard | 93.333333 [83.333333, 100.000000] | 93.333333 [83.333333, 100.000000] | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| force_feedback | 73.333333 [56.666667, 86.750000] | 73.333333 [56.666667, 86.750000] | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| reactive_guard | 73.333333 [56.666667, 86.750000] | 73.333333 [56.666667, 86.750000] | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |

Exact source keys: `sensor_stress_analysis/analysis.json: results.group_stats` (select `group=clean:<name>` or `stress:<name>`, `metric=joint_success`) and `results.contrasts` (select `first=stress:<name>`, `second=clean:<name>`, and the stated metric). Point/interval fields are `.estimate` or `.estimate_difference`, `.ci_low`, `.ci_high`.

Across the 330 executions, 4038 of 61,380 perturbed normal-channel observations were clipped at zero; 8715 observations became positive despite zero true normal contact. These are repeated observations, not independent sensor samples.

The paired 2,000-resample environment intervals are conditional on the fitted policies and the reused bank of 60 assigned residual blocks. They do not resample independent sensor datasets, CNN training, block assignment, or the sensor-error process. The library comes from held-out contact-derived regression errors with normal labels roughly 0.073–1.888 N, below the 8 N simulator threshold. Errors on two fingers, zero-contact states, and the imposed 20 Hz mapping are synthetic transports. A small or inconclusive change under this bank is not validated sensor robustness, sim-to-real transfer, hardware safety, or an identified noise model at 8 N.

## Guard activation and force-phase audit

Post hoc descriptive examination of 480 guard-on/off pairs independently checked all arrays and original trajectory SHA-256 values in 960 NPZ files. The frozen active-control interval comprises actions 44 through 134; recorded post-action force samples 45 through 135 are intervention peaks, and samples 136 through 150 are scripted lowering. The guard responds to a pre-action endpoint normal measurement at least 7.25 N, whereas the outcome uses all 25 simulator substeps per 0.05 s action interval and includes lowering. These are different temporal quantities.

**The guard was activated, but no paired binary outcome changed.** Model guards activated 23 times across 21 of 360 episodes (12/6/5 times for policy seeds 0/1/2); the matched reactive guard activated twice across two of 120 episodes. Exactly those 21 model and two reactive pairs have changed actions and trajectories; all remaining pairs are elementwise identical across every saved array. Across all 480 pairs, strict lift, joint success and full-episode force-violation labels are identical. Model action differences total 712 steps across the three policies, and reactive action differences total 174 steps, because an initial changed action alters later observations and actions. Thus the result is neither an inactive-guard experiment nor an identified protective improvement in the binary endpoint.

| Group | Executions | Guard trigger count / episodes | Setup / active / lowering violating episodes | Full-episode violations | Mean active peak (N) |
| --- | ---: | ---: | --- | ---: | ---: |
| model_guard | 360 | 23 / 21 | 0 / 57 / 45 | 100 | 6.573320594 |
| model_no_guard | 360 | 0 / 0 | 0 / 57 / 45 | 100 | 6.597287603 |
| reactive_guard | 120 | 2 / 2 | 0 / 3 / 39 | 41 | 6.021377110 |
| reactive_no_guard | 120 | 0 / 0 | 0 / 3 / 39 | 41 | 6.021377110 |
| force_feedback | 120 | 0 / 0 | 0 / 7 / 39 | 45 | 6.351157618 |

Phase counts overlap: model-guard counts include two episodes violating in both active control and lowering; reactive-guard counts include one. They must not be added as distinct failures. In model-guard runs, 139 of 150 active action intervals with a true substep peak above 8 N began with an endpoint measurement below 7.25 N. All four such reactive-guard intervals did too. This explains why a sampled measured-force guard need not prevent transient substep exceedances; it does not guard the later scripted lowering phase.

**ID and geometry joint gains occur in the lowering component.** Both model-assisted and matched reactive policies have 100% strict lift and no active-phase force violations in ID and geometry OOD. Model-assisted lowering violations are 6/90 ID executions and 24/90 geometry executions, versus 8/30 and 13/30 for the matched reactive policy. Pairing each trained model with the same reactive environment gives 18/90 ID joint gains and 15/90 geometry gains, with no losses; every gain is a reactive lowering-only exceedance that is absent under model assistance. These execution-pair counts retain only 30 independent environments per domain. The complete-episode joint gains therefore concern states carried into the shared scripted lowering phase, not observed prevention of active-phase violations. This is descriptive endpoint decomposition, not a causal mediation estimate.

| Domain | Model strict lift % | Reactive strict lift % | Model minus reactive strict-lift pp, paired 95% CI | Model minus reactive joint pp, paired 95% CI |
| --- | ---: | ---: | --- | --- |
| id | 100.000000 | 100.000000 | 0.000000 [0.000000, 0.000000] | 20.000000 [6.666667, 33.416667] |
| geometry_ood | 100.000000 | 100.000000 | 0.000000 [0.000000, 0.000000] | 16.666667 [3.333333, 30.000000] |
| physics_ood | 67.777778 | 96.666667 | -28.888889 [-45.555556, -14.444444] | -4.444444 [-18.888889, 8.888889] |
| combined_ood | 56.666667 | 93.333333 | -36.666667 [-53.333333, -20.000000] | -16.666667 [-36.666667, 3.333333] |
| all | 81.111111 | 97.500000 | -16.388889 [-22.229167, -10.833333] | 3.888889 [-4.451389, 11.666667] |

Exact audit keys: `completed_results_review.json: control.guard_details.phase_tables.<group>.<domain>`, `.guard_pairs.<on>_minus_<off>`, and `.model_minus_matched_reactive_joint_decomposition.<domain>`. Original stage values and trigger counts remain in `evaluations/<method>/episodes.jsonl`; phase indices and common lowering are specified in `evaluate.py`, while `adaptive_controller.py: guard` specifies the endpoint threshold. Paired strict/joint intervals above are the original `analysis/control_analysis.json: datasets.main.contrasts` rows, selected by first/second/domain/metric, independently recomputed by this audit.

**Main-text priorities.** The strongest control positive is the conditional ID/geometry joint improvement (+20.000 and +16.667 pp) of local model-assisted selection over matched reactive feedback, accompanied by the lowering decomposition. The pooled comparison is inconclusive (+3.889 pp, [-4.451, 11.667]), while strict lift falls substantially in physics and combined OOD. The learned WM actor achieves 11.944% pooled joint success versus 25.000% for reactive IQL (paired -13.056 pp, [-18.889, -7.778]); touch helps IQL pooled joint success by 8.889 pp [4.722, 13.333], whereas its pooled WM-actor effect is 0.556 pp [-2.500, 3.611]. Subsequent imagined policy improvement increases VT strict lift by 32.778 pp but also force violation by 40.833 pp; its joint change versus frozen-WM BC is 1.389 pp [-3.889, 6.389]. Report these negative and inconclusive findings alongside forecasting and local action-effect gains. Sensor-stress pooled-across-seeds ID joint changes range from -1.111 to +1.111 pp under the one fixed low-load residual bank, which supports only a narrowly scoped sensitivity observation.
