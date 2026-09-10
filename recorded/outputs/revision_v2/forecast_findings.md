# Completed forecasting evidence audit

**Conclusion:** tactile conditioning lowers endpoint/peak forecast error relative to vision in all four domains, but these force results do not establish that learned transition dynamics improve on persistence, nor any control benefit. Observed endpoint persistence has lower error than the visuotactile rollout in every domain; peak persistence differences include zero throughout. Height forecasting has a more favorable result, detailed below.

All values below come from the completed fixed forecast analysis. Values are printed to six decimal places; source CSVs retain full precision. Each domain contains 30 independent script-controlled environments, shared by all predictors and three fixed training seeds. History=3, horizon=5, stride=5, 30 windows/episode (3,600 windows/model). Endpoint MAE averages six signed/normal force components; peak MAE averages two per-finger substep interval maxima; both average all five future horizons. These are full-episode windows, including approach/lowering, not only grasp-contact windows.

Intervals are pointwise 95% percentile paired environment-bootstrap intervals (2,000 draws, seed 20401010), conditional on the fitted models. They do not estimate retraining uncertainty. Differences below are first method minus second; negative error differences favor the first.

## Force accuracy and paired contrasts

| Domain | Metric (N) | Vision MAE | VT MAE | VT minus vision, paired 95% CI | Persistence MAE | VT minus persistence, paired 95% CI |
| --- | --- | ---: | ---: | --- | ---: | --- |
| ID | Endpoint | 0.939294 | 0.188759 | -0.750535 [-0.999866, -0.525626] | 0.106492 | 0.082268 [0.060474, 0.106876] |
| ID | Interval peak | 2.566502 | 0.447606 | -2.118896 [-2.839316, -1.474436] | 0.443997 | 0.003609 [-0.065941, 0.062419] |
| Geometry OOD | Endpoint | 1.158857 | 0.630561 | -0.528296 [-0.775969, -0.298921] | 0.293298 | 0.337263 [0.208540, 0.482234] |
| Geometry OOD | Interval peak | 2.507895 | 1.133649 | -1.374246 [-2.100665, -0.681292] | 1.195154 | -0.061505 [-0.179351, 0.040298] |
| Physics OOD | Endpoint | 1.230400 | 0.402217 | -0.828183 [-1.040516, -0.607370] | 0.132581 | 0.269636 [0.204727, 0.342728] |
| Physics OOD | Interval peak | 2.977033 | 0.788700 | -2.188333 [-2.808095, -1.557965] | 0.788395 | 0.000305 [-0.084999, 0.083073] |
| Combined OOD | Endpoint | 0.771994 | 0.360726 | -0.411268 [-0.573198, -0.246870] | 0.132440 | 0.228286 [0.126460, 0.354781] |
| Combined OOD | Interval peak | 1.689529 | 0.667924 | -1.021605 [-1.445932, -0.601962] | 0.703753 | -0.035829 [-0.109911, 0.027339] |

**Persistence semantics matter:** endpoint persistence holds the current measured tactile vector constant and is independent of fitted model seed. Peak and height persistence hold each VT model's current decoded peak/height constant; these are learned-observer baselines, not persistence of measured interval peaks or oracle object height. Accordingly, tiny endpoint seed SD (about 1.7e-17 in ID) is floating-point noise, while peak/height baseline seed variation is real. The consistently worse endpoint rollout and inconclusive peak contrasts limit a claim that transition dynamics add force-forecasting value over these baselines.

## Shuffled-touch diagnostic

The same-domain cyclic donor assignment is a 30-environment permutation without fixed points in every domain. Only tactile history changes; targets, recorded actions, RGB and proprioception stay fixed. Positive shuffled-minus-VT differences show sensitivity to correctly paired touch, not a causal demonstration that action-conditioned dynamics improve control.

| Domain | Shuffled endpoint MAE (N) | Shuffled minus VT endpoint, paired 95% CI | Shuffled peak MAE (N) | Shuffled minus VT peak, paired 95% CI |
| --- | ---: | --- | ---: | --- |
| ID | 1.934074 | 1.745315 [1.405266, 2.102767] | 5.219987 | 4.772381 [3.726634, 5.846021] |
| Geometry OOD | 1.758189 | 1.127627 [0.813057, 1.446594] | 4.036123 | 2.902474 [1.987765, 3.799727] |
| Physics OOD | 1.813029 | 1.410812 [1.062610, 1.763420] | 4.591889 | 3.803189 [2.792519, 4.826221] |
| Combined OOD | 1.406145 | 1.045420 [0.692863, 1.399911] | 3.375346 | 2.707422 [1.759935, 3.688310] |

## Height and training-seed variation

| Domain | Vision height MAE (mm) | VT height MAE (mm) | VT minus vision, paired 95% CI (mm) | VT minus decoded-height persistence, paired 95% CI (mm) |
| --- | ---: | ---: | --- | --- |
| ID | 3.224770 | 2.112915 | -1.111855 [-1.500673, -0.691613] | -2.099061 [-2.576506, -1.567702] |
| Geometry OOD | 5.595799 | 5.164130 | -0.431670 [-1.176930, 0.265239] | -1.630433 [-2.351150, -0.889293] |
| Physics OOD | 4.663793 | 2.650142 | -2.013651 [-2.603316, -1.442824] | -3.029004 [-3.434441, -2.573293] |
| Combined OOD | 4.808955 | 3.858679 | -0.950276 [-1.922693, -0.192229] | -0.597522 [-1.435152, 0.221078] |

Height contrasts favor VT over vision in ID, physics OOD and combined OOD, while the geometry-only interval includes zero. Against its own decoded-height persistence baseline, VT rollout improves height MAE in ID, geometry OOD and physics OOD; the combined-shift interval includes zero. Thus the force-persistence limitation should not be generalized to every predicted quantity.

Seed columns below are sample SD across the three fixed model means; the bracket is the min–max across those means, not a confidence interval.

| Domain | Model | Endpoint SD [range] (N) | Peak SD [range] (N) | Height SD [range] (mm) |
| --- | --- | --- | --- | --- |
| ID | Vision | 0.020756 [0.921270, 0.961987] | 0.079670 [2.492059, 2.650531] | 0.898272 [2.688137, 4.261794] |
| ID | VT | 0.011681 [0.177498, 0.200819] | 0.027763 [0.421387, 0.476691] | 0.357288 [1.765894, 2.479652] |
| Geometry OOD | Vision | 0.055511 [1.105639, 1.216407] | 0.174501 [2.327445, 2.675767] | 0.757420 [4.723886, 6.091003] |
| Geometry OOD | VT | 0.060612 [0.568941, 0.690113] | 0.185220 [0.940138, 1.309281] | 1.544759 [4.210579, 6.946407] |
| Physics OOD | Vision | 0.042818 [1.198596, 1.279086] | 0.143205 [2.894326, 3.142391] | 1.367405 [3.777779, 6.238626] |
| Physics OOD | VT | 0.059072 [0.336582, 0.451115] | 0.111548 [0.680963, 0.903704] | 0.845550 [1.874933, 3.551791] |
| Combined OOD | Vision | 0.017790 [0.759373, 0.792341] | 0.089338 [1.623059, 1.791083] | 0.386039 [4.365181, 5.067234] |
| Combined OOD | VT | 0.035540 [0.340034, 0.401763] | 0.082971 [0.591829, 0.756384] | 1.102270 [2.858200, 5.040299] |

## Frozen nominal 90% joint coverage and interval size

Coverage requires every evaluated window, horizon and finger in an episode to satisfy the one-sided peak upper envelope (prediction + radius) and the two-sided height interval (prediction ± radius). Separate alpha/2 allocation uses the 47th order statistic among 48 calibration episodes. Coverage below averages the three fitted models within each environment; there are 30 independent environments per domain, not 90 independent trials.

| Domain | Vision joint coverage %, paired-environment 95% CI | VT joint coverage %, paired-environment 95% CI | VT per-seed joint coverage (%) |
| --- | --- | --- | --- |
| ID | 98.888889 [96.666667, 100.000000] | 90.000000 [78.888889, 97.777778] | 90.000000, 90.000000, 90.000000 |
| Geometry OOD | 83.333333 [74.444444, 91.111111] | 55.555556 [40.000000, 71.111111] | 40.000000, 63.333333, 63.333333 |
| Physics OOD | 91.111111 [83.333333, 97.777778] | 68.888889 [57.777778, 81.111111] | 76.666667, 63.333333, 66.666667 |
| Combined OOD | 81.111111 [71.111111, 90.027778] | 65.555556 [48.888889, 80.027778] | 63.333333, 66.666667, 66.666667 |

| Model seed | Peak upper-envelope radius (N) | Height radius (mm) | Full two-sided height width (mm) |
| --- | ---: | ---: | ---: |
| vision_seed0 | 15.478722 | 56.772277 | 113.544554 |
| vision_seed1 | 15.406090 | 49.534515 | 99.069029 |
| vision_seed2 | 14.679380 | 30.187272 | 60.374543 |
| visuotactile_seed0 | 6.854743 | 16.794741 | 33.589482 |
| visuotactile_seed1 | 8.698555 | 23.261502 | 46.523005 |
| visuotactile_seed2 | 8.117259 | 17.373286 | 34.746572 |

Across-seed radius/width ranges: vision: peak margin 14.679380–15.478722 N; height radius 30.187272–56.772277 mm; full height width 60.374543–113.544554 mm; visuotactile: peak margin 6.854743–8.698555 N; height radius 16.794741–23.261502 mm; full height width 33.589482–46.523005 mm. These force margins are large relative to the 8 N evaluation budget; empirical envelope coverage should not be equated with useful constraint feasibility.

VT marginal peak/height episode coverages are ID 94.444444% / 95.555556%; Geometry OOD 75.555556% / 55.555556%; Physics OOD 82.222222% / 84.444444%; Combined OOD 80.000000% / 66.666667%. A joint envelope failure can arise from the height interval and is not an observed force-budget violation.

**Coverage caveat:** calibration is a 50:50 script/force-feedback policy mixture, while all forecast test trajectories use script only. Thus even the nominal ID subset has a policy-mixture distribution shift; empirical ID coverage is not a verified exchangeability guarantee. OOD adds geometry and/or mass/friction shift. VT has narrower radii but markedly lower OOD joint coverage, so “90% calibrated under OOD,” “more accurate therefore safer,” and “better calibration than vision” are not supported. A force radius is an upper additive margin, not a symmetric interval half-width. No persistence or shuffled predictor is separately calibrated here.

The combined-shift set has lower force MAE than some single-shift sets. This is not evidence that combined shift is easier in general: contact prevalence and trajectory outcomes differ. Contact-forecast fractions are ID 46.888889%, Geometry OOD 49.288889%, Physics OOD 57.977778%, Combined OOD 33.977778%. Errors average many noncontact/slowly changing windows, which helps explain the strong observed-force persistence baseline.

## Data, compute, and provenance checks

Independent audit verified all 137 frozen protocol/source/checkpoint/calibration/trajectory byte hashes, 1,440 episode rows, 120 shuffled assignments, and the original metric aliases. It independently recomputed 596 aggregate/paired rows (point estimates, 95% CIs, per-seed means and SDs) from episode sufficient statistics; maximum absolute discrepancy was 2.66e-15. All six force/height radius pairs were independently recovered exactly as the 47th order statistics from their saved 48-episode calibration predictions. No training, tuning, new predictor evaluation, or control-result analysis was performed.

Each WM used 240 train / 48 validation / 48 calibration episodes, 25 epochs, batch 64, train stride 3 and validation stride 5. With 150 actions and five-step targets, this is 11,760 training windows and 1,440 validation windows; 184 optimizer batches/epoch (the final partial batch is retained), or 4,600 updates/model. Selection minimizes validation `total - 20*latent`, the supervised composite without the EMA latent term, not raw logged `val.total`. Every selected checkpoint matches that criterion and the frozen forecast lock. Both variants retain the same 652,157-parameter architecture; vision receives zeroed normalized tactile inputs rather than deleting the touch branch.

| WM run | Parameters | Selected epoch / 25 | Selected supervised validation loss | Fit + validation seconds | Including final evaluation seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| vision_seed0 | 652157 | 20 | 0.376095 | 145.837326 | 146.861764 |
| vision_seed1 | 652157 | 21 | 0.386606 | 170.721898 | 171.810316 |
| vision_seed2 | 652157 | 19 | 0.384359 | 173.592558 | 174.751024 |
| visuotactile_seed0 | 652157 | 22 | 0.065005 | 167.628059 | 168.666647 |
| visuotactile_seed1 | 652157 | 24 | 0.067302 | 175.813627 | 176.753121 |
| visuotactile_seed2 | 652157 | 24 | 0.061662 | 210.212660 | 211.557198 |

These elapsed times are local wall-clock observations, not hardware-normalized compute comparisons or isolated latency measurements. World-model training also uses an EMA target copy; inference parameter counts are reported above.

Completed imagined-actor metadata were available for 6/6 runs. Every inspected run uses the corresponding frozen WM hash, 1,000 BC plus 1,500 RL updates, batch 256, horizon 5 and the height reward. Reactive runs were not inspected because their matrix was incomplete.

| Imagined actor | Actor parameters | Train episodes | Replay states | Elapsed seconds |
| --- | ---: | ---: | ---: | ---: |
| vision_seed0 | 33282 | 240 | 10560 | 57.489279 |
| vision_seed1 | 33282 | 240 | 10560 | 53.384509 |
| vision_seed2 | 33282 | 240 | 10560 | 53.723154 |
| visuotactile_seed0 | 33282 | 240 | 10560 | 53.602214 |
| visuotactile_seed1 | 33282 | 240 | 10560 | 54.218609 |
| visuotactile_seed2 | 33282 | 240 | 10560 | 53.948518 |

## Suggested paper paragraph (~100 words)

Across 120 held-out scripted environments, visuotactile models reduced five-step endpoint-force and interval-peak MAE relative to vision in all four domains (paired 95% environment-bootstrap intervals excluded zero). ID errors fell from 0.939 to 0.189 N and 2.567 to 0.448 N, respectively. However, measured-force persistence achieved lower endpoint error in every domain, while peak differences from decoded-state persistence remained inconclusive. Shuffling tactile histories increased both errors. Frozen nominal-90% joint coverage was 90.0% ID but 55.6–68.9% across shifted domains. Calibration/test policy mixtures also differed. These findings support informative tactile conditioning, without establishing transition-model control benefits or safety guarantees.

Sources: `forecast_analysis/{aggregate_metrics,paired_differences,episode_metrics}.csv`, `analysis_metadata.json`, `locked_inputs.json`, six `world_models/*/{training.jsonl,summary.json,calibration.json}`, and completed `rl/*/run_metadata.json`. Machine-readable audit: `forecast_audit.json`.
