# Revision experiment protocol

Status: prospective local protocol, fixed before the new test outcomes are read.
This is not a public preregistration. Original paper, models and result cohorts
remain unchanged. New evidence may contradict the proposed controller; no
acceptance, safety or performance improvement is assumed.

## Questions

1. Under the same height-aligned reward and data, does touch improve executed
   control compared with vision alone?
2. Do action-conditioned dynamics add value beyond direct reactive policies?
3. Does a common measured-force feedback guard account for apparent improvements
   attributed to a learned world model?
4. Do conclusions persist across compiled object geometries and separately
   controlled, analytically feasible physical shifts?
5. Do learned predictions distinguish actual outcomes of alternative actions
   better than action-independent persistence?

## Data and test separation

- Expanded collection: 240 train, 48 validation, 48 calibration episodes.
- Collection mixes the common scripted controller and force feedback, 50:50.
- At least two training geometries, with disjoint held-out geometry families.
- Fresh control evaluation: 30 environments in each of ID, geometry-only OOD,
  physical-parameter-only OOD, and combined OOD, all controllers paired.
- A static friction-support check requires capacity/weight >= 1.5. This is an
  analytic screening condition, not a guarantee of achievable simulator success.
- Training seeds 0, 1, 2. Bootstrap units are environments, not policy executions.
- Calibration/test recordings never select weights or controller gains.
- Exact environment parameters, seeds and method matrix are saved in JSON before
  collection/evaluation, with hashes of relevant source and weights.

## Baselines and ablations

- Existing architecture, new multi-object data: vision and visuotactile world
  models, 25 epochs, training stride 3, identical architecture and loss weights.
- Behavior-initialized imagined actor-critic with height reward, three seeds,
  same 1000 BC/1500 RL update budget as the original study.
- Independently trained reactive vision/visuotactile BC and offline IQL:
  2000 BC updates, 10000 IQL updates, batch 256, Adam 3e-4, gamma .95,
  expectile .7, inverse temperature 3, maximum advantage weight 100,
  target update .005, gradient clip 10. BC selects minimum validation action
  MSE; IQL uses its fixed final update, with validation only diagnostic.
- Original script and 5 N feedback baselines.
- Exploratory fixed-grid model-assisted controller with common reactive base
  and measured-force guard; remove the model and remove the guard separately.
  No theorem or novel-algorithm claim is made before assessing existing work.
- Legacy vision-height ablation on the original data is reported as additional
  exploratory evidence, separately from the new confirmatory test set.

## Primary outcomes

Strict relative-height lift >= .10 m for ten consecutive 20 Hz observations
before lowering. Joint success additionally requires per-finger contact-force
peaks <= 8 N across every .002 s substep of the complete episode. Report each
component, all failed approaches, maximum height, peak force, inference cost,
training/data budgets and paired environment-bootstrap intervals.

## Counterfactual forecast diagnostic

Twelve additional environment seeds 20364000--20364011, with three fixed action
prefix lengths 44, 78 and 112. Reset and replay each identical prefix before
trying the fixed candidate actions for five steps. Compare learned force/height
forecasts, persistence, action-outcome ranking and selection regret. These
diagnostic outcomes do not tune the policy or filter the primary test set.

## Remaining scope

This work still concerns controlled grasp closure and lifting after a common
scripted approach, not general dexterous manipulation. The public GelSight
study is separate unless an actual transfer experiment is performed. Physical
robot results will not be inferred from other papers or presented as our own.
