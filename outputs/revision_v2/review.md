# Independent causal-comparison notes

This review concerns the prospectively fixed design and implementation. It does
not inspect new test outcomes or assert any performance improvement.

- **Common forecast distribution.** Evaluate every world model and diagnostic
  predictor on the same script trajectory for each protocol environment. This
  separates predictor quality from differences in controller-induced states.
  Actual future actions are supplied as forecast conditions; these metrics are
  not an online control evaluation or evidence of useful counterfactual ranking.
  The separate reset-and-branch diagnostic addresses local action sensitivity.
- **Persistence semantics.** Endpoint persistence holds the measured current
  tactile force. Peak and height persistence hold the model's current decoded
  values, matching the original implementation. They are not oracle current
  object-height inputs or measured transient-peak observations to the policy.
- **Calibration distribution.** The 48 calibration episodes mix script and
  feedback collection, whereas the common forecast test traces use script alone.
  Even the ID empirical coverage should not be advertised as a guaranteed 90%
  result under identical sampling distributions. All OOD groups add distribution
  shifts. Persistence and shuffled-touch predictors do not inherit the original
  world model's calibration, so their coverage is deliberately left unreported.
- **Touch ablation.** Vision and visuotactile world-model/actor comparisons have
  the same data, architecture size, task reward, and nominal optimization budget.
  A masked-touch improvement is a modality result in this protocol, not proof
  that action-conditioned dynamics are necessary. A matched reactive modality
  comparison helps establish whether touch alone supplies a similar advantage.
- **Reactive versus world-model control.** Reactive IQL and imagined actor-critic
  differ in optimization, representation training, regularization, and compute.
  Their comparison establishes performance of complete pipelines, not a pure
  causal effect of adding dynamics. Report both total trainable/inference counts
  and update/data budgets. A shared reactive base with explicit model/guard
  ablations more directly tests the added controller components.
- **Observed transition validity.** Expanded collection enforces the same two
  intervention action dimensions used by the reactive Q function. Original
  recordings contained other nonzero action components; projecting those actions
  would confound a causal two-action transition model. Keep them out of the
  revised primary IQL training set.
- **Sample units.** Thirty environments per domain remain thirty independent
  test conditions when used with three training seeds. Paired resampling must
  keep every horizon, window, coordinate, and fitted-model result for one
  environment together. Contact-conditioned errors require pooled error sums
  and contact counts, not an unweighted average of per-episode conditional MAE.
- **Geometry and physical shifts.** Compiled boxes test a limited family of
  aspect ratios/sizes, not arbitrary object categories. Physical-parameter OOD
  is a joint parameter shift unless factors are separately intervened upon.
  Friction/aperture screening is an analytic proxy and does not guarantee that
  the simulator policy can achieve a stable budget-compliant lift.
- **Failure accounting.** Keep setup failures, approach/lowering peaks, and
  unsuccessful strict lifts in all primary denominators. A common force guard
  must be compared both with and without model assistance before assigning its
  performance to learned predictions. An 8 N soft cost is not a hard guarantee.
- **Study status.** A local prospective protocol is not public preregistration.
  This revision was motivated by earlier results; distinguish untouched fresh
  evaluation from earlier exploratory cohorts. The separate public GelSight
  sensing study still supplies no demonstrated sensor-policy transfer.
