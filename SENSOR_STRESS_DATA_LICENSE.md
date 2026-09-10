# Embedded sensor-stress data attribution

The `normal_residual_N` blocks and their assignments in
`outputs/revision_v2/sensor_stress_protocol.json` and its archived counterpart
derive from **TacBench / Sparsh GelSight force data**, hosted by Meta / FAIR at
[facebook/gelsight-force-estimation, revision 136db55485b6501605b1d2648ce0fe44740e302e](https://huggingface.co/datasets/facebook/gelsight-force-estimation/tree/136db55485b6501605b1d2648ce0fe44740e302e).

The dataset is distributed under **Creative Commons Attribution-NonCommercial
4.0 International (CC BY-NC 4.0)**. The embedded derived data are provided under
those same terms, including attribution and noncommercial use. Read the
[license](https://creativecommons.org/licenses/by-nc/4.0/). The repository's
software MIT license does not replace these data terms.

Attribution: Carolina Higuera et al., *Sparsh: Self-supervised touch
representations for vision-based tactile sensing*, CoRL 2024 / PMLR 270 (2025).
The predecessor `THIRD_PARTY_NOTICES.md` and frozen download manifest retain
upstream attribution, dataset revision, download identifiers, and file hashes.

Changes: held-out sphere trajectories were processed by the existing force
regression pipeline using the first prespecified CNN seed, 17. Each residual
is predicted normal force minus the corresponding normal-force label. The
frozen protocol assigns two separately drawn, contiguous 93-frame blocks per
ID environment, retaining source trajectory/block identifiers and float32
digests. The archived source prediction NPZ digest is derivation provenance;
the NPZ and source raw tactile images/force arrays are not included.
The portable export changes no residual value or block assignment.

The imposed one-source-index-per-20-Hz-observation mapping and additive use on
simulated fingers are synthetic transformations. They do not imply endorsement
by the dataset authors, validated sensor dynamics, or measured cross-finger
correlation. Derived analysis tables describe this appendix diagnostic only.
