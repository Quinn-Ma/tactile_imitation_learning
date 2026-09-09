# Third-party sources

- **TD-MPC2**: https://github.com/nicklashansen/tdmpc2/tree/e9f5932 . The adapted layer definitions originate in `tdmpc2/common/layers.py`, commit e9f5932. The original MIT notice is retained verbatim in `outputs/world_model/LICENSE_TDMPC2.txt`. This package uses adapted building blocks, not an official TD-MPC2 reproduction or pretrained checkpoint.
- **TacBench / Sparsh GelSight force data**: https://huggingface.co/datasets/facebook/gelsight-force-estimation/tree/136db55485b6501605b1d2648ce0fe44740e302e . License: CC BY-NC 4.0. The archive includes download identifiers, hashes and derived numerical results; raw images/force arrays are downloaded separately. Cite Carolina Higuera et al., *Sparsh: Self-supervised touch representations for vision-based tactile sensing*, CoRL 2024 / PMLR 270 (2025).
- **robosuite 1.5.1**: https://github.com/ARISE-Initiative/robosuite . Installed as a dependency, not vendored in this archive. Paper: https://arxiv.org/abs/2009.12293 . The Lift task is modified with continuous gripper-rate control and the documented contact-force logging/protocol.
- **MuJoCo 3.3.7**: https://github.com/google-deepmind/mujoco . Installed as a dependency, not vendored in this archive. Project: https://mujoco.org/ . The solver's contact forces are simulator outputs, not physical damage measurements.

The manuscript's distribution terms do not replace these upstream licenses. No third-party raw dataset is represented as newly collected data.
