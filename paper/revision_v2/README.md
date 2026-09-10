# Revised manuscript and simulation supplement

[Named preprint PDF](preprint.pdf) | [LaTeX source bundle](arxiv_source_v2.zip) | [Simulation video](simulation_video.mp4)

**Compact Visuotactile World Models for Lifting: Prediction, Reward Alignment, and Force Constraints**

Qinzhen Ma (Rice University) and Sida Peng (Zhejiang University). Contact: qm18@rice.edu.

This seven-page revision contains two figures and three tables. It reports 4,080 main simulator executions on 120 independent environments, 324 action branches on 12 additional ID environments, and 330 sensor-stress executions reusing the 30 main ID environments. Results include negative comparisons and uncertainty intervals. There is no physical-robot control experiment or demonstrated sim-to-real transfer. The earlier exploratory manuscript remains at [../preprint.pdf](../preprint.pdf), with separate denominators.

The named PDF includes author affiliations. A separately checked anonymous ICRA review PDF is provided to the authors locally and is not published here. These files do not claim conference acceptance or a publicly assigned arXiv identifier. See [publication metadata](../../publication.json).

## Compile or upload the preprint source

Extract `arxiv_source_v2.zip` into an empty directory and compile `main.tex` twice with pdfLaTeX, with shell escape disabled. All seven required source files, including the official `ieeeconf.cls`, are included at the archive root. The bibliography is embedded in `main.tex`; no BibTeX run is needed. For arXiv, upload this source bundle rather than only the TeX-generated PDF. Local compilation used TeX Live 2025, PDF 1.4, US Letter, and embedded fonts.

## Video

The simulation supplement is 95 seconds, 1280 x 720, 20 fps, silent H.264 MP4, 4,351,457 bytes. Twenty replays cover five prespecified cases and two controller comparisons per case. Replaying recorded actions exactly reproduced all five checked observation arrays; full simulator qpos was not archived, so full-state equivalence is not claimed. Selected failures are retained. The results slide includes all fourteen method groups. See the [reproduction guide](../../README_REPRODUCTION_REVISION_V2_APPENDIX.md) and [full archived results](../../recorded/outputs/revision_v2/).

The manuscript and its original figures are released under CC0 1.0, consistent with the preprint license; adapted third-party code and derived data retain their own terms. See the repository [license](../../LICENSE), [third-party notices](../../THIRD_PARTY_NOTICES.md), and [sensor-data terms](../../SENSOR_STRESS_DATA_LICENSE.md). The bundled IEEE class retains its original notice.
