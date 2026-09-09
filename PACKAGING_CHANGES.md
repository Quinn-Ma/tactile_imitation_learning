# Package-only changes

1. Kept only the 21 Python files needed for dataset preparation, training, policy execution, numerical analysis and pairing audits. No raw data, weights, NPZ prediction/trajectory arrays, old manuscripts, synthetic fixtures or TeX tools are included.
2. Placed recorded JSON/JSONL/CSV results under `recorded/outputs/` so fresh commands do not overwrite them. Numeric values were checked for equality while copying. Source experiment files were not changed.
3. Replaced private Windows absolute paths in copied metadata with relative references or `LOCAL_PYTHON` / `EXTERNAL_LOCAL_PATH` markers. External HTTP(S) source URLs remain intact.
4. Changed both simulator matrix runners to require `--gpu-python`; they no longer default to a private executable path. Model, policy, reward, force and evaluation logic are unchanged.
5. Made the public downloader default to the included SHA-256 manifest and the pinned dataset revision. Added support for that manifest's schema, avoiding a dependency on an unpublished work-directory file.
6. Corrected the documented world-model command to include `--train-stride 3` and retained all other recorded options.
7. Defined strict lifting as ten consecutive 20-Hz observations, not a continuous 0.5-second dwell. This is a wording correction; no observation or outcome was recomputed.
8. Preserved the TD-MPC2 MIT notice and source revision. No new license has been substituted for upstream data or code licenses.

Validation checks file integrity, Python syntax, recorded split/controller counts, absence of private paths and excluded binary artifacts. It does not claim that every environment has been reinstalled or every experiment rerun from this portable copy.
