# Revision-v2 packaging changes

This package was staged without publishing. Predecessor indexed files are
copied byte-for-byte or retained in place; `FILES_SHA256.json` is unchanged.
The revision is a COMPLETE TEXT-RECORD ARCHIVE with reproduction source.

## Reused predecessor source audit

The module AST after removing the module docstring is identical for every
listed pair. The adapter changes one docstring phrase about Python packaging;
the remaining differences are line endings. This is a source-equivalence
audit, not a promise of cross-platform bitwise simulator replay.

| Path | Executed source SHA-256 | Portable source SHA-256 | Difference |
| --- | --- | --- | --- |
| `outputs/simulation/sim_adapter.py` | `6d8189ac2c7fd27d0c67074e6ca9a06fc042f8be88993411df85ed0d8f8a5847` | `407a7ba79fd2b4ec29219740e185a34b497ddf643b6565ab290b823e9591c321` | module docstring and line endings only |
| `outputs/world_model/model.py` | `a875e616d64c64dcece4c57a4b619fd621f2c56b8b0da7a6109e013e9619322b` | `9a1101eacb6c79f480131f536f3a4e24a25b2acc77cfc914eda5f71cf0b2873b` | line endings only |
| `outputs/world_model/data.py` | `6ee5d678ce8339d409c1ae9e137bd09be83c4c46bdc40e17b7f92725f110d05d` | `42ad941192674d2d857ed46d601ee97e04b1cbde562fe609665b0d25dfdfb834` | line endings only |
| `outputs/world_model/train.py` | `1b117e63e52a277e5a938701ed84d18f598633f45824491c7a38002cb5a5c736` | `6dd371f4daa3f106f08a68d8141cd531fef54db9aece524724645cad21a581df` | line endings only |
| `outputs/world_model/train_imagination_rl.py` | `1387694675078e0842e0515541f8ea9b6a38624dc293ee739689f6912f803324` | `ced01089c8005bd8a12ea7947a49aaba128c6c08b1c04917272f58a7fee3f766` | line endings only |
| `outputs/world_model/controller_server.py` | `fb80e01c96c79af12d0bd1e083396d62b06f20022b60070d2e58468f1a275078` | `89de207a5471fc3737b52d69029227571a6aa6a807299bf5061b9ba805fb6325` | line endings only |
| `outputs/world_model/tdmpc2_layers.py` | `0d59a2513aaabd52edd31d7842299a853065dce14c9bf4a09831e58e2006da8b` | `68d2ea8e332235559bbb8baa64b4452230fa68b9b0610c5dd757ce7b81b86a1f` | line endings only |

## New adapter and working protocol

Only the predecessor import/provenance block in the packaged
`outputs/revision_v2/sim_adapter_v2.py` changes. It accepts exactly the two
audited predecessor byte hashes, refuses any other predecessor, and records
the actual loaded predecessor as the expected hash. The original executed
hash remains `EXECUTED_ORIGINAL_SHA256`. Reversing this exact block replacement
recovers the original source after newline normalization. No environment,
observation, action, reward, controller, training, or evaluation calculation
is changed. The executed source is retained in a full archive under
`recorded/outputs/revision_v2/executed_source/sim_adapter_v2.py`.

The working `outputs/revision_v2/protocol.json` changes only its adapter
`source_sha256` to the packaged wrapper hash so `build_protocol.py` can verify
it. Its numerical parameters, seeds, split membership, and ordering remain
unchanged. The archived protocol retains its original bytes and execution hash.
Thus newly generated provenance differs intentionally from executed provenance.

## Sensor-stress provenance and embedded derived data

The working sensor protocol changes only inherited source hashes and the hashes
of the working main protocol/method list; its residual values, assignments,
environment cases, methods, numerical settings, and original source-data digest
are unchanged. The original sensor protocol is archived byte-for-byte. Its
packaged evaluator records the new working protocol digest, requires an explicit
GPU interpreter for execution, and accepts absence of the original prediction
NPZ. The embedded residual values remain authenticated by the frozen protocol
digest and the unchanged float32 block digests. If the optional original NPZ
exists, its original digest check still applies. These are only CLI/provenance
and source-availability changes; observation-error formulas, simulation,
controllers, and analysis are unchanged. Exact block substitutions are checked
as reversible before export. Original evaluator source hashes are retained.

The embedded 60 residual blocks are derived TacBench / Sparsh GelSight data,
under CC BY-NC 4.0, not the software MIT license. Attribution and transformations
are described in `SENSOR_STRESS_DATA_LICENSE.md`. Full packaging also requires
the complete 11-by-30 sensor-stress matrix and its paired analysis.

## Record export

Export uses a whitelist of JSON, JSONL, CSV, and Markdown records. It excludes
raw trajectories, arrays, checkpoints, media, ordinary process logs, and credentials.
The explicit exception is the complete failed counterfactual initialization
log set: every stack/error line is retained, with filename paths sanitized.
The infrastructure amendment, failed manifest, successful runtime provenance,
and separate main/branch runtime probes accompany it. The portable runner uses
CPU counterfactual inference and GPU training/main evaluation, matching the
recorded amendment. Existing simulation NumPy import-order differences are
documented rather than silently changed in scientific source.
Known checkout prefixes become relative paths; other external executable paths
are replaced by explicit placeholders. JSON keys containing paths are sanitized
as well. Numeric/boolean/null JSON values are unchanged; CSV numerical cells are
unchanged. Original source/record SHA-256 values are retained separately from
the hashes of exported files in `REVISION_V2_FILES_SHA256.json`.
Archived input hashes still describe the executed files, not their sanitized
exports. Use the new index to check archived export integrity.

The new standard-library runner and verifier, README, this document, and local
runtime `.gitignore` are packaging additions. The runner supplies explicit fixed
budgets and accepts user-supplied interpreter paths. It performs no installation
or publishing and never edits `recorded/`. It omits the separate exploratory
legacy rerun from the revised main pipeline. A complete package is accepted
only after all primary control, forecast, and counterfactual analyses exist.
