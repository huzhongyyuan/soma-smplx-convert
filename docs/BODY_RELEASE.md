# Body release v0.2.0-body.1

This release packages the verified body converter and diagnostic checks. It does not replace a live training pipeline or establish near-zero error.

## Included

- Fixed-shape forward body fitting with analytic attachment, skinning and pose Jacobians.
- Matched projected-stationarity inverse, avoiding nested forward fits.
- Explicit body CLI with preserved timing, separated shape/expression and deliberate shape normalization.
- Optional finger passthrough; no hand reconstruction claim.
- Numerical tests, actual asset/CLI checks, 65-frame TRAIN diagnostic scores and failure visualization.

The numerical class/function definitions in `body.py`, `_fastskin.py` and `_lmfit.py` are AST-identical to the verified snapshot; packaging changes only imports/module placement and comments. `meshfit.py` and `analytic.py` match the frozen dependencies. The CLI module and metadata paths were adjusted for the public package. The inherited legacy package version remains 0.1.0; this body release is identified by its Git tag and source hashes.

## Not adopted

NVIDIA mapping plus exact SMPL-X refinement did not dominate all metrics. A higher-order worst-point objective traded mean error for maximum error; a constrained tail fit had small gains and an unvalidated inverse. Neither is automatically selected or mixed per sample. No smoothing, original-pose lookup, per-frame shape change or difficult-frame exclusion improves the reported scores.

## Evidence and limits

`body_validation.json` records the newly packaged technical smoke and source hashes. `body_benchmark.json` reuses already-verified paired meshes and scores, with sample IDs and evidence hashes. It is not a new full-dataset run. Packaging equivalence and an actual one-frame CLI round trip link the frozen evidence to this release.

The torso/limb diagnostic excludes head-dominant query vertices and retains neck; original all-body scores remain available. Queries always use the complete opposite surface, avoiding inconsistent wrist cuts. Closest-surface distance differs from anatomical correspondence, joint error, round-trip error and official benchmark metrics.

Pending: lower collar/elbow residuals, explicit exterior-head semantics, long sequences, full-dataset checks and further independent native-pose bidirectional quality evaluation. The visualization shows the official comparator; it does not imply identical default results.

## Reproduction prerequisites

Validation used one CPU math thread, PyTorch 2.6.0+cu124 with CUDA hidden, NumPy 1.26.4, SciPy 1.15.3, trimesh 5.0.0 and the ARDY revision pinned in `requirements-body.txt`. Clean upstream asset downloads on every platform have not been tested. Obtain assets under their own licenses; regenerating a rest fit may change numerical scores.
