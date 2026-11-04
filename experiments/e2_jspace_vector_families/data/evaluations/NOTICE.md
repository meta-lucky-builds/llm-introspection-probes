# Attribution

`lens-eval-*.json` and `README.upstream.md` are unmodified copies from [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) at commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`, `data/evaluations/`. Copyright 2026 Anthropic PBC. Apache License 2.0, in `LICENSE.jacobian-lens`.

Git blob SHA-1 of each file at that commit, which `tests/test_e2_data.py` checks:

| File | Blob |
|---|---|
| `lens-eval-association.json` | `03a712344700b1f2bba8960995e998f04ded21e3` |
| `lens-eval-multihop.json` | `2d2d6e7bdf6fb63597ce11895dc55f5391b5f86f` |
| `lens-eval-multilingual.json` | `4aec3f4d32e1b4d0ed959ffe6b1ae27fcfeae650` |
| `lens-eval-order-ops.json` | `977e8ceac8d8030a21d4fcb8af4914d943c90ca9` |
| `lens-eval-poetry.json` | `5e06287e7fa49325506180aa4d2bb2a26ba29eaf` |
| `lens-eval-typo.json` | `7354faf759764c5837285ae6d35ac854b89135e6` |

`../order_ops_synonyms.json` and `../chinese_targets.json` are mine, not upstream's. No jacobian-lens code is vendored. `src/mechinterp/e2/lens.py` reimplements its readout and estimator in a few lines.
