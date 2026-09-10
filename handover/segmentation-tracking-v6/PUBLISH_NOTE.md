# Publication update and verification

## Parent advanced during review

The scientific review and source links initially refer to v5 commit
`0a3105a8f9d1b0954707675b42b9868832df4ac1` (eight completed configurations).
While this handover was being written, v5 advanced to
`0b6ce2d7a77baed54385d4c9b08acbefb6aa7c03`. The new branch is based on this later
commit, so none of that update is lost. The earlier REVIEW/SOURCES snapshots
remain identified rather than silently rewritten as a different experiment.

The added completed comparison is **H_probe_native_J** on all 199 clips:

| Item | Measured value |
|---|---:|
| Pooled score | 0.9125505234745164 |
| Delta against C0 | -0.02225185078606962 |
| 44b6 score | 0.8367204338897453 |
| 6bba score | 0.9279280377744168 |
| Edge TP / FP / FN | 121664 / 6207 / 7219 |
| Division TP / FP / FN | 30 / 175 / 121 |

Nine of eighteen configurations are now complete in the publication-parent
snapshot. V5 remains executing, with native comparisons/replication and full fresh
validation still pending. The first fresh clip ran all twelve paths in 674.73
seconds; C0 graph/CSV parity and the eleven other predecoder input fingerprints
passed. This is not the full six-clip graph/count validation. No newly validated
winner is claimed. C0 remains 0.934802374260586.

Source: [the parent commit](https://github.com/matheuspf/cell-tracking/commit/0b6ce2d7a77baed54385d4c9b08acbefb6aa7c03)
and its maintained v5 continuation. S600 must inspect current local receipts and
active jobs, not assume this progress snapshot is final.

## Authoring validation

The three reference Python files were tested under Python 3.13.5 with NumPy.
All **38 synthetic mask/graph contract tests passed**, including a repeat before
publication. The files compile, and preflight CLI help runs from a foreign working
directory. No real dataset, segmenter, tracker, model weights or GPU was used.

Remote blobs match the exact tested local files:

| File | Git blob SHA | Bytes |
|---|---|---:|
| region_contracts.py | 4fed6bc40aed34780a9aa5a2b4cbe227f345e7e1 | 7136 |
| test_region_contracts.py | 6693f4fc5c79b69c2edbc0ea472c4db0ac713250 | 5733 |
| preflight.py | aee01b7b5f5fd969859635188d6fd49751df36cb | 2379 |

These tests verify reference geometry and graph conventions only. Production
adapter/solver parity, actual mask quality, all-clip graph scores and runtime are
local execution tasks. No integration or >=0.95 achievement is claimed here.
