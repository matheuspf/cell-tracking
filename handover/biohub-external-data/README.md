# External-data handover for the next agent

Start with the committed [external-data guide](../../docs/external-data-guide/README.md).
It contains the complete factual handover for the September 8, 2026 download and
preparation snapshot, including every downloaded dataset, its label schema,
provenance, limitations, access routes and relevant results from the current
`handover/strong-tracker-v3` branch at `f67ba60`.

The next agent decides whether and how to use the data. This handover selects
no datasets, training order, model changes, experiments or future promotion rule.
No model was trained or evaluated using these external datasets during the
preparation work; no transfer gain is established.

- [Dataset facts and inspection scope](../../docs/external-data-guide/DATASETS.md).
- [Current models, measured outcomes and interface mismatches](../../docs/external-data-guide/BRANCH_CONTEXT.md).
- [Retrieval and preparation instructions](../../docs/external-data-guide/ACCESS.md).
- [Complete structured inventory](../../docs/external-data-guide/dataset_inventory.json).
- [Recorded verification evidence](../../docs/external-data-guide/verification_summary.json).

The original thread and ten replies were retrieved. All identified shared
dataset contents were recovered; the unavailable enriched Tribolium ZIP was
reconstructed from its public directory. Exact upstream ZIP bytes remain
unavailable. Separate raw microscopy linked from some study pages was not
downloaded and is not counted as acquired data.

Only authored code and the information package are committed. Raw datasets,
prepared arrays, model weights, original references, the local HTML guide and
private session exports remain outside Git. The historical artifact roots are
`work/biohub-forum-archive/` and `work/biohub-data-guide/`; a fresh checkout needs
separate data acquisition before executing the loaders. No private conversation
or machine-local file is needed to understand the committed handover.
