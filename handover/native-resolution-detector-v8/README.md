# Native-resolution center detector v8

One handover, one execution plan. Branch: `handover/native-resolution-detector-v8`,
based on `main@fb5521629eb41c8c485b291a5bcf344944c113ae`.

Give local Codex with the user's selected GPT 6 Pro model:

```text
Read handover/native-resolution-detector-v8/CODEX_PROMPT.md and execute it.
```

[PLAN.md](PLAN.md) is the only research/execution specification; [study.json](study.json)
is its bounded experiment registry. The task is actual detector training and
validation, not another plan, tracker replacement, or broad package survey.

**Main operating grid: native ZYX = 64 x 256 x 256.** Keep acquired XY detail;
no upsampling to 256 cubed. Compare a small number of resolution controls, three
native architectures, and incomplete-label training regimes. Predict subvoxel
centers in physical coordinates before the final integer submission conversion.

Treat the incumbent checkpoint's provenance as **unverified for this study** until
specific checkpoint receipts establish otherwise. It is a comparison, never an
implicit teacher or initialization for the source-only primary runs. Maintain
separate detector-specific validation and operational full-tracker comparisons.

No ZIP, patch, activation step, additional handover, dataset download, or external
checkpoint is required for the primary experiment. Use the installed PyTorch stack.
Already available FOCUS/Spotiflow assets can support optional, separately labelled
arms; missing assets must not prevent native detector training. Preserve v1-v7,
the original predictor, all inputs, active jobs and unrelated user work.

Included `reference.py` and `test_reference.py` are small, tested coordinate,
masked-loss and provenance utilities, not a trained detector or production scorer:

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s handover/native-resolution-detector-v8 -p 'test_*.py' -v
```

No microscopy inference, training, or new score is claimed by this handover.
