# Local execution prompt

Execute `handover/annotation-selection-v1/PROTOCOL.md` in this repository.
You are the local implementation-and-experiment agent, using the user's selected
GPT 6 Pro model. You have the competition data and one RTX 4090. This is an
execution request, not a request for another handover.

First read root `AGENTS.md`, relevant `.agents/skills/competition-*` files,
`docs/competition.md`, `docs/notebooks.md`, and all files in this handover.
Inspect the local reference pages and actual data. Preserve uncommitted user work,
raw inputs, downloaded notebook originals, credentials and existing environments.
Do not run a reset, clean, broad bootstrap, or full data download.

Your goals are to replace the hypothetical numbers from the discussion with real
measurements: annotation counts/coverage, baseline count ratios, achievable
annotation recall and correlation at useful keep budgets, and actual score gains
or losses after graph filtering. Establish the difference between matching sparse
GT, being a real cell, and being selected by an annotator.

Implement the missing modules in IMPLEMENTATION.md and run S000 through S110.
Start with metadata and cheap clean candidates; use mirrored public models only
in a separately labeled provenance-audited lane. Run a bounded image probe on
the 4090 even when tabular features are inconclusive, unless an integrity or
resource blocker makes it invalid. Do not replace the central question with
an unrelated general tracker search.

Freeze splits and experiment choices before outer score revelation. Train and
tune each direction using only its source embryo's labels; freeze both directions
before reading either direction's outer outcomes. Protect against checkpoint
contamination, crop overlap, GT-derived features, and post-hoc threshold selection.
The metadata-only overview can run first; defer label-versus-feature maps of outer
embryos to the final reporting phase. Never use visible test copies as validation.

Every scored graph variant must be matched and division-evaluated afresh with the
pinned official implementation. Verify the full expected sample list and do not
silently skip samples or missing count estimates. The helpers in this directory
are arithmetic checks, not evidence that the official pipeline has been reproduced.

Choose concrete adapters and dependency fixes yourself from the installed code;
record decisions and rerun tests. Keep changes additive and resumable. Do not ask
for another plan or stop after scaffolding. Integrity failures block affected
claims, not independent data audits and cheap diagnostics. Missing manual labels
must not block the automated study; report the remaining identifiability limit.

Write commands, versions, model/data/split hashes and stage results incrementally
under ignored work/output directories. Produce the final Markdown report,
self-contained HTML dashboard, CSV/JSON tables, plot data, reproduction commands,
and a local artifact manifest even if the hypothesis fails. Fill STATUS.json
truthfully and commit only sanitized code/configs/summary results after checking
Git's staged file list. Do not upload a Kaggle submission, publish notebooks,
post to the forum, merge the PR, or push data/weights. Do not initiate paid model
API calls or rent hardware.

Success is a reproducible answer supported by proper data, including a negative
answer. Do not claim a hidden-test gain, exact all-cell prevalence, or another
competitor's use of this idea without evidence.
