# Model and implementation contract

## 1. Interfaces and upstream boundary

Implement `tools/division_reliability_v11/` with small explicit modules for reconciliation, provenance, source data, upstream execution, action banks, labels, models, fitting, calibration, graph inference, evaluation and reporting. Reuse audited pure routines, not old experiment-global loaders. The planned interface is:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=tools:. python -m division_reliability_v11 reconcile --config handover/division-reliability-v11/study.json
PYTHONNOUSERSITE=1 PYTHONPATH=tools:. python -m division_reliability_v11 preflight
PYTHONNOUSERSITE=1 PYTHONPATH=tools:. python -m division_reliability_v11 pilot
PYTHONNOUSERSITE=1 PYTHONPATH=tools:. python -m division_reliability_v11 lock
PYTHONNOUSERSITE=1 PYTHONPATH=tools:. python -m division_reliability_v11 run --resume
PYTHONNOUSERSITE=1 PYTHONPATH=tools:. python -m division_reliability_v11 report
```

Also expose resumable `train-upstream`, `prepare-actions`, `fit-linear`, `train-compact`, `calibrate`, `predict`, `freeze`, `evaluate`, `validate`, `package`, and `infer --package PATH --images PATH --output PATH`. Source, seed, config and data root are explicit. These are interfaces to implement, not existing executable commands at authorship.

C00 follows the retained v10 upstream recipe exactly: random entire two-frame TemporalUNet3D/association family, channels 32/64/128, feature width 32, XY stride four with native-origin bookkeeping, fixed frame-local normalization, sparse masked center supervision, conservative background, supported incoming association labels, independent edge logits, and one-to-one continuation assignment with explicit nulls. Read the full inherited section rather than reconstructing it from this summary. It is a newly trained clean reference, not a reproduction of the public 0.946 notebook. No all-data checkpoint may be loaded even transiently.

At source-only preflight, inspect architecture source, import side effects, actual batch-8 memory, precision and loss implementation. Missing local architecture code must be resolved from its documented public source with provenance before training; do not silently switch architecture or use weights to supply it. Missing competition data is a concrete execution blocker, not a license to synthesize benchmark results.

## 2. Prediction-only complete actions

Adapt `pipeline_error_training/bank.py` and `actions.py` to explicit C00 graphs/features. Keep the actual frozen forward/reverse neighbor union, geometry gates, paths, ownership closure and bounded atomic solver. Snapshot the full ProposalConfig. Never generate deployment anchors from GT or use GT coordinates as proposals. Recompute every required edge score with the frozen clean encoder at the actual points, including event edges outside the original continuation bank. A missing score is an explicit invalid/unavailable action, not a high-confidence default or zero-valued fake measurement.

Every fork action contains its complete added/removed edges, donor continuation or termination, births, daughter continuation evidence and conflict resources. A group is one parent-side decision neighborhood. Use one keep alternative with utility zero; include explicit legal no-fork competitors. Their utility matters even though v11 only accepts fork edits. Canonicalize identical complete edits by sorted added/removed persisted edges and actual fork ID before scoring; choose one deterministic representation. Different daughter/time proposals that produce different graphs remain distinct. Deduplication must not erase multiple compatible GT timing labels in training.

No new observation coordinates or global continuation residual are learned in this study. C01/C11 independently edit C00, never each other. Existing fork protection applies when replaying historical graphs but does not create fictional forks in continuation-only C00. Keep the 2% changed-edge cap, full ownership closure, component bound and conservative solver-timeout abstention. Log all rejected actions and do not treat a resource cap as an accuracy result. Use a fixed 4096-unique-action parent cap: discovering a 4097th legal fork abstains for the whole parent. Report a lower-bound count, never an exact census. Such incomplete groups are excluded from supervised negative labels and calibration; their abstention remains in full-graph evaluation.

## 3. Sparse labels and sampling

Reuse the distinction in `pipeline_error_training/labels.py`: biological compatibility, supported identity contradiction, and official fork-risk evaluability are separate fields. Independent official local-window division matching supplies compatible positive forks. A matched parent with one annotated child may supply an official false-fork target; it does NOT prove biological nondivision. Unannotated objects and alternative second daughters remain unknown biologically.

For a group's legal fork actions, let risk labels be 1 for a supported compatible division, 0 for an officially evaluable incorrect fork, and -1 for unknown. The **occurrence target** is 1 when at least one fork has label 1; it is 0 only when the nonempty full fork set is entirely label 0; otherwise it is unknown. This is a supported metric-risk task, not an identified biological mitosis posterior. Do not label a group negative when it still contains an unknown legal fork. Censored or unreachable positive events are diagnosed separately.

Conditional action ranking trains only on positive-containing groups, with all supported compatible fork actions as a positive set. Unknown actions do not enter supervised loss denominators. At inference all legal actions enter the denominator; report this censored-training/deployment difference rather than pretending annotations identify the whole field.

Split and group using the inherited v10 source protocol. Build fit banks from complete source-fit clips only. Preserve positive event identity across timing-compatible groups; weight one biological event once in training, sampling one compatible anchor when appropriate. Source calibration instead uses the actual deployed-parent census, not a fabricated estimate of dense event prevalence. Neither source grouping nor multiple seeds creates independent embryos.

C11 effective batch is 32 accumulated groups. First 10% of E updates: 32 supported identity groups. Remaining updates: 16 identity, 8 positive-event, 4 uniform supported-negative event and 4 hard-negative event groups. Mine once at 50% of E, using only frozen current scores on source-fit data; before that use uniform supported negatives in the hard slots. Mine by parent neighborhood, not top individual rows, and balance clips. Negative occurrence groups must satisfy the complete-support rule above. Unknown high-score groups are logged, never promoted to negatives. Empty hard pools reuse uniform negatives with counts; no positive pool is a named head-training blocker.

Keep inclusion and within-group selection probabilities, unique group visits, annotation support and candidate multiplicity in receipts. Image augmentation consistency is label-free and reported separately; all-unknown batches have zero supervised loss. Do not manufacture a positive keep target simply to make listwise loss nonempty.

## 4. Shared features and C01 baseline

Define a named/versioned scalar schema from C00 and raw images: parent/daughter detection confidence; physical parent/daughter, sister and barycenter geometry; available history velocity; daughter persistence/separation and validity masks; local density; added/removed clean edge utilities; donor, birth and termination counts; and true image/time boundary support. Do not include teacher votes, GT matching, component IDs, annotation density, coarse GT node totals, source ID or filename features. Unavailable history/image values have explicit masks. Record all names and transforms in the lock; C01 and C11 use identical scalars.

Parent features use permutation-invariant mean/max summaries of available daughter/action scalar evidence, parent evidence and log(1 + unique action count). Compute normalization on source-fit only with standard-deviation floor 1e-4 and clip standardized values to [-10,10]. Image contrast/intensity features are fixed deterministic measurements, not an inherited neural representation.

C01 has two regularized linear models: occurrence logit a_g on parent features and action logit b_ga on action features. Use weighted logistic occurrence loss and masked compatible-set conditional loss, respectively; L2=1, unpenalized intercept, at most 1000 L-BFGS iterations, fixed initialization zero. Normalize each objective by summed eligible group weights. Log convergence and gradients; a nonconverged fit is not a claimed optimum. No hidden class-prior multiplier or parameter sweep. Source pools and sparse masks match C11; these CPU fits may use the complete supported fit census rather than neural minibatches, which must be disclosed.

C01 is a useful learned-fork control, not an architecture-matched control. If it already beats C11, prefer that evidence over automatically selecting the larger model. Its seed variation comes from the corresponding upstream outputs, not invented optimizer randomness.

## 5. Compact factorized model C11

Reuse the compact triplanar encoder family from `pipeline_error_training/models.py`: 3-plane 12x12 frame input, the audited three-frame compact view with actual validity masks, and a 128-dimensional temporal representation. Adapt pure crop code to C00 coordinates. Do not read P0-indexed historical crops. Verify physical support and frame origins against the original crop adapter on real source examples; input dimensions alone are not proof of equivalent preprocessing.

Initialize fresh per source/seed. The occurrence head is MLP [input,128,64,1] on parent embedding, masked mean/max daughter embeddings and parent scalars. The action head is MLP [input,128,64,1] on parent embedding, symmetric daughter sum/absolute difference, and complete-action scalars. Symmetry must also hold for all scalar feature ordering. An auxiliary identity head uses supported clean edge labels during fitting only; it never changes the frozen C00 association logits at inference.

Loss = occurrence BCE + compatible-set conditional ranking + auxiliary incoming-identity BCE + 0.1 same-image augmentation consistency. Average only eligible groups for each term; log their independent denominators. Negative-only groups have occurrence gradients even without a positive ranking action. There is no second, independently added metric-risk logit: occurrence already targets supported fork risk, avoiding an uncalibrated sum of two arbitrary event scores.

AdamW lr=3e-4, weight decay=1e-4, gradient norm clip 1. E is 4000 or 2000 chosen before fitting. Warmup W=min(200,floor(0.05*E)), followed sequentially by cosine to 1e-6. Do not multiply warmup and a concurrently running cosine. Final checkpoint only; save optimizer, scheduler, scaler, all RNG and sampler/mining state for exact resume. Log actual supervised-event update count. Unknown labels cannot be made negative to improve training loss.

## 6. Common action utility and calibrated deployment

For each group g, deduplicated legal fork a, and model logits a_g and b_ga:

```
conditional_log_probability(g,a) = b_ga - logsumexp(b_g over ALL legal fork actions)
structural_utility(a) = sum(clean_logits_added) - sum(clean_logits_removed)
                        - 2 * introduced_interior_births - 2 * introduced_interior_terminations
fork_utility(g,a) = calibrated_occurrence_logit(g) + conditional_log_probability(g,a)
                    + structural_utility(a)
no_fork_utility(g) = max(0, structural_utility of each complete legal no-fork competitor)
gain(g,a) = fork_utility(g,a) - no_fork_utility(g) - margin
```

Birth/termination costs apply only to changes introduced by the complete edit; real clip-boundary costs are zero. They are fixed modeling choices, not biological probabilities. Shared unchanged edges cancel. No-fork alternatives remain competitors, not unsolicited new continuation edits. Accept only positive-gain fork actions through the unchanged atomic conflict solver. Normalization is a proposed multiplicity-aware scoring design, not a posterior guarantee.

No per-row probability threshold is substituted for this graph objective. All legal actions, including unknown-at-training actions, participate in inference normalization. Stream exact logsumexp if needed, using stable arithmetic. Canonical duplicates cannot change probabilities or the graph; candidate order cannot change them. Oversized groups abstain with a recorded reason instead of silently truncating a denominator. Candidate gates are the same for C01/C11 and frozen before fitting.

Calibrate occurrence a_g/T + b on complete source-calibration deployed-parent groups with known occurrence labels. One deployed parent is one calibration unit; retain counts of correlated positive anchors. L2 penalty 0.01 on log T and b; bounds log T in [-2,2], b in [-12,12]. The conditional action temperature remains one. With fewer than two distinct supported positive events or 100 negative groups, freeze T=1,b=0 rather than fitting unsupported parameters.

For each model/source/seed, evaluate margins [2,4,6,8] through full source-calibration graphs with fresh official matching. Eligibility: combined score delta versus C00 >=0; adjusted edge contribution delta >=-0.001; newly introduced evaluable fork FP <=max(2,2*newly recovered division TP). Use event identities, not net TP. Rank by combined gain, ties within 1e-9 prefer larger margin. Under insufficient calibration support, test only margin 6. No safe margin means explicitly disabled edits, not an undisclosed baseline substitution. Raw model ranks/selected-tail errors must still be reported.

Report raw occurrence reliability, conditional rank, number of alternatives, highest-score tails, unknown-support rates, accepted actions and post-solver official outcomes separately. A supported-group census is not representative dense biological ground truth. Safety is established only on the inspected source subset; target behavior remains measured rather than assumed.

## 7. Training/inference correctness and implementation scope

The standard-library contracts in this handover define labels, score algebra, schedule endpoints and strict ancestry checks. They do not implement crop extraction, training, PyTorch gradients, official scoring or process isolation. Local Codex must test those independently with actual inputs.

Use pre-import read guards plus semantic parent manifests, including symlink/directory-FD/subprocess regression tests. A Python audit hook is not a complete security sandbox. Cold inference accepts only approved packages and images and must reproduce scored graphs after integer export. Do not read historical outputs to make the comparison match.

Preserve exact source hashes and versions for reused code. A parity-approved cache must be lane/source/seed/model/input-hashed. Never cache trainable embeddings as constants to make neural training cheaper. Every reported score must have a complete population manifest and a tested graph-to-CSV-to-graph path.
