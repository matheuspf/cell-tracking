None of the ten learned division configurations qualifies for promotion. Each was
frozen in both source directions and freshly scored on all 199 clips, and every
configuration lost against v2 in both embryos. The strongest division result,
`D_existing_p020`, scored **0.9153811415049362**, a change of
**−0.018825173465404133** from the preserved v2 incumbent
**0.9342063149703403**. Its 40 additional division true positives came with 1,646
additional division false positives; it also added 204 edge true positives and
1,772 edge false positives. The primary image configuration `D_image_p020` scored
0.9137057658219115 (−0.020500549148428804), with 11 additional division true
positives, 617 additional division false positives, 499 lost edge true positives,
and 1,396 additional edge false positives. Replacement/suppression did not reverse
the failure: `D_replace_p020` scored 0.9136496117884901. These are complete measured
results, not estimates from classifier losses or selected clips.

Candidate availability has substantially more potential than these learned
policies realized. An annotation-built, legal, add-only source diagnostic on the
existing pool scored 0.9577352819660182 (+0.023528966995677858 against v2): 57 net
additional division true positives, one additional division false positive,
41 additional edge true positives, and 13 additional edge false positives. The
expanded-pool source diagnostic scored 0.9672051875911127
(+0.03299887262077239), with 110 division true positives and 94 division false
positives overall: net changes of +81/+2, respectively. Its edge changes were
+36 true positives and +34 false positives. Both diagnostics improved both embryo
aggregates. They use source annotation labels to select edits and retain fixed
incumbent forks; they are **evaluation-only feasibility heuristics**, neither
deployable policies nor global optimization bounds. The gap supports better event
selection as a priority; it does not establish that calibration alone would close
the gap.

The coverage/cost measurements explain why simply expanding the pool is not an
adequate next step. The base pool had 78,849,206 alternatives and covered 106 of
151 observed event instances. The predeclared source-coverage fallback grew to
190,139,632 alternatives, about 2.41 times as many, and covered 112 instances.
Source coverage changed from 20/26 to 20/26 for 44b6 and from 86/125 to 92/125 for
6bba. Both remained below the 85% diagnostic target. Of the expanded alternatives,
187,091,750 (98.40%) had unknown or incomplete source annotation support and were
correctly ignored rather than converted into negatives. The training table held
1,182 positive alternatives in 112 observed event bags and 337,841 sampled,
supported negatives. These bags are per-clip observations; independent biological
events cannot be inferred when clips may overlap.

The tabular objective assigns equal total risk to event and nondivision groups,
and the image batches draw equal numbers of positive event bags and supported
negative groups. Consequently, sigmoid scores are not calibrated probabilities
for the natural frequency of events in the much larger proposal population.
The decoder also applies a fixed 1.5 log-odds margin, owner costs, compatibility
constraints, and an edit cap. For example, the nominal p.20 arm initially requires
a raw score above approximately 0.528 before additional action costs; raw p.20
counts in the seed table are not counts of decoder-eligible or accepted edits.
All four source/seed image fits completed 10,000 actual CUDA optimizer steps each,
and every supported positive alternative was seen. Training took 140.735 seconds
of optimizer-loop time in total; short runtime is not evidence of a skipped fit.
The 25/50/100% event-group fits are source training diagnostics, not independent
validation or calibration studies.

Several explanations remain hypotheses rather than demonstrated causes. Equal
class weighting, sparse supported supervision, and the large unknown deployment
population plausibly contribute to overly permissive event scores. The observed
false-positive and edge damage support that concern, but no ablation isolates its
share from weak discrimination or embryo transfer. The compact image encoder sees
nine parent-centered triplanar 12×12 crops at 2 µm per pixel: a 24 µm field.
Some daughters admitted by the 16–20 µm proposal gates lie outside that context,
so their discrimination depends on geometry/native link evidence and scalar
daughter image features. This is a concrete representation limitation, not a
measured explanation of the entire loss. Across the two fixed image seeds, mean
absolute sigmoid-score difference was 0.03505 over candidate rows, with 9,083,867
raw p.20 threshold disagreements. This is probability variability, not a
confidence interval, a calibration result, or a count of independent events.

Accepted edit counts are not monotonic in the nominal threshold. Lower thresholds
can connect many alternatives into components exceeding the fixed 256-alternative
limit, causing complete component abstention. Higher thresholds can leave smaller
components eligible for optimization. Every clip was retained, including clips
with oversized components, and the no-op alternative and 2% changed-edge cap were
preserved. A separate selective NPZ reader reduced the dense decode peak from
1.74 GiB to 0.599 GiB while reproducing all ten dense graph files byte for byte,
with identical action ledgers and solver counts. This was an I/O improvement and
did not change the frozen decoder or models.

The next agent should prioritize the following work before increasing proposal
volume or repeating optimizer steps:

- Audit source-supported false-positive events and edge damage using the frozen
  edit ledgers and fresh official matches. Separate wrong parent/daughter pairs,
  timing/path errors, and donor displacement to identify what the model fails to
  distinguish. Preserve the distinction between refuted and unknown cases.
- Test event-group ranking and conservative selection on source controls whose
  biological overlap can actually be certified. If independent controls cannot
  be established, retain that limitation explicitly. Do not convert unknown
  alternatives into negatives or use the opposite embryo's annotation frequency
  as a deployment prior. A calibrated threshold is only one hypothesis to test.
- Test image context that includes the proposed daughters and relevant owners,
  retaining common prediction-centered sampling, temporal masks, and daughter
  permutation invariance. Keep the current source bags and masks available so
  representation changes can be distinguished from label changes.
- Reuse the compatible-edit feasibility evidence and the tested owner guard.
  Freeze every new source direction and bounded graph round before comparative
  scoring, and require complete gains against the actual v2 incumbent in both
  embryos before promotion.

The frozen label NPZ field named `inclusion` records a parent-group sampled
fraction, not an exact row propensity when hard and ordinary negative strata mix.
The delivered table names it `parent_group_sampling_fraction` and documents the
limitation. Model fitting never used that field or inverse propensity weighting,
so this metadata defect is not evidence for the measured classifier failure. No
additional variants, thresholds, or refits were introduced after the scores were
exposed. All evidence remains operational exploratory because the embryos and
public checkpoints have already been reused.

Local evidence: `event_measured_summary.json`, `source_event_oracle_measured.json`,
`source_event_oracle_manifest.json`, `event_candidate_coverage.json`,
`event_seed_disagreement.parquet`, `event_score_disagreement.json`,
`event_edit_ledger.parquet`, `event_io_loader_parity.json`, and
`event_lane_findings.json` under the isolated strong-tracker-v3 output root.
