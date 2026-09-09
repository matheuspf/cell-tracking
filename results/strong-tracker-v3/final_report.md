# Strong tracker v3 — measured results

Decision: **small_local_gain**. Selected **A_residual_m3.0**, score **0.934802374260586**, delta against v2 **+0.000596059290246** on all **199 clips**.

The preserved v2 incumbent independently rescored to **0.9342063149703403**. V1's 0.9117740142186423 is historical context only; the selected delta versus v1 is +0.023028360041944, and was not used for promotion.

- 44b6: 0.931664468721842, delta versus v2 +0.000052081944511.
- 6bba: 0.935221784097327, delta versus v2 +0.000696534227794.

Operational exploratory. Both embryos repeatedly reused; public upstream contamination. Unknown crop overlaps; no independent inner folds or confidence intervals.

## Measurements and decision

Selected edge TP/FP/FN: 123135 / 4965 / 5748. Division TP/FP/FN: 29 / 92 / 122. Predicted nodes: 4108943.

32 complete scored graph configurations, 6368 rows. Every scored graph gets fresh official node matching and division assignment. Aggregation checks the exact expected sample set, fixed per-clip estimates and official denominator weights.

## V300–V310: incumbent, residual census and conditional controls

All 199 incumbent file hashes, the original notebook hash and the selected v2 lock were verified before experiments. The metric source matches current upstream byte-for-byte; the exact pinned revision remains unchanged.

[Pinned official metric source](https://github.com/royerlab/kaggle-cell-tracking-competition/tree/075fc5f5a52d11077f9dc2b074644618f26939e2/src/tracking_cellmot) and [metric documentation](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/metrics.md) define the evaluation. The per-file local/upstream SHA checks are in metric_verification.json.

The preserved v2 incumbent has 98 missed divisions with both daughter lineages matched but no surviving fork, 18 with missing daughter candidates, four with competing assignment and two missing parent-side evidence. The census covers all 151 GT division observations, 92 division FP, 5,860 edge FN and 4,930 edge FP. It does not reuse the old census.

The small safe-division/smoothing grid and no-pruning/no-gap controls all keep motion relinking off. Original and canonical teacher coordinates are accounted separately, and inserted donor IDs are excluded when correspondence is unproven.

## V320–V350: local arbitration, events and image rescue

Association features were rebuilt at incumbent coordinates and adjacency. Opposite-embryo native-residual logistic and seven-leaf tree models use supported positive and contradictory edges; unknown sparse contexts stay masked. Local edits include displaced owners, fixed boundaries, no-op, an edit cap and bounded MILP conflicts. Frozen v2 model transfer is a named control.

Division supervision targets supported events and their pair/timing/path alternatives. The expanded pool allows alternatives excluding the current continuation, image-derived temporal anchors and candidate daughter persistence. The individual-owner guard was tested against the free-daughter/owned-daughter counterexample. Models and both source directions were frozen before comparative scoring.

The image-rescue lane tests native reappearance, small image-centroid refinements and secondary maxima. A synthetic test exposed flat-background false persistence in the initial R_image_local arm; it is retained as an invalid-for-promotion control. R_image_persistent adds actual foreground-contrast evidence for every temporal peak. The DeepCenter-confirmed arm uses cached heatmaps and local checkpoint queries for missing frames. Continuous centroid shifts are capped at 0.75 um before integer serialization; realized shifts in the initial arm reach 0.908403 um, which is recorded explicitly.

## Event coverage and actual training

- base, source 44b6: 20/26 supported events (76.92%); 40,449,946 alternatives, 202 supported positive alternatives and 40,085,086 masked unknowns.
- base, source 6bba: 86/125 supported events (68.80%); 38,399,260 alternatives, 778 supported positive alternatives and 37,557,314 masked unknowns.
- fallback, source 44b6: 20/26 supported events (76.92%); 97,871,958 alternatives, 244 supported positive alternatives and 97,013,846 masked unknowns.
- fallback, source 6bba: 92/125 supported events (73.60%); 92,267,674 alternatives, 938 supported positive alternatives and 90,077,904 masked unknowns.

The four image fits used **40,000 actual optimizer steps** in total, 10,000 per source/seed. Seeds are 20260909 and 314159. The compact nine-frame parent-centered image encoder was trained from scratch on event bags. Daughter geometry is permutation invariant; cached image embedding inference was checked against direct inference. Both source directions and trained model hashes were frozen before comparative event scoring.

The image objective samples equal positive/negative event-group risk. Logistic fits normalize positive bag weight and balance classes. Their scores are not calibrated event-prevalence posteriors. All supported positive alternatives remain in training; incomplete sparse contexts are masked. The 25/50/100% source positive-group diagnostic fits are source resubstitution evidence, not independent validation. Secondary seeds were trained and compared at the probability level; they do not add a retrospectively selected deployment configuration.

The source-label audit found that the frozen negative-sampling inclusion field records a parent-group sampling fraction, not the exact row inclusion probability when hard and ordinary strata coexist. It was not used by training. The exported table names that field parent_group_sampling_fraction; it must not be used as an inverse-probability weight or calibration propensity. The audit preserved the original label/model fingerprints.

- 44b6_image_20260909: 10,000 steps, 32.5 seconds, 20 positive groups, 244/244 positive alternatives observed by the optimizer.
- 44b6_image_314159: 10,000 steps, 33.2 seconds, 20 positive groups, 244/244 positive alternatives observed by the optimizer.
- 6bba_image_20260909: 10,000 steps, 42.7 seconds, 92 positive groups, 938/938 positive alternatives observed by the optimizer.
- 6bba_image_314159: 10,000 steps, 32.3 seconds, 92 positive groups, 938/938 positive alternatives observed by the optimizer.

These are observed event bags, not certified distinct biological divisions across overlapping crops. Candidate counts measure search cost; they do not increase the number of independent positive events.

The wider pool costs 2.41× as many alternatives for 6 additional covered event observations (106 → 112). Of its 190,139,632 alternatives, 187,091,750 (98.40%) have unknown sparse labels and remain masked. This is candidate-label availability, not biological division prevalence.

## Association measurements and provenance

The stricter frozen residual setting **A_residual_m3.0** scored **0.9348023742605860**, **+0.0005960592902456 versus v2**. Its embryo deltas were +0.000052081945 for 44b6 and +0.000696534228 for 6bba. It recovered 112 new GT-edge identities and lost 0; it introduced 35 FP edges and removed 0. Its 2,904 accepted local actions changed 2,970 edges. Full-graph edge TP/FP/FN were 123,135 / 4,965 / 5,748.

The predeclared primary margin, **A_residual_m1.5**, scored 0.9348015373447275 (+0.0005952223743871 versus v2): 161 GT edges gained, 2 lost, 88 FP introduced and 2 removed. Both seven-leaf tree margins scored 0.9342137250752032 (+0.0000074101048628 versus v2; one extra TP, unchanged FP). Frozen v2 model transfer scored 0.9342513494629585 (+0.0000450344926182 versus v2; six extra TP, unchanged FP). Fresh node matching was identical in all five association variants, and each clip's division TP/FP/FN counts stayed unchanged. All five kept 4,108,943 nodes and pooled division counts 29 / 92 / 122. Choosing between these frozen settings remains exploratory; the small margin difference is not independent generalization evidence.

Current-center image, geometry and adjacency features covered 7,050,250 candidate pairs: 4,038,813 native/teacher pairs plus 3,011,437 additional image neighbors. The inference-only disagreement table has 241,181 rows and the five-variant edit ledger has 10,373 accepted actions. Source 44b6 contributed 19,316 supported positive and 33,982 contradictory negative edges from 71 clips; source 6bba contributed 105,451 positive and 84,590 negative edges from 128 clips. Unknown sparse contexts were masked. Both directions were frozen before comparative scoring.

The source-44b6 residual optimizer converged in 187 L-BFGS iterations (source loss 0.099117, AP 0.981118). The source-6bba fit reached its frozen 250-iteration cap with optimizer success=false, while retaining finite source loss 0.091551 and AP 0.993055. It was not refitted after outer scoring. These source-fit metrics measure resubstitution and do not establish calibration or held-out accuracy.

Historical teacher contamination limits the validation claim further. The reused v2 E_hgb teacher for a given source embryo was itself fit using the opposite embryo's annotations in v2. Thus target-embryo label information can already be present upstream of a v3 selector whose direct labels are source-only. Reused public neural checkpoints have their own embryo exposure. Frozen v3 source directions and annotation-free inference do not turn these operational comparisons into clean out-of-fold validation.

The source-resubstitution audit used the predeclared first five clips per source without changing any models or settings. At margin 3.0 it gained 2 and 1 GT edges on source 44b6 and 6bba, respectively, with no TP losses or additional FP. At margin 1.5 it gained three GT edges on each source, introducing three FP on 6bba. This small in-sample diagnostic is not an independent estimate of source bias.

Teacher identity was proven through raw/pre-ILP native IDs and timestamps; unproven inserted donor IDs were excluded. The edge census separately measured each original teacher, its edges on the incumbent node universe at donor centers, and those same edges at incumbent centers. The following TP counts therefore separate changes in node universe from changes in coordinates; they are edge diagnostics, not combined-score or division claims.

- raw_neural: original TP 122,654 → projected at donor centers 122,202 → canonical TP 122,552; unmapped nodes 3.531%, unmapped edges 2.929%; canonical GT edges versus v2: 66 gained, 537 lost.
- old_final: original TP 122,201 → projected at donor centers 121,337 → canonical TP 121,227; unmapped nodes 2.247%, unmapped edges 3.156%; canonical GT edges versus v2: 302 gained, 2,098 lost.
- v2_E_native: original TP 122,403 → projected at donor centers 121,540 → canonical TP 121,430; unmapped nodes 2.247%, unmapped edges 3.151%; canonical GT edges versus v2: 312 gained, 1,905 lost.
- v2_E_hgb: original TP 122,556 → projected at donor centers 121,694 → canonical TP 121,577; unmapped nodes 2.247%, unmapped edges 3.137%; canonical GT edges versus v2: 302 gained, 1,748 lost.

Of 1,634 old-final TP predicted pairs absent as TP pairs in v2, 625 had the same GT truth recovered through another predicted pair; 1,009 were genuinely missing GT edges. On the fixed incumbent universe, changing old-final centers to incumbent centers gained 206 GT edges and lost 316. Historical pair counts alone therefore overstate transferable association headroom.

The label-informed source association feasibility oracle scored **0.9386868070775238**, **+0.0044804921071835 versus v2**. It added 383 net edge TP and removed 299 FP, but pooled division TP/FP/FN changed from 29 / 92 / 122 to 28 / 93 / 123. This is a source-label heuristic feasibility result, not a global upper bound, deployable configuration or promotion candidate.

The exact division-regret audit found one early split whose two immediate fork edges stayed fixed while a downstream daughter continuation changed. Removing an edge counted globally as FP removed valid evidence within the official division timing window: the affected clip lost one division TP and gained one division FP and FN, despite one fewer edge FP. The implemented freeze protects immediate fork edges; it does not protect every downstream path used by the official division scorer. This diagnostic did not trigger another variant or a source/target refit.

The corrected contrast-gated image-rescue arm scored 0.9342333345063806 (+0.0000270195360402 versus v2), with embryo deltas +0.000249256418 and -0.000015168630. It fails the nonnegative-both-embryos promotion gate. Correcting the flat-background bug established valid candidate evidence but did not establish a qualifying standalone gain.

The aggregate evidence is preserved in [association_summary.json](association_summary.json), [association_regret_summary.json](association_regret_summary.json), [teacher_reconciliation_summary.json](teacher_reconciliation_summary.json), [association_source_bias_summary.json](association_source_bias_summary.json), [source_association_oracle_score_summary.json](source_association_oracle_score_summary.json) and [source_oracle_division_regret_summary.json](source_oracle_division_regret_summary.json). Detailed identities, coordinates and post-hoc labels remain local and are excluded from the selected inference package.

## Source-only oracle feasibility

These probes read source annotations and cannot run as inference. They are legal heuristic constructions, not global bounds. Both non-oracle prediction directions were frozen before the retrospective event-oracle diagnostics. Their deltas still use the v2 incumbent.

- source_existing_pool, 44b6: 0.947842231972 (+0.016229845195 versus v2); division TP/FP/FN 15/23/11.
- source_existing_pool, 6bba: 0.959873636403 (+0.025348386534 versus v2); division TP/FP/FN 71/70/54.
- source_existing_pool, pooled: 0.957735281966 (+0.023528966996 versus v2); division TP/FP/FN 86/93/65.
- source_expanded_pool, 44b6: 0.955596609059 (+0.023984222282 versus v2); division TP/FP/FN 19/23/7.
- source_expanded_pool, 6bba: 0.969751945872 (+0.035226696002 versus v2); division TP/FP/FN 91/71/34.
- source_expanded_pool, pooled: 0.967205187591 (+0.032998872621 versus v2); division TP/FP/FN 110/94/41.
- source_association_oracle, 44b6: 0.939139585410 (+0.007527198632 versus v2); division TP/FP/FN 7/22/19.
- source_association_oracle, 6bba: 0.938416327582 (+0.003891077712 versus v2); division TP/FP/FN 21/71/104.
- source_association_oracle, pooled: 0.938686807078 (+0.004480492107 versus v2); division TP/FP/FN 28/93/123.

## Measured event, rescue and combination outcomes

Every event arm failed the both-embryo promotion gate. The best pooled event arm, D_existing_p020, changed division TP by +40 and FP by +1646; its delta against v2 is -0.018825173465. The oracle contrast establishes useful candidate feasibility on the observed labels, while the learned policies fail to select those events reliably. Class-balanced uncalibrated scores and largely unlabelled candidate contexts are plausible contributors; these results do not isolate their causal contributions.

### Division

- D_existing_p020: 0.915381141505, delta versus v2 -0.018825173465; 44b6 -0.027995120338, 6bba -0.016820266674; division TP/FP/FN 69/1738/82; does not pass promotion gate.
- D_tabular_p020: 0.915192603176, delta versus v2 -0.019013711795; 44b6 -0.022519101726, 6bba -0.018225180580; division TP/FP/FN 44/697/107; does not pass promotion gate.
- D_tabular_p040: 0.914760464263, delta versus v2 -0.019445850707; 44b6 -0.025578017956, 6bba -0.018062786818; division TP/FP/FN 45/704/106; does not pass promotion gate.
- D_replace_p040: 0.914740620691, delta versus v2 -0.019465694280; 44b6 -0.028146097518, 6bba -0.016419240829; division TP/FP/FN 40/709/111; does not pass promotion gate.
- D_image_p040: 0.914698317260, delta versus v2 -0.019507997711; 44b6 -0.028096952961, 6bba -0.016480898131; division TP/FP/FN 41/718/110; does not pass promotion gate.
- D_tabular_p010: 0.914467629518, delta versus v2 -0.019738685452; 44b6 -0.025206012195, 6bba -0.018566264762; division TP/FP/FN 39/656/112; does not pass promotion gate.
- D_image_p010: 0.914113169802, delta versus v2 -0.020093145168; 44b6 -0.028975999128, 6bba -0.017455609921; division TP/FP/FN 42/663/109; does not pass promotion gate.
- D_image_p020: 0.913705765822, delta versus v2 -0.020500549148; 44b6 -0.030150826485, 6bba -0.017593626846; division TP/FP/FN 40/709/111; does not pass promotion gate.
- D_replace_p020: 0.913649611788, delta versus v2 -0.020556703182; 44b6 -0.030101066003, 6bba -0.017718656003; division TP/FP/FN 39/705/112; does not pass promotion gate.
- D_existing_p005_control: 0.912492020446, delta versus v2 -0.021714294524; 44b6 -0.034671755783, 6bba -0.019017765488; division TP/FP/FN 54/1442/97; does not pass promotion gate.

### Rescue

- R_image_persistent: 0.934233334506, delta versus v2 +0.000027019536; 44b6 +0.000249256418, 6bba -0.000015168630; division TP/FP/FN 29/91/122; does not pass promotion gate.
- R_image_local: 0.934227369670, delta versus v2 +0.000021054699; 44b6 +0.000236319928, 6bba -0.000019842356; division TP/FP/FN 29/91/122; invalid control.
- R_heatmap_local: 0.934206314970, delta versus v2 +0.000000000000; 44b6 +0.000000000000, 6bba +0.000000000000; division TP/FP/FN 29/92/122; does not pass promotion gate.
- R_native_restore: 0.934185864844, delta versus v2 -0.000020450126; 44b6 -0.000005782611, 6bba -0.000023175174; division TP/FP/FN 29/92/122; does not pass promotion gate.

### Combinations

- ADR_primary: 0.914555779005, delta versus v2 -0.019650535966; 44b6 -0.029343145218, 6bba -0.016645133803; division TP/FP/FN 42/706/109; does not pass promotion gate.
- AD_primary: 0.914476767786, delta versus v2 -0.019729547184; 44b6 -0.029333889292, 6bba -0.016799584990; division TP/FP/FN 41/708/110; does not pass promotion gate.

## V360: combinations and regret

Combination inputs are regenerated after association edits. Component scores are not added together. The selected regret audit distinguishes changed predicted pairs from recovered GT-edge identities using each full graph’s fresh matches. The local edit_ledger.parquet preserves prediction-time actions; evaluation/edit_effects.parquet and stage/variant regret tables remain separate. Per-action edge attribution requires unchanged nodes and matching plus exact reconstruction of the scored stage graph. Division credit and combined-score changes are reported for whole graphs, not allocated additively to individual actions.

Recovered GT-edge identity changes for the selected graph: 112 new, 0 lost. Predicted TP-pair changes: 112 new, 0 lost.

The five lowest per-clip adjusted-edge changes are listed below. This ordering is diagnostic: division Jaccard is pooled, so these changes are not additive per-clip combined-score contributions or independent biological replicates.

- 6bba_acd782a8: adjusted-edge delta -0.002515613; edge TP +0, FP +1; division TP +0, FP +0.
- 6bba_767a1e17: adjusted-edge delta -0.002231303; edge TP +0, FP +2; division TP +0, FP +0.
- 6bba_fc83837d: adjusted-edge delta -0.001737809; edge TP +0, FP +2; division TP +0, FP +0.
- 6bba_07477033: adjusted-edge delta -0.001593714; edge TP +0, FP +1; division TP +0, FP +0.
- 6bba_05db0fb1: adjusted-edge delta -0.001393371; edge TP +0, FP +2; division TP +0, FP +0.

## V370: inference, runtime and portability

The actual notebook neural and repair code was copied into an isolated v3 entry. The first two full 100-frame pilots were selected by median image contrast, one per embryo, before their scores were inspected. Both reproduced raw and repaired cached graphs exactly with annotations unavailable from process start.

All 199 selected fresh-image graphs received an additional independent official evaluation. Pooled and both embryo counts/scores agree with the selected cached experiment. This reproduction does not add a new inference policy to the 32-configuration experiment budget.

The final copied inference package also ran both preselected image pilots through its own run.sh from outside this checkout. Both selected graphs and CSV round trips matched, with annotations unavailable from process startup and zero cached graph inputs for prediction. The package smoke receipt pins its code, policy and dependencies.

The full fresh teacher audit found 8 old-teacher coordinate differences: 6 pre-existing bounds corrections and 2 documented half-integer serialization cases. There are zero unexplained changes. All old/E teacher edges and every current native feature array match exactly across 199 clips; the final selected graphs also match. The original old-teacher node arrays are therefore not claimed to be byte-identical.

The full fresh neural/baseline pass took 101.90 minutes of measured wall time. Per-clip neural/I/O time was median 47.08 seconds (p90 52.02); graph ILP/export was median 11.12 seconds (p90 29.30); teacher/native-feature reconstruction and decoding was median 60.25 seconds (p90 104.84). These stages overlapped across processes, so their sums are not end-to-end wall time.

The two copied-package pilots took 168.92 seconds concurrently. The separate warm-filesystem startup probe took 7.46 seconds; this is not a cold-cache measurement. See [fresh_delivery_receipt.json](fresh_delivery_receipt.json), [package_smoke_receipt.json](package_smoke_receipt.json) and [fresh_startup_benchmark.json](fresh_startup_benchmark.json) for exact timing boundaries and dependency hashes.

Observed peak study RSS 28.70 GiB; GPU memory 8.48 GiB. Full timing, source hashes, graph hashes and artifact dependencies are included in the local manifests.

See [dashboard.html](dashboard.html), [score_rows.csv](score_rows.csv), [operating_points.csv](operating_points.csv), [event diagnostic](event_diagnostic.md), [winning_config.json](winning_config.json), [NEXT_AGENT.md](NEXT_AGENT.md) and [reproduction instructions](../../docs/strong-tracker-v3.md).

Raw microscopy, detailed annotation evidence, crop tensors, graph predictions and weights stay outside Git. The complete local v3 root plus v1/v2 stores, raw competition inputs, official scorer, notebook code and local model weights are needed on another host. No external artifact backup or persistence is implied.
