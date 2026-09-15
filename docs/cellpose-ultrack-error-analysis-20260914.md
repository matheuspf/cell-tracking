**Latest result: temporal daughter-pair scoring plus one-frame daughter persistence scores 0.82416, versus 0.79986 for stock, but 0.82814 for its matched no-image control.** Evaluable false divisions fall from 219 to 22 versus stock, while true recovery remains one of five. The structural follow-up improves the optical-only variant by 0.00474 but does not establish a better division model than the simpler control. The six clips are a reused development panel, not independent validation.

The [current error audit](cellpose-ultrack-current-errors-20260914.md) separates the latest graph’s missed endpoints, rejected links and false temporal edges, with a fresh score comparison against the matched control and v3.

**On the same six complete clips, stock Cellpose + ultrack scores 0.79986, versus 0.97178 for selected v3.** The gap is 0.17192. The existing all-199 v3 score of 0.93480 is a different cohort. Public Harmonic Fusion scores 0.95765 on these same six clips.

This uses frozen Cellpose cpDINO-ViT-B and fresh official matching of full 100-frame graphs: 600 volumes, 4,141 annotated nodes, 4,026 annotated edges and five annotated divisions.

**Most of the original gap is temporal-edge accuracy.** Exact descriptive accounting assigns 0.09073 to excess edge FP, 0.05751 to missing true edges, 0.01163 to the change in node-count adjustment, and 0.01205 to the division bonus. The FP/FN terms average both substitution orders because Jaccard is nonlinear. They do not causally separate detector and linker effects.

Stock edge TP/FP/FN is 3,674/540/352 versus v3’s 3,923/91/103. The 352 missed edges comprise:

- 165 without an endpoint match in both raw Cellpose and selected graphs.
- 31 whose available raw endpoint matches are lost during selection.
- 115 with selected endpoint matches and a correct candidate edge that the solver does not choose.
- 35 pruned at the top-five IoU step, plus six excluded by the ten-nearest-hypothesis cap.

None of those 41 missing selected-endpoint candidates exceeds 15 µm. Widening the distance gate alone addresses none of them.

Raw annotated-node recall is 96.84%; after ultrack, 96.11%; v3, 98.91%. Of 540 evaluable false edges, 533 touch an unmatched endpoint; 262 of those endpoints remain within 7 µm of the expected GT node. Competing hypotheses or duplicates can affect matching, but a biological classification requires visual review. Missing-edge rates show no evident window-boundary spike: 8.43% at core boundaries versus 8.76% elsewhere. This is not a windowing ablation.

**Division recall needs improvement in both methods.** On this pilot, both recover one of five divisions, with 219 versus three evaluable false divisions. Stock’s 3,324 total predicted forks are not all known errors. Across all 199 clips, v3 recovers 29/151 annotated divisions with 92 FP; division Jaccard is 0.11934.

The refreshed [forum thread](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/740573) corrects its base-rate interpretation: sparse annotated track segments do not establish biological division prevalence. Displacement alone does not reliably separate divisions from continuations. Participants suggest temporal appearance and daughter-pair evidence; the small appearance probe is preliminary. Unknown branches must not automatically become negative training labels.

**Our stock objective favors extra branches.** For fixed selected nodes, a fork versus one continuation plus an interior new track differs by IoU(second link)^4 + division_weight − appear_weight. Both event weights were −0.001, so any positive extra overlap improves that local fork comparison. Five actual CBC tests confirm the sign and the effect of unequal event costs. Raising both penalties equally preserves this preference. [Solver source](/home/mpf/code/kaggle/ultrack/ultrack/core/solve/solver/mip_solver.py:150), [upstream tuning guidance](https://royerlab.github.io/ultrack/optimizing.html).

**Three controls were declared before new scoring.** The masks, hierarchy, candidate links, IoU values, 20-frame windows with overlap 5, eight CBC threads and 180-second per-window limit are unchanged:

- Extra division cost: division −0.011; appearance/disappearance −0.001. Its 0.01 surcharge needs IoU above approximately 0.316 in the local fixed-node comparison. This is an objective-scale control, not a fitted biological prior.
- Reference event costs: all three −0.1, copying only these weights from the pinned sparse-zebrafish configuration. This is not a paper replication.
- No-division diagnostic: division −1.01; appearance/disappearance −0.001. With IoU^4 bounded by one, replacing the extra branch with a new track strictly improves the objective. Zero exported forks are required; this is not a division model.

Measured full-panel results:

- **Stock ultrack: 0.79986197** (change +0.00000000); edge TP/FP/FN 3674/540/352; division TP/FP/FN 1/219/4; 3,324 total forks; 96,609 selected nodes.

- **Extra division cost: 0.82831501** (change +0.02845304); edge TP/FP/FN 3650/353/376; division TP/FP/FN 0/20/5; 153 total forks; 96,423 selected nodes.

- **Reference event costs: 0.78257932** (change -0.01728264); edge TP/FP/FN 3586/555/440; division TP/FP/FN 1/89/4; 1,050 total forks; 90,142 selected nodes.

- **No divisions (diagnostic): 0.82987357** (change +0.03001160); edge TP/FP/FN 3645/339/381; division TP/FP/FN 0/0/5; 0 total forks; 96,405 selected nodes.

**Temporal image evidence is now integrated into ultrack’s joint optimization.** An explicit binary variable identifies the actual parent and two chosen daughter edges; a fork must select exactly one compatible pair. The existing v4 C4 optical model is evaluated at the current region centers, using each observation’s preceding/current/following images. Its symmetric pair-versus-no-fork logit is divided by the original source model’s temperature, clipped to ±8 and scaled by 0.01. The comparison sets this term to zero with identical pair constraints and candidates. No geometric mitosis gate, top-k pair cap or target-label fitting was used. Pairs are discarded only when replacing their weaker edge by a new track provably improves both objectives.

- **Explicit pairs, no image: 0.82655625**; edge TP/FP/FN 3649/361/377; division TP/FP/FN 0/20/5; 153 total forks.

- **Temporal image pairs: 0.81941949**; edge TP/FP/FN 3630/391/396; division TP/FP/FN 1/29/4; 1,110 total forks.

Image-term score change versus the exact pair control: **-0.00713676**, with embryo changes 44b6 **-0.01965902**, 6bba **-0.00434043**. [Learned-division evidence and checks](../results/cellpose-ultrack-learned-divisions-20260914/comparison.json). Direct optical fits are routed from the opposite embryo; synthetic initialization/replay and original generator temperature remain inherited. This tests an adapter, not new source calibration or an independent biological posterior.

**The frozen optical extension fails its matched control.** It gains one division TP but adds nine division FP, loses 19 edge TP and adds 30 edge FP. The recovered event is 44b6_a21120c2 at frame 52; stock instead recovered the frame-zero event in 6bba_fe670320. The optical arm therefore demonstrates changed event recognition, not increased aggregate division recall. Its logits do not reliably rank errors below the recovered event: six evaluable false divisions have a larger image logit than that true division.

The [post-prediction branch audit](../results/cellpose-ultrack-learned-divisions-20260914/branch-audit.json) finds nine of 29 evaluable false divisions with no continuation on one daughter, versus none of the single recovered true division. Twenty false divisions already continue for at least one frame. Requiring two continuation steps would remove 12 of the current false divisions and the recovered true division; these are graph diagnostics, not rescored counterfactual predictions.

**A separately declared follow-up adds one-frame daughter persistence.** Every selected daughter must have a selected outgoing edge when a following movie frame exists. Clip-final daughters are exempt. Committed outgoing edges constrain right window boundaries, and daughter continuation is enforced for inherited forks at left boundaries. The final exported graph must satisfy the rule everywhere. This is motivated by the inspected pilot errors; it is not an unseen test or a duration sweep. The unchanged image evidence and pair bank are reused in both arms.

- **Persistence, no image: 0.82813536**; edge TP/FP/FN 3651/355/375; division TP/FP/FN 0/18/5; 128 total forks.

- **Image + persistence: 0.82415820**; edge TP/FP/FN 3635/376/391; division TP/FP/FN 1/22/4; 994 total forks.

Image-term score change with persistence held fixed: **-0.00397716**; 44b6 **-0.00860360**, 6bba **-0.00483719**. [Persistence plan](../results/cellpose-ultrack-persistent-divisions-20260914/plan.json), [results and checks](../results/cellpose-ultrack-persistent-divisions-20260914/comparison.json). The no-division control remains a diagnostic; suppressing all forks does not solve mitosis recognition.

Persistence improves the optical arm by **+0.00473872**, removes seven evaluable division FP, retains one TP, recovers five temporal edges and removes 15 edge FP. However, the optical term with persistence still loses 16 correct temporal edges and adds 21 false temporal edges versus its matched no-image control. These gains therefore do not justify replacing the simpler control. All 994 optical-arm forks obey the final-graph rule; 55 occur at the last transition, where future daughter persistence cannot be observed. All forks include unannotated cells and are not all known false divisions.

The forum screenshot uses earlier figure numbering: its sparse-zebrafish Figure 6 is Figure 4 in the [published paper](https://www.nature.com/articles/s41592-025-02778-0). The [linked implementation](https://github.com/royerlab/ultrack_supplementary/tree/14d24aa3cada922d2ff7ac5b8adf39df85658b44/configuration/sparse_zebrafish) uses a learned foreground/contour U-Net on the dense channel, supplying different hypotheses from our hard Cellpose masks. The published Figure 6 is a separate neuromast experiment using fine-tuned Cellpose, evaluated with a different metric. [External image metadata](https://public.czbiohub.org/royerlab/ultrack/zebrafish_embryo.ome.zarr/.zattrs) confirms Kaggle’s ZYX spacing of 1.625/0.40625/0.40625 µm. Same spacing does not establish independent embryo provenance.

**The remaining modeling gap is continuation identity and calibration at Cellpose proposals.** The temporal pair reward is implemented and tested; continuation links still use overlap. A further model should evaluate learned continuation scores at the actual Cellpose anchors and train division-versus-continuation-plus-birth competition on observed source-embryo events. Valid daughter proposals excluded by the top-IoU link shortlist also need coverage. Fit event costs on the same score scale; signed association logits require an identity transform, not a fourth power. See the repo’s [existing adapter design](focus-detector-ultrack.md).

For fitting, use observed continuations and divisions as supervised evidence while keeping unknown branches and track truncations unknown. Group crops and every learned component by embryo, and audit external-volume overlap before using it for transfer validation. These reused six clips cannot independently validate a division model. v3 has documented upstream label exposure, so this comparison cannot isolate architecture from training-data effects.

All 42 new complete graphs were scored afresh. Solver statuses: 1 FEASIBLE, 209 OPTIMAL. Largest relative gap: 0.0012. Eighteen real-CBC event, pair and persistence checks passed.

The first cache-reuse attempt omitted ultrack’s geometry metadata file and failed before optimization. Copying and verifying the original metadata fixed the runner without changing costs or candidate banks. Failure logs remain beside each run. No mask inference was rerun, and no selected baseline was replaced.

[Declared plan](../results/cellpose-ultrack-event-costs-20260914/plan.json), [all cost controls and validation](../results/cellpose-ultrack-event-costs-20260914/comparison.json), [matched error audit](../results/cellpose-ultrack-20260914/error-analysis.json), [all-199 v3 receipt](../results/strong-tracker-v3/fresh_selected_score_summary.json), [reproduction](../tools/cellpose_ultrack/README.md).
