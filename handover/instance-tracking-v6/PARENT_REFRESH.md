# Parent refresh before publication

The source branch advanced during authoring. This v6 branch starts from
`handover/image-native-tracking-v5@fe00326ea6ca79a5b5c54910de03925d353e0629`,
not the older initial review snapshot. All newer v5 files are inherited unchanged.

The initial REVIEW.md, SOURCES.md and baseline.json record evidence at
`0a3105a8f9d1b0954707675b42b9868832df4ac1`. Their historical measurements remain
valid; this addendum updates the parent state and takes precedence for completeness.
Read it with CODEX_PROMPT.md before M600. No v6 model/data experiment ran here.

The refreshed v5 CONTINUATION.md reports eleven complete configurations, including
two GT-assisted oracles. It is still explicitly an in-progress study. The added
199-clip results are:

| Added complete configuration | Pooled score | Delta versus C0 |
|---|---:|---:|
| H_probe_native_J | 0.9125505234745164 | -0.02225185078606962 |
| N_head_J | 0.8817718421042569 | -0.05303053215632914 |
| P_image_ablation | 0.8697528094106733 | -0.06504956484991264 |

N_head_J recovers 56 division TP but has 693 division FP; edge TP/FP/FN are
118,887/7,352/9,996. The other new comparisons also regress. C0 remains
0.934802374260586; no new adopted incumbent is documented in this snapshot.
The HOCT unit-contract limitation is unchanged, and full native N2/replication
and final multi-clip inference work still have pending stages. One fresh clip has
input-fingerprint parity for new paths, not a complete final six-clip graph report.

These additional results reinforce the need to separate object extent, candidate
population and decoder effects. They do not evaluate independent FOCUS/Cellpose
masks or Ultrack's mask hierarchy. The v6 scientific plan and tests are unchanged.
Do not stop, restart or mutate the active v5 jobs. M600 must inspect actual local
state and use a separate worktree/resource schedule when needed.

Verified source:
https://github.com/matheuspf/cell-tracking/blob/fe00326ea6ca79a5b5c54910de03925d353e0629/handover/image-native-tracking-v5/CONTINUATION.md
