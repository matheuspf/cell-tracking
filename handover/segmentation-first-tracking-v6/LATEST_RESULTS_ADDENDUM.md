# Late source update observed during publication

Read after initial plan publication on 10 September 2026. The v5 branch advanced
to `0a3105a8f9d1b0954707675b42b9868832df4ac1` while this review was underway.
Its [continuation](https://github.com/matheuspf/cell-tracking/blob/0a3105a8f9d1b0954707675b42b9868832df4ac1/handover/image-native-tracking-v5/CONTINUATION.md)
Git blob is `6505a11f9c4a5e00ab96a8bcba7f6a78d01e5d35`.

At 20:51 UTC, H_probe_J completed all 199 clips:

- Pooled score: **0.9135868579084195**; delta C0 **-0.02121551635216645**.
- 44b6: **0.8440421993352256**; 6bba: **0.92770934806879**.
- Edge TP/FP/FN: **121603/6053/7280**.
- Division TP/FP/FN: **29/149/122**.

Eight of eighteen configurations were measured in this later receipt. Primary N2
score extraction finished all 199 clips around 20:53 UTC, but its graph decoder
was still queued. A fresh-image validation child started; the complete six-clip
parity result did not yet exist. Both N2 replicas and other evaluations remained
in progress. This is not a final v5 failure or completion report.

This addendum supersedes the seven-of-eighteen count in the earlier REVIEW.md as
of this later observation. C0 and the segmentation-first rationale are unchanged.
H_probe_J also inherits the explicitly documented HOCT feature-unit limitation;
it is not a clean verdict on the pretrained backbone.

V6 remains intentionally based on the frozen `03ab557...` source, with no merge of
live v5 code or jobs. X600 must inventory any newer local receipts and resources.
Do not rewrite the original source anchors or mix late selected weights into the
registered comparison. The addendum is reporting only and introduces no new
trained model, detector, score or author-run microscopy experiment.
