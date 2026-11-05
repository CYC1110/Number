# Cohort and pairing contract

| States | Networks | Runs | Explicit initialization seeds |
|---|---:|---|---|
| Untrained, Pretrained, Label-shuffled, DVD | 20 each | out_04–out_23 | See mapping below |
| SimCLR, RotNet, DINO | 5 each | out_04–out_08 | 5962–5966 |
| Noise, Grating, Statistical, Shaders | 5 each | out_04–out_08 | 5962–5966 |
| correct_05/10/20/50, ResNet-50 Untrained/Pretrained | 5 each | out_04–out_08 | 5962–5966; retained reference setting |

The final row preserves the prior reference configuration. It is not a correction or claim about those controls' historical cohort sizes. Cohorts are assigned by state in `configs/protocol.json`; there is no global network count.

| Run | Seed | Run | Seed |
|---|---:|---|---:|
| out_04 | 5962 | out_14 | 5972 |
| out_05 | 5963 | out_15 | 5973 |
| out_06 | 5964 | out_16 | 5974 |
| out_07 | 5965 | out_17 | 5975 |
| out_08 | 5966 | out_18 | 5976 |
| out_09 | 5967 | out_19 | 5977 |
| out_10 | 5968 | out_20 | 5978 |
| out_11 | 5969 | out_21 | 5959 |
| out_12 | 5970 | out_22 | 5960 |
| out_13 | 5971 | out_23 | 5961 |

Reviewed evidence: `figures/iclr/methods/METHODS_NOTES.md` explicitly retains 20 networks for Untrained and supervised conditions while selecting five for SSL/synthetic diets. `subspace/agg/results/pa_overlap_p6_p7_n1p2_20260921/source_inventory.json` and `subspace/agg/results/all_state_p7_trajectories_20260925/inventory.json` contain the four 20-run cohorts and the mapping above. These relative paths identify source evidence; this checkout does not import or require that research tree.

`subspace/agg/scripts/plot_untrained_pretrained_paired_geometry.py` pivots on run, requires out_04–out_23, and calls `ttest_rel(pretrained, untrained, alternative="two-sided")`. Thus the preserved comparison uses 20 common initialization seeds, df=19, and an after-minus-before t sign. Historical pairing for unequal cohort sizes was not established from this source.

Analysis verifies the configured seed in each checkpoint, stores it in the metrics and source provenance, and stores the condition's expected network count. Figure collection requires the whole configured cohort and verifies every identity, seed and count. Network SEM uses sample SD / sqrt(n), where n is the number of complete configured records. Figure sidecars record actual state counts. S4 and S5 retain each condition's configured runs across epochs.

`paired_statistics` accepts explicit seed-to-value mappings. It reports n_before, n_after, n_pairs, df, paired seeds and both unmatched seed sets. Unequal cohorts use the shared-seed intersection; at least two pairs are required. Nonfinite values are rejected before matching. Array positions and shorter-array truncation cannot define pairs. The resulting effect concerns only matched seeds, not the difference between complete cohort means; sharing an initialization does not establish identical training experience. F3 uses the original 20 matched pairs. F6/S2 correlate complete condition means and do not pair a 20-network array against a five-network array.

No image generation, checkpoint creation, network training or empirical result reconstruction was performed for this correction.
