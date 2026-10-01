# Non-witness sufficient statistics

`endpoint_counts.csv`: 4,992 policy-question rows = 13 policies × (256 historical FINAL +128 CONFIRM). Populations remain separate. M10 contributes shared SFT and nine main endpoints once; M11 contributes only the three fixed-zero actors. M11 reused controls were checked against M10 rather than counted twice.

| Field | Meaning |
|---|---|
| policy | SFT, GRPO/RFT/PPO + seed, or PPO_ZERO + seed |
| seed | blank for the one SFT policy; otherwise17/29/43 |
| population | historical identifiers; CONFIRM128_RESERVED is now observed |
| original_band | add_sub_solvable or requires_mul_or_div |
| question_id | original stable identifier, no prompt or solution |
| greedy_correct | one separate greedy outcome, 0 or1 |
| c | successes among exactly32 samples, integer0..32 |
| n | always32; failures and duplicate responses retained |

The source table subset contains numeric rates/histograms/diagnostics, original paired rows, secondary17/29 rows and measured resource phases. No raw response, prompt, oracle witness, token sequence, private tensor or infrastructure identity is included. Preserving generic oracle code is distinct from exporting held-out solutions. Exact evaluation prompts and SFT demonstrations require separate owner release choices.

pass@k is the question mean of `1−C(32−c,k)/C(32,k)` for k=1,2,4,8,16,32 (zero numerator when32−c<k). pass@32 is finite observed coverage. pass@4 is a subset estimator from32, not the historical realized four-draw DEV metric. Sampling keys shared between policies do not imply shared answers. Samples/k values are not extra independent questions or training seeds.

Method summaries use three training-seed points and sample SD(ddof1); SFT has no training-seed SD. Primary M11 pairs are zero minus learned at the same seed. Pair differences are formed before computing SD; marginal SDs are never added. Matched17/29 is secondary for both branches and does not replace primary results. Learned PPO43 and pair43 carry the recovery caveat in the statistics JSON and technical report.
