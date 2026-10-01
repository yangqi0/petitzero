# Technical report: frozen recipes and a later value intervention

Author/owner: Yang Qi; coding-agent assistance disclosed in README. This report synthesizes completed measurements, with no new experiment. One RTX4090, Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, LoRA and a shared M1A initializer define the scope. There is no learned reward model, long-CoT objective, distributed training or universal mathematics benchmark. PetitZero is separate from PetitGPT.

## Formal M8–M10 study

Three predeclared training seeds17/29/43 for GRPO, online verified-positive RFT and learned-value PPO; fixed slot256 endpoints, never best-checkpoint selection. Each arm used128 rollouts ×8 questions ×4 samples =4,096 TRAIN responses and256 nominal actor slots, with two passes per frozen batch. Actor microbatch2. DEV at slots128/256 used64 questions ×(one greedy+four sampled)=320 responses each,640 per arm; DEV artifacts are not the endpoint subset estimator. Main nine arms therefore requested36,864 TRAIN and5,760 arm DEV responses, excluding the separate shared baseline and engineering smokes. These are matched-request/update-budget **recipes**, not equal FLOPs or a critic-only intervention. RFT accounting distinguishes nominal slots and actual empty-positive skips.

M10 evaluated ten policies: one SFT and nine endpoints. For each policy, historical FINAL256 had8,448 responses and CONFIRM128 had4,224 (one greedy+32 samples/question); total126,720. Each population has its own SFT pool, not three invented SFT seeds. CONFIRM was first model-evaluated in M10; FINAL was already observed. CONFIRM is a same-range residual slice from unused multisets, not an independent full-universe draw, distribution-shift benchmark or proof of pretraining non-exposure. No pooled384-question headline is used.

Historical BF16 V1 results, M7 engineering pilot and all excluded smokes are outside this formal seed set. M1A itself used BF16 base/autocast; the formal downstream F-M profile uses original BF16 checkpoint values promoted to FP32, with active unmerged FP32 LoRA, zero dropout, native cached physical B1 generation, math SDPA, no autocast/TF32 and highest matmul precision. Sampling temperature1/top_p1/top_k0 and128-action cap are unchanged. Genuine EOS is an action distinct from padding; scoring uses C−1+t positions and real masks. Whole decoded expression reward is exact and binary, with no extraction, repair or learned judge.

| Recipe | Fixed objective/denominator | Roles |
|---|---|---|
| GRPO | Group4 population std+1e−4; constant groups retained with zero advantage; policy clip.2 plus sampled k3 beta.02; response token mean /32 responses | actor, frozen SFT reference |
| Online RFT | Own verified-positive response CE; duplicates retained; response token mean /whole-rollout positive count; true empty skip before optimizer/backward | actor |
| Learned PPO | Sampled k1 shaping−.02 plus final-action task reward; gamma1/lambda.95; frozen GAE/returns, global valid-token population whitening epsilon1e−8; policy/value clip.2; value factor.5; full-rollout valid-token D | independent actor/reference/critic; zero-initialized FP32 1536→1 biased head |
| Fixed-zero PPO (M11) | Same PPO shaping/GAE/whitening/actor loss with every real old value FP32 zero | actor/reference only; no critic/head/critic optimizer |

AdamW actor LR1e−5, critic LR1e−4 when present, eight nominal-slot warmup, betas.9/.999, epsilon1e−8, weight decay0, norm cap1, two passes with fixed targets. k1/k3 are sampled-prefix quantities, not exact global KL. Conditional categorical entropy describes next-token distributions at sampled prefixes, not sequence diversity.

## Endpoint results, populations kept separate

Percentages below are means ± sample SD across three seed endpoints (ddof1), except SFT's single pool. All individual seeds and original bands are retained in the linked [seed points](../results/seed_points.csv) and [method summaries](../results/method_summaries.csv). No intervals, significance tests or winner selection are added.

### CONFIRM128 — first evaluated at M10, now observed

| Policy/recipe | pass@1 (%) | pass@32 (%) |
|---|---:|---:|
| SFT (one pool) | 21.777343750 | 85.156250000 |
| GRPO | 31.778971354 ± 1.357635554 | 64.062500000 ± 2.343750000 |
| RFT | 33.430989583 ± 2.102635295 | 46.093750000 ± 0.781250000 |
| PPO | 31.038411458 ± 1.690457143 | 54.947916667 ± 7.507049638 |
| PPO_ZERO | 34.301757812 ± 0.423568154 | 59.895833333 ± 0.902109796 |

### Historical FINAL256 — previously observed

| Policy/recipe | pass@1 (%) | pass@32 (%) |
|---|---:|---:|
| SFT (one pool) | 22.204589844 | 81.640625000 |
| GRPO | 35.510253906 ± 2.489785547 | 64.192708333 ± 4.320484381 |
| RFT | 35.034179688 ± 2.357822135 | 46.614583333 ± 2.386758174 |
| PPO | 32.401529948 ± 2.529252493 | 57.291666667 ± 7.893460712 |
| PPO_ZERO | 35.750325521 ± 1.288193689 | 61.848958333 ± 2.600909421 |

Increased sampled pass@1 accompanied reduced observed finite pass@32 in the tested main endpoints. This does not establish complete support shrinkage, a universal method ordering, or that RL cannot learn new skills. Fixed-zero PPO did not restore SFT's CONFIRM coverage.

## M11: zero minus learned, all three pairs primary

The detailed M11 protocol was set after both populations were observed in M10. An earlier preferred intervention proposal does not make this an untouched confirmation experiment or fully blinded development. Three new zero arms independently initialized from M1A; no learned endpoint was used as a zero initializer. Primary pairs are seed17/29/43 at slot256. New formal TRAIN12,288 +DEV1,920 +FINAL25,344 +CONFIRM12,672 =52,224 outputs; the excluded smoke adds64. Formal actor768 plus excluded smoke4 =772 historical actor steps, critic0. These are M11 measurements, **not M12 activity**.

| Population | Metric | ZERO−LEARNED mean (pp) | Sample SD (pp) | seed17 | seed29 | seed43 |
|---|---|---:|---:|---:|---:|---:|
| Historical FINAL256 | pass@1 | +3.348795573 | 3.740620274 | -0.610351562 | +3.833007812 | +6.823730469 |
| Historical FINAL256 | pass@32 | +4.557291667 | 5.303492670 | -1.562500000 | +7.812500000 | +7.421875000 |
| CONFIRM128 | pass@1 | +3.263346354 | 1.683921505 | +1.440429688 | +4.760742188 | +3.588867188 |
| CONFIRM128 | pass@32 | +4.947916667 | 8.353653611 | -2.343750000 | +14.062500000 | +3.125000000 |

Seed17 is negative for CONFIRM pass@32 and for both listed historical FINAL metrics. Coverage improvement is not uniform. Across-seed endpoint SD is not gradient variance, a confidence interval, a powered equivalence test or proof of robustness. At lambda.95, removing learned values changes the estimator and temporal credit as well as potentially variance; whitening, clipping and online trajectories respond. This is not a pure unbiased baseline-variance intervention and does not show that critics are useless. Fewer score fluctuations alone cannot demonstrate reduced gradient variance.

**Learned PPO43** is a documented post-failure recovered trajectory with missing original slot192 commit identity. Historical memory-to-disk roundtrip and uninterrupted equivalence are not established. Zero43 had no such interruption, but pair43 and learned/paired primary aggregates inherit this limitation. All three pairs remain primary. The existing [matched17/29 secondary table](../results/secondary_matched17_29.csv) applies to BOTH branches; no new exclusion rule was introduced.

## Statistical units

Per question c successes among n=32; pass@k=`1−C(32−c,k)/C(32,k)`, for k1,2,4,8,16,32, averaged across questions. Greedy is separate. pass@32 is observed finite coverage, not full support. Endpoint subset pass@4 is not the old realized four-draw DEV artifact. Same sampling keys do not mean policies produced shared answers. Samples and k values do not add independent questions or training seeds. Paired differences are computed first; their actual sample SD is not the sum of marginal SDs. Original bands and negative exceptions remain visible. SFT pools are reused once as controls.

## Numerical validation and resources

Historical M8 native-generation/teacher-forced action-logp gate used `2e−4+1e−5*abs(native logp)` with maximum observed TRAIN absolute difference0.000115990639. This is finite engineering agreement, not identity of distributions. M11's first normal rollout and first two actor updates matched the learned initial boundary exactly; it did not introduce extra model probes. Existing M10/M11 scientific audits are reused as historical evidence, not reexecuted here. M12 checks mathematical code, export bindings and saved-count statistics; it neither repeats rewards nor loads models.

The following are saved measured phases, not complete matched end-to-end times. Learned PPO carries additional critic compute/memory, so reductions cannot be promoted into general speedup claims. Checkpoint bytes are recorded historical footprint, not a new backup.

| Branch/seed | Recorded phase seconds | Peak allocated GiB | Checkpoint GiB | Actor/critic steps |
|---|---:|---:|---:|---:|
| learned/17 | 2322.101 | 18.235 | 6.618 | 256/256 |
| zero/17 | 1932.813 | 12.224 | 3.309 | 256/0 |
| learned/29 | 2337.918 | 18.281 | 6.618 | 256/256 |
| zero/29 | 1945.019 | 12.255 | 3.309 | 256/0 |
| learned/43 | 2326.206 | 18.238 | 6.618 | 256/256 |
| zero/43 | 1965.089 | 12.212 | 3.309 | 256/0 |

Short realized responses fitting24GB do not imply every padded128-token shape fits. Recovered learned43 phase records do not establish uninterrupted-run equivalence or full wall-time matching.

## Reproducibility and rights limits

The package reproduces counts-based results and synthetic CPU calculations. M12A implements a new explicit-input runtime with fresh identities, journals and checkpoint validation. M12C validated the portable learned-PPO path on the recorded RTX4090/environment using64 synthetic TRAIN responses and a planned save/process-exit/fresh-process resume, with actor4/critic4 updates and no retries. Actual adapter/head, ordered optimizer and RNG/request restoration checks passed. GRPO/RFT/ppo_zero native execution, standalone actor evaluation, other devices/shapes, full historical replay and uninterrupted equivalence remain unvalidated. This engineering evidence contributes no scientific result rows. Exact prompts, SFT demonstrations and tensors remain external; there is no full historical GPU replay or independent backup. The final packaging pass makes two help/status string-only edits, documented in the numerical-consistency guide; it performs no model/GPU work. Results are conditional on one SFT initializer, a small task and three trajectories. Code and statistics cannot resolve the historical PPO43 provenance gap.

Existing local references, carried forward without network retrieval: [PPO and GAE sources](references/M3.md), [RFT/GRPO/numerical behavior](references/M2.md), [model and libraries](references/SOURCES.md), [diagnostic interpretation](references/M4.md). These provide background, not measured result evidence or official affiliation. No bibliographic details or new license permission are invented. The Qwen notice is preserved. PetitZero project code is released under [Apache-2.0](../LICENSE), subject to applicable upstream component/model licenses and notices. The first public release excludes exact private prompts, private SFT demonstrations, weights/adapters and private engineering evidence.
