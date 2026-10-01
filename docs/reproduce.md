# Reproduction and bounded M12C validation

## M12C bounded learned-PPO validation

Status: **M12C_PORTABLE_LEARNED_PPO_SMOKE_RESUME_PASSED_OTHER_GPU_PATHS_UNTESTED**.

M12C used byte-identical M12B production Python, runtime config and synthetic input. The M12B nested-journal snapshot fix remains intact. A standalone private preparation helper materialized only the seven authenticated base files into independent regular files, preserving original stored BF16 bytes. The recorded M1A adapter remained read-only. Capacity was accepted from the existing owner-confirmed allocation increment minus fresh measured growth and conservative allowances; this was an estimate, not an enforced quota guarantee.

Two fresh public-CLI processes completed the one authorized learned-PPO smoke: process1 committed32 responses and actor2/critic2 then exited; process2 verified/restored actual actor/critic/head, ordered optimizer state and RNG and completed the second rollout. Totals:64 unique TRAIN responses,640 actions, actor4/critic4 steps,128 backward calls,224 normal top-level forwards,640 native autoregressive forwards and3,584 separately counted decoder-layer recomputations (1,792 per trainable role). No evaluation/probe responses or retries occurred. Native/teacher-forced alignment and same-layout gates passed. This engineering result adds no scientific conclusion.

The existing38 guarded CPU tests and five separate standard-library preparation fixtures passed. CPU fakes still do not validate other native paths. M12B remains historically FIXED_CPU_ONLY_GPU_NOT_RUN; M12C is a new authorization, directory and prospective source/input freeze. Historical results and all scientific helpers remain unchanged.

| Operation | Validation |
| --- | --- |
| Offline saved-count/statistical workflow | CPU verified; inherited results unchanged, no new scientific scores |
| Learned-PPO smoke, actual save/exit/resume | PASSED for seed9001, the16 synthetic prompts, recorded RTX4090/environment only |
| GRPO / RFT / ppo_zero native execution | NOT RUN for the portable candidate |
| Standalone actor-only evaluation | NOT RUN |
| Other shapes/devices/environments and full historical replay | NOT RUN |
| Uninterrupted-trajectory equivalence | Not established |

## Recorded environment and CPU commands

Use the existing Python3.10.12, torch2.8.0+cu126, Transformers4.56.2, PEFT0.17.1, NumPy2.2.6 and safetensors0.8.0 environment. The real run used one visible RTX4090, driver580.126.09. No installation, upgrade, download or alternate dependency resolution occurred. Requirements record this environment, not arbitrary-version compatibility.

From the candidate root:

```bash
python tests/guarded.py tests
python pz.py verify --numbers 2 3 4 --target 14 --expression '2*(3+4)'
python pz.py fixture
python pz.py analyze --counts data/endpoint_counts.csv --output offline_results
```

The unchanged guarded suite has38 tests, including M12B mutation/reopen/tamper regressions. It blocks model/tokenizer imports, model/optimizer construction, forwards, CUDA operations and network connections. Five additional tiny standard-library materialization fixtures are provided in the private engineering evidence outside runtime sources. They test independent regular copies, unchanged original bytes and rejection of mismatched, missing, escaping or looping targets. Historical statistics/figures were not regenerated in M12C.

## Authenticated local assets

Supply seven ordinary files: config.json, tokenizer_config.json, tokenizer.json, generation_config.json, vocab.json, merges.txt and model.safetensors. The revision is989aa7980e4cf806f80c7fef2b1adb7bc71aa306. The six metadata hashes are fixed in protocol.BASE_METADATA; runtime.json pins the original base weight SHA256 dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee. Shared M1A weight/config hashes remain bf7e362147437e90c97f0e4675cbb724f89323fc02b7722a501fd975ec1049b7 and4b299f5a0985c94eb4322a69a4f49089ad641c4bcb8b946915de02f43bba1e9a.

M12C authenticated the recorded local symlink targets strictly inside the model cache's blobs directory, copied only the seven whitelisted files once into a temporary sibling, fsynced, rehashed and atomically published the ordinary-file directory. Total materialized payload:3,098,955,668 bytes. Destinations have independent inodes and are neither symlinks nor hardlinks. Original source contents were rehashed before/after copying and before each native process. No BF16 conversion, LoRA merge, metadata edit or original-cache change occurred. The same-volume derived copy was retained privately and is not a backup; archives contain no weights.

The initial capacity gate required at least8GiB, covering the copy plus checkpoint/log/delivery budget. Fresh baseline-growth accounting and additional allowances supported an estimated23,582,558,469 bytes before materialization. Enforceable quota remained unavailable and was explicitly null. This engineering estimate does not guarantee protection against concurrent writes. Small write/fsync/replace and remaining-work estimates were rechecked after materialization and between processes.

## Completed two-process sequence

The following generic paths illustrate the actual command shape; exact private paths, commands/exits, hashes and prospective receipt are in the separate evidence archive. The completed authorization is consumed. These examples do not schedule another run.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true PYTHONDONTWRITEBYTECODE=1
python pz.py smoke --execute --method ppo \
  --base ../assets/base_989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --sft ../assets/m1a \
  --data data/synthetic_prompts.jsonl --config configs/runtime.json \
  --seed 9001 --rollouts 2 --max-responses 64 --max-updates 4 \
  --stop-after 1 --output ../execution/engineering_ppo
python pz.py resume --execute --operation smoke --method ppo \
  --base ../assets/base_989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --sft ../assets/m1a \
  --data data/synthetic_prompts.jsonl --config configs/runtime.json \
  --seed 9001 --rollouts 2 --max-responses 64 --max-updates 4 \
  --checkpoint ../execution/engineering_ppo/checkpoints/rollout0001 \
  --output ../execution/engineering_ppo
```

The16 prompt-only rows are unchanged, four per band/exposure stratum. Exposure flags are synthetic scheduling labels, not real M1A-membership claims. Data SHA256 is2c995bca3d7640aa83bf64aac78370687b4901a528f1e4661c7f335223a72ee3; config SHA256 is5a1f8f691f9e4cade93719ecb825cbb770a68d1b5b06ae142de534e2b28ee0c7; the full64-request canonical digest is301b0d51251490a2e945b171ed959dace1fb156704d5d7f93565ce0f509c1c4b. The dataset is engineering-only and cannot enter study tables.

Process1 exited0 at a clean committed pause. A fresh CPU read verified the journal prefix, payload hashes,32 unique requests, slot2, actor2/critic2 and next index32 before process2. Process2 exited0 after restoring actual adapter/head, ordered Adam states/moments/steps and RNG without a probe forward or fresh-state fallback. Initial, rollout1 and rollout2 checkpoints remain retained. The final CPU audit checked only these64 new scores with the frozen verifier, request order/uniqueness, journal chain, frozen source/assets, enriched targets, numeric records and checkpoint payload hashes.

F-M remained unchanged: stored BF16 values promoted in memory toFP32, active unmerged FP32 LoRA, zero dropout, math SDPA, no autocast/TF32, independent role stores, genuine EOS and C−1+t positions, cached generation B1/max128, full-rollout targets/denominator and original Adam settings. Both native/teacher-forced gates passed (maximum absolute discrepancy5.435943603515625e−5); both pre-pass1 same-layout differences were zero under the unchanged2e−4+1e−5×abs(native-logp) tolerance. No minimum reward, improvement or nonzero-update condition was added.

The journal owns detached JSON event snapshots and separate reconstructed response records. Mutable training targets remain saved separately. Hash-chain/prefix/payload/cursor/latest-checkpoint/no-replay guards stay active. An unfinished request/update or unexpected runtime failure must stop; the only continuation exercised was the preauthorized clean boundary. Production Python/config/data are byte-identical to M12B before, during and after execution; these documentation changes were made afterward and are distinct from the prospective executed-source receipt.

## Original data/M1A and release limitations

Original multiset construction remains in [data logic](../petitzero/data_logic.py): exclude64 calibration groups, hash-sort1,476 remaining groups into TRAIN1024/DEV64/FINAL256/unused132; alternate mathematically defined bands and hash-select targets. Cold start selects256 TRAIN rows/band. CONFIRM128 is a residual same-range slice after all1,408 historical groups were excluded, not an independent universe draw. Exact study prompts and supervision are not bundled; those release choices do not block explicit-input execution.

[Recorded M1A recipe](../configs/m1a_recorded.json):512 TRAIN demonstrations, two epochs,64 updates, microbatch2/accumulation8, completion-plus-EOS labels, BF16 base/autocast and FP32 LoRA r16/alpha32/dropout0; four-update LR warmup to1e−4 then decay to1e−5. Exact supervision and the full M1A rebuild driver remain external; the required shared adapter is explicit and hash-checked. No base-initialized demo is called a formal reproduction.

Project code uses [Apache-2.0](license_status.md). The first public release excludes exact private prompts, private SFT demonstrations, weights/adapters and private engineering evidence. The RunPod network volume remains retained. Public code archives contain no tensors and are not weight backups. M11's observed-data and learned-PPO43/pair43 limitations remain in the report. No new model scores are generated by the CPU tests.

## Final packaging boundary

The public package incorporates only two previously accepted display string literal changes in CLI help and planning status after M12C. Scientific helpers and all execution-runtime files retain M12C bytes. No model/GPU work is performed in release preparation, and the changed help/planning sources are CPU-verified rather than newly GPU-validated. Source identity includes planning.py and pz.py, so existing M12C run resumption requires its retained original source; do not bypass identity checks with this packaging copy.

`plan` uses `configs/new_execution.json` and only emits an intention. Its one-rollout32-response smoke proposal predates the separate two-rollout M12C execution; it is not the native execution validator. For a model-free parsing example use the README plan command. Explicit `--execute` commands require supplied local assets and a separate model-work authorization. See the [license and release scope](license_status.md).
