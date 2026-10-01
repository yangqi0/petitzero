# Numerical consistency and evidence boundaries

This document indexes accepted evidence; it reports no new model calculation.

The formal F-M profile loads stored BF16 base values and promotes them toFP32 in memory, with independent actor/reference/critic stores, active unmerged FP32 LoRA, zero dropout, math SDPA, no autocast/TF32 and native cached generation at physical batch1. Real EOS is an action; padding is not. Action log-probabilities and critic states use C−1+t positions and valid-action masks. Frozen target denominators and the original optimizer settings are specified in the [technical report](technical_report.md).

Native-generation versus teacher-forced selected log-probabilities use the original bound `2e-4 + 1e-5 * abs(native_logp)`. Historical M8's maximum observed TRAIN difference was0.000115990639. M12C's two engineering rollouts had maxima4.8160552978515625e−5 and5.435943603515625e−5, with zero same-layout pre-pass1 maximum differences. Passing this finite gate is not identity of whole distributions or validation of other shapes/devices.

The M12B persistence fix detached JSON snapshots from mutable target-enrichment records. Its mutation, reopen, reconciliation and tamper regressions are part of the38-test CPU suite. M12C then exercised actual model/head, ordered optimizer state/moments/steps, RNG and request restoration across a process exit. It verified a committed32-response boundary before resuming to64 unique requests, preserving all three checkpoints. No probe forward or uninterrupted comparison was performed.

The release-preparation pass only changes two user-facing help/status string literals in `pz.py` and `petitzero/planning.py`; scientific helpers and the complete `petitzero/execution/` package remain byte-identical to the M12C execution. CPU verification of these packaging changes does not constitute a new GPU validation. Source hashes are part of run identity: retain the original M12C source for that run's existing checkpoints.

See [portable coverage](reproduce.md), [limitations](limitations.md) and [provenance](../provenance.json). Historical learned-PPO43's missing original slot192 commit identity and unproven historical roundtrip/uninterrupted equivalence are separate issues and remain unresolved.
