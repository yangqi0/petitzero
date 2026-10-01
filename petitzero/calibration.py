"""Small, fixed TRAIN-SIDE calibration set and rollout accounting."""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
import statistics
from typing import Iterable

from .countdown import TASK_VERSION, check_expression, make_messages, reachable_targets

SEED = 20260928
BANDS = ('add_sub_solvable', 'requires_mul_or_div')


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def build_calibration(seed: int = SEED) -> list[dict]:
    """32 instances/band; every numerical multiset is unique across all64.

    Bands are defined by the oracle, NOT model scores. Withholding these64 numeric
    groups from the later final test is mandatory. They may later enter TRAIN.
    There is no validation or test generation in this milestone.
    """
    triples = list(itertools.combinations_with_replacement(range(1, 21), 3))
    triples.sort(key=lambda x: digest([seed, 'number-order', x]))
    selected: list[dict] = []
    counts: Counter = Counter()
    for triple in triples:
        full = reachable_targets(triple)
        simple = reachable_targets(triple, '+-')
        # Alternate target band so the first band does not consume every number set.
        band = BANDS[len(selected) % 2]
        if counts[band] >= 32:
            band = BANDS[1 - BANDS.index(band)]
        choices = sorted(simple if band == BANDS[0] else set(full) - set(simple))
        if not choices:
            continue
        target = min(choices, key=lambda y: digest([seed, 'target', triple, y]))
        n = len(selected) + 1
        row = {'id': f'PZ0-{n:03}', 'task_version': TASK_VERSION,
               'role': 'TRAIN_SIDE_CALIBRATION', 'band': band,
               'numbers': list(triple), 'target': target,
               'number_group': digest(list(triple)),
               'puzzle_key': digest([list(triple), target]),
               'messages': make_messages(triple, target),
               'oracle_witness': full[target]}
        assert check_expression(row['oracle_witness'], triple, target).correct
        selected.append(row)
        counts[band] += 1
        if len(selected) == 64:
            break
    if len(selected) != 64 or set(counts.values()) != {32}:
        raise RuntimeError('Could not build prescribed calibration set')
    return selected


def summarize(records: Iterable[dict], rows: list[dict]) -> dict:
    """All-or-nothing accounting for64 x (greedy1 + sampled4), no silent drop."""
    records = list(records)
    ids = {x['id'] for x in rows}
    expected = {(i, 'greedy', 0) for i in ids} | {(i, 'sample', j) for i in ids for j in range(4)}
    keys = [(x['id'], x['mode'], x['sample_index']) for x in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Duplicate, missing, or extra rollout records')
    row_by_id = {x['id']: x for x in rows}
    for x in records:
        row = row_by_id[x['id']]
        wanted = check_expression(x['scoring_text'], row['numbers'], row['target']).as_dict()
        if x['score'] != wanted:
            raise ValueError(f'Stored reward mismatch: {x["id"]}')
    result = {}
    for label in ('all',) + BANDS:
        use_ids = ids if label == 'all' else {x['id'] for x in rows if x['band'] == label}
        group = [x for x in records if x['id'] in use_ids]
        greedy = [x for x in group if x['mode'] == 'greedy']
        sampled = [x for x in group if x['mode'] == 'sample']
        by_prompt = {i: [x for x in sampled if x['id'] == i] for i in sorted(use_ids)}
        success_counts = [sum(x['score']['correct'] for x in xs) for xs in by_prompt.values()]
        result[label] = {
            'prompts': len(use_ids), 'greedy_correct': sum(x['score']['correct'] for x in greedy),
            'greedy_total': len(greedy), 'sample_correct': sum(success_counts),
            'sample_total': len(sampled), 'sample_success_rate': sum(success_counts)/len(sampled),
            'empirical_any_success_in_four': sum(k > 0 for k in success_counts)/len(use_ids),
            'all_zero_groups': sum(k == 0 for k in success_counts),
            'all_one_groups': sum(k == 4 for k in success_counts),
            'mixed_reward_groups': sum(0 < k < 4 for k in success_counts),
            'zero_reward_variance_fraction': sum(k in (0, 4) for k in success_counts)/len(use_ids),
            'sample_format_rate': sum(x['score']['format_valid'] for x in sampled)/len(sampled),
            'sample_number_usage_rate': sum(x['score']['numbers_valid'] for x in sampled)/len(sampled),
            'sample_mean_unique_texts_per_group': statistics.mean(len(set(x['scoring_text'] for x in xs))
                                                                    for xs in by_prompt.values()),
            'sample_reason_counts': dict(Counter(x['score']['reason'] for x in sampled)),
            'sample_stop_counts': dict(Counter(x['stop_reason'] for x in sampled)),
            'mean_sample_output_tokens': statistics.mean(x['generated_tokens'] for x in sampled),
        }
    return {'status': 'CALIBRATION_COMPLETE_NOT_A_TEST_RESULT', 'groups': result,
            'completions': len(records), 'training_updates': 0,
            'interpretation': 'TRAIN-side task/sampler calibration only. No policy gradient was run.'}
