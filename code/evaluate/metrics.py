"""Micro-averaged metrics. Never average sampler F1s or hide pending rows."""

from __future__ import annotations


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for tier in ('Overall', 'L1', 'L2', 'L3', 'L4'):
        selected = [r for r in rows if tier == 'Overall' or r['tier'] == tier]
        bucket = {
            'samplers': len(selected),
            'gt_needs': sum(r['gt_count'] for r in selected),
            'predictions': sum(r['prediction_count'] for r in selected),
            'noneed_samplers': sum(r['gt_count'] == 0 for r in selected),
            'correct_rejections': sum(
                not r.get('protocol_failure', False)
                and r['gt_count'] == r['prediction_count'] == 0
                for r in selected
            ),
        }
        for metric in ('need', 'time', 'overall'):
            pending = sum(r['scores'][metric]['status'] == 'pending' for r in selected)
            not_evaluated = sum(r['scores'][metric]['status'] == 'not_evaluated' for r in selected)
            counts = None if pending or not_evaluated else {
                key: sum(r['scores'][metric]['counts'][key] for r in selected)
                for key in ('tp', 'fp', 'fn')
            }
            denominator = None if counts is None else 2 * counts['tp'] + counts['fp'] + counts['fn']
            bucket[metric + '_counts'] = counts
            bucket[metric + '_pending_samplers'] = pending
            bucket[metric + '_not_evaluated_samplers'] = not_evaluated
            bucket[metric + '_f1'] = (
                None if counts is None else (2 * counts['tp'] / denominator if denominator else 0.0)
            )
            if metric == 'need':
                recall_denominator = None if counts is None else counts['tp'] + counts['fn']
                bucket['need_recall'] = (
                    None if not recall_denominator else counts['tp'] / recall_denominator
                )
        missing = sum(r['cost_units'] is None for r in selected)
        bucket['missing_cost_samplers'] = missing
        bucket['avg_cost'] = (
            None if missing or not selected else sum(r['cost_units'] for r in selected) / len(selected)
        )
        result[tier] = bucket
    return result
