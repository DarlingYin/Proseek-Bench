"""Input and integrity utilities for the Need, Time and Claims evaluators."""

from __future__ import annotations

import math
from pathlib import Path
import re
import sys

from judge_client import file_hash, read_json
from whole_file_judge import build_input  # Re-exported for the Stage 1 runner.

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from prediction_layout import CANONICAL_RUN_IDS, validate_setting_metadata

def component(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(f'Invalid path component: {value!r}')
    return value


def inside(path: Path, parent: Path) -> Path:
    path = path.resolve()
    path.relative_to(parent.resolve())
    return path


def text_field(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value


def window(value) -> list[str]:
    if (not isinstance(value, list) or not value
            or any(not isinstance(x, str) or not re.fullmatch(r'o[0-9]+', x) for x in value)
            or len(set(value)) != len(value)):
        raise ValueError('Commit window must contain unique observation IDs')
    return value


def expand_gt(data: dict) -> tuple[list[dict], int]:
    if not isinstance(data, dict) or not isinstance(data.get('needs'), list):
        raise ValueError('GT needs array is missing')
    expanded, ids = [], set()
    for need in data['needs']:
        gid = text_field(need.get('need_id'), 'need_id')
        if gid in ids:
            raise ValueError('Duplicate GT need_id')
        ids.add(gid)
        outer = window(need.get('commit_time_sets'))
        alternatives = need.get('alternatives')
        if alternatives is None:
            options = [need]
        else:
            if (not isinstance(alternatives, dict) or alternatives.get('relation') != 'OR'
                    or alternatives.get('match_policy') != 'any_one_supported_option_satisfies_need'):
                raise ValueError('Unsupported alternative policy')
            options = alternatives.get('options')
            if not isinstance(options, list) or len(options) < 2:
                raise ValueError('OR requires at least two options')
        option_ids, union = set(), set()
        for option in options:
            aid = text_field(option.get('alternative_id'), 'alternative_id') if alternatives is not None else None
            if aid in option_ids:
                raise ValueError('Duplicate alternative_id')
            option_ids.add(aid)
            times = window(option['commit_time_sets'] if 'commit_time_sets' in option else outer)
            if not set(times) <= set(outer):
                raise ValueError('Option window is outside the parent window')
            union.update(times)
            expanded.append({'gt_need_id': gid, 'alternative_id': aid,
                             'need': text_field(option.get('need'), 'GT Need'),
                             'claims': text_field(option.get('need_claims'), 'GT Claims'),
                             'window': times})
        if union != set(outer):
            raise ValueError('Parent window must equal union of option windows')
    return expanded, len(ids)


def discover(repo: Path, runs: dict, user_ids=(), sampler_ids=(),
             protocol_failure_policy: str = "reject") -> tuple[list[dict], dict]:
    if protocol_failure_policy not in {"reject", "error_as_failure"}:
        raise ValueError("Invalid protocol_failure_policy")
    if not isinstance(runs, dict) or not runs:
        raise ValueError('--runs must map labels to evaluate/runs-relative directories')
    rows, hashes, seen, found_users, found_samplers = [], {}, set(), set(), set()
    selected = []
    for sp in sorted((repo / 'users').glob('*/L*/*/sampler.json')):
        s = read_json(sp)
        uid, sid = component(s['user_id']), component(s['sampler_id'])
        if user_ids and uid not in user_ids or sampler_ids and sid not in sampler_ids:
            continue
        if sp.parent.name != sid or sp.parents[2].name != uid:
            raise ValueError('Sampler identity differs from its directory')
        tier = s.get('task_tier')
        if tier not in ('L1', 'L2', 'L3', 'L4') or sp.parents[1].name != tier:
            raise ValueError('Sampler tier differs from its directory')
        declared = s.get('episodes')
        if not isinstance(declared, list) or len(declared) != 1 or not isinstance(declared[0], dict):
            raise ValueError('This release runner requires one declared Episode per Sampler')
        episode_ref = declared[0].get('episode_ref')
        if not isinstance(episode_ref, str) or not episode_ref:
            raise ValueError('Declared Episode reference is missing')
        ep = inside(sp.parent / episode_ref, sp.parent)
        episode = read_json(ep)
        eid = component(episode['episode_id'])
        if eid != ep.parent.name or (uid, sid, eid) in seen:
            raise ValueError('Duplicate or inconsistent Episode identity')
        seen.add((uid, sid, eid))
        gt_ref = episode.get('ground_truth_ref') or declared[0].get('ground_truth_ref')
        if not isinstance(gt_ref, str) or not gt_ref:
            raise ValueError('Declared Ground Truth reference is missing')
        gt_path = inside(sp.parent / gt_ref, sp.parent)
        options, count = expand_gt(read_json(gt_path))
        for p in (sp, ep, gt_path):
            hashes[str(p.resolve())] = file_hash(p)
        selected.append((uid, sid, eid, tier, options, count))
        found_users.add(uid)
        found_samplers.add(sid)
    if not selected or set(user_ids) - found_users or set(sampler_ids) - found_samplers:
        raise ValueError('No samples selected or some requested filters were not found')
    for label, relative in runs.items():
        component(label)
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError('Run path must be relative')
        run = inside(repo / 'evaluate/runs' / relative, repo / 'evaluate/runs')
        for uid, sid, eid, tier, options, count in selected:
            pp = inside(run / uid / sid / 'prediction.json', run)
            failed = pp.with_name('prediction.protocol-failure.json')
            if failed.exists():
                if protocol_failure_policy != "error_as_failure":
                    raise ValueError(f'Unresolved protocol failure; do not score as NoNeed or use a stale Prediction: {failed}')
                failure = read_json(failed)
                if (failure.get('schema_version') != 'proactivebench.online-run-result.v1'
                        or failure.get('status') != 'protocol_failed'
                        or failure.get('episode_id') != eid):
                    raise ValueError(f'Invalid protocol failure receipt: {failed}')
                query_cost = failure.get('query_cost', {})
                cost = query_cost.get('cost_units')
                if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
                    raise ValueError(f'Invalid protocol-failure cost: {failed}')
                hashes[str(failed)] = file_hash(failed)
                call_log = pp.with_name(pp.name + '.model-calls.jsonl')
                if call_log.exists():
                    hashes[str(call_log)] = file_hash(call_log)
                rows.append({
                    'model': label, 'run': relative, 'user_id': uid, 'sampler_id': sid,
                    'episode_id': eid, 'tier': tier, 'predictions': [], 'options': options,
                    'gt_count': count, 'cost_units': cost, 'cost_warning': None,
                    'prediction_count': 0, 'protocol_failure': True,
                    'protocol_failure_ref': str(failed.resolve().relative_to(repo.resolve())),
                })
                continue
            prediction = read_json(pp)
            if (prediction.get('schema_version') != 'proseekbench.prediction.v1'
                    or prediction.get('sampler_id') != sid or prediction.get('episode_id') != eid
                    or not isinstance(prediction.get('predictions'), list)):
                raise ValueError(f'Invalid prediction identity/schema: {pp}')
            if type(prediction.get('search_count')) is not int or prediction['search_count'] < 0:
                raise ValueError(f'Invalid search_count: {pp}')
            for p in prediction['predictions']:
                text_field(p.get('need'), 'Predicted Need')
                text_field(p.get('claims'), 'Predicted Claims')
                window([p.get('committed_after')])
            rp = inside(pp.with_name('prediction.runtime.json'), run)
            cost, cost_warning = None, 'runtime_cost_missing'
            if rp.exists():
                runtime = read_json(rp)
                if runtime.get('status') != 'complete' or runtime.get('episode_id') != eid:
                    raise ValueError(f'Incomplete or inconsistent Runtime: {rp}')
                if run.name in CANONICAL_RUN_IDS:
                    config = runtime.get('configuration', {})
                    validate_setting_metadata(
                        run.name, inference_mode=config.get('inference_mode'),
                        image_window=config.get('image_window'),
                        prompt_version=config.get('experiment_condition', {}).get('prompt_version'),
                    )
                cost = runtime.get('query_cost', {}).get('cost_units')
                if cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
                    raise ValueError(f'Invalid cost: {rp}')
                cost_warning = None if cost is not None else 'runtime_cost_missing'
                hashes[str(rp)] = file_hash(rp)
            elif run.name.startswith('framework-v2-'):
                raise ValueError(f'Framework v2 requires its versioned Runtime receipt: {rp}')
            hashes[str(pp)] = file_hash(pp)
            call_log = pp.with_name(pp.name + '.model-calls.jsonl')
            if call_log.exists():
                hashes[str(call_log)] = file_hash(call_log)
            rows.append({'model': label, 'run': relative, 'user_id': uid, 'sampler_id': sid,
                         'episode_id': eid, 'tier': tier,
                         'predictions': [{k: p[k] for k in ('need', 'claims', 'committed_after')}
                                         for p in prediction['predictions']],
                         'options': options, 'gt_count': count, 'cost_units': cost,
                         'cost_warning': cost_warning, 'prediction_count': len(prediction['predictions']),
                         'protocol_failure': False, 'protocol_failure_ref': None})
        # Selected Prediction call logs are bound above. Do not hash unrelated
        # users or Samplers merely because they share the model run directory.
    return rows, hashes


def check_integrity(before: dict) -> dict:
    changed = []
    for path, expected in before.items():
        try:
            actual = file_hash(Path(path))
        except OSError:
            actual = None
        if actual != expected:
            changed.append(path)
    return {'status': 'pass' if not changed else 'fail', 'file_count': len(before), 'changed': changed}


