"""Need/Claims prompt contracts and deterministic sampler evaluation."""

from __future__ import annotations

import re

from judge_client import canonical, file_hash, read_json

METHODS = ('score-twice',)
WHOLE_FILE_METHODS = ()
DEFAULT_VERSIONS = {'score-twice': 'v1'}
CLAIMS_VERSION = 'v1'
SCORE_TWICE_PREFILTER_MINIMUM = 3
SCORE_TWICE_ISOLATED_SCALE = (1, 4)
SCORE_TWICE_ISOLATED_ACCEPTANCE_MINIMUM = 3


class WholeFilePromptBundle:
    def __init__(self, root, method, version=None):
        from jsonschema import Draft202012Validator
        if method not in METHODS:
            raise ValueError('Unknown evaluation method')
        version = version or DEFAULT_VERSIONS[method]
        self.method, self.version, self.claims_version = method, version, CLAIMS_VERSION
        if method == 'score-twice':
            self.paths = {
                'need_prefilter.md': root / method / version / 'prefilter.md',
                'need_prefilter.schema.json': root / method / version / 'prefilter.output.schema.json',
                'need_isolated.md': root / method / version / 'isolated.md',
                'need_isolated.schema.json': root / method / version / 'isolated.output.schema.json',
                'claims.md': root / 'shared' / CLAIMS_VERSION / 'claims.md',
                'claims.schema.json': root / 'shared' / CLAIMS_VERSION / 'output.schema.json',
            }
            stages = ('need_prefilter', 'need_isolated', 'claims')
        else:
            self.paths = {
                'need.md': root / method / version / 'need.md',
                'need.schema.json': root / method / version / 'output.schema.json',
                'claims.md': root / 'shared' / CLAIMS_VERSION / 'claims.md',
                'claims.schema.json': root / 'shared' / CLAIMS_VERSION / 'output.schema.json',
            }
            stages = ('need', 'claims')
        self.hashes = {name: file_hash(path) for name, path in self.paths.items()}
        self.texts = {stage: self.paths[stage + '.md'].read_text(encoding='utf-8') for stage in stages}
        self.validators = {}
        for stage in stages:
            schema = read_json(self.paths[stage + '.schema.json'])
            Draft202012Validator.check_schema(schema)
            self.validators[stage] = Draft202012Validator(schema)

    def system(self, stage):
        return self.texts[stage]

    def validate_input(self, stage, inputs):
        if self.method == 'score-twice' and stage == 'need_isolated':
            if set(inputs) != {'prediction', 'ground_truth'}:
                raise ValueError('Unexpected isolated-pair input fields')
            prediction, reference = inputs['prediction'], inputs['ground_truth']
            if set(prediction) != {'prediction_id', 'need'} or set(reference) != {'reference_id', 'need'}:
                raise ValueError('Unexpected isolated-pair envelope fields')
            if prediction['prediction_id'] != 'p0001' or reference['reference_id'] != 'r0001':
                raise ValueError('Isolated-pair IDs must be canonical')
            if (not isinstance(prediction['need'], str) or not prediction['need'].strip()
                    or not isinstance(reference['need'], str) or not reference['need'].strip()):
                raise ValueError('Isolated Need text must be nonempty')
            return
        whole_stage = stage == 'claims' or stage == 'need' or (
            self.method == 'score-twice' and stage == 'need_prefilter')
        expected = {'prediction', 'ground_truth', 'pairs'}
        if not whole_stage or set(inputs) != expected:
            raise ValueError('Unexpected whole-file input fields')
        if set(inputs['prediction']) != {'predictions'} or set(inputs['ground_truth']) != {'needs'}:
            raise ValueError('Unexpected file envelope fields')
        predictions, references = inputs['prediction']['predictions'], inputs['ground_truth']['needs']
        pids = [p['prediction_id'] for p in predictions]
        rids = [r['reference_id'] for r in references]
        if not pids or not rids or len(set(pids)) != len(pids) or len(set(rids)) != len(rids):
            raise ValueError('Empty or duplicate item IDs')
        for p in predictions:
            if set(p) != {'prediction_id', 'need', 'claims', 'committed_after'}:
                raise ValueError('Invalid prediction fields')
        for r in references:
            if set(r) != {'reference_id', 'gt_need_id', 'alternative_id', 'need', 'need_claims', 'commit_time_sets'}:
                raise ValueError('Invalid reference fields')
        seen = set()
        for pair in inputs['pairs']:
            if set(pair) != {'prediction_id', 'reference_id'}:
                raise ValueError('Invalid requested pair')
            key = pair['prediction_id'], pair['reference_id']
            if key[0] not in pids or key[1] not in rids or key in seen:
                raise ValueError('Unknown or duplicated requested pair')
            seen.add(key)
        if not seen:
            raise ValueError('No pairs requested')
        if stage in ('need', 'need_prefilter') and seen != {(p, r) for p in pids for r in rids}:
            raise ValueError('Need requests must include the full Cartesian pair list')

    def validate(self, stage, inputs, output):
        self.validate_input(stage, inputs)
        self.validators[stage].validate(output)
        if stage == 'need_isolated':
            wanted = {('p0001', 'r0001')}
        elif stage in ('need', 'need_prefilter'):
            wanted = {(p['prediction_id'], r['reference_id'])
                      for p in inputs['prediction']['predictions'] for r in inputs['ground_truth']['needs']}
        else:
            wanted = {(p['prediction_id'], p['reference_id']) for p in inputs['pairs']}
        got = [(p['prediction_id'], p['reference_id']) for p in output['results']]
        if len(got) != len(set(got)) or set(got) != wanted:
            raise ValueError('Missing, extra, duplicate, or unknown result pairs')
        # Python booleans are ints, but a boolean is never a Judge score.
        if stage != 'claims' and self.method in ('score', 'score-twice'):
            if any(p['score'] is not None and type(p['score']) is not int for p in output['results']):
                scale = '1 to 4' if stage == 'need_isolated' else '0 to 5'
                raise ValueError(f'Score must be an integer from {scale}')
            if 'material_deviation' in self.validators[stage].schema['properties']['results']['items']['properties']:
                if stage == 'need_isolated':
                    refs = {'r0001': inputs['ground_truth']['need']}
                else:
                    refs = {r['reference_id']: r['need'] for r in inputs['ground_truth']['needs']}
                for result in output['results']:
                    deviation = result['material_deviation']
                    if deviation is not None:
                        requirement = deviation.get('gt_requirement', deviation.get('requirement', ''))
                        if not requirement.strip() or not deviation['consequence'].strip():
                            raise ValueError('Material deviation must identify a requirement and consequence')
                        if requirement not in refs[result['reference_id']]:
                            raise ValueError('Cited requirement must be an exact reference-Need substring')
                        evidence = deviation.get('prediction_evidence')
                        if evidence is not None:
                            predictions = ({'p0001': inputs['prediction']['need']}
                                           if stage == 'need_isolated' else
                                           {p['prediction_id']: p['need']
                                            for p in inputs['prediction']['predictions']})
                            if evidence not in predictions[result['prediction_id']]:
                                raise ValueError('Cited evidence must be an exact Prediction-Need substring')

    def repair_substring_witnesses(self, stage, inputs, output):
        """Apply semantics-preserving witness formatting and schema-length repairs.

        A Judge sometimes paraphrases the conflicting object when the frozen
        contract requires an exact substring.  If punctuation/span recovery is
        impossible and the complete source Need fits the witness field, use
        that complete source verbatim.  This changes no score, slot, or task
        semantics and keeps the deletion witness auditable.
        """
        if stage not in ('need', 'need_prefilter', 'need_isolated'):
            return []
        if 'material_deviation' not in self.validators[stage].schema['properties']['results']['items']['properties']:
            return []
        if stage == 'need_isolated':
            references = {'r0001': inputs['ground_truth']['need']}
            predictions = {'p0001': inputs['prediction']['need']}
        else:
            references = {item['reference_id']: item['need'] for item in inputs['ground_truth']['needs']}
            predictions = {item['prediction_id']: item['need'] for item in inputs['prediction']['predictions']}
        punctuation = str.maketrans({
            '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
            '\u2013': '-', '\u2014': '-', '\u00a0': ' ',
        })
        changes = []
        for index, result in enumerate(output['results']):
            reason = result.get('reason')
            if isinstance(reason, str) and len(reason) > 1024:
                result['reason'] = reason[:1024]
                changes.append(f'result[{index}]:truncate_reason_to_schema_limit')
            deviation = result.get('material_deviation')
            if not isinstance(deviation, dict):
                continue
            consequence = deviation.get('consequence')
            if isinstance(consequence, str) and len(consequence) > 400:
                deviation['consequence'] = consequence[:400]
                changes.append(f'result[{index}]:truncate_consequence_to_schema_limit')
            for field, source in (
                ('gt_requirement', references[result['reference_id']]),
                ('prediction_evidence', predictions[result['prediction_id']]),
            ):
                candidate = deviation.get(field)
                if not isinstance(candidate, str) or not candidate or candidate in source:
                    if isinstance(candidate, str) and candidate in source and len(candidate) > 300:
                        deviation[field] = candidate[:300]
                        changes.append(f'result[{index}]:truncate_{field}_to_schema_limit')
                    continue
                folded_source = source.translate(punctuation)
                folded_candidate = candidate.translate(punctuation)
                start = folded_source.find(folded_candidate)
                exact = None
                if start >= 0 and folded_source.find(folded_candidate, start + 1) < 0:
                    direct = source[start:start + len(candidate)]
                    if direct.translate(punctuation) == folded_candidate:
                        exact = direct
                if exact is None and ('...' in folded_candidate or '\u2026' in folded_candidate):
                    fragments = [part.strip() for part in re.split(r'(?:\.\.\.|\u2026)', folded_candidate)
                                 if part.strip()]
                    if len(fragments) >= 2 and all(len(part) >= 4 for part in fragments):
                        pattern = '.*?'.join(re.escape(part) for part in fragments)
                        matches = list(re.finditer(pattern, folded_source))
                        if len(matches) == 1:
                            exact = source[matches[0].start():matches[0].end()]
                if exact is None and len(source) <= 300:
                    exact = source
                    changes.append(f'result[{index}]:replace_{field}_with_full_exact_need')
                if exact is None:
                    continue
                deviation[field] = exact[:300]
                if not changes or changes[-1] != f'result[{index}]:replace_{field}_with_full_exact_need':
                    changes.append(f'result[{index}]:repair_{field}_exact_span')
        return changes

    def normalize(self, stage, result):
        if stage == 'claims' or self.method == 'binary':
            passed = result['pass']
        elif self.method == 'score':
            passed = None if result['score'] is None else result['score'] >= 3
        elif self.method == 'score-twice':
            score = result['score']
            if score is None:
                passed = None
            elif stage == 'need_prefilter':
                passed = score >= SCORE_TWICE_PREFILTER_MINIMUM
            else:
                passed = score >= SCORE_TWICE_ISOLATED_ACCEPTANCE_MINIMUM
        else:
            found = result['counterexample_found']
            passed = None if found is None else not found
        return {'prompt_version': self.claims_version if stage == 'claims' else self.version, 'method': self.method, 'stage': stage,
                'status': 'review_required' if passed is None else 'evaluated',
                'pass': passed, 'reason': result['reason'], 'raw_result': result}


def build_isolated_input(prediction, option):
    """Build a canonical one-Prediction/one-GT Need request.

    Model/run/tier, Claims, timing, and neighboring pairs are deliberately
    absent so that the semantic verdict depends only on the two Need strings.
    """
    return {
        'prediction': {'prediction_id': 'p0001', 'need': prediction['need']},
        'ground_truth': {'reference_id': 'r0001', 'need': option['need']},
    }


def build_input(row):
    """Keep all substantive prediction/GT content, remove author/run identity.

    OR options remain grouped by gt_need_id with a unique reference_id each.
    Window inheritance has already been validated by the loader.
    """
    return {
        'pairs': [{'prediction_id': f'p{i:04d}', 'reference_id': f'r{j:04d}'}
                  for i in range(1, len(row['predictions']) + 1)
                  for j in range(1, len(row['options']) + 1)],
        'prediction': {'predictions': [
            {'prediction_id': f'p{i:04d}', **p} for i, p in enumerate(row['predictions'], 1)
        ]},
        'ground_truth': {'needs': [
            {'reference_id': f'r{i:04d}', 'gt_need_id': o['gt_need_id'],
             'alternative_id': o['alternative_id'], 'need': o['need'],
             'need_claims': o['claims'], 'commit_time_sets': o['window']}
            for i, o in enumerate(row['options'], 1)
        ]},
    }


def judge_once(stage, payload, judge, max_input_chars):
    if len(canonical(payload)) > max_input_chars:
        return {'verdict': None, 'error': 'whole_sampler_input_too_large', 'cache_hit': False}
    return judge.judge_once(stage, payload)


def evaluate_sampler(row, judge, max_input_chars=120000, evaluate_claims=True):
    from matching import score_pairs
    payload = build_input(row)
    pairs = []
    need_response = None
    claims_response = None
    if row['predictions'] and row['options']:
        # Exactly one logical Need request, irrespective of Cartesian pair count.
        need_response = judge_once('need', payload, judge, max_input_chars)
        values = {} if need_response['verdict'] is None else {
            (v['prediction_id'], v['reference_id']): v for v in need_response['verdict']['results']}
        eligible = []
        for i, prediction in enumerate(row['predictions'], 1):
            for j, option in enumerate(row['options'], 1):
                pid, rid = f'p{i:04d}', f'r{j:04d}'
                raw = values.get((pid, rid))
                verdict = judge.bundle.normalize('need', raw) if raw is not None else None
                time_pass = prediction['committed_after'] in option['window']
                skipped = 'skipped_need_unresolved'
                if verdict is not None and verdict['status'] == 'evaluated':
                    if not verdict['pass']:
                        skipped = 'skipped_need_failed'
                    elif not time_pass:
                        skipped = 'skipped_time_failed'
                    else:
                        eligible.append({'prediction_id': pid, 'reference_id': rid})
                pairs.append({'prediction_index': i, 'option_index': j,
                              'gt_need_id': option['gt_need_id'], 'alternative_id': option['alternative_id'],
                              'committed_after': prediction['committed_after'], 'window': option['window'],
                              'time_pass': time_pass,
                              'need': {**{k: v for k, v in need_response.items() if k != 'verdict'}, 'verdict': verdict},
                              'claims': {'status': skipped}})
        if not evaluate_claims:
            eligible_keys = {(item['prediction_id'], item['reference_id']) for item in eligible}
            for pair in pairs:
                key = f"p{pair['prediction_index']:04d}", f"r{pair['option_index']:04d}"
                if key in eligible_keys:
                    pair['claims'] = {'status': 'not_evaluated'}
            claims_response = {'status': 'not_evaluated'}
        elif eligible:
            claims_response = judge_once('claims', {**payload, 'pairs': eligible}, judge, max_input_chars)
            claims_values = {} if claims_response['verdict'] is None else {
                (v['prediction_id'], v['reference_id']): v for v in claims_response['verdict']['results']}
            for pair in pairs:
                key = f"p{pair['prediction_index']:04d}", f"r{pair['option_index']:04d}"
                if {'prediction_id': key[0], 'reference_id': key[1]} not in eligible:
                    continue
                raw = claims_values.get(key)
                pair['claims'] = {
                    **{k: v for k, v in claims_response.items() if k != 'verdict'},
                    'verdict': judge.bundle.normalize('claims', raw) if raw is not None else None,
                }
    result = {k: v for k, v in row.items() if k not in ('predictions', 'options')}
    result['method'] = judge.bundle.method
    result['pairs'] = pairs
    result['stage_responses'] = {'need': need_response, 'claims': claims_response}
    result['scores'] = score_pairs(pairs, row['prediction_count'], row['gt_count'],
                                   evaluate_claims=evaluate_claims)
    terminal_metric = 'overall' if evaluate_claims else 'time'
    result['status'] = 'complete' if result['scores'][terminal_metric]['status'] == 'complete' else 'pending'
    return result


def evaluate_sampler_score_twice(row, judge, max_input_chars=120000, evaluate_claims=True):
    """Whole-sampler 0-2 prefilter, then isolated scoring for every 3-5 pair.

    The prefilter is routing only. A 0-2 is a terminal rejection; a 3-5
    receives a separate request containing only the two Need strings. The
    isolated 1-4 scale rejects 1-2 and accepts 3-4.
    """
    from matching import score_pairs

    if judge.bundle.method != 'score-twice':
        raise ValueError('Two-stage evaluator requires the score-twice prompt bundle')
    payload = build_input(row)
    pairs = []
    prefilter_response = None
    isolated_responses = []
    claims_response = None
    if row['predictions'] and row['options']:
        prefilter_response = judge_once('need_prefilter', payload, judge, max_input_chars)
        prefilter_values = {} if prefilter_response['verdict'] is None else {
            (v['prediction_id'], v['reference_id']): v for v in prefilter_response['verdict']['results']}
        eligible = []
        for i, prediction in enumerate(row['predictions'], 1):
            for j, option in enumerate(row['options'], 1):
                pid, rid = f'p{i:04d}', f'r{j:04d}'
                raw_prefilter = prefilter_values.get((pid, rid))
                prefilter_verdict = (
                    judge.bundle.normalize('need_prefilter', raw_prefilter)
                    if raw_prefilter is not None else None
                )
                need_transport = {k: v for k, v in prefilter_response.items() if k != 'verdict'}
                final_verdict = prefilter_verdict
                match_score = None
                source = 'prefilter_unresolved'
                if prefilter_verdict is not None and prefilter_verdict['status'] == 'evaluated':
                    if raw_prefilter['score'] < SCORE_TWICE_PREFILTER_MINIMUM:
                        source = 'prefilter_rejected'
                    else:
                        isolated_payload = build_isolated_input(prediction, option)
                        isolated_response = judge_once('need_isolated', isolated_payload, judge, max_input_chars)
                        raw_isolated = None
                        if isolated_response['verdict'] is not None:
                            raw_isolated = isolated_response['verdict']['results'][0]
                        final_verdict = (
                            judge.bundle.normalize('need_isolated', raw_isolated)
                            if raw_isolated is not None else None
                        )
                        match_score = raw_isolated['score'] if raw_isolated is not None else None
                        need_transport = {k: v for k, v in isolated_response.items() if k != 'verdict'}
                        source = 'isolated'
                        isolated_responses.append({
                            'prediction_index': i,
                            'option_index': j,
                            'gt_need_id': option['gt_need_id'],
                            'alternative_id': option['alternative_id'],
                            **isolated_response,
                        })
                time_pass = prediction['committed_after'] in option['window']
                skipped = 'skipped_need_unresolved'
                if final_verdict is not None and final_verdict['status'] == 'evaluated':
                    if not final_verdict['pass']:
                        skipped = 'skipped_need_failed'
                    elif not time_pass:
                        skipped = 'skipped_time_failed'
                    else:
                        eligible.append({'prediction_id': pid, 'reference_id': rid})
                pairs.append({
                    'prediction_index': i,
                    'option_index': j,
                    'gt_need_id': option['gt_need_id'],
                    'alternative_id': option['alternative_id'],
                    'committed_after': prediction['committed_after'],
                    'window': option['window'],
                    'time_pass': time_pass,
                    'match_score': match_score,
                    'prefilter': {
                        **{k: v for k, v in prefilter_response.items() if k != 'verdict'},
                        'verdict': prefilter_verdict,
                    },
                    'need': {**need_transport, 'source': source, 'verdict': final_verdict},
                    'claims': {'status': skipped},
                })
        if not evaluate_claims:
            eligible_keys = {(item['prediction_id'], item['reference_id']) for item in eligible}
            for pair in pairs:
                key = f"p{pair['prediction_index']:04d}", f"r{pair['option_index']:04d}"
                if key in eligible_keys:
                    pair['claims'] = {'status': 'not_evaluated'}
            claims_response = {'status': 'not_evaluated'}
        elif eligible:
            claims_response = judge_once('claims', {**payload, 'pairs': eligible}, judge, max_input_chars)
            claims_values = {} if claims_response['verdict'] is None else {
                (v['prediction_id'], v['reference_id']): v for v in claims_response['verdict']['results']}
            eligible_keys = {(item['prediction_id'], item['reference_id']) for item in eligible}
            for pair in pairs:
                key = f"p{pair['prediction_index']:04d}", f"r{pair['option_index']:04d}"
                if key not in eligible_keys:
                    continue
                raw = claims_values.get(key)
                pair['claims'] = {
                    **{k: v for k, v in claims_response.items() if k != 'verdict'},
                    'verdict': judge.bundle.normalize('claims', raw) if raw is not None else None,
                }
    result = {k: v for k, v in row.items() if k not in ('predictions', 'options')}
    result['method'] = judge.bundle.method
    result['pairs'] = pairs
    result['stage_responses'] = {
        'need_prefilter': prefilter_response,
        'need_isolated': isolated_responses,
        'claims': claims_response,
    }
    result['scores'] = score_pairs(
        pairs, row['prediction_count'], row['gt_count'], evaluate_claims=evaluate_claims)
    terminal_metric = 'overall' if evaluate_claims else 'time'
    result['status'] = 'complete' if result['scores'][terminal_metric]['status'] == 'complete' else 'pending'
    return result
