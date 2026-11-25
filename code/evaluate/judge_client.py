"""Whole-sampler Judge transport, strict validation, cache and request audit."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
import re


_CURL_STATUS_MARKER = b"\n__PROSEEKBENCH_HTTP_STATUS__:"


class CurlHTTPError(RuntimeError):
    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"HTTP {status}")


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def normalize_score_twice_transport(stage: str, verdict, *, default_reason="Same normalized task."):
    """Repair only deterministic isolated-score envelope/label formatting."""
    changes = []
    if isinstance(verdict, list):
        verdict = {'results': verdict}
        changes.append('wrap_bare_result_array')
    elif isinstance(verdict, dict) and 'results' not in verdict:
        if isinstance(verdict.get('result'), list):
            verdict = {'results': verdict['result']}
            changes.append('rename_result_array_to_results')
        elif isinstance(verdict.get('result'), dict):
            verdict = {'results': [verdict['result']]}
            changes.append('wrap_result_object_as_results')
        elif {'prediction_id', 'reference_id', 'score'} <= set(verdict):
            verdict = {'results': [verdict]}
            changes.append('wrap_direct_result_object')
    if not isinstance(verdict, dict) or not isinstance(verdict.get('results'), list):
        return verdict, changes
    slot_aliases = {
        'assistance action': 'action', 'action family': 'action',
        'primary object': 'object', 'primary object or candidate': 'object',
        'target state': 'repair_direction', 'repair direction': 'repair_direction',
        'target state / repair direction': 'repair_direction',
        'authority boundary': 'authority', 'authority mode': 'authority',
    }
    for index, row in enumerate(verdict['results']):
        if not isinstance(row, dict):
            continue
        pair_id = row.pop('pair_id', None)
        if (isinstance(pair_id, str) and 'prediction_id' not in row
                and 'reference_id' not in row):
            match = re.fullmatch(r'(p[0-9]{4,})-(r[0-9]{4,})', pair_id)
            if match:
                row['prediction_id'], row['reference_id'] = match.groups()
                changes.append(f'result[{index}]:split_pair_id')
        deviation = row.get('material_deviation')
        if stage == 'need_prefilter' and isinstance(deviation, dict):
            slot = deviation.get('conflict_slot')
            normalized = slot_aliases.get(slot)
            if normalized is not None:
                deviation['conflict_slot'] = normalized
                changes.append(f'result[{index}]:normalize_conflict_slot')
        if row.get('reason') is None or 'reason' not in row:
            row['reason'] = (deviation.get('consequence', '')
                             if isinstance(deviation, dict) else default_reason)
            changes.append(f'result[{index}]:fill_nonsemantic_reason')
        if stage == 'claims_isolated':
            checks = row.get('completion_checks')
            if not isinstance(checks, list):
                continue
            role_aliases = {
                'target_action': 'action',
                'post_commit_action': 'action',
                'assistance_action': 'action',
                'assistance': 'action',
                'final_action': 'action',
                'final_assistance': 'action',
                'action_boundary': 'boundary',
                'hard_boundary': 'boundary',
                'derived_resolution': 'target',
                'resolution': 'target',
                'required_resolution': 'target',
                'assistance_target': 'target',
                'applicability': 'prerequisite',
                'compatibility': 'prerequisite',
                'unresolved_gap': 'prerequisite',
                'detail': 'prerequisite',
                'context': 'prerequisite',
                'supporting': 'prerequisite',
                'supplementary': 'prerequisite',
            }
            for check_index, check in enumerate(checks):
                if not isinstance(check, dict):
                    continue
                if check.get('materiality') == 'non-decisive':
                    check['materiality'] = 'non_decisive'
                    changes.append(
                        f'result[{index}].completion_checks[{check_index}]:'
                        'normalize_non_decisive'
                    )
                if (check.get('evidence_status') == 'not_required'
                        and 'evidence_source' not in check):
                    check['evidence_source'] = 'not_required'
                    changes.append(
                        f'result[{index}].completion_checks[{check_index}]:'
                        'fill_not_required_evidence_source'
                    )
                role = check.get('completion_role')
                if role in role_aliases:
                    check['completion_role'] = role_aliases[role]
                    changes.append(
                        f'result[{index}].completion_checks[{check_index}]:'
                        'normalize_completion_role_alias'
                    )
    return verdict, changes


class JudgeClient:
    def __init__(self, bundle, output: Path, *, api_base: str,
                 model: str, api_key: str, temperature: float = 0.0,
                 max_tokens: int = 8192, timeout: float = 180,
                 retries: int = 0, response_format: str = 'json_object',
                 reasoning_effort: str | None = 'low',
                 http_transport: str = 'urllib', curl_resolve: str | None = None,
                 transport=None):
        if http_transport not in {'urllib', 'curl', 'curl_direct'}:
            raise ValueError('http_transport must be urllib, curl, or curl_direct')
        self.bundle, self.output, self.api_key = bundle, output, api_key
        self.settings = dict(api_base=api_base.rstrip('/'), model=model, temperature=temperature,
                             max_tokens=max_tokens, timeout=timeout, retries=retries,
                             response_format=response_format, reasoning_effort=reasoning_effort,
                             http_transport=http_transport)
        if curl_resolve:
            self.settings['curl_resolve'] = curl_resolve
        self.transport = transport or self._http
        self.lock = threading.Lock()
        self.pair_locks = {}
        self.auth_failed = threading.Event()

    def _http(self, payload: dict) -> dict:
        if self.settings['http_transport'] in {'curl', 'curl_direct'}:
            return self._curl(payload)
        request = urllib.request.Request(
            self.settings['api_base'] + '/chat/completions',
            data=canonical(payload).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.api_key},
        )
        with urllib.request.urlopen(request, timeout=self.settings['timeout']) as response:
            return json.load(response)

    def _curl(self, payload: dict) -> dict:
        """POST JSON through the system SecureTransport curl without exposing the key."""
        read_fd, write_fd = os.pipe()
        try:
            escaped_key = self.api_key.replace('\\', '\\\\').replace('"', '\\"')
            config = (
                f'header = "Authorization: Bearer {escaped_key}"\n'
                'header = "Content-Type: application/json"\n'
            ).encode('utf-8')
            os.write(write_fd, config)
        finally:
            os.close(write_fd)
        command = [
            '/usr/bin/curl', '--silent', '--show-error', '--connect-timeout', '30',
            '--max-time', str(int(self.settings['timeout'])), '--config', f'/dev/fd/{read_fd}',
            '--data-binary', '@-', '--write-out',
            _CURL_STATUS_MARKER.decode('ascii') + '%{http_code}',
            self.settings['api_base'] + '/chat/completions',
        ]
        if self.settings['http_transport'] == 'curl_direct':
            command[1:1] = ['--noproxy', '*']
        if self.settings.get('curl_resolve'):
            command[1:1] = ['--resolve', self.settings['curl_resolve']]
        try:
            completed = subprocess.run(
                command, input=canonical(payload).encode('utf-8'), capture_output=True,
                check=False, pass_fds=(read_fd,), timeout=self.settings['timeout'] + 15,
            )
        finally:
            os.close(read_fd)
        if completed.returncode != 0:
            raise OSError(f'curl transport failed with exit code {completed.returncode}')
        try:
            body, status_raw = completed.stdout.rsplit(_CURL_STATUS_MARKER, 1)
            status = int(status_raw.decode('ascii'))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError('curl response is missing the HTTP status marker') from exc
        if status < 200 or status >= 300:
            raise CurlHTTPError(status)
        return json.loads(body.decode('utf-8'))

    def _log(self, record: dict) -> None:
        # Credentials and headers are never serialized. Redact accidental key echoes.
        line = canonical(record)
        if self.api_key:
            line = line.replace(self.api_key, '[REDACTED]')
        with self.lock:
            with (self.output / 'judge-calls.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(line + '\n')

    def judge_once(self, stage: str, inputs: dict) -> dict:
        self.bundle.validate_input(stage, inputs)
        return self._request(stage, inputs)

    def _request(self, stage: str, inputs: dict) -> dict:
        key = digest({'stage': stage, 'method': self.bundle.method, 'inputs': inputs,
                      'prompts': self.bundle.hashes, 'settings': self.settings})
        with self.lock:
            pair_lock = self.pair_locks.setdefault(key, threading.Lock())
        with pair_lock:
            cache = self.output / 'cache' / (key + '.json')
            if cache.exists():
                record = read_json(cache)
                if record.get('cache_key') != key:
                    raise ValueError('Invalid cache key')
                self.bundle.validate(stage, inputs, record['verdict'])
                return {**record, 'cache_hit': True}
            if self.auth_failed.is_set():
                return {'cache_key': key, 'verdict': None, 'error': 'endpoint_authentication_failed', 'cache_hit': False}
            payload = {
                'model': self.settings['model'], 'temperature': self.settings['temperature'],
                'max_tokens': self.settings['max_tokens'],
                'messages': [{'role': 'system', 'content': self.bundle.system(stage)},
                             {'role': 'user', 'content': canonical(inputs)}],
            }
            if self.settings['response_format'] == 'json_object':
                payload['response_format'] = {'type': 'json_object'}
            if self.settings['reasoning_effort'] is not None:
                payload['reasoning_effort'] = self.settings['reasoning_effort']
            error = 'no_response'
            for attempt in range(self.settings['retries'] + 1):
                start = time.perf_counter()
                response = None
                retryable = True
                try:
                    response = self.transport(payload)
                    choice = response['choices'][0]
                    if choice.get('finish_reason') not in ('stop', 'end_turn'):
                        raise ValueError('Judge response did not finish normally')
                    verdict = json.loads(choice['message']['content'])
                    repairs = []
                    if self.bundle.method == 'score-twice':
                        verdict, repairs = normalize_score_twice_transport(stage, verdict)
                    elif self.bundle.method == 'claims-1-4':
                        verdict, repairs = normalize_score_twice_transport(
                            stage, verdict, default_reason='Adequate Claims support.'
                        )
                    repairs.extend(self.bundle.repair_substring_witnesses(stage, inputs, verdict))
                    self.bundle.validate(stage, inputs, verdict)
                    self._log({'cache_key': key, 'stage': stage, 'attempt': attempt + 1,
                               'request': payload, 'response': response, 'status': 'valid',
                               'transport_repairs': repairs,
                               'elapsed_seconds': time.perf_counter() - start})
                    record = {'cache_key': key, 'verdict': verdict, 'cache_hit': False}
                    # Cache both pass/fail and review_required. Do not reroll a valid verdict.
                    write_json(cache, record)
                    return record
                except urllib.error.HTTPError as exc:
                    error = f'http_{exc.code}'
                    retryable = exc.code in (408, 429) or exc.code >= 500
                    if exc.code in (401, 403):
                        self.auth_failed.set()
                except CurlHTTPError as exc:
                    error = f'http_{exc.status}'
                    retryable = exc.status in (408, 429) or exc.status >= 500
                    if exc.status in (401, 403):
                        self.auth_failed.set()
                except Exception as exc:
                    # Response body already records the invalid output without interpolating
                    # exception messages that could contain private transport data.
                    error = type(exc).__name__
                self._log({'cache_key': key, 'stage': stage, 'attempt': attempt + 1,
                           'request': payload, 'response': response, 'status': 'invalid_or_error',
                           'error': error, 'elapsed_seconds': time.perf_counter() - start})
                if not retryable or attempt == self.settings['retries']:
                    break
                time.sleep(min(2 ** attempt, 8))
            return {'cache_key': key, 'verdict': None, 'error': error, 'cache_hit': False}
