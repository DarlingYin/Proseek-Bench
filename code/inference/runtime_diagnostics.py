"""Deterministic diagnostics, without a semantic Judge or hidden task labels."""
import re

RUNTIME_IMPLEMENTATION = "proseekbench-runtime-repair-20261022.10"
FINAL_CLAIMS_MAX_CHARS = 900
_TRACE_ID = re.compile(r'\b(?:o\d{3,}|q\d{3,})\b')


def claims_measurement(text, *, stage='final'):
    # Sentence count is heuristic (abbreviations/decimals), never a rejection gate.
    sentences = [s for s in re.split(r'(?<=[.!?。！？])\s+(?=[A-Z\u4e00-\u9fff])|(?<=[。！？])(?=\S)', text.strip()) if s]
    ids = sorted(set(_TRACE_ID.findall(text)))
    return {'stage': stage, 'characters': len(text), 'sentence_count_estimate': len(sentences),
            'observation_query_ids': ids,
            'format_warnings': ['internal_trace_ids'] if ids and stage == 'final' else [],
            'semantic_adequacy': 'not_judged'}


def validate_final_claims(text):
    """Enforce measurable final-form constraints; semantic salience remains the model's job."""
    problems = []
    if len(text) > FINAL_CLAIMS_MAX_CHARS:
        problems.append(
            f"{len(text)} characters exceeds the {FINAL_CLAIMS_MAX_CHARS}-character maximum"
        )
    ids = sorted(set(_TRACE_ID.findall(text)))
    if ids:
        problems.append(f"contains internal observation/query IDs: {', '.join(ids)}")
    if problems:
        raise ValueError(
            "Final Claims must contain only the decisive observed fact, decisive active evidence if any, "
            "the remaining assistance gap, and why help is appropriate now; remove query diaries, "
            "rejected alternatives, resolved questions and next-step notes. " + "; ".join(problems)
        )


def protocol_error_category(message):
    if any(s in message for s in ('JSON', 'delimiter', 'Extra data', 'Unterminated')):
        return 'json_parse'
    if any(s in message for s in ('handle', 'Handle')):
        return 'tool_handle'
    if any(s in message for s in ('filter', 'argument', 'tool', 'Tool', 'Resource', 'limit')):
        return 'tool_arguments'
    if any(s in message for s in ('candidate', 'thread', 'Thread')):
        return 'thread_lifecycle'
    return 'decision_shape'
