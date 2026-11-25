"""Data-independent rule registry, returned only in execution provenance.

Every new rule needs a unique ID, trigger, transformation, invariant and negative
example. Exact event patterns link observed corrections to the declared rules.
Query accounting and Claims semantics intentionally are not compatibility rules.
"""
from copy import deepcopy
import hashlib
import json

REGISTRY_VERSION = 'proseekbench.compatibility-rules.20261021.1'


def rule(id, events, trigger, transformation, invariant, positive, negative):
    introduced = ('proseekbench-runtime-repair-20261021' if id in {'typed_relation_handle', 'single_id_to_array'}
                  else 'proseekbench-runtime-repair-20261021.2' if id == 'json_envelope'
                  else 'legacy_runtime_version_unrecorded')
    return {'rule_id': id, 'introduced_in_registry': REGISTRY_VERSION,
            'introduced_in_runtime_version': introduced,
            'registration_runtime_version': 'proseekbench-runtime-repair-20261021.2',
            'event_prefixes': events, 'trigger': trigger, 'transformation': transformation,
            'semantic_invariant': invariant, 'positive_example': positive, 'negative_example': negative}


RULES = [
    rule('operation_object_to_array', ['thread_ops_object_to_array'],
         'One thread_ops object.', 'Wrap the object as a one-element list.',
         'Same operation and field values.', '{thread_ops:{op:update,...}}', 'Two competing operations are not selected.'),
    rule('control_string_to_object', ['control_string_to_object', 'nest_top_level_thread_ids'],
         'Explicit string control with unambiguous associated IDs.', 'Wrap as type and move the supplied IDs.',
         'No action or target is inferred.', 'control=wait -> control={type:wait}', 'A missing control is not converted to Wait.'),
    rule('misplaced_control', ['move_wait_op_to_control', 'move_search_op_to_control', 'move_commit_op_to_control'],
         'Exactly one control is misplaced in thread_ops; control is absent.', 'Move that control to its correct field.',
         'Preserves the one explicitly expressed control.', 'thread_ops=[{op:wait}] -> control={type:wait}',
         'Two controls or a conflicting top-level control remain invalid.'),
    rule('readonly_lifecycle_fields', ['strip_runtime_field:'],
         'An operation copies Runtime-owned status/timestamps.', 'Remove only explicitly listed read-only fields.',
         'Runtime owns lifecycle; Need and Claims are untouched.', 'update with status=candidate loses status only.',
         'Unknown claims_note is not silently discarded.'),
    rule('empty_update', ['drop_empty_update'],
         'Update contains only op and thread_id.', 'Drop the no-op.',
         'No content is updated.', '{op:update,thread_id:n}', 'An update with an unknown content field remains invalid.'),
    rule('missing_linkage', ['default_linked_thread_ids'],
         'Explicit Search lacks optional semantic linkage.', 'Supply an empty linked_thread_ids list.',
         'No candidate binding is guessed.', 'Search with no linked_thread_ids gets [].', 'Unknown supplied thread IDs are not dropped.'),
    rule('exact_resource_alias', ['map_resource_alias:'],
         'Tool equals one exact resource in the fixed catalog; fields are non-conflicting.',
         'Translate to search_records(resource,filters,limit).', 'Preserve resource, filter values and limit.',
         'tool=maps.route -> search_records(resource=maps.route)', 'babaquna.train is not guessed to mean starrail.train.'),
    rule('limit_placement', ['move_limit_outside_filters'],
         'Only one limit value exists, inside filters.', 'Move it to arguments.limit.',
         'Same query and requested maximum result count.', 'filters={limit:10} -> limit=10,filters={}',
         'Conflicting nested and outer limits are not arbitrarily resolved.'),
    rule('single_id_to_array', ['linked_thread_id_to_array', 'commit_thread_id_to_array'],
         'An ID-list field is a single string.', 'Wrap as a one-element list.',
         'Same target; no target added or removed.', 'thread_ids=n -> [n]', 'An empty Commit list is not filled from candidate state.'),
    rule('json_envelope', ['strip_code_fence', 'extract_json_envelope', 'unwrap_singleton_decision'],
         'One complete unambiguous JSON decision with a presentation wrapper.',
         'Remove presentation wrapper or singleton array.', 'No JSON content is completed or chosen among alternatives.',
         '[{thread_ops:[],control:{type:wait}}] -> its one object', 'Multiple objects, duplicate keys and truncated JSON fail.'),
    rule('typed_relation_handle', ['resolve_relation_handle:'],
         'Declared relation filter has an issued, accessible handle of an allowed type.',
         'Resolve the private key within that typed namespace.', 'Same identified parent object and exact child filter.',
         'merchant_id=chilemei:merchant:001 resolves only that merchant.',
         'A Tuantuan merchant handle cannot fill a Chilemei merchant filter.'),
]


def registry():
    return {'version': REGISTRY_VERSION, 'rules': deepcopy(RULES),
            'regression_suites': ['test_runtime_repairs.py', 'test_runtime_incremental.py', 'test_experiment_modes.py'],
            'legacy_stateful': 'Historical non-safe recovery only; may infer controls/targets. Not used by safe mode.'}


def registry_hash():
    return hashlib.sha256(json.dumps(registry(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def rule_ids(events):
    matches = set()
    for event in events:
        matching = [r['rule_id'] for r in RULES if any(event.startswith(p) for p in r['event_prefixes'])]
        if len(matching) != 1:
            raise ValueError(f'Unregistered or ambiguous safe compatibility event: {event}')
        matches.update(matching)
    return sorted(matches)
