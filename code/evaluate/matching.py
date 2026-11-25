"""Deterministic maximum-cardinality matching, with OR options sharing a GT slot."""

from __future__ import annotations


def maximum_matching(edges: list[dict]) -> list[dict]:
    """An edge identifies a prediction, GT Need, and one supporting alternative.

    Stable prediction/GT/alternative ordering breaks ties without inspecting
    model identity, prose similarity, or a more favorable experiment group.
    Augmenting paths (not greedy selection) maximize cardinality.
    """
    adjacency: dict[int, dict[str, dict]] = {}
    for edge in sorted(edges, key=lambda x: (
        x['prediction_index'], x['gt_need_id'], x.get('alternative_id') or ''
    )):
        adjacency.setdefault(edge['prediction_index'], {}).setdefault(edge['gt_need_id'], edge)
    assigned: dict[str, dict] = {}

    def augment(prediction: int, visited: set[str]) -> bool:
        for gt, edge in sorted(adjacency[prediction].items()):
            if gt in visited:
                continue
            visited.add(gt)
            if gt not in assigned or augment(assigned[gt]['prediction_index'], visited):
                assigned[gt] = edge
                return True
        return False

    for prediction in sorted(adjacency):
        augment(prediction, set())
    return sorted(assigned.values(), key=lambda x: (x['prediction_index'], x['gt_need_id']))


def maximum_weight_matching(edges: list[dict]) -> list[dict]:
    """Maximize cardinality first, then the sum of integer match_score values.

    Score Twice uses this after its frozen threshold has selected eligible
    edges. Stable edge insertion and strict shortest-path improvements make
    equal-weight ties deterministic.
    """
    if not edges:
        return []
    best: dict[tuple[int, str], dict] = {}
    for edge in sorted(edges, key=lambda x: (
        x['prediction_index'], x['gt_need_id'], x.get('alternative_id') or ''
    )):
        key = edge['prediction_index'], edge['gt_need_id']
        current = best.get(key)
        if current is None or edge.get('match_score', 0) > current.get('match_score', 0):
            best[key] = edge
    candidates = list(best.values())
    target_flow = len(maximum_matching(candidates))
    if not target_flow:
        return []

    predictions = sorted({edge['prediction_index'] for edge in candidates})
    gt_ids = sorted({edge['gt_need_id'] for edge in candidates})
    source = 0
    prediction_node = {p: i + 1 for i, p in enumerate(predictions)}
    gt_offset = 1 + len(predictions)
    gt_node = {g: gt_offset + i for i, g in enumerate(gt_ids)}
    sink = gt_offset + len(gt_ids)
    graph: list[list[dict]] = [[] for _ in range(sink + 1)]

    def add_edge(start: int, end: int, capacity: int, cost: int) -> int:
        forward_index = len(graph[start])
        reverse_index = len(graph[end])
        graph[start].append({'to': end, 'rev': reverse_index, 'cap': capacity, 'cost': cost})
        graph[end].append({'to': start, 'rev': forward_index, 'cap': 0, 'cost': -cost})
        return forward_index

    for prediction in predictions:
        add_edge(source, prediction_node[prediction], 1, 0)
    links = []
    for edge in sorted(candidates, key=lambda x: (
        x['prediction_index'], x['gt_need_id'], x.get('alternative_id') or ''
    )):
        start = prediction_node[edge['prediction_index']]
        index = add_edge(start, gt_node[edge['gt_need_id']], 1, -int(edge.get('match_score', 0)))
        links.append((start, index, edge))
    for gt_id in gt_ids:
        add_edge(gt_node[gt_id], sink, 1, 0)

    infinity = 10 ** 18
    for _ in range(target_flow):
        distance = [infinity] * len(graph)
        parent = [None] * len(graph)
        distance[source] = 0
        for _ in range(len(graph) - 1):
            changed = False
            for start in range(len(graph)):
                if distance[start] == infinity:
                    continue
                for index, residual in enumerate(graph[start]):
                    if residual['cap'] <= 0:
                        continue
                    candidate = distance[start] + residual['cost']
                    if candidate < distance[residual['to']]:
                        distance[residual['to']] = candidate
                        parent[residual['to']] = (start, index)
                        changed = True
            if not changed:
                break
        if parent[sink] is None:
            raise ValueError('Weighted matching could not realize maximum cardinality')
        node = sink
        while node != source:
            start, index = parent[node]
            residual = graph[start][index]
            residual['cap'] -= 1
            graph[node][residual['rev']]['cap'] += 1
            node = start

    selected = [edge for start, index, edge in links if graph[start][index]['cap'] == 0]
    return sorted(selected, key=lambda x: (x['prediction_index'], x['gt_need_id']))


def route_global_top1(entries: list[dict]) -> tuple[list[dict], dict[tuple[int, int], str]]:
    """Apply the frozen Stage-1 threshold, OR collapse and one-to-one routing.

    The input route_weight is produced by the frozen Judge. This function is
    deterministic and never infers or changes semantic scores.
    """
    statuses = {(entry["prediction_index"], entry["option_index"]): "filtered_score_0_2"
                for entry in entries}
    eligible = [entry for entry in entries if entry["level"] >= 3]
    for entry in eligible:
        statuses[(entry["prediction_index"], entry["option_index"])] = "eligible_not_selected"

    collapsed: dict[tuple[int, str], dict] = {}
    for entry in sorted(eligible, key=lambda item: (
        item["prediction_index"], item["gt_need_id"], item.get("alternative_id") or "",
        item["option_index"],
    )):
        key = entry["prediction_index"], entry["gt_need_id"]
        current = collapsed.get(key)
        if current is None or entry["route_weight"] > current["route_weight"]:
            if current is not None:
                statuses[(current["prediction_index"], current["option_index"])] = "or_alternative_not_selected"
            collapsed[key] = entry
        elif current is not entry:
            statuses[(entry["prediction_index"], entry["option_index"])] = "or_alternative_not_selected"

    matching_edges = [{
        "prediction_index": entry["prediction_index"],
        "gt_need_id": entry["gt_need_id"],
        "alternative_id": entry.get("alternative_id"),
        "match_score": entry["route_weight"],
        "option_index": entry["option_index"],
    } for entry in collapsed.values()]
    selected = maximum_weight_matching(matching_edges)
    selected_keys = {(edge["prediction_index"], edge["gt_need_id"]): edge
                     for edge in selected}
    selected_entries = []
    for key, entry in collapsed.items():
        chosen = selected_keys.get(key)
        if chosen is not None and chosen["option_index"] == entry["option_index"]:
            statuses[(entry["prediction_index"], entry["option_index"])] = "selected_global_top1"
            selected_entries.append(entry)
        else:
            statuses[(entry["prediction_index"], entry["option_index"])] = "global_conflict_not_selected"
    return sorted(selected_entries, key=lambda item: (
        item["prediction_index"], item["gt_need_id"], item.get("alternative_id") or ""
    )), statuses


def score_pairs(pairs: list[dict], prediction_count: int, gt_count: int,
                evaluate_claims: bool = True) -> dict:
    """Unresolved judgments do not silently become negative edges."""
    need_edges, time_edges, overall_edges = [], [], []
    need_pending = claims_pending = 0
    for pair in pairs:
        verdict = pair['need']['verdict']
        if verdict is None or verdict['status'] != 'evaluated':
            need_pending += 1
            continue
        if not verdict['pass']:
            continue
        edge = {k: pair[k] for k in ('prediction_index', 'gt_need_id', 'alternative_id')}
        if type(pair.get('match_score')) is int:
            edge['match_score'] = pair['match_score']
        need_edges.append(edge)
        if not pair['time_pass']:
            continue
        time_edges.append(edge)
        if evaluate_claims:
            claims = pair['claims'].get('verdict')
            if claims is None or claims['status'] != 'evaluated':
                claims_pending += 1
            elif claims['pass']:
                overall_edges.append(edge)
    out = {'pending_need_pairs': need_pending, 'pending_claims_pairs': claims_pending}
    for metric, edges, pending in (
        ('need', need_edges, need_pending),
        ('time', time_edges, need_pending),
        ('overall', overall_edges, need_pending + claims_pending),
    ):
        if metric == 'overall' and not evaluate_claims:
            out[metric] = {'status': 'not_evaluated', 'counts': None, 'matches': None}
            continue
        if pending:
            out[metric] = {'status': 'pending', 'counts': None, 'matches': None}
            continue
        selected = (maximum_weight_matching(edges)
                    if any('match_score' in edge for edge in edges)
                    else maximum_matching(edges))
        tp = len(selected)
        out[metric] = {
            'status': 'complete', 'matches': selected,
            'counts': {'tp': tp, 'fp': prediction_count - tp, 'fn': gt_count - tp},
        }
    return out
