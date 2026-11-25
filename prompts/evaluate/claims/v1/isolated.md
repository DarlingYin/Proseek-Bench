# Claims v4 Gate B: Reference completion at Commit

Judge one Prediction whose Need identity, Commit time and source-grounded causal
adequacy already passed Gate A. Do not redo Need identity, writing style,
intervention usefulness or harmless speech-act wording. Judge only whether the
Prediction has reached the Reference Need's required resolution at Commit.

Ground-truth text defines the target and decision-changing prerequisites but is
never factual evidence. Use only pre-Commit Passive evidence, visual receipts
and actual query receipts. A fact matching the Reference is unsupported when it
is absent from admissible evidence.

First classify `reference_resolution_type`:

- `information_action`: asking, checking, locating, opening, inspecting or
  requesting confirmation is itself the final assistance target;
- `resolved_output`: the Reference requires the result of information gathering,
  such as a concrete value, identified object, verified state, supported choice,
  recommendation, comparison, modification or plan;
- `visible_direct_action`: decisive evidence is already visible and directly
  supports offering the Reference assistance.

Completion is evaluated at Commit. A promise to retrieve, verify, confirm,
compare or use a source during later assistance does not supply a source-derived
value or state needed for a `resolved_output`. Conversely, do not require the
post-Commit user-facing action itself to have been executed: an evidence-backed
offer to fill, modify, book, send or otherwise act may be complete when that is
the Reference assistance.

For every decisive completion requirement, separately judge:

- `prediction_status`: `resolved`, `deferred`, `missing`, or `different`;
- `evidence_status`: `supported`, `unsupported`, `contradicted`, or
  `not_required`.

Use `deferred` when the Prediction leaves a required internal lookup or
decision for later. Use `not_required` only for an action/boundary that has no
factual prerequisite beyond already supported evidence.

Include an applicability or unresolved-gap fact when changing it could make the
assistance unnecessary, duplicate, infeasible or unsafe. For absence, coverage,
availability, current-state, identity, capacity, compatibility, route, price or
historical-coverage requirements, demand evidence with appropriate scope.

Exclude incidental wording, names versus pronouns, redundant motivation,
already-tested Commit timing, observation citations and details whose removal
does not change the target, decision or hard boundary. Advisory expressions
such as flag, alert, surface, offer help, help coordinate or help confirm can
cover the same assistance when the corrective direction and boundaries match.

Set `completion_status`:

- `complete`: every decisive requirement is resolved and supported/not_required;
- `incomplete`: at least one decisive requirement is deferred, missing or
  unsupported;
- `different`: the target or resolution is materially different;
- `contradicted`: decisive evidence conflicts with the proposed resolution.

Scores: 4 = complete; 3 = complete with only non-decisive omissions; 2 =
incomplete; 1 = different or contradicted. Scores 3--4 pass.

Return exactly this JSON shape and no prose:

```json
{"results":[{"prediction_id":"p0001","reference_id":"r0001","reference_resolution_type":"resolved_output","completion_checks":[{"requirement":"decision-changing completion requirement","materiality":"decisive","completion_role":"prerequisite","prediction_status":"resolved","evidence_status":"supported","evidence_source":"query_evidence_before_commit","reason":"specific coverage and evidence basis"}],"completion_status":"complete","score":4,"reason":"concise completion judgment"}]}
```
