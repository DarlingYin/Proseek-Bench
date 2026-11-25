# Score Twice - Stage 1 high-recall router

Compare every supplied Prediction Need with every supplied reference Need.
Return every requested pair exactly once. References with the same gt_need_id
are alternatives, not jointly required variants. Do not rank pairs or perform
matching; code handles OR collapse and global one-to-one assignment.

Judge Need identity only. Do not use Claims, evidence quality, Commit time,
model identity, neighboring predictions, or unstated world facts to repair or
penalize a Need.

## Role of this stage

This is a high-recall router, not the final semantic judge. Scores 0-2 delete a
pair permanently; scores 3-5 send it to isolated Stage 2. Therefore, when the
boundary between 2 and 3 is genuinely uncertain, use 3.

Privately normalize each Need into four semantic slots:

1. assistance action;
2. primary object or candidate;
3. desired target state or repair direction;
4. explicit authority boundary.

Speech-act framing is not an assistance-action difference. A notification,
reminder, suggestion, offer, request, guidance statement, or direct imperative
may express the same embedded action. Authority differs only when the text
explicitly performs or claims an operation, or explicitly requires or forbids
one. Do not infer execution merely from grammatical mood.

A reference written as a direct imperative does not by itself assert execution
authority or completion. Need text describes the assistance goal, not a tool
call. Treat a direct repair statement and a communication that the same repair
must or should occur as the same action family unless one side explicitly
claims completion, unilateral execution, or a prohibited operation.

When a communication wrapper states that a particular object must, should, or
needs to reach a repair state, treat that stated repair as the embedded
assistance action. Do not reduce the pair merely because one Need communicates
the repair while the other states it directly.

Enabling a prerequisite so that a later task can become possible does not
claim that the later task is performed. A purpose or capability clause does
not expand authority. It conflicts with a stop boundary only when the
Prediction explicitly instructs, performs, or claims the prohibited later
operation.

A negative boundary on a later operation is not an additional instruction to
undo the prerequisite. Do not invent a restoration or revocation action from
wording that merely says to stop before the later operation.

If the assistance action, primary object, and repair direction match, and the
Prediction does not explicitly assert an incompatible operation or target
state, the score MUST be at least 3.

This minimum also applies when the Prediction is a compatible superset: once
it contains the reference's action, object, and repair direction, additional
context or subordinate actions cannot lower it below 3 unless they introduce
an explicit conflicting object, target state, or authority. Do not require the
Prediction to focus solely on the reference action.

Missing exact values, identifiers, evidence, timing estimates, auxiliary
steps, restoration instructions, repeated negative clauses, or unchanged
context do not justify 0-2 after the concrete task is identifiable. A missing
co-primary branch justifies 0-2 only when the Prediction explicitly narrows,
substitutes, or conflicts with that branch; mere non-mention is routed onward.

## Deletion witness required for scores 0-2

Every score from 0 through 2 must provide a decisive failure witness with:

- `conflict_slot`: exactly one of `unrelated_task`, `action`, `object`,
  `repair_direction`, or `authority`;
- `gt_requirement`: an exact substring of the reference Need;
- `prediction_evidence`: an exact substring of the Prediction Need showing the
  different action/object/direction or explicit authority conflict;
- `consequence`: how fully satisfying the Prediction can leave the referenced
  task-defining requirement materially unsatisfied.

The witness is invalid if it relies only on different framing, omitted
recoverable detail, omitted evidence, or an operation the Prediction never
claims. If no valid witness can be written from the two Need strings, assign
3-5.

## Scale

0 - Unrelated or explicitly opposite task.

1 - Related context, but a different assistance action, primary object, or
repair direction is explicitly expressed.

2 - The concrete task is related, but the Prediction explicitly commits to an
incompatible target state or authority boundary, substitutes a different
task-defining branch, or narrows scope so the GT task can remain undone.

3 - Same concrete task after normalization; task identity is clear, while
recoverable details or boundaries may be incomplete.

4 - Same concrete task with the decisive conditions substantially present.

5 - Semantically equivalent task, direction, scope, and authority.

Return exactly one JSON object with a `results` array; never return a bare
array. Every result must contain `prediction_id`, `reference_id`, `score`,
`reason`, and `material_deviation`. For scores 0-2, `material_deviation` must
contain the four witness fields. For scores 3-5, it must be null. Use null score
only when malformed or ambiguous reference text prevents any defensible
comparison; explain why in `reason`.

