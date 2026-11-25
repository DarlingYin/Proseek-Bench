# Score Twice - Stage 2 isolated 1-4 Need judgement

Judge exactly one Prediction Need against exactly one Reference Need.

Judge Need identity only. Do not judge Claims, evidence sufficiency, factual
support, Commit timing, writing quality, or whether every execution detail is
repeated.

## Mandatory decision order

First decide whether both Needs identify the same concrete assistance task.
Compare the intended help outcome, the primary target, and the desired change
or resolution.

Ignore surface phrasing and grammatical mood. Communication, reminder,
notification, suggestion, offer, request, guidance, help, and direct imperative
wording may express the same embedded assistance task. When a Need states that
a target must, should, or needs to reach a state, treat that state change as
the embedded assistance goal.

Need identity concerns the intended assistance outcome, not whether the user,
assistant, or another component performs the eventual interface operation. Do
not distinguish an explicit communication that a repair should occur from a
direct statement of that same repair. Actor or delivery-mode wording is not an
authority conflict. Authority conflicts require an explicit completion claim,
unilateral execution requirement, or prohibited operation.

For task identity, apply this domain-independent normalization before scoring:

`COMMUNICATE(USER, ACTION(TARGET), PURPOSE) -> ACTION(TARGET)`

Strip the communication wrapper and subordinate purpose, then compare the
embedded ACTION and TARGET. If the other Need directly states the same action
on the same target in the same direction, the identity gate passes and the
score must be 3 or 4. Additional compatible purpose details or a less-specific
time do not reverse that identity decision.

A direct imperative does not by itself claim execution or completion. Making a
later operation possible does not claim that the later operation has been
performed. Additional compatible context or subordinate actions do not change
Need identity.

If the intended help outcome, primary target, and desired resolution match,
and neither Need explicitly requires an incompatible operation or target
state, the score MUST be 3 or 4. Only then judge completeness.

## Scale

**1 - Different or incompatible Need.**

The assistance goal, primary target, desired resolution, or explicit authority
is materially different. Use 1 only for a clear mismatch or conflict, not for
an omission.

**2 - Related situation, but the concrete assistance Need is not identified.**

The Prediction recognizes the same situation or broad problem but does not
identify the Reference's core assistance task. A valid score-2 reason must
explain how the Prediction could be fully satisfied while the Reference's core
task remains entirely unaddressed.

Do not use 2 merely because the Prediction omits evidence, exact values,
identifiers, auxiliary steps, repeated restrictions, restoration instructions,
or other execution details after the same concrete task is identifiable.

**3 - Same concrete Need, incomplete specification.**

The intended help outcome, primary target, and desired resolution match. Some
parameters, evidence-derived values, execution details, auxiliary conditions,
or repeated boundaries are absent, broader, or less specific.

**4 - Same concrete Need, sufficiently complete.**

The same task and the main conditions needed to distinguish it from materially
different alternatives are clear. Exact wording and complete Claims evidence
are not required.

## Core versus detail

A condition is task-defining only when changing or omitting it permits a
materially different assistance task, primary target, or repair outcome. It is
a detail when the concrete task remains identifiable and the missing
information can be recovered without changing what help is needed.

Candidate identity is task-defining when the Reference has already selected a
particular candidate. A Prediction that names only a broader class containing
multiple materially different candidates has not identified that concrete
Need and should receive 2. By contrast, omitted attributes of an already fixed
object are details and normally receive 3.

Do not treat every Reference clause as a separate mandatory branch. A missing
clause supports score 2 only when it defines an independent co-primary outcome
and fully satisfying the Prediction can leave that outcome entirely undone.

For score 1 or 2, the reason must identify the core task difference and explain
how the Reference Need can remain unmet. If no such explanation can be given,
use score 3.

Do not use actor choice, communication-versus-execution framing, grammatical
mood, compatible extra context, or omitted non-core details as the core
difference for score 1 or 2.

Scores 1-2 fail. Scores 3-4 pass.

Return only JSON with exactly one result:

{"results":[{"prediction_id":"p0001","reference_id":"r0001","score":3,"reason":"Same concrete assistance task; only execution details are less specific."}]}

The result must contain exactly prediction_id, reference_id, integer score 1-4,
and a concise nonempty reason. Do not browse or output surrounding prose.
