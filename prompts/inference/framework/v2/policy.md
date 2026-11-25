# Framework Decision and Memory Policy v5

## Choose one control

Use the current evidence to choose among these controls. None is a default response:

- **Search** when an observed activity or object raises a specific unknown that could change whether,
  how or when to help, and a proportionate legal query can investigate it. A visible name, returned
  fact or issued handle may provide the starting point. A complete Need, existing thread or known
  answer is not required. When such a query is available and needed for the decision, use it rather
  than guessing or waiting solely to save cost.
- **Commit** when evidence supports a concrete, useful, still-unmet need that is appropriate now.
  Its action, target, constraints and timing must be specific enough to help. Resolve material
  uncertainty through proportionate available queries first; an unchecked list or a narrow empty
  search alone does not establish a gap. When help is time-sensitive and further evidence cannot
  reasonably be obtained, a grounded proposal may retain explicit uncertainty. Certainty that the
  user will accept is not required. Sufficient evidence does not require another Search or Wait.
- **Wait** when neither a mature Need nor a useful grounded query is available, or the decision
  genuinely needs future passive evidence. Assess independent questions separately; ending one
  unproductive inquiry does not prevent another justified Search or Commit.

Search stays at the current observation and returns evidence for the next decision. Wait advances
one observation while retaining memory. Commit freezes the selected Need/Claims pairs and advances
one observation; it does not end the Episode. Follow `allowed_control_actions` and query availability.
Thread operations are applied before control, so create/update and Commit can occur in one decision
at the same observation.

## Maintain NeedThreads

Create a candidate when an observed activity and the available evidence support a specific possible
help gap. Update it as facts or uncertainty change; delete it when the gap becomes unnecessary or
incorrect. A candidate may remain uncommitted. Keep one thread per assistance goal, using updates for
refinements instead of committing a vague precursor and later submitting the same need again.
Every `thread_id` is unique for the entire Episode. Once a thread is committed or deleted, its ID is
retired permanently and must never be created, updated, deleted or committed again. The current
`thread_lifecycle.candidate_thread_ids` are the only IDs eligible for update/delete/commit; use a
fresh ID only for a genuinely new candidate.

Each thread contains:

- **Need:** one concise action sentence with a concrete help goal, target and necessary known
  parameters or conditions. Keep independent goals separate and do not invent missing parameters.
- **Candidate Claims:** working evidence memory containing relevant observed/query facts,
  constraints, the current hypothesis and material unanswered questions. Distinguish facts from
  inferences. Preserve information needed across observations and searches; this may be more detailed
  than final Claims and need not yet justify Commit. Replace the paragraph as evidence changes,
  rather than appending a query diary.
- **Final Claims:** at Commit, a concise causal paragraph explaining what was observed, what relevant
  queries established if any, why help remains unmet and why now. Preserve decisive constraints and
  uncertainty; omit resolved questions, observation/query IDs and redundant narration. There is no
  fixed sentence count, but Final Claims must be at most 900 characters. Omit the query part when no
  query evidence was used. Do not include rejected alternatives, a query diary, next-search plans or
  material already resolved. Summarization must not turn hypotheses into facts or invent urgency.
  Rewrite only when needed; content created and committed together must already satisfy this final form.

Use the existing `need` and `claims` fields without additional fields or checklist labels. Runtime
status and timestamps are read-only. Committed predictions cannot be edited or deleted, and the
same need must not be committed twice.
