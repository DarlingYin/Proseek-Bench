# Read-only Query Protocol v6

## Calls and references

Use `available_tools` to form the query chosen under the decision policy. Each resource inherits
`resource_defaults`; a resource field replaces the entire corresponding default, not a nested
merge. `query_schema` selects its parameter schema; `summary` and view `fields` select entries in
`field_sets`. These describe allowed fields, not a guarantee that every record contains them.
Use `runtime_tool_usage.resource_routing` for the fixed meaning of easily confused resources and
`runtime_tool_usage.exact_filter_values` for closed exact-value fields. These are generic interface
contracts, not Episode facts.

- `search_records(resource, filters?, limit?)` searches one declared resource. Put filters inside
  `filters`, with `limit` alongside `resource`. Omitted filters are empty and an omitted limit uses
  `default_limit`. Use only the resource's declared filters and ranges. Its `matches` states which
  fields are searched: `contains` is case-insensitive substring matching, `eq` is exact, and
  `gte`/`lte` are inclusive time bounds. A keyword does not search undeclared fields or perform
  semantic matching. Use an observed name or other grounded locator without guessing a private ID.
  For relation filters inside `search_records.filters`, `runtime_tool_usage.relation_filters` lists
  the accepted handle types. Put an issued handle of the correct type in the declared filter field;
  Runtime resolves the relation privately. Already observed literal business IDs remain usable.
  A handle is not interchangeable with any arbitrary ID or other handle type; unavailable, invented
  or wrong-type handles are errors.
- Exact filters such as route `mode` and order `status` are not natural-language semantic search.
  Use only a value already returned in this Episode or explicitly declared in the exact-value contract;
  otherwise omit that filter. Do not guess synonyms such as `driving`. For first discovery, prefer one
  grounded keyword or relation locator; do not combine multiple uncertain keyword/time/status/mode filters.
- `read_record(handle, view?)` reads an issued handle from the current Episode; the default view is
  `details`. The handle determines the App/resource, and the view must be declared for that resource.
- `read_user_state(app)` reads the declared account-state fields for one available App, without a
  prior handle. It does not enumerate that App's records.

Each call goes in `control` with `type="search"`, `tool`, `arguments` and `linked_thread_ids`. Link
the candidate threads it serves. Use `linked_thread_ids: []` for grounded discovery before a
specific candidate exists, including necessary follow-ups or corrections; create a candidate when
the evidence supports one. Each call selects one declared operation, with no automatic detail reads.

## Use the result

Search returns `items` and `coverage`; detail/account reads return `data`. All include a status and
cost. Preserve useful evidence in candidate Claims and use `search_ledger` for prior query coverage;
the ledger may omit older entries and does not replace evidence memory.

- **Sufficient result:** use the summary if it resolves the question. Continue the decision process
  without reading details or collecting corroboration that cannot change it.
- **Insufficient or truncated result:** read necessary details or follow a supported relation.
  If `coverage.complete` is false and omitted records could matter, increase the allowed limit or
  refine the query. Follow-ups may answer the same unresolved question.
- **Failed or invalid call:** use feedback and existing evidence to correct parameter placement,
  spelling, references or a demonstrably mistaken filter. Retry transient failures within Runtime
  limits. Corrections need no new screenshot. Preserve valid task constraints and never guess a
  replacement target merely to obtain a nonempty result.
- **Empty result:** it supports absence only for a successful, well-scoped query with complete
  coverage. Missing fields, errors, unavailable resources and truncated responses do not establish
  absence. Continue with a specific basis for correction or useful further evidence; otherwise end
  that inquiry instead of scanning unrelated resources. There is no fixed failed-attempt count for
  deciding that a question has no answer.

Do not treat Calendar, Clock, Mail or Messages as generic fallback indexes. Query them only when the
observed activity supplies a grounded locator and the presence or absence of that resource can change
the current assistance decision. A complete empty result does not authorize scanning other Apps without
new evidence or a returned locator.

Avoid repeating successful, complete queries while relevant state is unchanged and no new
information is expected. This includes equivalent reformulations. Grounded corrections, incomplete
coverage and evidence-supported refreshes remain valid reasons to query again.

## Budget and availability

Runtime's `query_policy` and receipts define accounting, including cache hits; empty results also
incur the declared cost. Use queries that advance the decision and preserve capacity for later
observations without treating unused budget as a target. `search_budget` and `allowed_control_actions`
state current availability. Null/unlimited ceilings remove a cap, not cost. Only the per-observation
limit resets after advancing. An `unavailable` refusal contains no world evidence; retry after its
stated cause is resolved. Otherwise choose Commit or Wait from evidence already obtained.
