# ProSeek-Bench: Do Mobile Agents Know What to Seek and When to Proact?

**arXiv:** <!-- Add the arXiv link after the preprint is public. -->

ProSeek-Bench evaluates whether mobile agents can recognize unspoken user needs, seek the evidence they need, and offer help at the right time. Each episode pairs a GUI stream with an evolving, queryable phone world. As the episode unfolds, an agent can **Search** using read-only tools, **Wait**, or **Commit** an assistance proposal.

## Benchmark

ProSeek-Bench contains **373 replayable episodes**, including **37 no-need controls**, across **GUI-Only**, **Active**, and **Long-Term** settings. The main experiments use a shared **NeedThread** harness that tracks candidate needs and evidence across observations.

Human-adjudicated references support three nested checks:

- **Need** — Does the proposal match a reference need?
- **Time** — Was the proposal committed within the intervention window for the matched option?
- **Evidence** — Does the proposal resolve the decisive requirements, with factual premises supported by evidence acquired before commitment?

## Findings

Across **13 multimodal models**, the strongest model reaches **61.0 Need F1** and **42.6 Evidence F1**. Search is especially helpful in Active episodes. Removing persistent state reduces Long-Term Evidence F1 by **63.9–78.2%** across four models.
