# Proseek-Bench

CVPR under review

## NeedThread evaluation

This source package runs the Framework v2 NeedThread agent and evaluates its
predictions for Need, Time and Claims. The default observation window is three
images and eight actions. The default search budget and per-observation limit
are both 20. The model creates or updates candidate threads, searches for
evidence, and commits Needs with Claims. Commit and Wait advance the Episode.

## Setup

Use Python 3.10 or newer. For Claims scoring with live OCR, install Tesseract
with English language data and make it available on `PATH`.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s code/inference
python -m unittest discover -s code/evaluate
```

Install the approved benchmark data alongside this source tree. Preserve its
directory layout, including `users/`, `platform/`, `release_manifest.json`,
`SHA256SUMS`, and the selected release manifest. Include the selected release's
capability mapping at the path it declares for Claims evaluation. The source
package contains code, prompts, schemas and provider routing configuration.

## Run inference

From the package root, set `RELEASE_MANIFEST` to the selected frozen manifest
path and `MODEL` and `API_BASE` to your model and OpenAI-compatible endpoint.
For an authenticated custom endpoint, set `PROACTIVE_AGENT_API_KEY` in the
environment. For built-in providers, see `config/model_providers.json` for the
corresponding environment variable.

```sh
python code/inference/run_batch.py \
  --model "$MODEL" --provider custom --api-base "$API_BASE" \
  --inventory-manifest "$RELEASE_MANIFEST" \
  --inference-mode framework --prompt-version v2 --image-window 3 --dry-run
```

Remove `--dry-run` to run the selected inventory. The W3 setting writes to
`evaluate/runs/<model>/framework-v2-w3/`. Add `--limit 1` for a single-Sampler
check. Keep each Sampler's prediction, runtime receipt, and model call log
together for scoring and evidence auditing.

For offline replay, use `code/inference/run_online_episode.py` with
`--sampler-root`, `--decisions` (one Decision JSON per line), and `--output`.

## Score predictions

Create `evaluate/run-map.json` as a mapping from an evaluation label to a path
relative to `evaluate/runs/`, for example
`{"my-model":"my-model/framework-v2-w3"}`. The same mapping works with a
merged recovery run when the path points to that run directory.

The default Judge uses `QIANWEN_API_KEY`. Follow the sequence below for a new
evaluation. Stage 1 routes candidate pairs and selects one-to-one matches,
Stage 2 judges the selected pairs and uses the frozen Commit window for Time,
and Claims v1 scores grounded completion after Need and Time pass.

```sh
python code/evaluate/run_text_stage1_router.py \
  --runs evaluate/run-map.json --release-manifest "$RELEASE_MANIFEST" --routing-id need-router
python code/evaluate/export_stage1_selected_edges.py \
  --routing-dir evaluate/reviews/need-router \
  --output-json evaluate/selected-edges.json --output-csv evaluate/selected-edges.csv
python code/evaluate/run_text_stage2_selected_edges.py \
  --runs evaluate/run-map.json --release-manifest "$RELEASE_MANIFEST" \
  --selected-edges evaluate/selected-edges.json --evaluation-id need-time
python code/evaluate/run_text_claims_isolated.py \
  --need-evaluation-dir evaluate/reviews/need-time \
  --release-manifest "$RELEASE_MANIFEST" --evaluation-id claims --prompt-version v1
```

For the error-inclusive protocol, pass `--protocol-failure-policy error_as_failure`
to Stage 1, Stage 2, and Claims. Stage 1 and Claims accept `--dry-run` to check
their inputs before making Judge requests. The entry points provide further
options through `--help`.
