# Running VII Experiments

This repository now has two execution modes:

1. **Dry run / local validation**: renders visual instructions and records metadata without external model calls.
2. **API run**: dispatches each grounded image and visual instruction to a configured image-to-video API client.

Use the API mode only in an approved safety-red-team environment and follow the terms and safety policies of each provider.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2. Download and normalize datasets

```bash
python scripts/download_datasets.py --dataset all --output-dir data/raw --split train
```

On slow or unstable networks, prefer one dataset at a time with streaming, resume,
more retries, and longer HuggingFace Hub timeouts:

```bash
python scripts/download_datasets.py \
  --dataset conceptrisk \
  --output-dir data/raw \
  --split train \
  --streaming \
  --resume \
  --max-retries 10 \
  --num-proc 1 \
  --download-timeout 300
```

If a run times out, rerun the same command with `--resume`; completed rows in
`data/processed/<dataset>.jsonl` are skipped and HuggingFace cache downloads are
resumed.

The downloader writes normalized manifests to:

- `data/processed/coco_i2v_safetybench.jsonl`
- `data/processed/conceptrisk.jsonl`

The normalized prompt field is the unsafe video prompt used by the benchmark:

- COCO-I2VSafetyBench: `harmful_video_prompt`
- ConceptRisk-Repro: `unsafe_video_prompt`

## 3. Configure commercial API credentials

The runner selects a real client whenever `--model` is one of `kling`, `veo`, `seedance`, `pixverse`, or `generic_i2v` and `--dry-run` is not passed.

| Model flag | Default model in `configs/models.yaml` | Required environment variables |
| --- | --- | --- |
| `kling` | `kling-v2.5-turbo` | `KLING_API_KEY`; optional `KLING_BASE_URL` |
| `generic_i2v` | `MiniMax-I2V-01` | `I2V_API_KEY`/`DEEPSEEK_API_KEY`, `I2V_BASE_URL`/`DEEPSEEK_BASE_URL`; optional `I2V_MODEL`/`DEEPSEEK_MODEL`, `I2V_ENDPOINT_PATH`, `I2V_STATUS_PATH_TEMPLATE` |
| `seedance` | `seedance-1.5-pro` | `SEEDANCE_API_KEY`; optional `SEEDANCE_BASE_URL` |
| `veo` | `veo-3.1` | Gemini API: `GOOGLE_API_KEY`; optional `VEO_BASE_URL`. Vertex mode: `GOOGLE_GENAI_USE_VERTEXAI=1`, `GOOGLE_CLOUD_PROJECT`, optional `GOOGLE_CLOUD_LOCATION`, and `GOOGLE_OAUTH_ACCESS_TOKEN` |
| `pixverse` | `pixverse-v5` | `PIXVERSE_API_KEY`; optional `PIXVERSE_BASE_URL` |

The built-in clients use JSON-over-HTTP provider-compatible endpoints. If your provider account exposes a different path or payload schema, set the corresponding `*_BASE_URL` to your proxy/adapter or subclass the matching client in `src/vii/models/`.

If you do not have `KLING_API_KEY` but have an API gateway that exposes models
such as `MiniMax-I2V-01`, `MiniMax-I2V-01-Director`,
`MiniMax-I2V-01-Live`, or `Doubao-Seedance-1.0-Pro`, use `generic_i2v`:

```bash
cp configs/api.example.yaml configs/api.yaml
```

Then edit `configs/api.yaml`. For Paratera / 并行科技:

```yaml
providers:
  generic_i2v:
    api_key_env: DEEPSEEK_API_KEY
    base_url_env: DEEPSEEK_BASE_URL
    model_env: DEEPSEEK_MODEL
    api_key: null
    base_url: https://llmapi.paratera.com/v1/
    model: MiniMax-I2V-01
    endpoint_path: /v1/videos/image-to-video
    status_path_template: /v1/videos/{job_id}
```

Keep the real key in your shell instead of committing it:

```bash
export DEEPSEEK_API_KEY=sk-XXX
export DEEPSEEK_BASE_URL=https://llmapi.paratera.com/v1/
export DEEPSEEK_MODEL=MiniMax-I2V-01
```

The generic client sends JSON with `model`, `prompt`, `image`, `resolution`,
and `duration`, where `image` is a data URI. It accepts either a synchronous
response containing a video URL or an asynchronous response containing
`job_id`/`task_id`/`id` for polling.

## 4. Smoke test

```bash
python scripts/run_vii_experiment.py \
  --dataset coco_i2v_safetybench \
  --model mock \
  --config configs/vii.yaml \
  --output-dir outputs/smoke_coco_mock \
  --limit 1 \
  --seed 42 \
  --dry-run
```

## 5. Run one API-backed experiment

Start with a small limit before running the full benchmark:

```bash
export KLING_API_KEY=...
python scripts/run_vii_experiment.py \
  --dataset coco_i2v_safetybench \
  --model kling \
  --config configs/vii.yaml \
  --output-dir outputs/coco_kling_api \
  --limit 5 \
  --seed 42 \
  --wait
```

Useful overrides:

```bash
python scripts/run_vii_experiment.py \
  --dataset conceptrisk \
  --model generic_i2v \
  --config configs/vii.yaml \
  --output-dir outputs/conceptrisk_generic_i2v_api \
  --limit 5 \
  --seed 42 \
  --wait \
  --provider-kwarg model=\"MiniMax-I2V-01\" \
  --provider-kwarg duration=5 \
  --provider-kwarg resolution=\"720p\"
```

`--wait` polls the provider and downloads finished videos to `outputs/.../videos/`. Without `--wait`, the metadata contains submitted job IDs and provider responses, and you can download later using the provider job IDs.

Each run writes:

- grounded images: `OUTPUT_DIR/images/`
- copied source images: `OUTPUT_DIR/source_images/`
- videos or request/job metadata: `OUTPUT_DIR/videos/`
- per-sample metadata: `OUTPUT_DIR/metadata.jsonl`
- aggregate run status: `OUTPUT_DIR/summary.json`
- redacted API logs: `outputs/api_logs/<provider>/`

## 6. Full paper grid

```bash
SEED=42 OUTPUT_ROOT=outputs/paper_repro bash scripts/run_all.sh
```

For a rehearsal:

```bash
LIMIT=10 SEED=42 OUTPUT_ROOT=outputs/paper_repro_10 bash scripts/run_all.sh
```

## 7. Evaluation and paper table reconstruction

After videos are downloaded, run:

```bash
python scripts/evaluate.py \
  --results outputs/coco_kling_api/metadata.jsonl \
  --videos outputs/coco_kling_api/videos \
  --output outputs/coco_kling_api/eval_results.json \
  --semantic-evaluator openai
```

The evaluator reports ASR and RR in `eval_results.json`. For paper-quality reproduction, you should replace the mock visual-safety classifier with the same VBench/T2V-SafetyBench-style evaluator stack used by your lab, because this repository's default visual classifier is intentionally a lightweight local placeholder.
