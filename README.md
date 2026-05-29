# Semantic Triplet Restoration (STR)

[English](README.md) | [中文](README_zh.md)

This repository contains the public implementation for **Semantic Triplet
Restoration (STR)** and the **TripletQL** table-QA router.

Current public release status:

- Supported input formats: **HTML**, **Excel**, and **image with local TABLET weights**.
- TABLET image weights are **not public** and are not included in this repository.
- TABLET training code is included for reproducibility, but checkpoints and
  trained weights are intentionally not included.

STR converts a table into semantic facts:

```text
<item path, feature path, value>
```

TripletQL answers questions over these facts through the paper-aligned routing
paths: metadata direct-answer, conservative filtering, and full STR reasoning.

## Repository Layout

```text
STR/
├── str_core/              # HTML/Excel/image -> STR, CLI, FastAPI, QA wrapper
├── triplet_agent/         # TripletQL routing, filtering, rendering, memory
├── learnable_routing/     # Router SFT utilities from the paper experiments
├── tablet/                # TABLET data/model/training code, no weights
├── examples/              # Minimal examples
├── scripts/               # Convenience CLI wrappers
└── tests/                 # Offline smoke tests
```

## Deploy

Use the existing `fintab` conda environment:

```bash
cd /home/zyb/baidu/TTE/STR
conda activate fintab
pip install -e .
```

Set an OpenAI-compatible LLM only when using LLM-backed triplet restoration or
QA:

```bash
export STR_LLM_API_KEY=...
export STR_LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
export STR_LLM_MODEL=your-model-name
```

Optional TripletQL feature probe:

```bash
export STR_QWEN_URL=http://localhost:8001/v1
```

## Convert Tables To STR

HTML:

```bash
python -m str_core convert examples/simple_table.html \
  --input-type html \
  --use-llm auto \
  -o table.str.json
```

Excel:

```bash
python -m str_core convert path/to/table.xlsx \
  --input-type excel \
  --use-llm auto \
  -o table.str.json
```

Image with local TABLET checkpoints:

```bash
python -m str_core convert path/to/table.png \
  --input-type image \
  --split-checkpoint tablet/checkpoints/split/best.pth \
  --merge-checkpoint tablet/checkpoints/merge/best.pth \
  --ocr-words path/to/table_words.json \
  --use-llm auto \
  -o table.str.json
```

Image conversion will fail unless you provide local TABLET weights. The public
repository does not include those weights.

`--use-llm auto` uses `STR_LLM_*` if configured. Without an LLM key, conversion
falls back to a deterministic heuristic for simple first-row-header tables.

## Ask Questions

```bash
python -m str_core qa --triplet table.str.json \
  -q "Which company has the highest revenue?" \
  -q "What is its profit?"
```

The QA path requires an OpenAI-compatible LLM client through `STR_LLM_*`.

## Run API Server

```bash
python -m str_core serve --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `POST /v1/convert`: multipart upload for HTML, Excel, or image with local TABLET weights
- `POST /v1/qa`: JSON body with `triplet_data` and `question` or `questions`

Example conversion request:

```bash
curl -X POST http://localhost:8000/v1/convert \
  -F "file=@examples/simple_table.html" \
  -F "input_type=html" \
  -F "use_llm=auto"
```

For images, also pass `input_type=image`, `split_checkpoint`,
`merge_checkpoint`, and optionally `ocr_words_json`.

Example QA request:

```bash
curl -X POST http://localhost:8000/v1/qa \
  -H "Content-Type: application/json" \
  -d '{"triplet_data": {"shape": "2*2", "group": []}, "question": "How many rows and columns?"}'
```

## TABLET Training Code

`tablet/` contains the training and inference source code used by the visual
foundation:

```bash
cd tablet
python download_data.py
python train_split.py --batch-size 32 --epochs 16 --device cuda
python train_merge.py --batch-size 32 --epochs 24 --device cuda
```

No TABLET weights are published in this repository. Image conversion logic is
kept, but deployment must mount/provide local checkpoints before accepting raw
images.

## Smoke Test

```bash
conda run -n fintab python tests/test_smoke.py
```

## GitHub Push

```bash
cd /home/zyb/baidu/TTE/STR
git add .
git commit -m "Initial STR reference implementation"
git remote add origin git@github.com:<your-user-or-org>/STR.git
git push -u origin main
```
