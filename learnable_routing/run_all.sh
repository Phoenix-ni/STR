#!/usr/bin/env bash
# 跑完整 learnable routing 实验的 orchestrator。
# 第一次跑前请打开下面变量按你的实际路径修改。
# 默认在 conda env `fintab` 下运行；用 PY 环境变量可覆盖。
set -euo pipefail

# ── 路径与服务配置 ────────────────────────────────────────────
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$HERE"

PY="${PY:-conda run -n fintab python3}"

# benchmark 数据路径：公开仓库不携带数据，默认指向本仓库 data/ 下的约定位置。
ROOT="$( cd "$HERE/.." >/dev/null 2>&1 && pwd )"
TABLEEVAL_TEST="${TABLEEVAL_TEST:-$ROOT/data/TableEval-test.jsonl}"
TABLEEVAL_TRIP="${TABLEEVAL_TRIP:-$ROOT/data/TableEval-triplet.jsonl}"
WTQ_TEST="${WTQ_TEST:-$ROOT/data/wtq-test.jsonl}"
TABLEBENCH_TEST="${TABLEBENCH_TEST:-$ROOT/data/TableBench_triplet.jsonl}"

# LLM 配置。默认使用 STR_LLM_* 环境变量；CONFIG_FILE 仅用于兼容旧 TableEval OpenAIClient。
MODEL_NAME="${MODEL_NAME:-${STR_LLM_MODEL:-gpt-4o-mini}}"
CONFIG_FILE="${CONFIG_FILE:-}"
QWEN_URL="${QWEN_URL:-${STR_QWEN_URL:-}}"

# 训练超参
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
LORA_CKPT="${LORA_CKPT:-checkpoints/router_v1}"

MAX_WORKERS="${MAX_WORKERS:-4}"
LIMIT="${LIMIT:-}"  # 调试时设小一点，例如 LIMIT=100；正式跑空字符串

mkdir -p out data checkpoints

LIMIT_ARG=()
[ -n "$LIMIT" ] && LIMIT_ARG=(--limit "$LIMIT")
CONFIG_ARG=()
[ -n "$CONFIG_FILE" ] && CONFIG_ARG=(--config_file "$CONFIG_FILE")
QWEN_ARG=()
[ -n "$QWEN_URL" ] && QWEN_ARG=(--qwen_url "$QWEN_URL")

# ── Step 2: Oracle 标注 ──────────────────────────────────────
echo "[Step 2/5] Build oracle labels on 3 benchmarks"

$PY build_oracle_labels.py --benchmark tableeval \
    --test_jsonl "$TABLEEVAL_TEST" --triplet_jsonl "$TABLEEVAL_TRIP" \
    --model_name "$MODEL_NAME" "${CONFIG_ARG[@]}" "${QWEN_ARG[@]}" \
    --output_jsonl out/routing_labels_tableeval.jsonl \
    --max_workers "$MAX_WORKERS" --resume "${LIMIT_ARG[@]}"

$PY build_oracle_labels.py --benchmark wtq \
    --test_jsonl "$WTQ_TEST" \
    --model_name "$MODEL_NAME" "${CONFIG_ARG[@]}" "${QWEN_ARG[@]}" \
    --output_jsonl out/routing_labels_wtq.jsonl \
    --max_workers "$MAX_WORKERS" --resume "${LIMIT_ARG[@]}"

$PY build_oracle_labels.py --benchmark tablebench \
    --test_jsonl "$TABLEBENCH_TEST" \
    --model_name "$MODEL_NAME" "${CONFIG_ARG[@]}" "${QWEN_ARG[@]}" \
    --output_jsonl out/routing_labels_tablebench.jsonl \
    --max_workers "$MAX_WORKERS" --resume "${LIMIT_ARG[@]}"

# ── Step 3a: merge labels ────────────────────────────────────
echo "[Step 3a] Merge labels"
$PY merge_labels.py \
    --inputs out/routing_labels_tableeval.jsonl out/routing_labels_wtq.jsonl out/routing_labels_tablebench.jsonl \
    --out_train data/routing_train.jsonl \
    --out_val data/routing_val.jsonl \
    --val_ratio 0.1

# ── Step 3b: SFT ─────────────────────────────────────────────
echo "[Step 3b] LoRA SFT on Qwen3-0.6B"
$PY sft_train.py \
    --base_model "$BASE_MODEL" \
    --train_file data/routing_train.jsonl --val_file data/routing_val.jsonl \
    --output_dir "$LORA_CKPT" \
    --epochs 3 --bs 16 --lr 1e-4 --lora_r 16

# ── Step 5: 双对照评测 ───────────────────────────────────────
echo "[Step 5] Eval Fixed vs Learnable on 3 benchmarks"
for BENCH in tableeval wtq tablebench; do
    case "$BENCH" in
        tableeval) TEST="$TABLEEVAL_TEST"; TRIP_ARG=(--triplet_jsonl "$TABLEEVAL_TRIP") ;;
        wtq)       TEST="$WTQ_TEST"; TRIP_ARG=() ;;
        tablebench)TEST="$TABLEBENCH_TEST"; TRIP_ARG=() ;;
    esac

    for ROUTER in fixed learnable; do
        EXTRA=()
        [ "$ROUTER" = "learnable" ] && EXTRA=(--lora_ckpt "$LORA_CKPT" --base_model "$BASE_MODEL")
        echo "--- $BENCH / $ROUTER ---"
        $PY eval_learnable.py --benchmark "$BENCH" --test_jsonl "$TEST" "${TRIP_ARG[@]}" \
            --model_name "$MODEL_NAME" "${CONFIG_ARG[@]}" "${QWEN_ARG[@]}" \
            --router "$ROUTER" "${EXTRA[@]}" \
            --output_jsonl "out/${BENCH}_${ROUTER}.jsonl" \
            --max_workers "$MAX_WORKERS" "${LIMIT_ARG[@]}"
    done
done

echo "==== Table 6 inputs ===="
for BENCH in tableeval wtq tablebench; do
    for ROUTER in fixed learnable; do
        echo "[$BENCH / $ROUTER]"
        cat "out/${BENCH}_${ROUTER}_summary.json"
        echo
    done
done
