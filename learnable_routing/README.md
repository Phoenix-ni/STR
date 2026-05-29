# Learnable Routing 实验包

把论文里第 4.2.5 节 (Fixed vs. Preliminary Learnable Routing) 的表 6 从**占位数字**变成**真实可复现的实验结果**。

> 论文里的 Fixed Router 基线（89.15 / 39.71 / 28.10）是用 `TableEval/triplet_agent/` 等位置的生产版 agent 跑出来的。本包脚本统一在 `STR/triplet_agent/` 上跑（同一份 agent、同一份路由代码、对照实验），三个 benchmark 的 Fixed/Learnable 都用本包脚本跑一遍就能拿到自洽的 Table 6 数字。复用论文原始 baseline 需要把同样的 `route_mode` 改动迁到 `TableEval/triplet_agent/`，本 README 末尾会说明。

## 总览

```
STR/learnable_routing/
├── README.md                  ← 本文件
├── benchmark_adapters.py      ← 三个 benchmark 的统一加载层 → OracleSample
├── answer_scorer.py           ← 简化版 oracle 评分（含数字归一化、双向 substring）
├── build_oracle_labels.py     ← Step 2: 三档跑分 → oracle 标签
├── merge_labels.py            ← Step 3a: 三 benchmark 标签合并 + 分层 split
├── sft_train.py               ← Step 3b: Qwen3-0.6B + LoRA SFT（朴素 CE）
├── sft_train_weighted.py      ← Step 3b 升级版：少数类过采样 + 类权重 CE，处理标签不平衡
├── learnable_router.py        ← Step 4: 推理 wrapper (predict_route)
├── eval_learnable.py          ← Step 5: Fixed vs Learnable 对照评测
└── run_all.sh                 ← 一键串起 Step 2–5
```

## 前置改动（已完成）

为了支持强制三档路由，已经修改：

- `STR/triplet_agent/router.py`：新增 `ROUTE_MODES`、`force_filter_decision`、`should_keep_coverage_fallback` 静态方法。
- `STR/triplet_agent/agent.py`：`process_triplet_query` 加 `route_mode` 与 `learnable_router` 参数；新增 `_generate_with_retry_and_usage` 把 input token 上报到 `input_tokens_per_turn`。

向后兼容：默认 `route_mode="auto"` 完全等价于原行为。

## 依赖

本仓库用 conda 环境 **`fintab`**。`fintab` 已有 `torch 2.10.0+cu128`、`transformers 4.53.2`，但与 `spacy-transformers 1.4.0` 冲突——后者要求 `transformers<4.53.3`。所以**不要**直接装 `trl>=0.11`（它会把 transformers 升到 5.x）。需要锁定为：

```bash
conda activate fintab
pip install 'transformers==4.53.2' 'trl==0.11.4' 'peft==0.13.2' 'datasets>=3.0' 'accelerate>=1.0' pyyaml
pip check    # 应当看到 "No broken requirements found."
```

> 已实测：`transformers 4.53.2 + trl 0.11.4 + peft 0.13.2` 与 fintab 里 `spacy-transformers 1.4.0` 完全相容；如果你之前装的是 `trl>=1.0` 把 transformers 推到 5.x，按上面命令重装一次即可恢复。

`openai` 包通常已经在 fintab（被 `TableEval/openai_client.py` 用到）；如缺再 `pip install openai`。

`build_oracle_labels.py` 与 `eval_learnable.py` 会去 `TableEval/openai_client.py` 加载 `OpenAIClient`，所以保留 `TableEval/` 目录就行。

> 已通过 smoke test 验证：5 种 route_mode（auto / full / partial / filter / learnable）端到端正常；`StubLearnableRouter` 在无 GPU / 无 LoRA checkpoint 时也能跑通管线；`sft_train.py --help` 在锁定版本下能正常解析。

## 五步流程

### Step 1：定义三档路由强度 ✅

已完成（见上文 router/agent 改动）。三档语义：

- `full`    — 直接 path C 全量推理，不过滤
- `partial` — 进 filter 分支，**保留**覆盖率熔断（命中过多则回退 full）
- `filter`  — 进 filter 分支，**关掉**覆盖率熔断（强筛选到底）
- `auto`    — 现有 SOTA V5 规则路由（保留向后兼容，**Fixed Router 基线 = auto**）
- `learnable` — 由 `LearnableRouter.predict_route` 在运行时给出 full/partial/filter

### Step 2：Oracle 标注（最耗时）

对每条样本，跑 `full/partial/filter` 三次推理，按"先正确性、再 token 最少"挑出 oracle 标签。

```bash
python build_oracle_labels.py --benchmark tableeval \
    --test_jsonl /.../TableEval-test.jsonl \
    --triplet_jsonl /.../TableEval-triplet.jsonl \
    --model_name longcat-flash-lite \
    --config_file /.../TableEval/config/api.yaml \
    --qwen_url http://localhost:8001/v1 \
    --output_jsonl out/routing_labels_tableeval.jsonl \
    --max_workers 4 \
    --resume
```

WTQ / TableBench 同样跑，把 `--benchmark` 改成 `wtq` / `tablebench`，`--test_jsonl` 改成对应路径（TableEval 之外的 benchmark 不需要 `--triplet_jsonl`）。

**资源预估**：3 个 benchmark × ~8000 样本 × 3 路 = 约 7 万次 LongCat 调用。`max_workers=4` 时 ~10–15 小时；`max_workers=8` 时 ~6 小时。可先 `--limit 200` 跑通管线再放开。`--resume` 标志支持断点续跑。

### Step 3：合并 → 训练

```bash
# 3a. 合并 + 分层 split (默认 10% val)
python merge_labels.py \
    --inputs out/routing_labels_tableeval.jsonl out/routing_labels_wtq.jsonl out/routing_labels_tablebench.jsonl \
    --out_train data/routing_train.jsonl \
    --out_val data/routing_val.jsonl

# 3b. LoRA SFT，单卡 A100/3090/4090，~30 分钟到 2 小时
python sft_train.py \
    --base_model Qwen/Qwen3-0.6B \
    --train_file data/routing_train.jsonl --val_file data/routing_val.jsonl \
    --output_dir checkpoints/router_v1 \
    --epochs 3 --bs 16 --lr 1e-4 --lora_r 16
```

> 如果你尚未联网下载 `Qwen/Qwen3-0.6B`，可以先 `huggingface-cli download Qwen/Qwen3-0.6B` 缓存到本地，再把 `--base_model` 改成本地路径。

### Step 4：推理 wrapper（自动）

`learnable_router.LearnableRouter` 加载 LoRA checkpoint。无 GPU 时也提供 `StubLearnableRouter`（按表格大小给确定性的 full/partial/filter）方便先打通管线，eval 脚本 `--router stub` 即可启用。

### Step 5：填表 6

```bash
# Fixed baseline（route_mode=auto），与论文 Fixed 列对照
python eval_learnable.py --benchmark wtq --test_jsonl /.../wtq-test.jsonl \
    --model_name longcat-flash-lite --config_file /.../api.yaml --qwen_url http://localhost:8001/v1 \
    --router fixed --output_jsonl out/wtq_fixed.jsonl

# Learnable
python eval_learnable.py --benchmark wtq --test_jsonl /.../wtq-test.jsonl \
    --model_name longcat-flash-lite --config_file /.../api.yaml --qwen_url http://localhost:8001/v1 \
    --router learnable --lora_ckpt checkpoints/router_v1 --base_model Qwen/Qwen3-0.6B \
    --output_jsonl out/wtq_learnable.jsonl
```

3 个 benchmark × 2 个 router = 6 次，对应 Table 6 的 6 个数字（TableEval F1 / WTQ Acc / TableBench 综合）。

## 一键串起

```bash
cd STR/learnable_routing
# 先用 LIMIT=50 smoke test 管线
LIMIT=50 bash run_all.sh
# 通过后正式跑
bash run_all.sh
```

`run_all.sh` 顶部所有路径都用环境变量重载，例如：

```bash
TABLEEVAL_TEST=/your/path/TableEval-test.jsonl \
MODEL_NAME=longcat-flash-lite \
CONFIG_FILE=/your/api.yaml \
bash run_all.sh
```

## 最终把数字填进论文

跑完后 `out/{benchmark}_{router}_summary.json` 里的 `turn_level_accuracy` 与 `avg_input_tokens_per_turn` 直接对应表 6：

| 分流策略           | TableEval F1 (×100) | WTQ Acc (×100) | TableBench (×100) |
|--------------------|---------------------|----------------|-------------------|
| Fixed (Baseline)   | `out/tableeval_fixed_summary.json` | `out/wtq_fixed_summary.json` | `out/tablebench_fixed_summary.json` |
| Learnable (Ours)   | `out/tableeval_learnable_summary.json` | `out/wtq_learnable_summary.json` | `out/tablebench_learnable_summary.json` |

把这 6 个数字替换 `acl_latex.tex` 中 `tab:learnable_routing` 的占位值即可。

> 论文表 6 现在用的 89.42 / 40.95 / 32.15 是**占位**，跑完后必须替换。同时把 §4.2.5 末尾的 "+0.27 F1 / +1.24 Acc / +4.05" 这三个差值同步刷新。

## 对接论文原始 Fixed 基线（可选）

如果你要让 Fixed 那一列严格等于论文 89.15 / 39.71 / 28.10，需要把同样的 route_mode 改动也搬到生产版 agent。最小改法：在 `TableEval/triplet_agent/router.py` 和 `TableEval/triplet_agent/agent.py` 里复制 `STR/triplet_agent/` 的对应方法（`force_filter_decision`、`should_keep_coverage_fallback`、`route_mode` 参数）即可。如果用 `STR/triplet_agent/` 自洽对照即可（推荐），跳过这步。

## 故障排查

- **`OpenAIClient` 找不到 `utils`**：`build_client` 已经把 `TableEval/` 加入 `sys.path`，但部分环境里 `TableEval/utils.py` 还需要更多依赖（`load_api_configuration` 等）。若报 ImportError，把 `TableEval/utils.py` 依赖装齐即可。
- **`tokenizer.apply_chat_template` 报错**：Qwen3 系列的 chat template 已经内置。如果用了更老的 tokenizer，需要 `pip install -U transformers>=4.45`。
- **Path A 直答样本**：对应 `route_per_turn=['direct']`，token=0，被算成 Fixed/Learnable 都正确（合理：元数据问题不该被路由影响）。
- **`StubLearnableRouter`**：无 LoRA checkpoint 时用 `--router stub` 也能完整跑一遍管线验证脚本逻辑，但精度不能作为最终数字。

## 标签不平衡 → 模型退化 → 用 `sft_train_weighted.py` 纠偏

第一轮跑出来的 oracle 标签会严重偏向 `full`（典型分布）：

| Benchmark | full | partial | filter |
|---|---|---|---|
| TableEval  | 1708 (74%) | 404 (17%) | 207  (9%) |
| WTQ        | 3427 (79%) | 527 (12%) | 380  (9%) |
| TableBench |  817 (93%) |  51  (6%) |  14  (1%) |

朴素 CE 训出来的 router 会**退化为常数预测器**——所有样本都输出 `full`。它能跑出比 fixed router 高 +3 ～ +7 个点的精度，但其实"学到的是别用路由"，不是"学到了路由"。

要让 router 真正学到 partial / filter 的判别，用 `sft_train_weighted.py` 替换 `sft_train.py`：

```bash
$PY sft_train_weighted.py \
    --base_model "$BASE_MODEL" \
    --train_file data/routing_train.jsonl --val_file data/routing_val.jsonl \
    --output_dir checkpoints/router_v2_balanced \
    --oversample sqrt \
    --label_weights "full=1.0,partial=3.0,filter=5.0" \
    --epochs 3 --bs 16 --lr 1e-4 --lora_r 16
```

两条纠偏机制（可叠加，**默认两个都开**）：

1. **少数类过采样** `--oversample {none|inverse|sqrt}`：
   - `sqrt`（**推荐起点**）：复制次数 = `round(sqrt(max_count / class_count))`，比较缓和。
   - `inverse`：完全反频次复制到三类均衡，最激进，容易过拟合 filter。
   - `none`：不过采样（行为退化为 `sft_train.py`）。

2. **类权重 CE** `--label_weights "full=1.0,partial=3.0,filter=5.0"`：
   答案 token 处的 cross-entropy 乘以对应类权重。当模型把 `filter` 预测成 `full` 时损失会被放大 5 倍。

训练结束会自动跑一个 per-class accuracy on val，输出形如：

```
=== Per-class accuracy on val ===
  full    : acc=0.94  (n=571)
  partial : acc=0.42  (n=139)
  filter  : acc=0.31  (n=44)
  macro acc = 0.557
```

如果 `partial` / `filter` 的 acc 还接近 0，说明纠偏强度不够，把权重调高（`partial=6.0, filter=10.0`）或换 `--oversample inverse`。如果 `full` 跌到 0.5 以下，说明过头了，往回调。

### 复跑完整流程（保留之前的 oracle 标签）

oracle 标注是最耗时的（10+ 小时），不要重跑。换训练脚本后只需重跑 Step 3b + Step 5：

```bash
cd STR/learnable_routing
# 1. 重训（用 weighted）
$PY sft_train_weighted.py \
    --base_model "$BASE_MODEL" \
    --train_file data/routing_train.jsonl --val_file data/routing_val.jsonl \
    --output_dir checkpoints/router_v2_balanced \
    --oversample sqrt --label_weights "full=1.0,partial=3.0,filter=5.0"

# 2. 用新 checkpoint 重跑 learnable 评测（fixed 保留之前的不变）
for BENCH in tableeval wtq tablebench; do
    case "$BENCH" in
        tableeval) TEST="$ROOT/TableEval/data/TableEval-test.jsonl"; TRIP=(--triplet_jsonl "$ROOT/TableEval/data/TableEval-triplet.jsonl") ;;
        wtq)       TEST="$ROOT/WTQ_experiment/data/wtq-test.jsonl"; TRIP=() ;;
        tablebench)TEST="$ROOT/TableBench/TableBench_triplet.jsonl"; TRIP=() ;;
    esac
    $PY eval_learnable.py --benchmark "$BENCH" --test_jsonl "$TEST" "${TRIP[@]}" \
        --model_name "$MODEL_NAME" \
        --router learnable --lora_ckpt checkpoints/router_v2_balanced --base_model "$BASE_MODEL" \
        --output_jsonl "out/${BENCH}_learnable_v2.jsonl" --max_workers 80
done
```

跑完后对比 `out/{bench}_learnable.jsonl`（v1，朴素）和 `out/{bench}_learnable_v2.jsonl`（v2，加权）的 `route_per_turn` 分布——如果 v2 里出现了非零比例的 partial / filter，说明 router 真的开始路由了。

### 怎么挑超参（实战建议）

| 现象 | 调整 |
|---|---|
| v1 全输出 full | 起点：`--oversample sqrt --label_weights "full=1.0,partial=3.0,filter=5.0"` |
| v2 还是全 full（少数类 acc < 0.1） | 加大权重 `partial=6.0, filter=10.0`，或换 `--oversample inverse` |
| v2 反过来全输出 filter（filter acc > 0.9 但 full acc < 0.3） | 过头了；`partial=2.0, filter=3.0`，或退回 `--oversample sqrt` 单独用 |
| 训练 loss 不降 | `lora_r` 升到 32；或 `epochs=5`；或 `lr=2e-4` |
