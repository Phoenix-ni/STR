# Semantic Triplet Restoration (STR)

[English](README.md) | [中文](README_zh.md)

本仓库提供 **Semantic Triplet Restoration (STR)** 与 **TripletQL** 表格问答路由器的公开实现。

当前公开版本状态：

- 支持输入格式：**HTML**、**Excel**，以及**本地提供 TABLET 权重时的图片输入**。
- TABLET 图片结构恢复权重**不公开**，仓库中也**不包含**这些权重。
- 仓库保留 TABLET 训练代码，用于复现和后续扩展，但默认不附带 checkpoint 或训练结果。

STR 会把表格转换为语义事实：

```text
<item path, feature path, value>
```

TripletQL 基于这些语义事实执行与论文一致的三条路径：元信息直答、保守过滤、全量 STR 推理。

## 仓库结构

```text
STR/
├── str_core/              # HTML/Excel/image -> STR，CLI、FastAPI、QA 封装
├── triplet_agent/         # TripletQL 路由、过滤、渲染、记忆
├── learnable_routing/     # 论文中的可学习路由训练/评测工具
├── tablet/                # TABLET 数据、模型、训练代码，不含权重
├── examples/              # 最小示例
├── scripts/               # 便捷脚本
└── tests/                 # 离线 smoke test
```

## 部署

使用现有的 `fintab` conda 环境：

```bash
cd /home/zyb/baidu/TTE/STR
conda activate fintab
pip install -e .
```

只有在使用 LLM 驱动的 triplet 恢复或 QA 时，才需要配置 OpenAI-compatible 接口：

```bash
export STR_LLM_API_KEY=...
export STR_LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
export STR_LLM_MODEL=your-model-name
```

可选的 TripletQL 特征探测服务：

```bash
export STR_QWEN_URL=http://localhost:8001/v1
```

## 表格转 STR

HTML：

```bash
python -m str_core convert examples/simple_table.html \
  --input-type html \
  --use-llm auto \
  -o table.str.json
```

Excel：

```bash
python -m str_core convert path/to/table.xlsx \
  --input-type excel \
  --use-llm auto \
  -o table.str.json
```

图片输入（需要本地 TABLET checkpoint）：

```bash
python -m str_core convert path/to/table.png \
  --input-type image \
  --split-checkpoint tablet/checkpoints/split/best.pth \
  --merge-checkpoint tablet/checkpoints/merge/best.pth \
  --ocr-words path/to/table_words.json \
  --use-llm auto \
  -o table.str.json
```

如果没有本地提供 TABLET 权重，图片转换会直接失败。公开仓库只保留逻辑，不提供这些权重。

`--use-llm auto` 会在配置了 `STR_LLM_*` 环境变量时自动调用 LLM。未配置时，会退回简单表格可用的确定性 heuristic 模式。

## 问答

```bash
python -m str_core qa --triplet table.str.json \
  -q "Which company has the highest revenue?" \
  -q "What is its profit?"
```

QA 路径需要通过 `STR_LLM_*` 配置 OpenAI-compatible LLM。

## 启动 API 服务

```bash
python -m str_core serve --host 0.0.0.0 --port 8000
```

接口：

- `GET /health`
- `POST /v1/convert`：接收 HTML、Excel，或带本地 TABLET 权重配置的图片输入
- `POST /v1/qa`：JSON 请求体，传入 `triplet_data` 和 `question` / `questions`

转换接口示例：

```bash
curl -X POST http://localhost:8000/v1/convert \
  -F "file=@examples/simple_table.html" \
  -F "input_type=html" \
  -F "use_llm=auto"
```

如果是图片输入，还需要额外传：

- `input_type=image`
- `split_checkpoint`
- `merge_checkpoint`
- 可选 `ocr_words_json`

QA 接口示例：

```bash
curl -X POST http://localhost:8000/v1/qa \
  -H "Content-Type: application/json" \
  -d '{"triplet_data": {"shape": "2*2", "group": []}, "question": "How many rows and columns?"}'
```

## TABLET 训练代码

`tablet/` 目录中包含视觉基础模型的训练与推理源码：

```bash
cd tablet
python download_data.py
python train_split.py --batch-size 32 --epochs 16 --device cuda
python train_merge.py --batch-size 32 --epochs 24 --device cuda
```

仓库不公开 TABLET 权重。图片转换逻辑被保留，但真正部署图片入口时，你需要自行挂载或提供本地 checkpoint。

## Smoke Test

```bash
conda run -n fintab python tests/test_smoke.py
```

## 推送到 GitHub

```bash
cd /home/zyb/baidu/TTE/STR
git add .
git commit -m "Initial STR reference implementation"
git remote add origin git@github.com:<your-user-or-org>/STR.git
git push -u origin main
```
