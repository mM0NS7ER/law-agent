# 刑法知识问答 Agent

基于 RAG（检索增强生成）的《中华人民共和国刑法》智能问答系统。支持双路召回（BM25 + FAISS）、CrossEncoder 重排序、意图识别与拒答，通过 Streamlit 提供 Web 交互界面。

## 架构

```
用户提问
  → 意图识别（Qwen-3：刑法/超出范围）
  → BM25 关键词召回 + FAISS 语义召回（BGE-large-zh-v1.5）
  → RRF 融合 → bge-reranker-base 重排序
  → DeepSeek 生成回答 + 法条引用
  → Streamlit 展示
```

## 快速开始

### 环境要求

- Python >= 3.13
- CUDA 可选（模型默认 CPU 运行）

### 1. 安装依赖

```bash
pip install -e ".[dev]"
```

### 2. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env`，填入真实的 API Key：

```env
DEEPSEEK_API_KEY=sk-your-deepseek-key
QWEN_API_KEY=sk-your-qwen-key
HF_ENDPOINT=https://hf-mirror.com  # 国内用户建议保留
```

### 3. 构建索引

```bash
python build_index.py
```

首次运行会从 HuggingFace 下载 `BAAI/bge-large-zh-v1.5`（约 1.3GB），并对 505 条刑法条文编码构建 FAISS 索引。

### 4. 启动服务

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`。

## 使用方式

| 入口 | 命令 | 说明 |
|------|------|------|
| Web 界面 | `streamlit run app.py` | 交互式问答，含法条引用卡片 |
| 索引构建 | `python build_index.py` | 生成 BM25 + FAISS 索引到 `vector_store/` |
| 自动标注 | `python -c "from src.annotator import Annotator; ..."` | 半自动标注测试集（生成检索候选供人工核对） |
| 检索评测 | `python evaluate.py --mode retrieval` | 计算 Recall@k / MRR / NDCG |
| 端到端评测 | `python evaluate.py --mode end2end` | 完整流水线 + LLM-as-Judge 评分 |

## 项目结构

```
law-agent/
├── src/
│   ├── exceptions.py          # 异常层次（LawAgentError 基类）
│   ├── models.py              # Pydantic v2 数据模型（10 个）
│   ├── config.py              # pydantic-settings 配置（自动读取 .env）
│   ├── logger.py              # 日志系统（控制台 + 文件）
│   ├── prompt_loader.py       # Prompt 模板加载与缓存
│   ├── embedding_engine.py    # BGE-large-zh-v1.5 向量编码
│   ├── bm25_engine.py         # BM25 关键词检索引擎（jieba 分词）
│   ├── faiss_engine.py        # FAISS 语义检索引擎（IndexFlatIP）
│   ├── retriever.py           # 双路召回 + RRF 融合
│   ├── reranker.py            # bge-reranker-base CrossEncoder 重排序
│   ├── llm_client.py          # OpenAI SDK 兼容 LLM 客户端
│   ├── intent_classifier.py   # 意图分类器（含 API 异常降级）
│   ├── generator.py           # 回答生成器（含 JSON 解析重试）
│   ├── annotator.py           # 半自动标注工具
│   └── evaluator.py           # 检索评测 + LLM-as-Judge 生成评测
├── prompts/                   # Prompt 模板
│   ├── intent_classification.txt
│   ├── answer_generation.txt
│   └── llm_as_judge.txt
├── tests/                     # 88 个测试用例
├── data/
│   ├── crimina_law_china.jsonl   # 刑法语料（505 条）
│   ├── test.json                 # 测试问题集（3450 条）
│   └── extra/                    # 额外法律数据（28 万条）
├── vector_store/              # 构建后的索引文件
├── build_index.py             # 索引构建脚本
├── app.py                     # Streamlit Web 界面
├── evaluate.py                # 评测 CLI
├── pyproject.toml             # 项目配置与依赖
└── .env.example               # 环境变量模板
```

## 配置

所有配置项通过 `.env` 文件或环境变量注入，由 `src/config.py` 的 `AppConfig` 统一管理：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `embedding_model` | `BAAI/bge-large-zh-v1.5` | Embedding 模型 |
| `reranker_model` | `BAAI/bge-reranker-base` | 重排序模型 |
| `bm25_topk` | 20 | BM25 召回数量 |
| `vector_topk` | 20 | 向量召回数量 |
| `rrf_k` | 60 | RRF 融合平滑参数 |
| `rrf_topk` | 20 | RRF 融合后保留数量 |
| `rerank_topk` | 5 | 重排序后保留数量 |
| `intent_threshold` | 0.7 | 意图识别置信度阈值 |
| `generation_max_retries` | 3 | 生成 JSON 解析失败重试次数 |
| `deepseek_model` | `deepseek-chat` | DeepSeek 模型名称 |
| `qwen_model` | `qwen3` | Qwen 模型名称 |

## 核心模块

### 双路检索 + RRF 融合

- **BM25**：基于 jieba 分词的关键词召回，对精确法条号匹配效果好
- **FAISS**：基于 BGE-large-zh-v1.5 的语义召回，覆盖同义表述
- **RRF**：`score = 1/(k + rank)`，对两路结果去重融合，记录每条结果的来源（bm25 / vector / both）

### 重排序

使用 `bge-reranker-base` CrossEncoder 对融合结果二次打分，按相关性降序输出 top-k。

### 意图识别

Qwen-3 分类器判断用户问题是否属于刑法范畴。置信度低于阈值自动归为 `out_of_scope`。API 异常时降级为 `criminal_law`（可用回答优先）。

### 回答生成

DeepSeek 结合检索到的法条生成结构化回答，要求 JSON 格式输出（含 answer + citations）。JSON 解析失败自动重试，全部失败抛出 `GenerationError`。

### 评测

- **检索评测**：Recall@k、MRR、NDCG@k
- **生成评测**：LLM-as-Judge 对正确性和完整性打分（需 API Key）
- **半自动标注**：检索器生成候选列表，人工填写 ground truth

## 测试

```bash
# 运行全部单元测试
pytest tests/ -v

# 跳过需要模型下载的慢测试
pytest tests/ -v -k "not slow"

# 类型检查
mypy src/

# 代码风格
ruff check src/
```

当前状态：84 passed, 1 skipped, ruff 零告警, mypy strict 模式零错误。

## 模块进度

| 模块 | 状态 |
|------|------|
| 01 基础设施（异常、模型、配置、日志、Prompt） | ✅ |
| 02 本地模型层（Embedding、BM25、FAISS） | ✅ |
| 03 RAG 核心层（双路召回 + RRF、重排序） | ✅ |
| 04 API 交互层（LLM 客户端、意图识别、生成） | ✅ |
| 05 工具与评测层（标注、评测指标） | ✅ |
| 06 入口层（Streamlit、索引构建、评测脚本） | ✅ |
| 07 测试（单元 + 集成） | ✅ |
