# SEMCR

本项目实现了基于历史修复经验（HRE）与大模型反思机制的代码修复流水线，包含：构建向量索引、检索并重排序历史经验、基于检索与评论生成修复、反思迭代、汇总与更新经验等完整流程。

## 目录结构
- `memory_build.py`：构建向量索引及元数据（`faiss`、`.npy`、`.jsonl`）。
- `eval_code_sim.py`：评估指标实现（EM / BLEU / CodeBLEU / ROUGE-L / EditProgress）。
- 主脚本（当前工程中为顶层运行段）：执行检索、生成、反思、更新经验与评估并保存结果。
- `README.md`：本文件。
- 数据与结果路径示例：
  - 训练数据：`../data/{repo}/{repo}_train.jsonl`
  - 索引：`./hre/{repo}/{repo}_code_refinement_hre_index_plus_0.faiss`
  - 元数据：`./hre/{repo}/{repo}_code_refinement_hre_meta_plus_0.jsonl`
  - 向量：`./hre/{repo}/{repo}_code_refinement_hre_vec_plus_0.npy`
  - 结果表格：`./result/{repo}/{repo}_qwen3_8B_train_HRE_outputs_plus_0.xlsx`
  - 评估报告：`./result/{repo}/{repo}_qwen3_8B_train_HRE_metrics_plus_0.txt`

---

## 环境与依赖
- Python 3.9+（建议使用 3.10）
- macOS（代码已对 `mps` 做兼容判断）/ Windows需自行修改配置
- 主要第三方库：
  - `torch`（在 macOS 上如果支持 MPS，可使用）
  - `transformers`
  - `faiss` 或 `faiss-cpu`
  - `numpy`, `pandas`, `openpyxl`
  - `nltk`, `rouge`, `codebleu`（或等价实现）
  - `tqdm`
- 建议使用虚拟环境：
  - python -m venv .venv
  - source .venv/bin/activate
  - pip install -r requirements.txt
- 注意：`codebleu` 可能需要 C/C++ 依赖或额外安装步骤。若不便，可替换为本地实现或跳过该指标。

---

## 快速开始

1. 配置嵌入模型路径
   - 在 `memory_build.py` 与主脚本中，设置 `MODEL_DIR` 为本地嵌入模型目录（示例：`/path/to/Qwen3-Embedding-0.6B`）。

2. 构建索引（从训练数据生成向量索引）
   - 准备训练数据：每行 JSON 表示一个 patch，需包含字段 `old`（修复前代码）与 `original_item`（可选完整条目）。
   - 运行：
     - `python memory_build.py`
   - 输出会生成：`*.faiss`、`*.npy` 与 `*.jsonl`（meta）。

3. 运行主流程（检索 + 生成 + 反思 + 更新经验）
   - 配置主脚本顶部参数，例如 `REPO_List`、`MODEL_DIR`、`TOPK_HRE` 等。
   - 运行脚本（示例在主文件末尾有 `if __name__ == "__main__"`）：
     - `python <主脚本>.py`
   - 运行期间将逐条处理数据、保存每条结果到 `./result/...xlsx` 并写入总体评估到 `*.txt`。

---

## 主要模块功能说明

- 嵌入与索引
  - `load_embedding_model(model_dir)`：加载 `transformers` 模型与 tokenizer，送入指定设备（`cpu` / `mps` / GPU）。
  - `mean_pooling` 与 `embed_texts`：将模型输出池化并归一化为向量。
  - `memory_build.py` 中的 `build_combined_index`：读取训练 `jsonl`，提取 `old` 字段批量嵌入后，用 `faiss` 构建索引，保存 `.faiss`, `.npy`, `.jsonl`。

- 检索与重排序
  - `load_index_and_meta(index_path, meta_path, vec_path)`：加载索引、meta 文件与向量阵列，建立 `text_to_vecs` 映射（用于 trigger_snippet 相似度计算）。
  - `retrieve_and_rerank_experiences(...)`：检索 top-K 锚点（含额外 margin），为每个候选经验计算基于 trigger snippet 的相似度并重排序，支持结合评论向量再加权排序。

- 文本处理
  - `process_diff_code(diff_code)`：处理带有 diff 标记（`+`, `-`）的代码片段，返回修复后的代码（仅保留 `+` 与普通行，不包含 `-`）。

- LLM 交互
  - `load_model()`：初始化大模型客户端（示例使用兼容接口）。
  - `gen_with_messages(messages, client)`：发送对话消息给模型（stream=False），返回原始文本、处理时间与 token 计数等。包含对 thinker 标签与 json 封装的解析。
  - 生成相关函数：
    - `generate_refinement_code(...)`：基于检索到的 HRE 与 review comment 生成初始修复结果。
    - `generate_reflection(...)`：当模型输出不理想时，生成一段短的“反思”文本，指出改进点。
    - `generate_fix_with_reflection(...)`：将最新反思与历史证据合并以重新生成修复。
    - `summarize_experience_v2(...)`：从最终高质量修复中提炼出一条可复用的经验文本。
    - `update_experience_v2(...)`：对检索到的旧经验进行细化并通过 `RHE_search_subprocess(..., operation='update')` 写回到 `meta` 文件中（索引本身未变，只会修改 `meta`）。

- 评估与指标（见 `eval_code_sim.py`）
  - 指标包括：EM（精确匹配）、BLEU、CodeBLEU、ROUGE-L、EditProgress（基于 Levenshtein 距离的编辑进度）。
  - `compute_metrics(prediction, reference, input_code, lang, repo)`：统一封装返回字典。
---

## 输入输出格式（简述）
- 训练/试验数据：JSONL，每行格式类似：
  ```json
  {
    "ghid": "...", 
    "proj": "...", 
    "old": "...", 
    "new": "...", 
    "comment": "...", 
    "lang": "..."
  }
  ```
- `meta` 文件：每行为一个 JSON 元对象，字段 `original_item` 保存原始条目，`experiences` 字段保存 HRE 列表（每个经验包含 `experience`, `trigger_snippet`）。
- 输出：每条样本会追加写入 `./result/*.xlsx`，同时总体统计写入 `*.txt`。

---

## 参数与阈值
- `EXTRA_RETRIEVAL_MARGIN`：检索时对 top-K 的额外扩展量，用于增加候选多样性。
- 反思与经验更新阈值（示例默认值在代码中）：
  - `REFLECTION_CODEBLEU_THRESHOLD`，`REFLECTION_HIGH_CODEBLEU_THRESHOLD`
  - `REFLECTION_EDIT_PROGRESS_THRESHOLD`
  - `MEMORY_UPDATE_CODEBLEU_THRESHOLD`，`MEMORY_UPDATE_EDIT_PROGRESS_THRESHOLD`
- 可以通过修改这些阈值观察系统行为差异。

---

## 常见问题
- faiss 无法安装或导入：
  - macOS 建议安装 `faiss-cpu` 或在 Linux/GPU 环境安装 `faiss-gpu`。
- CodeBLEU 报错或无法运行：
  - 检查 `codebleu` 依赖是否安装，不同设备所使用的tree-sitter-X的版本不一，需自行测试。
- 模型 API（`load_model`）部分：
  - 可用本地小模型替代。

