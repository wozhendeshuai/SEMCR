# SEMCR

## 简要说明
本仓库实现基于检索的历史修复经验 (HRE) + 反思循环的代码修复流水线。主要脚本：
- `code/memory_build.py`：把训练数据嵌入并构建 Faiss 索引（为后续检索准备）
- `code/code_refinement_HRE_train_rag_repo.py`：主流程，检索、生成、反思、更新经验并记录结果
- 其它：评估工具在 `code/eval_code_sim.py` 中

## 目录结构（示例）
- `code/` — 源代码
  - `memory_build.py`
  - `code_refinement_HRE_train_rag_repo.py`
  - `eval_code_sim.py`
- `data/{repo}/{repo}_train.jsonl` — 每个仓库的训练数据（jsonl）
- `hre_self_improve/{repo}/` — index/meta/vec 输出目录（由 `memory_build.py` 生成）
- `result/{repo}/` — 运行结果与度量文件
- `MODEL_DIR` — 本地嵌入模型路径（在代码中指定）

## 前提与环境
- 操作系统：macOS（说明以 macOS 为主）
- Python：推荐 Python 3.9+
- GPU/加速：若有 Apple Silicon 请使用 MPS（代码已检测）
- 本地嵌入模型：`Qwen3-Embedding-0.6B`（或你自己的 embedding 模型），并将其解压到本地路径，然后在代码中设置 `MODEL_DIR`
- 需要的磁盘空间：模型和索引文件可能较大（数 GB）

## 依赖（示例）
在虚拟环境中安装（示例 `requirements.txt` ）：

`requirements.txt`
```
torch
transformers
numpy
pandas
tqdm
openai
faiss-cpu    # 或根据系统使用 faiss 或 conda 安装 faiss-cpu
openpyxl
```

安装命令（macOS）：
- 使用 pip（先创建并激活虚拟环境）：
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - 安装 PyTorch（按官方首页选择 MPS/CPU/metal 方案），例如：
    - `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` （或参考官方命令）
  - `pip install -r requirements.txt`
- 如果 `faiss-cpu` 在 pip 上安装失败，建议使用 conda：
  - `conda install -c pytorch faiss-cpu`

## 配置说明（必须）
在运行前请检查/配置以下变量（可在脚本顶部修改或改为使用环境变量）：

- 嵌入模型路径（在两个脚本中都使用）：
  - `MODEL_DIR = /path/to/Qwen3-Embedding-0.6B`
- 训练数据路径（示例，放在 repo_data）：
  - `repo_data/{repo}/{repo}_train.jsonl`
- 在 `code/code_refinement_HRE_train_rag_repo.py` 中：
  - `REPO_List`：要处理的仓库列表（脚本末尾）
  - `RHE_index_path`、`RHE_meta_path`、`RHE_vec_path`：会基于 `repo` 自动构造，确保对应目录存在或可写
  - `RESULT_XLSX` 与 `METRIC_FILE`：输出路径
- LLM 接口（API）：脚本内 `load_model()` 当前硬编码了 `api_key` 与 `base_url`，务必替换为你自己的凭证或改为读取环境变量：
  - 建议设置环境变量：`export OPENAI_API_KEY=sk-...`、`export OPENAI_BASE_URL=https://...`，并在 `load_model()` 中读取（推荐修改）

示例（建议将 `load_model()` 改为读取环境变量）：
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

## 构建 Faiss 索引（流程）
在索引构建前，确保 `TRAIN_DATA` 文件存在（`repo_train.jsonl`），且 `MODEL_DIR` 已配置并可用。

命令示例：
1. 激活虚拟环境（见上）
2. 运行构建脚本：
   - `python3 code/memory_build.py`

`memory_build.py` 会遍历 `REPO_List`，为每个 repo 读取训练数据并生成：
- `hre_self_improve/{repo}/{repo}_code_refinement_hre_index_plus_0.faiss`
- `hre_self_improve/{repo}/{repo}_code_refinement_hre_meta_plus_0.jsonl`
- `hre_self_improve/{repo}/{repo}_code_refinement_hre_vec_plus_0.npy`

注意：
- 若需要单独构建某个仓库可临时修改脚本中 `REPO_List` 或直接在文件顶部指定路径并调用 `build_combined_index(...)`。

## 运行主流程（检索 + 生成 + 反思 + 更新）
在索引和模型准备就绪后，运行主脚本：

1. 激活虚拟环境
2. 设置必要的环境变量（建议）：
   - `export OPENAI_API_KEY=sk-...`
   - `export OPENAI_BASE_URL=https://your-llm-endpoint`
3. 修改 `code/code_refinement_HRE_train_rag_repo.py` 中的 `MODEL_DIR` 与 `REPO_List`（如果需要）
4. 运行：
   - `python3 code/code_refinement_HRE_train_rag_repo.py`

脚本运行说明：
- 会读取 `repo_data/{repo}/{repo}_train.jsonl`
- 使用 `RHE_index_path` / `RHE_meta_path` / `RHE_vec_path` 进行检索
- 调用 `gen_with_messages()` 与外部 LLM（通过 `load_model()` 返回的 client）
- 生成结果会以追加方式写入到 `result/{repo}/{repo}_qwen3_8B_train_HRE_outputs_plus_0.xlsx`
- 全局平均统计写入 `result/{repo}/{repo}_qwen3_8B_train_HRE_metrics_plus_0.txt`

## 在 PyCharm 中运行（简单步骤）
1. 打开 PyCharm，打开项目根目录
2. 设置 Python 解释器为刚创建的虚拟环境
3. 在 Run/Debug Configurations 中新建 Python 运行配置：
   - Script path：`code/code_refinement_HRE_train_rag_repo.py`
   - Working directory：项目根目录
   - Environment variables：添加 `OPENAI_API_KEY`、`OPENAI_BASE_URL`（如果你使用环境变量）
4. 运行或调试

## 调试与注意事项
- 如果模型在 MPS 上出现显存/精度问题，可在 `load_embedding_model` 中强制使用 `torch.float32` 或改为 CPU。
- Faiss 在 macOS 上安装可能出错，优先使用 `conda` 安装 `faiss-cpu`。
- 若你的 LLM endpoint 不兼容 `OpenAI(OpenAI...)` 的 `chat.completions.create` 接口，请根据提供端点 SDK 调整 `gen_with_messages()`。
- 强烈建议把真实 API Key 从代码中移除，改用环境变量或 secrets 管理。

## 快速示例命令（完整）
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 配置模型路径：编辑 code/memory_build.py 与 code/code_refinement_HRE_train_rag_repo.py 中的 MODEL_DIR
# 构建索引
python3 code/memory_build.py
# 设置 API 凭证（或在代码中设置）
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://your-llm-endpoint
# 运行主流程
python3 code/code_refinement_HRE_train_rag_repo.py
```

## 常见问题
- Faiss 安装失败：尝试 `conda install -c pytorch faiss-cpu`
- GPU/加速不可用：确认 PyTorch 与硬件（MPS/CUDA）兼容
- 模型加载慢或 OOM：降低 `BATCH_SIZE` 或在 `AutoModel.from_pretrained` 中使用更小的 dtype（谨慎）

## 许可与安全
- 请勿在代码仓库里硬编码 `api_key`。生产或公开仓库请使用环境变量或 secret 管理。
- 本仓库用于研究/实验目的，确保遵守所下载模型与第三方服务的使用条款。


