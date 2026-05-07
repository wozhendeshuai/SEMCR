# python
import json
import os
from typing import List

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_DIR = "/Users/....../llm_models/Qwen3-Embedding-0.6B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 8
MAX_LENGTH = 4096


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    emb = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return emb


def embed_texts(texts: List[str], tokenizer, model) -> np.ndarray:
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        print(f"[embed] batch {i} - {min(i + BATCH_SIZE, len(texts))}/{len(texts)}", flush=True)

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=MAX_LENGTH,
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        with torch.no_grad():
            out = model(**encoded)
            emb = mean_pooling(out, encoded["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            emb = emb.detach().float().cpu().numpy()
            all_embeddings.append(emb)

        if DEVICE == "mps":
            torch.mps.empty_cache()

    if all_embeddings:
        return np.vstack(all_embeddings).astype(np.float32)

    return np.zeros((0, model.config.hidden_size), dtype=np.float32)


def build_combined_index(train_path: str, out_dir: str, index_name: str, meta_name: str, vec_name: str):
    os.makedirs(out_dir, exist_ok=True)

    print(f"[build] train_path = {train_path}", flush=True)
    print(f"[build] device = {DEVICE}", flush=True)

    print("[build] before tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    print("[build] tokenizer ok", flush=True)

    print("[build] before model load", flush=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32,
    )
    print("[build] model load ok", flush=True)

    print("[build] before model.to(device)", flush=True)
    model = model.to(DEVICE)
    print("[build] model.to ok", flush=True)

    model.eval()
    print("[build] model eval ok", flush=True)

    texts = []
    metas = []

    print("[build] 开始读取训练数据", flush=True)

    with open(train_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line.strip())
            except Exception:
                continue

            patch_text = item.get("old")
            # patch_text = process_diff_code(patch_text)
            if not patch_text:
                continue
            texts.append(patch_text)
            meta = {
                "pr_number": item.get("pr_number"),
                "need_check": item.get("y"),
            }
            # 若需完整原始记录可放开下一行（会增大 meta 文件）
            meta["original_item"] = item
            metas.append(meta)

    if not texts:
        print("没有可用的 patch 文本，跳过索引构建。")
        return

    vecs = embed_texts(texts, tokenizer, model).astype('float32')
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # 向量已归一化，可用内积进行余弦检索
    index.add(vecs)

    idx_path = os.path.join(out_dir, index_name)
    meta_path = os.path.join(out_dir, meta_name)
    vec_path = os.path.join(out_dir, vec_name)
    faiss.write_index(index, idx_path)
    np.save(vec_path, vecs)

    with open(meta_path, 'w', encoding='utf-8') as mf:
        for m in metas:
            mf.write(json.dumps(m, ensure_ascii=False) + '\n')

    print(f"保存索引到 `{idx_path}`，元数据到 `{meta_path}`，向量数: {len(metas)}")


if __name__ == "__main__":
    REPO_List = [
        "apache-beam",
        "EOSIO-eos",
        "home-assistant-core",
        "pachyderm-pachyderm",
    ]
    #
    for repo in REPO_List:
        TRAIN_DATA = f"/Users/data/{repo}/{repo}_train.jsonl"

        OUT_DIR = f"./hre/{repo}"
        INDEX_NAME = f"{repo}_code_refinement_hre_index_plus_0.faiss"
        META_NAME = f"{repo}_code_refinement_hre_meta_plus_0.jsonl"
        VEC_NAME = f"{repo}_code_refinement_hre_vec_plus_0.npy"
        os.makedirs(OUT_DIR, exist_ok=True)
        build_combined_index(TRAIN_DATA, OUT_DIR, INDEX_NAME, META_NAME, VEC_NAME)
