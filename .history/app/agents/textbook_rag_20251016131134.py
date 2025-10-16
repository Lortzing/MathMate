from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import os
import re
import json
import uuid
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import asyncio

import numpy as np
import faiss

import dashscope
# --- 必须项 2：dashscope + TextEmbedding 做查询向量 ---
from dashscope import TextEmbedding

from config import Config

PAGE_OPEN_RE = Config.PAGE_OPEN_RE
PAGE_CLOESE_RE = Config.PAGE_CLOSE_RE
HEAD_RE = Config.HEAD_RE

INDEX_PATH = Config.INDEX_PATH

API_KEY = Config.EMBED_API_KEY
BASE_URL = Config.EMBED_BASE_URL
DIM = Config.EMBED_DIM
MODEL = Config.EMBED_MODEL


def normalize(vecs: np.ndarray) -> np.ndarray:
    """L2 归一化（用于余弦相似度的内积检索）。"""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _pick_most_frequent_heading_level(text: str, prefer_deeper_on_tie: bool = True) -> int | None:
    """返回文档中出现次数最多的标题级别（1..6）。若无标题返回 None。"""
    cnt = Counter(len(m.group(1)) for m in HEAD_RE.finditer(text))
    if not cnt:
        return None
    max_freq = max(cnt.values())
    cands = [lvl for lvl, n in cnt.items() if n == max_freq]
    print(f"出现次数最多的标题级别：{cands}")
    print(f"次数为：{max_freq}")
    return (max(cands) if prefer_deeper_on_tie else min(cands))

def _align_to_page_boundaries(s: str, start: int, end: int) -> Tuple[int, int]:
    """
    将 [start, end) 向左对齐到最近的 <page ...>，向右对齐到最近的 </page ...>。
    若未命中任一标签，则保留原边界。确保最终 start < end。
    """
    # 最近的上一 <page ...>
    adj_start = start
    last_open = None
    for m in PAGE_OPEN_RE.finditer(s, 0, max(0, start)):
        last_open = m
    if last_open:
        adj_start = last_open.start()

    # 最近的下一 </page ...>
    adj_end = end
    next_close = PAGE_CLOSE_RE.search(s, min(len(s), end))
    if next_close:
        adj_end = next_close.end()

    if adj_start >= adj_end:  # 兜底，避免非法切片
        adj_start, adj_end = start, end
    return adj_start, adj_end

def md_smart_chunks(text: str, max_chars=4096, overlap=0) -> List[Dict[str, Any]]:
    """
    改进版 Markdown 切片：
    1) 自动选取文档中“出现次数最多”的标题级别做一级切分；
    2) 对超长块进行定长滑窗，但每个窗口会对齐到最近的 <page ...> 与 </page ...>；
    3) 去重相同对齐区间的窗口；保留标题与级别作为元信息。
    """
    # 0) 选取“最多”的标题级别
    split_level = _pick_most_frequent_heading_level(text, prefer_deeper_on_tie=True)

    # 1) 按选定级别切分为 block
    blocks: List[Tuple[str, str, int]] = []  # (title_line, body, level)
    current_title = f"H{split_level or 0}:ROOT"
    current_buf: List[str] = []

    def flush():
        if current_buf:
            blocks.append((current_title, "\n".join(current_buf).strip(), split_level or 0))

    if split_level is None:
        # 文档没有标题，整个文本作为一个 block
        current_buf = [text]
        flush()
    else:
        for line in text.splitlines():
            m = re.match(r"^(#{1,6})\s+(.*)$", line)
            if m and len(m.group(1)) == split_level:
                flush()
                current_title = line.strip()
                current_buf = []
            else:
                current_buf.append(line)
        flush()

    # 2) 对每个 block 做滑窗 + 按页标签对齐
    chunks: List[Dict[str, Any]] = []
    for title, body, lvl in blocks:
        body = body.strip()
        if not body:
            continue

        # 小块直接输出（可选：也可对齐页边界，这里保持简单不对齐）
        if len(body) <= max_chars:
            chunks.append({
                "title": title,
                "title_level": lvl,
                "text": body
            })
            continue

        # 大块滑窗，对齐到页边界
        seen_ranges = set()  # 去重 (adj_start, adj_end)
        start = 0
        L = len(body)
        while start < L:
            end = min(L, start + max_chars)
            adj_start, adj_end = _align_to_page_boundaries(body, start, end)

            key = (adj_start, adj_end)
            if adj_end - adj_start > 0 and key not in seen_ranges:
                seen_ranges.add(key)
                sub = body[adj_start:adj_end]
                chunks.append({
                    "title": title,
                    "title_level": lvl,
                    "text": sub
                })

            if end >= L:
                break
            start = max(0, end - overlap)

    return chunks

def embed_documents(documents: list[str]) -> np.ndarray:
    emb_matrix, errors = asyncio.run(embed_documents_async(documents))
    print(emb_matrix.shape)   # -> (N, 1024)
    if errors:
        print(f"有 {len(errors)} 条样本失败，示例：", errors[:3])
    return emb_matrix
    
    
def _parse_embeddings(resp) -> List[Dict[str, Any]]:
    """将 TextEmbedding.call 的返回转为按 text_index 升序的列表。"""
    try: 
        items = resp.output.get("embeddings", [])
    except Exception as e:
        print(resp)
        raise TypeError("无embeddings字段")
    if not isinstance(items, list):
        raise TypeError("resp.output['embeddings'] 应为 list。")
    if all(isinstance(x, dict) and 'text_index' in x for x in items):
        items = sorted(items, key=lambda x: x['text_index'])
    return items

async def _embed_batch(
    batch_docs: List[str],
    batch_idx: List[int],
    *,
    model: str,
    dim: int,
    api_key: Optional[str],
    text_type: str = "document",
    max_retries: int = 3,
    base_delay: float = 0.6,
) -> Tuple[List[int], List[np.ndarray], List[Tuple[int, Exception]]]:
    """
    单批嵌入（放在线程池里执行 SDK 同步调用），带重试。
    返回：(成功的全局索引列表, 对应向量列表, 错误列表[ (全局索引, 异常) ])。
    """
    attempt = 0
    while True:
        try:
            resp = await asyncio.to_thread(
                TextEmbedding.call,
                model=model,
                input=batch_docs,
                dimension=dim,
                text_type=text_type,
                api_key=EMBED_API_KEY
            )
            items = _parse_embeddings(resp)

            vecs: List[np.ndarray] = []
            # items 顺序与 batch_docs 保持一致（已按 text_index 排序）
            for it in items:
                emb = it.get("embedding")
                if emb is None:
                    raise ValueError("返回项缺少 'embedding'。")
                vecs.append(np.asarray(emb, dtype=np.float32))
            # 校验维度（个别模型版本可能返回不同维度）
            for i, v in enumerate(vecs):
                if v.shape[0] != dim:
                    raise ValueError(f"返回向量维度 {v.shape[0]} 与期望 {dim} 不一致（全局索引 {batch_idx[i]}）。")
            return batch_idx, vecs, []
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                # 该批整体失败：把这一批里每个元素都记为错误
                errs = [(gidx, e) for gidx in batch_idx]
                return [], [], errs
            # 指数退避
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)))

async def embed_documents_async(
    documents: List[str],
    *,
    batch_size: int = 10,
    concurrency: int = 5,
    model: str = EMBED_MODEL,
    dim: int = EMBED_DIM,
    api_key: Optional[str] = EMBED_API_KEY,
    text_type: str = "document",
) -> Tuple[np.ndarray, List[Tuple[int, str]]]:
    """
    异步并发向量化：
    - 返回 (emb_matrix, errors)
      emb_matrix: [N, dim]，失败项为 np.nan
      errors: [(全局索引, 错误消息)]
    """
    N = len(documents)
    out = np.full((N, dim), np.nan, dtype=np.float32)
    errors: List[Tuple[int, str]] = []

    # 组批
    batches: List[Tuple[List[str], List[int]]] = []
    buf_docs: List[str] = []
    buf_idx: List[int] = []

    for i, doc in enumerate(documents):
        buf_docs.append(doc)
        buf_idx.append(i)
        if len(buf_docs) == batch_size:
            batches.append((buf_docs, buf_idx))
            buf_docs, buf_idx = [], []
    if buf_docs:
        batches.append((buf_docs, buf_idx))

    sem = asyncio.Semaphore(concurrency)

    async def runner(b_docs, b_idx):
        async with sem:
            return await _embed_batch(
                b_docs, b_idx, model=model, dim=dim,
                api_key=api_key, text_type=text_type
            )

    tasks = [asyncio.create_task(runner(b_docs, b_idx)) for (b_docs, b_idx) in batches]

    for task in asyncio.as_completed(tasks):
        ok_idx, vecs, errs = await task
        # 写入成功结果
        for gidx, v in zip(ok_idx, vecs):
            out[gidx, :] = v
        # 记录错误
        for gidx, exc in errs:
            errors.append((gidx, repr(exc)))

    return out, errors

    
    

def embed_query(query: str) -> np.ndarray:
    resp = TextEmbedding.call(
        model=EMBED_MODEL,
        input=query,
        dimension=EMBED_DIM,
        text_type="query",
        instruct="Given a knowledge point, retrieve relevant sectionn in textbook",
        api_key=EMBED_API_KEY
    )
    emb = resp.output["embeddings"][0]["embedding"]
    return np.array(emb, dtype=np.float32)


# ========== FAISS 索引 ==========
def build_faiss(dim: int) -> faiss.IndexFlatIP:
    """使用内积 + 归一化实现余弦相似度检索。"""
    return faiss.IndexFlatIP(dim)


def save_index(index: faiss.Index, meta: Dict[str, Any]) -> None:
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index() -> Tuple[faiss.Index, Dict[str, Any]]:
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("索引或元数据不存在，请先运行 `index` 子命令构建。")
    index = faiss.read_index(str(INDEX_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return index, meta


# ========== 索引构建 ==========
def index_corpus(paths: List[Path]) -> None:
    """
    读取 Markdown → 切片 → OpenAI 兼容接口生成嵌入 → 归一化 → 写入 FAISS
    """
    all_chunks: List[Dict[str, Any]] = []
    for p in paths:
        raw = read_text(p)
        chunks = md_smart_chunks(raw)
        for i, ch in enumerate(chunks):
            all_chunks.append({
                "doc_id": p.name,
                "chunk_id": f"{p.name}::{i}",
                "title": ch["title"],
                "text": ch["text"]
            })

    documents = [c["text"] for c in all_chunks]
    print(f"[索引] 文本切片数：{len(documents)}（开始向量化）")

    vecs = embed_documents(documents)  # (N, D)
    vecs = normalize(vecs)

    index = build_faiss(EMBED_DIM)
    index.add(vecs)

    meta = {
        "dim": EMBED_DIM,
        "size": len(all_chunks),
        "chunks": all_chunks,  # 简单用 json 存，生产可换 SQLite/Parquet
        "source": paths.__repr__(),
        "model": EMBED_MODEL
    }
    save_index(index, meta)
    print(f"[索引] 完成：向量条目 {meta['size']}，写入 {INDEX_PATH} / {META_PATH}")


# ========== 召回 & 生成 ==========
def search(query: str, k: int = 5) -> List[Tuple[int, float]]:
    """
    用 dashscope.TextEmbedding 生成查询向量（必须项 2），做 FAISS Top-k 检索。
    返回 [(chunk_idx, score), ...] 其中 chunk_idx 是 meta['chunks'] 的下标。
    """
    index, meta = load_index()

    q = embed_query(query)
    q = q.reshape(1, -1).astype(np.float32)
    q = normalize(q)

    scores, idxs = index.search(q, k)
    # 内积即余弦相似度（归一化后）
    return list(zip(idxs[0].tolist(), scores[0].tolist()))


def build_context(meta: Dict[str, Any], hits: List[Tuple[int, float]]) -> str:
    """
    把命中的切片拼成 Prompt 上下文，同时保留来源标注。
    """
    lines = []
    for rank, (i, score) in enumerate(hits, 1):
        ch = meta["chunks"][i]
        header = f"[{rank}] {ch['doc_id']} | {ch['title']} | sim={score:.3f}"
        body = ch["text"].strip()
        lines.append(header + "\n" + body)
    return "\n\n-----\n\n".join(lines)


# ========== CLI ==========
def main():
    parser = argparse.ArgumentParser(description="RAG over ctt1.md + ctt2.md (DashScope + FAISS)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="构建索引")

    p_query = sub.add_parser("query", help="查询并生成答案")
    p_query.add_argument("q", type=str, help="用户问题")
    p_query.add_argument("--k", type=int, default=5, help="召回条数")

    args = parser.parse_args()
    
    if args.cmd == "index":
        # 以你上传的两份文件为语料
        paths = [Path("./ctt1.md"), Path("./ctt2.md")]
        index_corpus(paths)

    elif args.cmd == "query":
        _, meta = load_index()
        hits = search(args.q, k=args.k)
        ctx = build_context(meta, hits)
        print("==== 命中切片（用于答案的上下文） ====")
        print(ctx)


if __name__ == "__main__":
    main()
