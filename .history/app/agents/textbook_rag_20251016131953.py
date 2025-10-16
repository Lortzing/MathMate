from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from collections import Counter
import argparse
import asyncio
import json
import re

import numpy as np
import faiss
from dashscope import TextEmbedding

from config import Config


class TextBoookRAG:
    """RAG pipeline: build() to index, search() to query. Only expose these two methods."""

    # -------------------- public API --------------------
    def __init__(self, *, top_k: int = 5) -> None:
        # config
        self.PAGE_OPEN_RE = Config.PAGE_OPEN_RE
        self.PAGE_CLOSE_RE = Config.PAGE_CLOSE_RE
        self.HEAD_RE = Config.HEAD_RE

        self.INDEX_PATH: Path = Config.INDEX_PATH
        self.META_PATH: Path = Config.META_PATH

        self.API_KEY: Optional[str] = Config.EMBED_API_KEY
        self.BASE_URL: Optional[str] = Config.EMBED_BASE_URL  # 未直接使用，但保留
        self.DIM: int = Config.EMBED_DIM
        self.MODEL: str = Config.EMBED_MODEL

        self.top_k = top_k

    def build(self, document_path: Path) -> None:
        """
        构建/重建索引。
        - 如果传入目录：递归收集其中所有 .md 文件
        - 如果传入文件：仅索引该文件（要求 .md）
        输出：写入 INDEX_PATH / META_PATH
        """
        paths = self._resolve_paths(document_path)
        if not paths:
            raise FileNotFoundError(f"未找到可用的 .md 文件：{document_path}")

        # 1) 读取 + 切片
        all_chunks: List[Dict[str, Any]] = []
        for p in paths:
            raw = self._read_text(p)
            chunks = self._md_smart_chunks(raw)
            for i, ch in enumerate(chunks):
                all_chunks.append({
                    "doc_id": p.name,
                    "chunk_id": f"{p.name}::{i}",
                    "title": ch["title"],
                    "text": ch["text"],
                })

        documents = [c["text"] for c in all_chunks]
        print(f"[索引] 文本切片数：{len(documents)}（开始向量化）")

        # 2) 向量化 + 归一化
        vecs = self._embed_documents(documents)  # (N, D)
        vecs = self._normalize(vecs)

        # 3) FAISS 索引写入
        index = self._build_faiss(self.DIM)
        index.add(vecs)

        meta = {
            "dim": self.DIM,
            "size": len(all_chunks),
            "chunks": all_chunks,   # 简单 JSON 存储；生产可换 SQLite/Parquet
            "source": [str(p) for p in paths],
            "model": self.MODEL,
        }
        self._save_index(index, meta)
        print(f"[索引] 完成：向量条目 {meta['size']}，写入 {self.INDEX_PATH} / {self.META_PATH}")

    def search(self, query: str) -> List[Tuple[int, float]]:
        """
        用 DashScope 生成查询向量 → FAISS Top-k 检索。
        返回 [(chunk_idx, score), ...]，chunk_idx 可到 meta['chunks'][i] 取对应内容。
        """
        index, meta = self._load_index()

        q = self._embed_query(query)
        q = q.reshape(1, -1).astype(np.float32)
        q = self._normalize(q)

        scores, idxs = index.search(q, self.top_k)
        hits = list(zip(idxs[0].tolist(), scores[0].tolist()))

        # 打印一个可视化的命中列表（非必须）
        print("==== 命中切片（用于答案的上下文） ====")
        print(self._build_context(meta, hits))

        return hits

    # -------------------- private helpers --------------------
    def _resolve_paths(self, p: Path) -> List[Path]:
        if p.is_dir():
            return sorted([x for x in p.rglob("*.md") if x.is_file()])
        if p.is_file() and p.suffix.lower() == ".md":
            return [p]
        return []

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs / norms

    @staticmethod
    def _read_text(path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")

    def _pick_most_frequent_heading_level(self, text: str, prefer_deeper_on_tie: bool = True) -> Optional[int]:
        cnt = Counter(len(m.group(1)) for m in self.HEAD_RE.finditer(text))
        if not cnt:
            return None
        max_freq = max(cnt.values())
        cands = [lvl for lvl, n in cnt.items() if n == max_freq]
        print(f"出现次数最多的标题级别：{cands}（次数={max_freq}）")
        return (max(cands) if prefer_deeper_on_tie else min(cands))

    def _align_to_page_boundaries(self, s: str, start: int, end: int) -> Tuple[int, int]:
        adj_start = start
        last_open = None
        for m in self.PAGE_OPEN_RE.finditer(s, 0, max(0, start)):
            last_open = m
        if last_open:
            adj_start = last_open.start()

        adj_end = end
        next_close = self.PAGE_CLOSE_RE.search(s, min(len(s), end))
        if next_close:
            adj_end = next_close.end()

        if adj_start >= adj_end:
            adj_start, adj_end = start, end
        return adj_start, adj_end

    def _md_smart_chunks(self, text: str, max_chars: int = 4096, overlap: int = 0) -> List[Dict[str, Any]]:
        split_level = self._pick_most_frequent_heading_level(text, prefer_deeper_on_tie=True)

        blocks: List[Tuple[str, str, int]] = []
        current_title = f"H{split_level or 0}:ROOT"
        current_buf: List[str] = []

        def flush():
            if current_buf:
                blocks.append((current_title, "\n".join(current_buf).strip(), split_level or 0))

        if split_level is None:
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

        chunks: List[Dict[str, Any]] = []
        for title, body, lvl in blocks:
            body = body.strip()
            if not body:
                continue

            if len(body) <= max_chars:
                chunks.append({"title": title, "title_level": lvl, "text": body})
                continue

            seen_ranges = set()
            start = 0
            L = len(body)
            while start < L:
                end = min(L, start + max_chars)
                adj_start, adj_end = self._align_to_page_boundaries(body, start, end)

                key = (adj_start, adj_end)
                if adj_end - adj_start > 0 and key not in seen_ranges:
                    seen_ranges.add(key)
                    sub = body[adj_start:adj_end]
                    chunks.append({"title": title, "title_level": lvl, "text": sub})

                if end >= L:
                    break
                start = max(0, end - overlap)

        return chunks

    # ---- embedding batch (async) ----
    @staticmethod
    def _parse_embeddings(resp) -> List[Dict[str, Any]]:
        try:
            items = resp.output.get("embeddings", [])
        except Exception:
            print(resp)
            raise TypeError("无embeddings字段")
        if not isinstance(items, list):
            raise TypeError("resp.output['embeddings'] 应为 list。")
        if all(isinstance(x, dict) and 'text_index' in x for x in items):
            items = sorted(items, key=lambda x: x['text_index'])
        return items

    async def _embed_batch(
        self,
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
        attempt = 0
        while True:
            try:
                resp = await asyncio.to_thread(
                    TextEmbedding.call,
                    model=model,
                    input=batch_docs,
                    dimension=dim,
                    text_type=text_type,
                    api_key=api_key,
                )
                items = self._parse_embeddings(resp)

                vecs: List[np.ndarray] = []
                for it in items:
                    emb = it.get("embedding")
                    if emb is None:
                        raise ValueError("返回项缺少 'embedding'。")
                    vecs.append(np.asarray(emb, dtype=np.float32))

                for i, v in enumerate(vecs):
                    if v.shape[0] != dim:
                        raise ValueError(f"返回向量维度 {v.shape[0]} 与期望 {dim} 不一致（全局索引 {batch_idx[i]}）。")
                return batch_idx, vecs, []
            except Exception as e:
                attempt += 1
                if attempt >= max_retries:
                    errs = [(gidx, e) for gidx in batch_idx]
                    return [], [], errs
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))

    async def _embed_documents_async(
        self,
        documents: List[str],
        *,
        batch_size: int = 10,
        concurrency: int = 5,
        model: str,
        dim: int,
        api_key: Optional[str],
        text_type: str = "document",
    ) -> Tuple[np.ndarray, List[Tuple[int, str]]]:
        N = len(documents)
        out = np.full((N, dim), np.nan, dtype=np.float32)
        errors: List[Tuple[int, str]] = []

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
                return await self._embed_batch(
                    b_docs, b_idx, model=model, dim=dim, api_key=api_key, text_type=text_type
                )

        tasks = [asyncio.create_task(runner(b_docs, b_idx)) for (b_docs, b_idx) in batches]

        for task in asyncio.as_completed(tasks):
            ok_idx, vecs, errs = await task
            for gidx, v in zip(ok_idx, vecs):
                out[gidx, :] = v
            for gidx, exc in errs:
                errors.append((gidx, repr(exc)))

        return out, errors

    def _embed_documents(self, documents: List[str]) -> np.ndarray:
        emb_matrix, errors = asyncio.run(
            self._embed_documents_async(
                documents,
                model=self.MODEL,
                dim=self.DIM,
                api_key=self.API_KEY,
                text_type="document",
            )
        )
        print(emb_matrix.shape)  # -> (N, D)
        if errors:
            print(f"有 {len(errors)} 条样本失败，示例：", errors[:3])
        return emb_matrix

    def _embed_query(self, query: str) -> np.ndarray:
        resp = TextEmbedding.call(
            model=self.MODEL,
            input=query,
            dimension=self.DIM,
            text_type="query",
            instruct="Given a knowledge point, retrieve relevant sectionn in textbook",
            api_key=self.API_KEY,
        )
        emb = resp.output["embeddings"][0]["embedding"]
        return np.array(emb, dtype=np.float32)

    # ---- FAISS I/O ----
    @staticmethod
    def _build_faiss(dim: int) -> faiss.IndexFlatIP:
        return faiss.IndexFlatIP(dim)

    def _save_index(self, index: faiss.Index, meta: Dict[str, Any]) -> None:
        faiss.write_index(index, str(self.INDEX_PATH))
        self.META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_index(self) -> Tuple[faiss.Index, Dict[str, Any]]:
        if not self.INDEX_PATH.exists() or not self.META_PATH.exists():
            raise FileNotFoundError("索引或元数据不存在，请先调用 build() 构建。")
        index = faiss.read_index(str(self.INDEX_PATH))
        meta = json.loads(self.META_PATH.read_text(encoding="utf-8"))
        return index, meta

    # ---- pretty print (optional) ----
    @staticmethod
    def _build_context(meta: Dict[str, Any], hits: List[Tuple[int, float]]) -> str:
        lines = []
        for rank, (i, score) in enumerate(hits, 1):
            ch = meta["chunks"][i]
            header = f"[{rank}] {ch['doc_id']} | {ch['title']} | sim={score:.3f}"
            body = ch["text"].strip()
            lines.append(header + "\n" + body)
        return "\n\n-----\n\n".join(lines)


# -------------------- CLI (可选) --------------------
def _main():
    parser = argparse.ArgumentParser(description="RAG over Markdown (DashScope + FAISS)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="构建索引")
    p_build.add_argument("path", type=Path, help="Markdown 文件或目录")

    p_search = sub.add_parser("search", help="查询")
    p_search.add_argument("q", type=str, help="用户问题")

    args = parser.parse_args()
    rag = RAG(top_k=5)

    if args.cmd == "build":
        rag.build(args.path)
    elif args.cmd == "search":
        rag.search(args.q)


if __name__ == "__main__":
    _main()
