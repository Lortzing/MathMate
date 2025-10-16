import pytest
from pathlib import Path
from app.agents.textbook_rag import TextbookRAG


def test_build_creates_index(tmp_path):
    """测试 build 能正常运行并生成索引文件"""
    # 构造一个简单的测试文档
    md_file = tmp_path / "test.md"
    md_file.write_text("# 测试章节\n这里是测试内容", encoding="utf-8")

    rag = TextbookRAG()
    rag.INDEX_PATH = tmp_path / "index.json"
    rag.META_PATH = tmp_path / "meta.json"
    rag.build(tmp_path)  # 传入目录

    # 断言 build 后生成了索引和元数据
    assert rag.INDEX_PATH.exists(), "向量索引文件未创建"
    assert rag.META_PATH.exists(), "元数据文件未创建"


def test_search():
    """测试 search 能在 build 之后正常运行"""
    rag = TextbookRAG(top_k=3)
    print(rag.INDEX_PATH)
    print(rag.META_PATH)

    # 运行搜索，不要求结果，只要能跑通
    hits = rag.search("微分方程")
    print(hits)