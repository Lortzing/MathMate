from __future__ import annotations
import argparse

from app.agents import MathAgent, MockVideoRAG, OCRAgent, TextbookRAG
from app.graph import RootGraphDeps, build_root_graph
from app.utils import get_logger, load_image_bytes


logger = get_logger()


def run_cli() -> None:
    """Entry point for the command-line interface."""

    parser = argparse.ArgumentParser(
        description="Run the multi-agent root orchestrator via CLI."
    )
    parser.add_argument("--query", type=str, required=True, help="User question")
    parser.add_argument(
        "--images",
        nargs="*",
        default=[],
        help="Optional image file paths to include in the request",
    )
    args = parser.parse_args()

    deps = RootGraphDeps(
        video_rag=MockVideoRAG(),
        textbook_rag=TextbookRAG(),
        ocr_agent=OCRAgent(),
        math_agent=MathAgent(),
    )
    graph_app = build_root_graph(deps)

    state_in: dict[str, object] = {"user_query": args.query}
    if args.images:
        state_in["images"] = load_image_bytes(args.images)

    logger.info("Invoking root graph...")
    result = graph_app.invoke(state_in)

    payload = {
        "video": result.get("video", {}),
        "textbook": result.get("textbook", {}),
        "reply_to_user": result.get("reply_to_user", ""),
    }
    print(payload)


if __name__ == "__main__":
    run_cli()
