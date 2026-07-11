import argparse

from backend.app.multi_agent_rag import MultiAgentRAG, load_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run simple planner/retrieval/response RAG"
    )
    parser.add_argument("question", nargs="+", help="Question to ask the local RAG")
    parser.add_argument(
        "--config",
        default="backend/config/rag_runtime_config.json",
        help="Path to runtime config JSON",
    )
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Print planner output before the answer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question = " ".join(args.question).strip()
    cfg = load_runtime_config(runtime_config_path=args.config)
    app = MultiAgentRAG(cfg)

    result = app.run(question)

    if args.show_plan:
        print("Plan")
        print("-" * 60)
        for key, value in result["plan"].items():
            print(f"{key}: {value}")
        print()

    print("Answer")
    print("-" * 60)
    print(result["answer"])
    print()
    print("Sources")
    print("-" * 60)
    if result["sources"]:
        for src in result["sources"]:
            print(
                f"{src['source_file']}#row{src['row_index']} "
                f"chunk{src['chunk_index']} score={src['score']}"
            )
    else:
        print("No sources returned")


if __name__ == "__main__":
    main()
