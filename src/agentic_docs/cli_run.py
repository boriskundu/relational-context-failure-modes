import argparse
from pathlib import Path

from agentic_docs.config import DOCUMENTS_PATH, EXTRACTIONS_DIR
from agentic_docs.funsd.parse import load_documents
from agentic_docs.llm_clients import get_available_models
from agentic_docs.runner import CONDITIONS, run_grid, status


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="docs run")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names from config.MODELS (default: all with an API key set)",
    )
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N documents")
    parser.add_argument(
        "--out", default=None, help="Output path (default: results/extractions/run.json)"
    )
    parser.add_argument("--status", action="store_true", help="Report progress only, no API calls")
    args = parser.parse_args(argv)

    documents = load_documents(DOCUMENTS_PATH)
    if args.limit:
        documents = documents[: args.limit]

    models = args.models or get_available_models()
    out_path = Path(args.out) if args.out else EXTRACTIONS_DIR / "run.json"

    if args.status:
        report = status(models, documents, args.conditions, out_path)
        for model_name, counts in report.items():
            print(
                f"  {model_name}: {counts['done']}/{counts['total']} done, {counts['failed']} failed "
                f"(extract), {counts['verify_failed']} verify-stage fallback"
            )
        return

    if not models:
        print("No models available -- set at least one API key in .env")
        return

    print(
        f"Running {len(documents)} documents x {len(args.conditions)} conditions x {len(models)} models"
    )
    run_grid(models, documents, args.conditions, out_path)
    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
