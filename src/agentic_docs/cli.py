"""Thin CLI dispatcher — each subcommand's module owns its own argparse parser."""
import argparse
import sys

_COMMANDS = {
    "download": "agentic_docs.funsd.download",
    "run": "agentic_docs.cli_run",
    "analyze": "agentic_docs.analyze",
    "visualize": "agentic_docs.visualize",
}


def main(argv: list[str] | None = None) -> None:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] not in _COMMANDS:
        parser = argparse.ArgumentParser(prog="docs")
        parser.add_argument("command", choices=list(_COMMANDS))
        parser.parse_args(argv[:1])
        return
    command, rest = argv[0], argv[1:]
    import importlib
    module = importlib.import_module(_COMMANDS[command])
    module.main(rest)


if __name__ == "__main__":
    main()
