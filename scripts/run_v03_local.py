"""Local user entry point; no database access happens on import."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jra_racing_ai.ml_v03_cli import main


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        args = ["first-run", "--execute-local", *args]
    raise SystemExit(main(args))
