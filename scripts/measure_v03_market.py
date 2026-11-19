"""Run locally by the owner; import never connects to PostgreSQL."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jra_racing_ai.market_baseline_cli import main


if __name__ == "__main__":
    raise SystemExit(main(["--execute-local", *sys.argv[1:]]))
