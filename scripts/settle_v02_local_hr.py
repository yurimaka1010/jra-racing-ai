"""User-operated HR settlement of the already frozen 2026-10-04 virtual tickets."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jra_racing_ai.win_sim_cli import main


if __name__ == "__main__":
    raise SystemExit(main(["settle-hr", "--execute-local", "--simulation-id",
                          "a8b9aed2aafd4afba66aa30c019273bc"] + sys.argv[1:]))
