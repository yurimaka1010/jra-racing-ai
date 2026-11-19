"""User-run shortcut for the already prepared 2026-10-04 prediction artifacts."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jra_racing_ai.win_sim_cli import main


if __name__ == "__main__":
    root = Path(r"C:\Users\acbd5\AppData\Local\JraRacingAI")
    arguments = ["publication-run", "--execute-local",
                 "--policy", str(root / "ml-v02/policies/bb20dc4a58804796b6dab81998192fe4/policy.json"),
                 "--predictions", str(root / "ml-v02/predictions/0bc0baa4431748308e94c416a515e3ed"),
                 "--entries", str(root / "ml-v01/datasets/5216fc96135a44639806ae5b5e020479/runners.csv")]
    raise SystemExit(main(arguments + sys.argv[1:]))
