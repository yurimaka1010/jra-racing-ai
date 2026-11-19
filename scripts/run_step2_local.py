"""Run the fixed STEP 2 V2 plan against its already collected private snapshot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jra_racing_ai.learning_curve_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
