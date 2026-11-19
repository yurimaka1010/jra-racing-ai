"""No credentials, DB connections, downloads, or task changes."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from jra_racing_ai.objective_pilot.runner import main

if __name__ == '__main__':
    raise SystemExit(main())
