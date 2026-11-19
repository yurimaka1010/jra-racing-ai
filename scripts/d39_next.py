"""Separate research entry point. No change to any existing scheduled action."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from jra_racing_ai.d39_next.cli import main
if __name__ == '__main__':
    raise SystemExit(main())
