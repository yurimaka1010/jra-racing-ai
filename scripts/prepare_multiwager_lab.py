"""Local-only saved-development preparation; never train or read live results."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from jra_racing_ai.multiwager.lab import main
if __name__ == '__main__':
    raise SystemExit(main())
