"""A10/B20: maximum16 newfits; C19/D39 and Boruta are read-only references."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from jra_racing_ai.tuner_four_models import main

if __name__ == '__main__':
    raise SystemExit(main())
