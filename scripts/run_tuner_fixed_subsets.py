"""Independent fixed 39-model Gain ranking comparison; no deployment."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from jra_racing_ai.tuner_fixed_subsets import main

if __name__ == '__main__':
    raise SystemExit(main())
