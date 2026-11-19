"""Delete only Gain ranks 24, 28, 30 from the saved A39 Tuner model."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from jra_racing_ai.tuner_rank_ablation import main

if __name__ == '__main__':
    raise SystemExit(main())
