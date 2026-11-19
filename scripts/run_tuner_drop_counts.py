"""Corrected deletion-count comparison; remaining counts are 15/11/9."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from jra_racing_ai.tuner_drop_counts import main
if __name__=="__main__":
    raise SystemExit(main())
