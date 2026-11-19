"""Isolated one-fit candidate; existing recorder tasks are unchanged."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from jra_racing_ai.d39_enrichment.cli import main
if __name__=='__main__':raise SystemExit(main())
