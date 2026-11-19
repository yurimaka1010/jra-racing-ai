"""Project-local entry point; output is fixed status codes and private paths only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from jra_racing_ai.keiba_lab_live import main
if __name__ == '__main__':
    raise SystemExit(main())
