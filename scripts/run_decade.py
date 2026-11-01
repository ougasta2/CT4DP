"""Run the eight-year training/two-year holdout pilot from the repository root."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from energy_pilot.decade import main

if __name__=='__main__':
    main()
