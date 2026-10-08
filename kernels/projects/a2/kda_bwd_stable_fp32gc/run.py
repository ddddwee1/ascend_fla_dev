"""Standalone runner for the D-PM-60 FP32 cumulative-gate backward unit."""
from pathlib import Path
from _unit_runner import main

if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve().parent))
