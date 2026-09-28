"""Canonical raw diagnostic runner; classified acceptance uses the task qualifier."""
from pathlib import Path
from _unit_runner import main
if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve().parent))
