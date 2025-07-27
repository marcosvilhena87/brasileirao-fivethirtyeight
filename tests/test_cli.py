import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_accepts_rating_method():
    subprocess.run(
        [sys.executable, "-m", "brasileirao", "--rating-method", "elo", "--simulations", "1"],
        check=True,
        cwd=ROOT,
        capture_output=True,
    )


def test_cli_accepts_seasons():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "brasileirao",
            "--rating-method",
            "spi",
            "--seasons",
            "2023",
            "2024",
            "--simulations",
            "1",
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
    )
