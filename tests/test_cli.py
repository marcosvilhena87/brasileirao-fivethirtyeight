import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_accepts_rating_method():
    subprocess.run(
        [sys.executable, "main.py", "--rating-method", "elo", "--simulations", "1"],
        check=True,
        cwd=ROOT,
        capture_output=True,
    )


def test_cli_accepts_seasons():
    subprocess.run(
        [
            sys.executable,
            "main.py",
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


def test_spi_coeffs_cli_accepts_logistic_decay():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "brasileirao.spi_coeffs",
            "--logistic-decay",
            "0.01",
        ],
        check=True,
        cwd=ROOT,
        env=env,
        capture_output=True,
    )
