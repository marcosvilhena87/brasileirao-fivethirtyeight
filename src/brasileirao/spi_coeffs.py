from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .simulator import parse_matches, estimate_spi_strengths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate SPI logistic regression coefficients"
    )
    parser.add_argument(
        "--market-path",
        default="data/Brasileirao2024A.csv",
        help="CSV with team market values",
    )
    parser.add_argument(
        "--out",
        type=argparse.FileType("w"),
        default="-",
        help="File to write coefficients (default: stdout)",
    )
    args = parser.parse_args()

    matches_2023 = parse_matches("data/Brasileirao2023A.txt")
    matches_2024 = parse_matches("data/Brasileirao2024A.txt")
    matches = pd.concat([matches_2023, matches_2024], ignore_index=True)

    _, _, _, intercept, slope = estimate_spi_strengths(
        matches, market_path=args.market_path
    )

    args.out.write(f"{intercept:.6f} {slope:.6f}\n")
    args.out.flush()


if __name__ == "__main__":
    main()
