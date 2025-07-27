from __future__ import annotations

import argparse
import os
import pathlib
import pandas as pd

from .simulator import (
    parse_matches,
    estimate_spi_strengths,
    SPI_DEFAULT_INTERCEPT,
    SPI_DEFAULT_SLOPE,
)


def available_seasons(data_dir: str | pathlib.Path = "data") -> list[str]:
    """Return a sorted list of seasons found in ``data_dir``."""
    seasons: list[str] = []
    for txt in pathlib.Path(data_dir).glob("Brasileirao????A.txt"):
        year = txt.stem[10:14]
        seasons.append(year)
    seasons.sort()
    return seasons


def compute_spi_coeffs(
    seasons: list[str] | None = None,
    *,
    data_dir: str | pathlib.Path = "data",
    market_path: str | pathlib.Path = "data/Brasileirao2025A.csv",
    smooth: float = 1.0,
) -> tuple[float, float]:
    """Return fitted intercept and slope from historical seasons.

    ``seasons`` may be provided as a list of years.  If omitted the value of the
    ``BRASILEIRAO_SEASONS`` environment variable is used.  When that is also
    unset all seasons found in ``data_dir`` are processed.  If no match files are
    available the default SPI coefficients are returned.
    """

    if seasons is None:
        env = os.getenv("BRASILEIRAO_SEASONS")
        if env:
            seasons = [s.strip() for s in env.split(",") if s.strip()]
    if not seasons:
        seasons = available_seasons(data_dir)

    frames: list[pd.DataFrame] = []
    for season in seasons:
        path = pathlib.Path(data_dir) / f"Brasileirao{season}A.txt"
        if path.exists():
            frames.append(parse_matches(path))

    if not frames:
        return SPI_DEFAULT_INTERCEPT, SPI_DEFAULT_SLOPE

    matches = pd.concat(frames, ignore_index=True)

    _, _, _, intercept, slope = estimate_spi_strengths(
        matches, market_path=market_path, smooth=smooth
    )

    return intercept, slope


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate SPI logistic regression coefficients"
    )
    parser.add_argument(
        "--market-path",
        default="data/Brasileirao2025A.csv",
        help="CSV with team market values",
    )
    parser.add_argument(
        "--seasons",
        nargs="*",
        help="Seasons to include (default: all in data/ or $BRASILEIRAO_SEASONS)",
    )
    parser.add_argument(
        "--out",
        type=argparse.FileType("w"),
        default="-",
        help="File to write coefficients (default: stdout)",
    )
    args = parser.parse_args()

    intercept, slope = compute_spi_coeffs(
        seasons=args.seasons, market_path=args.market_path
    )

    args.out.write(f"{intercept:.6f} {slope:.6f}\n")
    args.out.flush()


if __name__ == "__main__":
    main()
