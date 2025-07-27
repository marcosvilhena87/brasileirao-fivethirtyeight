from __future__ import annotations

import argparse
import os
import pathlib
import numpy as np
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
        year = txt.stem[11:15]
        seasons.append(year)
    seasons.sort()
    return seasons


def compute_spi_coeffs(
    seasons: list[str] | None = None,
    *,
    data_dir: str | pathlib.Path = "data",
    market_path: str | pathlib.Path = "data/Brasileirao2025A.csv",
    smooth: float = 1.0,
    decay_rate: float = 0.0,
) -> tuple[float, float]:
    """Return fitted intercept and slope from historical seasons.

    ``seasons`` may be provided as a list of years.  If omitted the value of the
    ``BRASILEIRAO_SEASONS`` environment variable is used.  When that is also
    unset all seasons found in ``data_dir`` are processed.  ``decay_rate``
    applies an exponential weight ``exp(-decay_rate * age)`` to each season
    where ``age`` counts seasons back from the most recent.  If no match files
    are available the default SPI coefficients are returned.
    """

    if seasons is None:
        env = os.getenv("BRASILEIRAO_SEASONS")
        if env:
            seasons = [s.strip() for s in env.split(",") if s.strip()]
    elif len(seasons) == 0:
        return SPI_DEFAULT_INTERCEPT, SPI_DEFAULT_SLOPE
    if not seasons:
        seasons = available_seasons(data_dir)

    frames: list[pd.DataFrame] = []
    if seasons:
        max_year = max(int(s) for s in seasons)
    else:
        max_year = 0
    for season in seasons:
        path = pathlib.Path(data_dir) / f"Brasileirao{season}A.txt"
        if path.exists():
            age = max_year - int(season)
            weight = float(np.exp(-decay_rate * age)) if decay_rate else 1.0
            df = parse_matches(path)
            df["weight"] = weight
            frames.append(df)

    if not frames:
        return SPI_DEFAULT_INTERCEPT, SPI_DEFAULT_SLOPE

    matches = pd.concat(frames, ignore_index=True)
    weights = matches.pop("weight") if "weight" in matches else None

    _, _, _, intercept, slope = estimate_spi_strengths(
        matches, market_path=market_path, smooth=smooth, match_weights=weights
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
        "--decay-rate",
        type=float,
        default=0.0,
        help="Exponential decay rate for historical seasons",
    )
    parser.add_argument(
        "--out",
        type=argparse.FileType("w"),
        default="-",
        help="File to write coefficients (default: stdout)",
    )
    args = parser.parse_args()

    intercept, slope = compute_spi_coeffs(
        seasons=args.seasons,
        market_path=args.market_path,
        decay_rate=args.decay_rate,
    )

    args.out.write(f"{intercept:.6f} {slope:.6f}\n")
    args.out.flush()


if __name__ == "__main__":
    main()
