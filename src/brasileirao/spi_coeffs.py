from __future__ import annotations

import argparse
import os
import pathlib
import numpy as np
import pandas as pd
from collections.abc import Mapping

from .simulator import (
    parse_matches,
    estimate_spi_strengths,
    SPI_DEFAULT_INTERCEPT,
    SPI_DEFAULT_SLOPE,
)


def season_complete(path: pathlib.Path) -> bool:
    """Return ``True`` if all fixtures in ``path`` have final scores."""
    df = parse_matches(path)
    return not (df["home_score"].isna().any() or df["away_score"].isna().any())


def available_seasons(
    data_dir: str | pathlib.Path = "data", *, only_complete: bool = False
) -> list[str]:
    """Return a sorted list of seasons found in ``data_dir``."""
    seasons: list[str] = []
    for txt in pathlib.Path(data_dir).glob("Brasileirao????A.txt"):
        if only_complete and not season_complete(txt):
            continue
        year = txt.stem[11:15]
        seasons.append(year)
    seasons.sort()
    return seasons


def compute_spi_coeffs(
    seasons: list[str] | None = None,
    *,
    data_dir: str | pathlib.Path = "data",
    market_path: str | pathlib.Path | dict[str, str | pathlib.Path] = (
        "data/Brasileirao{season}A.csv"
    ),
    smooth: float = 1.0,
    decay_rate: float = 0.0,
    logistic_decay: float | None = 0.007,
) -> tuple[float, float]:
    """Return fitted intercept and slope from historical seasons.

    ``seasons`` may be provided as a list of years.  If omitted the value of the
    ``BRASILEIRAO_SEASONS`` environment variable is used.  When that is also
    unset all seasons found in ``data_dir`` are processed. ``decay_rate``
    applies an exponential weight ``exp(-decay_rate * age)`` to each season
    where ``age`` counts seasons back from the most recent. Team market values
    are loaded from ``data/Brasileirao{season}A.csv`` by default. ``market_path``
    may be a custom pattern containing ``{season}`` or a mapping from season to
    CSV path. All matches from the selected seasons are merged into a single
    DataFrame and passed once to :func:`estimate_spi_strengths` using the
    computed weights. ``logistic_decay`` applies an exponential weight to
    recent fixtures when fitting the logistic regression. When both logistic
    weighting and the computed match weights are present they are multiplied.
    The default decay rate is ``0.007``. If no match files are
    available the default SPI coefficients are returned.
    """

    if seasons is None:
        env = os.getenv("BRASILEIRAO_SEASONS")
        if env:
            seasons = [s.strip() for s in env.split(",") if s.strip()]
    elif len(seasons) == 0:
        return SPI_DEFAULT_INTERCEPT, SPI_DEFAULT_SLOPE
    if not seasons:
        seasons = available_seasons(data_dir, only_complete=True)

    if isinstance(market_path, Mapping):
        market_map = {str(k): str(v) for k, v in market_path.items()}
        pattern = "data/Brasileirao{season}A.csv"
    else:
        market_map = {}
        pattern = str(market_path)

    dfs: list[pd.DataFrame] = []
    match_weights: list[pd.Series] = []

    if seasons:
        max_year = max(int(s) for s in seasons)
    else:
        max_year = 0

    for season in seasons:
        txt_path = pathlib.Path(data_dir) / f"Brasileirao{season}A.txt"
        if not txt_path.exists():
            continue
        df = parse_matches(txt_path)
        age = max_year - int(season)
        w = float(np.exp(-decay_rate * age)) if decay_rate else 1.0
        dfs.append(df)
        match_weights.append(pd.Series(w, index=df.index))

    if not dfs:
        return SPI_DEFAULT_INTERCEPT, SPI_DEFAULT_SLOPE

    all_matches = pd.concat(dfs, ignore_index=True)
    weights = pd.concat(match_weights, ignore_index=True)

    target_season = str(max_year) if seasons else ""
    csv_path = market_map.get(target_season, pattern.format(season=target_season))

    _, _, _, intercept, slope = estimate_spi_strengths(
        all_matches,
        market_path=csv_path,
        smooth=smooth,
        logistic_decay=logistic_decay,
        match_weights=weights,
    )

    return intercept, slope


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate SPI logistic regression coefficients"
    )
    parser.add_argument(
        "--market-path",
        action="append",
        dest="market_paths",
        help=(
            "Pattern with {season} or YEAR=CSV pair. "
            "Defaults to data/Brasileirao{season}A.csv"
        ),
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
        "--logistic-decay",
        type=float,
        default=None,
        help=(
            "exponential weight for recent matches when fitting the logistic"
            " regression"
        ),
    )
    parser.add_argument(
        "--out",
        type=argparse.FileType("w"),
        default="-",
        help="File to write coefficients (default: stdout)",
    )
    args = parser.parse_args()

    if not args.market_paths:
        market = "data/Brasileirao{season}A.csv"
    elif len(args.market_paths) == 1 and "=" not in args.market_paths[0]:
        market = args.market_paths[0]
    else:
        mapping: dict[str, str] = {}
        for spec in args.market_paths:
            if "=" not in spec:
                parser.error("Mapping entries must use SEASON=PATH syntax")
            season, path = spec.split("=", 1)
            mapping[season] = path
        market = mapping

    intercept, slope = compute_spi_coeffs(
        seasons=args.seasons,
        market_path=market,
        decay_rate=args.decay_rate,
        logistic_decay=args.logistic_decay,
    )

    args.out.write(f"{intercept:.6f} {slope:.6f}\n")
    args.out.flush()


if __name__ == "__main__":
    main()
