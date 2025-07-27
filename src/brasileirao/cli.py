import argparse
import json
from . import (
    parse_matches,
    simulate_chances,
    simulate_relegation_chances,
    simulate_final_table,
)
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate Brasileir\u00e3o 2025 title odds. "
            "The simulator uses FiveThirtyEight's SPI ratings by default and "
            "recomputes the logistic regression coefficients from the "
            "historical seasons in 'data/'. For a new campaign seed the ratings "
            "with `initial_spi_strengths`."
        )
    )
    parser.add_argument("--file", default="data/Brasileirao2025A.txt", help="fixture file path")
    parser.add_argument("--simulations", type=int, default=1000, help="number of simulation runs")
    parser.add_argument(
        "--market-path",
        default="data/Brasileirao2025A.csv",
        help="CSV with team market values for the FiveThirtyEight model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for repeatable simulations",
    )
    parser.add_argument(
        "--home-advantage-file",
        default=None,
        help="JSON mapping teams to home advantage multipliers",
    )
    parser.add_argument(
        "--rating-method",
        default="spi",
        choices=[
            "spi",
            "elo",
            "poisson",
            "neg_binom",
            "skellam",
            "historic_ratio",
            "dixon_coles",
            "initial_spi",
            "initial_ratio",
            "initial_points",
            "initial_points_market",
            "leader_history",
        ],
        help=(
            "algorithm used to rate teams (default: spi). "
            "SPI-based methods recompute the coefficients from the historical "
            "data."
        ),
    )
    parser.add_argument(
        "--seasons",
        nargs="*",
        help="historical seasons for SPI-based methods",
    )
    args = parser.parse_args()

    matches = parse_matches(args.file)
    rng = np.random.default_rng(args.seed) if args.seed is not None else None
    if args.home_advantage_file:
        with open(args.home_advantage_file, "r", encoding="utf-8") as f:
            team_home = json.load(f)
    else:
        team_home = None
    chances = simulate_chances(
        matches,
        iterations=args.simulations,
        rating_method=args.rating_method,
        rng=rng,
        market_path=args.market_path,
        team_home_advantages=team_home,
        seasons=args.seasons,
    )

    relegation = simulate_relegation_chances(
        matches,
        iterations=args.simulations,
        rating_method=args.rating_method,
        rng=rng,
        market_path=args.market_path,
        team_home_advantages=team_home,
        seasons=args.seasons,
    )

    table_proj = simulate_final_table(
        matches,
        iterations=args.simulations,
        rating_method=args.rating_method,
        rng=rng,
        market_path=args.market_path,
        team_home_advantages=team_home,
        seasons=args.seasons,
    )

    summary = table_proj.copy()
    summary["title"] = summary["team"].map(chances)
    summary["relegation"] = summary["team"].map(relegation)
    summary = summary.sort_values("position").reset_index(drop=True)
    summary["position"] = range(1, len(summary) + 1)
    summary["points"] = summary["points"].round().astype(int)

    TITLE_W = 7
    REL_W = 10
    print(f"{'Pos':>3}  {'Team':15s} {'Points':>6} {'Title':^{TITLE_W}} {'Relegation':^{REL_W}}")
    for _, row in summary.iterrows():
        title = f"{row['title']:.2%}"
        releg = f"{row['relegation']:.2%}"
        print(
            f"{row['position']:>2d}   {row['team']:15s} {row['points']:6d} {title:^{TITLE_W}} {releg:^{REL_W}}"
        )


