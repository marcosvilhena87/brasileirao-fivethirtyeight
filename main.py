import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from brasileirao import (
    parse_matches,
    simulate_chances,
    simulate_relegation_chances,
    simulate_final_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Brasileirão 2025 title odds")
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
    args = parser.parse_args()

    matches = parse_matches(args.file)
    rng = np.random.default_rng(args.seed) if args.seed is not None else None
    if args.home_advantage_file:
        import json

        with open(args.home_advantage_file, "r", encoding="utf-8") as f:
            team_home = json.load(f)
    else:
        team_home = None
    chances = simulate_chances(
        matches,
        iterations=args.simulations,
        rng=rng,
        market_path=args.market_path,
        team_home_advantages=team_home,
    )

    relegation = simulate_relegation_chances(
        matches,
        iterations=args.simulations,
        rng=rng,
        market_path=args.market_path,
        team_home_advantages=team_home,
    )

    table_proj = simulate_final_table(
        matches,
        iterations=args.simulations,
        rng=rng,
        market_path=args.market_path,
        team_home_advantages=team_home,
    )

    # print("Title chances:")
    # for team, prob in sorted(chances.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{team:15s} {prob:.2%}")

    # print("\nRelegation chances:")
    # for team, prob in sorted(relegation.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{team:15s} {prob:.2%}")

    # print("\nExpected final position and points:")
    # for _, row in table_proj.iterrows():
    #     print(f"{row['team']:15s} {row['position']:5.1f} {row['points']:5.1f}")

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


if __name__ == "__main__":
    main()
