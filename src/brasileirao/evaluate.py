"""Evaluate rating methods against historical results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom

from .simulator import (
    parse_matches,
    get_strengths,
    _estimate_team_home_advantages,
    _spi_probs,
)


_DEF_METHODS = [
    "ratio",
    "historic_ratio",
    "poisson",
    "neg_binom",
    "skellam",
    "elo",
    "dixon_coles",
    "spi",
    "leader_history",
]


def _poisson_probs(lam: float, mu: float, max_goals: int = 10) -> tuple[float, float, float]:
    """Return win/draw/loss probabilities for Poisson rates."""
    probs = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            probs[i, j] = poisson.pmf(i, lam) * poisson.pmf(j, mu)
    home = np.tril(probs, -1).sum()
    draw = np.trace(probs)
    away = np.triu(probs, 1).sum()
    total = home + draw + away
    if total < 1.0:
        tail = 1.0 - total
        home += tail / 3
        draw += tail / 3
        away += tail / 3
    return float(home), float(draw), float(away)


def _nbinom_probs(r: float, p_home: float, p_away: float, max_goals: int = 10) -> tuple[float, float, float]:
    """Return win/draw/loss probabilities for Negative Binomial parameters."""
    probs = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            probs[i, j] = nbinom.pmf(i, r, p_home) * nbinom.pmf(j, r, p_away)
    home = np.tril(probs, -1).sum()
    draw = np.trace(probs)
    away = np.triu(probs, 1).sum()
    total = home + draw + away
    if total < 1.0:
        tail = 1.0 - total
        home += tail / 3
        draw += tail / 3
        away += tail / 3
    return float(home), float(draw), float(away)


def _dixon_coles_probs(lam: float, mu: float, rho: float, max_goals: int = 10) -> tuple[float, float, float]:
    """Return win/draw/loss probabilities for Dixon-Coles parameters."""
    probs = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            tau = 1.0
            if i == 0 and j == 0:
                tau = 1 - lam * mu * rho
            elif i == 0 and j == 1:
                tau = 1 + lam * rho
            elif i == 1 and j == 0:
                tau = 1 + mu * rho
            elif i == 1 and j == 1:
                tau = 1 - rho
            tau = max(tau, 0.0)
            probs[i, j] = poisson.pmf(i, lam) * poisson.pmf(j, mu) * tau
    home = np.tril(probs, -1).sum()
    draw = np.trace(probs)
    away = np.triu(probs, 1).sum()
    total = home + draw + away
    if total < 1.0:
        tail = 1.0 - total
        home += tail / 3
        draw += tail / 3
        away += tail / 3
    return float(home), float(draw), float(away)


def _match_probs(
    home: str,
    away: str,
    strengths: dict[str, dict[str, float]],
    avg_goals: float,
    home_adv: float,
    team_home_adv: dict[str, float],
    rating_method: str,
    extra_param: float | tuple[float, float],
) -> tuple[float, float, float]:
    """Return win/draw/loss probabilities for a single match."""
    factor = team_home_adv.get(home, 1.0)
    lam = avg_goals * strengths[home]["attack"] * strengths[away]["defense"] * home_adv * factor
    mu = avg_goals * strengths[away]["attack"] * strengths[home]["defense"]

    if rating_method == "dixon_coles" and isinstance(extra_param, float):
        return _dixon_coles_probs(lam, mu, extra_param)
    if rating_method == "neg_binom" and isinstance(extra_param, float) and extra_param > 0:
        r = 1.0 / extra_param
        p_home = r / (r + lam)
        p_away = r / (r + mu)
        return _nbinom_probs(r, p_home, p_away)
    if rating_method == "spi" and isinstance(extra_param, tuple):
        return _spi_probs(lam - mu, extra_param)
    return _poisson_probs(lam, mu)


def evaluate_methods(
    path: str | Path,
    methods: list[str] = _DEF_METHODS,
    *,
    elo_k: float = 20.0,
    elo_home_advantage: float = 0.0,
    leader_history_paths: list[str | Path] | None = None,
    leader_weight: float = 0.5,
    smooth: float = 1.0,
    market_path: str | Path = "data/Brasileirao2025A.csv",
) -> pd.DataFrame:
    """Evaluate ``methods`` on the season file at ``path``."""

    matches = parse_matches(path)
    played = matches.dropna(subset=["home_score", "away_score"]).sort_values("date").reset_index(drop=True)

    results: list[dict[str, float]] = []

    for method in methods:
        brier = 0.0
        n = 0
        for idx, row in played.iterrows():
            history = played.iloc[:idx]
            strengths, avg_goals, home_adv, extra = get_strengths(
                history,
                method,
                elo_k=elo_k,
                home_field_advantage=elo_home_advantage,
                leader_history_paths=leader_history_paths,
                leader_history_weight=leader_weight,
                smooth=smooth,
                market_path=market_path,
            )
            team_home_adv = _estimate_team_home_advantages(history)
            probs = _match_probs(
                row["home_team"],
                row["away_team"],
                strengths,
                avg_goals,
                home_adv,
                team_home_adv,
                method,
                extra,
            )
            outcome = "D"
            if row["home_score"] > row["away_score"]:
                outcome = "H"
            elif row["home_score"] < row["away_score"]:
                outcome = "A"
            obs = np.array([outcome == "H", outcome == "D", outcome == "A"], dtype=float)
            brier += np.sum((obs - np.array(probs)) ** 2)
            n += 1
        results.append({"method": method, "brier": brier / n if n else np.nan})

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rating methods on a season file")
    parser.add_argument("file", help="season results file")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=_DEF_METHODS,
        help="rating methods to evaluate",
    )
    parser.add_argument("--elo-k", type=float, default=20.0, help="K factor for Elo")
    parser.add_argument(
        "--elo-home-advantage",
        type=float,
        default=0.0,
        help="home advantage in Elo calculations",
    )
    parser.add_argument(
        "--leader-history-paths",
        nargs="*",
        default=["data/Brasileirao2024A.txt"],
        help="season files used for leader_history rating",
    )
    parser.add_argument(
        "--leader-weight",
        type=float,
        default=0.5,
        help="leader_history weighting factor",
    )
    parser.add_argument(
        "--market-path",
        default="data/Brasileirao2025A.csv",
        help="CSV with team market values for spi",
    )
    parser.add_argument("--smooth", type=float, default=1.0, help="smoothing constant")
    args = parser.parse_args()

    df = evaluate_methods(
        args.file,
        args.methods,
        elo_k=args.elo_k,
        elo_home_advantage=args.elo_home_advantage,
        leader_history_paths=args.leader_history_paths,
        leader_weight=args.leader_weight,
        smooth=args.smooth,
        market_path=args.market_path,
    )
    for _, row in df.iterrows():
        print(f"{row['method']:15s} {row['brier']:.4f}")


if __name__ == "__main__":
    main()
