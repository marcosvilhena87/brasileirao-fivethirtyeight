from __future__ import annotations

import numpy as np
import pandas as pd
import re
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import poisson


def _parse_date(date_str: str) -> pd.Timestamp:
    """Parse dates from multiple formats."""
    parts = date_str.split("/")
    year = parts[-1]
    if len(year) == 4:
        return pd.to_datetime(date_str, format="%d/%m/%Y")
    return pd.to_datetime(date_str, format="%m/%d/%y")

SCORE_PATTERN = re.compile(r"(\d+/\d+/\d+)\s+(.+?)\s+(\d+)-(\d+)\s+(.+?)\s*(?:\(ID:.*)?$")
NOSCORE_PATTERN = re.compile(r"(\d+/\d+/\d+)\s+(.+?)\s{2,}(.+?)\s*(?:\(ID:.*)?$")


def parse_matches(path: str | Path) -> pd.DataFrame:
    """Parse the fixture text file into a DataFrame."""
    rows: list[dict] = []
    in_games = False
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == 'GamesBegin':
                in_games = True
                continue
            if line.strip() == 'GamesEnd':
                break
            if not in_games:
                continue
            line = line.rstrip('\n')
            m = SCORE_PATTERN.match(line)
            if m:
                date_str, home, hs, as_, away = m.groups()
                rows.append({
                    'date': _parse_date(date_str),
                    'home_team': home.strip(),
                    'away_team': away.strip(),
                    'home_score': int(hs),
                    'away_score': int(as_),
                })
                continue
            m = NOSCORE_PATTERN.match(line)
            if m:
                date_str, home, away = m.groups()
                rows.append({
                    'date': _parse_date(date_str),
                    'home_team': home.strip(),
                    'away_team': away.strip(),
                    'home_score': np.nan,
                    'away_score': np.nan,
                })
    return pd.DataFrame(rows)


def _head_to_head_points(matches: pd.DataFrame, teams: list[str]) -> dict[str, int]:
    """Return points won in games among ``teams``."""
    points = {t: 0 for t in teams}
    df = matches.dropna(subset=["home_score", "away_score"])
    df = df[df["home_team"].isin(teams) & df["away_team"].isin(teams)]
    for _, row in df.iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        if hs > as_:
            points[ht] += 3
        elif hs < as_:
            points[at] += 3
        else:
            points[ht] += 1
            points[at] += 1
    return points


def league_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute league standings from match results."""
    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    table = {
        t: {"team": t, "played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0}
        for t in teams
    }

    played = matches.dropna(subset=['home_score', 'away_score'])
    for _, row in played.iterrows():
        home = row['home_team']
        away = row['away_team']
        hs = int(row['home_score'])
        as_ = int(row['away_score'])
        table[home]['played'] += 1
        table[away]['played'] += 1
        table[home]['gf'] += hs
        table[home]['ga'] += as_
        table[away]['gf'] += as_
        table[away]['ga'] += hs
        if hs > as_:
            table[home]['wins'] += 1
            table[home]['points'] = table[home].get('points', 0) + 3
            table[away]['losses'] += 1
            table[away].setdefault('points', 0)
        elif hs < as_:
            table[away]['wins'] += 1
            table[away]['points'] = table[away].get('points', 0) + 3
            table[home]['losses'] += 1
            table[home].setdefault('points', 0)
        else:
            table[home]['draws'] += 1
            table[away]['draws'] += 1
            table[home]['points'] = table[home].get('points', 0) + 1
            table[away]['points'] = table[away].get('points', 0) + 1

    for t in table.values():
        t.setdefault('points', 0)
        t['gd'] = t['gf'] - t['ga']

    df = pd.DataFrame(table.values())

    df["head_to_head"] = 0
    for _, group in df.groupby(["points", "wins", "gd", "gf"]):
        if len(group) <= 1:
            continue
        teams = group["team"].tolist()
        h2h = _head_to_head_points(played, teams)
        for t, val in h2h.items():
            df.loc[df["team"] == t, "head_to_head"] = val

    df = df.sort_values(
        ["points", "wins", "gd", "gf", "head_to_head", "team"],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    return df


def compute_leader_stats(matches: pd.DataFrame) -> dict[str, int]:
    """Return how often each team led the table as the season progressed."""
    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    leader_counts = {t: 0 for t in teams}

    stats = {
        t: {
            "team": t,
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "gf": 0,
            "ga": 0,
            "points": 0,
        }
        for t in teams
    }

    played_rows: list[dict] = []

    for _, row in matches.sort_values("date").iterrows():
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue

        ht = row["home_team"]
        at = row["away_team"]
        hs = int(row["home_score"])
        as_ = int(row["away_score"])

        played_rows.append(row.to_dict())

        h = stats[ht]
        a = stats[at]
        h["played"] += 1
        a["played"] += 1
        h["gf"] += hs
        h["ga"] += as_
        a["gf"] += as_
        a["ga"] += hs

        if hs > as_:
            h["wins"] += 1
            a["losses"] += 1
            h["points"] += 3
        elif hs < as_:
            a["wins"] += 1
            h["losses"] += 1
            a["points"] += 3
        else:
            h["draws"] += 1
            a["draws"] += 1
            h["points"] += 1
            a["points"] += 1

        df = pd.DataFrame(stats.values())
        df["gd"] = df["gf"] - df["ga"]
        df["head_to_head"] = 0
        played_df = pd.DataFrame(played_rows)

        for _, group in df.groupby(["points", "wins", "gd", "gf"]):
            if len(group) <= 1:
                continue
            teams_tied = group["team"].tolist()
            h2h = _head_to_head_points(played_df, teams_tied)
            for t, val in h2h.items():
                df.loc[df["team"] == t, "head_to_head"] = val

        df = df.sort_values(
            ["points", "wins", "gd", "gf", "head_to_head", "team"],
            ascending=[False, False, False, False, False, True],
        ).reset_index(drop=True)

        leader_counts[df.iloc[0]["team"]] += 1

    return leader_counts


def _estimate_strengths(matches: pd.DataFrame, smooth: float = 1.0):
    played = matches.dropna(subset=['home_score', 'away_score'])
    total_goals = played['home_score'].sum() + played['away_score'].sum()
    total_games = len(played)
    avg_goals = total_goals / total_games if total_games else 2.5
    home_adv = played['home_score'].sum() / played['away_score'].sum() if played['away_score'].sum() else 1.0

    teams = pd.unique(matches[['home_team', 'away_team']].values.ravel())
    strengths = {}
    for team in teams:
        gf = (
            played.loc[played.home_team == team, 'home_score'].sum() +
            played.loc[played.away_team == team, 'away_score'].sum()
        )
        ga = (
            played.loc[played.home_team == team, 'away_score'].sum() +
            played.loc[played.away_team == team, 'home_score'].sum()
        )
        gp = played.loc[(played.home_team == team) | (played.away_team == team)].shape[0]
        if gp == 0:
            attack = defense = 1.0
        else:
            attack = ((gf + smooth) / (gp + smooth)) / avg_goals
            defense = ((ga + smooth) / (gp + smooth)) / avg_goals
        strengths[team] = {'attack': attack, 'defense': defense}
    return strengths, avg_goals, home_adv


def _estimate_team_home_advantages(matches: pd.DataFrame) -> dict[str, float]:
    """Return relative home advantage factors for each team."""
    played = matches.dropna(subset=["home_score", "away_score"])
    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())

    total_home = played["home_score"].sum()
    total_away = played["away_score"].sum()
    baseline = total_home / total_away if total_away else 1.0

    factors = {}
    for t in teams:
        home_games = played[played.home_team == t]
        away_games = played[played.away_team == t]
        if not len(home_games) or not len(away_games):
            factors[t] = 1.0
            continue
        home_gpg = home_games["home_score"].mean()
        away_gpg = away_games["away_score"].mean()
        if away_gpg == 0 or np.isnan(home_gpg) or np.isnan(away_gpg):
            factors[t] = 1.0
        else:
            factors[t] = float((home_gpg / away_gpg) / baseline)
    return factors


def _estimate_dispersion(matches: pd.DataFrame) -> float:
    """Return method-of-moments dispersion for Negative Binomial sampling."""
    played = matches.dropna(subset=["home_score", "away_score"])
    if played.empty:
        return 0.0
    mean_home = played["home_score"].mean()
    var_home = played["home_score"].var()
    mean_away = played["away_score"].mean()
    var_away = played["away_score"].var()
    alpha_home = max(var_home - mean_home, 0.0) / (mean_home ** 2) if mean_home else 0.0
    alpha_away = max(var_away - mean_away, 0.0) / (mean_away ** 2) if mean_away else 0.0
    return (alpha_home + alpha_away) / 2


def estimate_strengths_with_history(
    current_matches: pd.DataFrame | None = None,
    past_path: str | Path = "data/Brasileirao2024A.txt",
    past_weight: float = 0.5,
    smooth: float = 1.0,
) -> tuple[dict[str, dict[str, float]], float, float]:
    """Estimate strengths using current season matches and weighted history."""
    if current_matches is None:
        current_matches = parse_matches("data/Brasileirao2025A.txt")
    past_matches = parse_matches(past_path)
    if 0 < past_weight < 1:
        past_matches = past_matches.sample(frac=past_weight, random_state=0).reset_index(drop=True)
    combined = pd.concat([current_matches, past_matches], ignore_index=True)
    return _estimate_strengths(combined, smooth=smooth)


def estimate_leader_history_strengths(
    current_matches: pd.DataFrame | None = None,
    past_paths: list[str | Path] | str | Path = "data/Brasileirao2024A.txt",
    weight: float = 0.5,
    smooth: float = 1.0,
) -> tuple[dict[str, dict[str, float]], float, float]:
    """Estimate strengths influenced by historical league leaders."""
    if current_matches is None:
        current_matches = parse_matches("data/Brasileirao2025A.txt")
    strengths, avg_goals, home_adv = _estimate_strengths(current_matches, smooth=smooth)

    if isinstance(past_paths, (str, Path)):
        past_paths = [past_paths]

    leader_counts: dict[str, int] = {t: 0 for t in strengths}
    for p in past_paths:
        past_matches = parse_matches(p)
        counts = compute_leader_stats(past_matches)
        for team, val in counts.items():
            leader_counts[team] = leader_counts.get(team, 0) + val

    if leader_counts:
        max_count = max(leader_counts.values()) or 1
        for team in strengths:
            factor = leader_counts.get(team, 0) / max_count
            mult = 1.0 + weight * factor
            strengths[team]["attack"] *= mult
            strengths[team]["defense"] /= mult

    return strengths, avg_goals, home_adv


def estimate_poisson_strengths(matches: pd.DataFrame):
    """Fit a Poisson regression model to estimate team strengths."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    played = matches.dropna(subset=["home_score", "away_score"])

    rows: list[dict] = []
    for _, row in played.iterrows():
        rows.append(
            {
                "team": row["home_team"],
                "opponent": row["away_team"],
                "home": 1,
                "goals": row["home_score"],
            }
        )
        rows.append(
            {
                "team": row["away_team"],
                "opponent": row["home_team"],
                "home": 0,
                "goals": row["away_score"],
            }
        )

    df = pd.DataFrame(rows)

    model = smf.glm(
        "goals ~ home + C(team) + C(opponent)",
        data=df,
        family=sm.families.Poisson(),
    ).fit()

    base_mu = float(np.exp(model.params["Intercept"]))
    home_adv = float(np.exp(model.params.get("home", 0.0)))

    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    strengths: dict[str, dict[str, float]] = {}
    for t in teams:
        atk_coef = model.params.get(f"C(team)[T.{t}]", 0.0)
        def_coef = model.params.get(f"C(opponent)[T.{t}]", 0.0)
        strengths[t] = {
            "attack": float(np.exp(atk_coef)),
            "defense": float(np.exp(def_coef)),
        }

    return strengths, base_mu, home_adv


def estimate_negative_binomial_strengths(matches: pd.DataFrame):
    """Fit a Negative Binomial regression model to estimate team strengths.

    The dispersion parameter is estimated from the data and supplied to the
    :class:`statsmodels.families.NegativeBinomial` family when fitting the GLM.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    played = matches.dropna(subset=["home_score", "away_score"])

    rows: list[dict] = []
    for _, row in played.iterrows():
        rows.append({
            "team": row["home_team"],
            "opponent": row["away_team"],
            "home": 1,
            "goals": row["home_score"],
        })
        rows.append({
            "team": row["away_team"],
            "opponent": row["home_team"],
            "home": 0,
            "goals": row["away_score"],
        })

    df = pd.DataFrame(rows)

    dispersion = _estimate_dispersion(matches)

    model = smf.glm(
        "goals ~ home + C(team) + C(opponent)",
        data=df,
        family=sm.families.NegativeBinomial(alpha=dispersion),
    ).fit()

    base_mu = float(np.exp(model.params["Intercept"]))
    home_adv = float(np.exp(model.params.get("home", 0.0)))

    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    strengths: dict[str, dict[str, float]] = {}
    for t in teams:
        atk_coef = model.params.get(f"C(team)[T.{t}]", 0.0)
        def_coef = model.params.get(f"C(opponent)[T.{t}]", 0.0)
        strengths[t] = {
            "attack": float(np.exp(atk_coef)),
            "defense": float(np.exp(def_coef)),
        }

    return strengths, base_mu, home_adv, dispersion


def estimate_skellam_strengths(matches: pd.DataFrame):
    """Estimate team strengths via a simple Skellam regression.

    This implementation fits independent Poisson models for home and away
    goals using :mod:`statsmodels`. The resulting attack and defence factors are
    interpreted under a Skellam framework where goal difference is the
    difference of the two Poisson rates.
    """

    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    played = matches.dropna(subset=["home_score", "away_score"])

    rows: list[dict] = []
    for _, row in played.iterrows():
        rows.append(
            {
                "team": row["home_team"],
                "opponent": row["away_team"],
                "home": 1,
                "goals": row["home_score"],
            }
        )
        rows.append(
            {
                "team": row["away_team"],
                "opponent": row["home_team"],
                "home": 0,
                "goals": row["away_score"],
            }
        )

    df = pd.DataFrame(rows)

    model = smf.glm(
        "goals ~ home + C(team) + C(opponent)",
        data=df,
        family=sm.families.Poisson(),
    ).fit()

    base_mu = float(np.exp(model.params["Intercept"]))
    home_adv = float(np.exp(model.params.get("home", 0.0)))

    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    strengths: dict[str, dict[str, float]] = {}
    for t in teams:
        atk_coef = model.params.get(f"C(team)[T.{t}]", 0.0)
        def_coef = model.params.get(f"C(opponent)[T.{t}]", 0.0)
        strengths[t] = {
            "attack": float(np.exp(atk_coef)),
            "defense": float(np.exp(def_coef)),
        }

    return strengths, base_mu, home_adv


def estimate_elo_strengths(
    matches: pd.DataFrame, K: float = 20.0, home_field_advantage: float = 0.0
):
    """Estimate team strengths using an Elo ratings approach.

    Parameters
    ----------
    matches : pd.DataFrame
        Fixture list including results for played games.
    K : float, default 20.0
        Rating update factor.
    home_field_advantage : float, default 0.0
        Extra rating points given to the home side when computing the expected
        result. This biases Elo updates in favour of the hosts without altering
        final ratings directly.

    Returns
    -------
    dict[str, dict[str, float]]
        Attack and defense multipliers derived from final Elo ratings.
    float
        Average goals per game of played matches.
    float
        Ratio of home to away goals.
    """

    played = matches.dropna(subset=["home_score", "away_score"]).copy()
    played = played.sort_values("date")

    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    ratings = {t: 1500.0 for t in teams}

    for _, row in played.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hs = int(row["home_score"])
        as_ = int(row["away_score"])

        r_home = ratings[home]
        r_away = ratings[away]
        expected_home = 1 / (
            1 + 10 ** ((r_away - (r_home + home_field_advantage)) / 400)
        )
        score_home = 1.0 if hs > as_ else 0.5 if hs == as_ else 0.0

        ratings[home] = r_home + K * (score_home - expected_home)
        ratings[away] = r_away + K * ((1 - score_home) - (1 - expected_home))

    baseline = float(np.mean(list(ratings.values())))
    strengths: dict[str, dict[str, float]] = {}
    for t, r in ratings.items():
        factor = 10 ** ((r - baseline) / 400)
        strengths[t] = {"attack": factor, "defense": 1 / factor}

    total_goals = played["home_score"].sum() + played["away_score"].sum()
    total_games = len(played)
    avg_goals = total_goals / total_games if total_games else 2.5
    home_adv = (
        played["home_score"].sum() / played["away_score"].sum()
        if played["away_score"].sum()
        else 1.0
    )

    return strengths, avg_goals, home_adv


def estimate_dixon_coles_strengths(matches: pd.DataFrame):
    """Estimate team strengths using the Dixon-Coles model."""
    played = matches.dropna(subset=["home_score", "away_score"])

    teams = pd.unique(played[["home_team", "away_team"]].values.ravel())
    team_index = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    def _nll(params: np.ndarray) -> float:
        attack = np.zeros(n)
        defense = np.zeros(n)
        attack[1:] = params[: n - 1]
        defense[1:] = params[n - 1 : 2 * (n - 1)]
        home_param = params[-2]
        rho = params[-1]

        ll = 0.0
        for _, row in played.iterrows():
            i = team_index[row["home_team"]]
            j = team_index[row["away_team"]]
            lam = np.exp(attack[i] + defense[j] + home_param)
            mu = np.exp(attack[j] + defense[i])
            x = int(row["home_score"])
            y = int(row["away_score"])
            tau = 1.0
            if x == 0 and y == 0:
                tau = 1 - lam * mu * rho
            elif x == 0 and y == 1:
                tau = 1 + lam * rho
            elif x == 1 and y == 0:
                tau = 1 + mu * rho
            elif x == 1 and y == 1:
                tau = 1 - rho
            ll += poisson.logpmf(x, lam) + poisson.logpmf(y, mu) + np.log(tau)
        return -float(ll)

    res = minimize(_nll, np.zeros(2 * (n - 1) + 2), method="L-BFGS-B")
    params = res.x

    attack = np.zeros(n)
    defense = np.zeros(n)
    attack[1:] = params[: n - 1]
    defense[1:] = params[n - 1 : 2 * (n - 1)]
    attack -= np.mean(attack)
    defense -= np.mean(defense)

    home_adv = float(np.exp(params[-2]))
    rho = float(params[-1])

    strengths: dict[str, dict[str, float]] = {}
    for t, idx in team_index.items():
        strengths[t] = {
            "attack": float(np.exp(attack[idx])),
            "defense": float(np.exp(defense[idx])),
        }

    total_goals = played["home_score"].sum() + played["away_score"].sum()
    total_games = len(played)
    avg_goals = total_goals / total_games if total_games else 2.5

    return strengths, avg_goals, home_adv, rho


def _dixon_coles_sample(
    lam: float, mu: float, rho: float, rng: np.random.Generator, max_goals: int = 6
) -> tuple[int, int]:
    """Sample a scoreline using the Dixon-Coles adjustment."""
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
    total = probs.sum()
    if total <= 0:
        hs = rng.poisson(lam)
        as_ = rng.poisson(mu)
        return int(hs), int(as_)
    probs = probs / total
    flat = rng.choice((max_goals + 1) ** 2, p=probs.ravel())
    return int(flat // (max_goals + 1)), int(flat % (max_goals + 1))


def get_strengths(
    matches: pd.DataFrame,
    rating_method: str,
    *,
    elo_k: float = 20.0,
    home_field_advantage: float = 0.0,
    leader_history_paths: list[str | Path] | None = None,
    leader_history_weight: float = 0.5,
    smooth: float = 1.0,
) -> tuple[dict[str, dict[str, float]], float, float, float]:
    """Return strength estimates for ``matches`` using ``rating_method``.

    Parameters other than ``matches`` and ``rating_method`` mirror those of the
    :func:`simulate_chances` routine.  The returned tuple contains the attack and
    defence multipliers for each team, the average goals per game, the overall
    home advantage factor and an additional value depending on the chosen
    ``rating_method``:

    ``dixon_coles``
        The Dixon--Coles correlation parameter ``rho``.
    ``neg_binom``
        The estimated dispersion parameter of the Negative Binomial model.
    otherwise
        ``0.0``.
    """

    extra_param = 0.0
    if rating_method == "poisson":
        strengths, avg_goals, home_adv = estimate_poisson_strengths(matches)
    elif rating_method == "neg_binom":
        strengths, avg_goals, home_adv, extra_param = estimate_negative_binomial_strengths(matches)
    elif rating_method == "skellam":
        strengths, avg_goals, home_adv = estimate_skellam_strengths(matches)
    elif rating_method == "historic_ratio":
        strengths, avg_goals, home_adv = estimate_strengths_with_history(matches, smooth=smooth)
    elif rating_method == "elo":
        strengths, avg_goals, home_adv = estimate_elo_strengths(
            matches, K=elo_k, home_field_advantage=home_field_advantage
        )
    elif rating_method == "dixon_coles":
        strengths, avg_goals, home_adv, extra_param = estimate_dixon_coles_strengths(matches)
    elif rating_method == "leader_history":
        paths = leader_history_paths or ["data/Brasileirao2024A.txt"]
        strengths, avg_goals, home_adv = estimate_leader_history_strengths(
            matches, paths, weight=leader_history_weight, smooth=smooth
        )
    else:
        strengths, avg_goals, home_adv = _estimate_strengths(matches, smooth=smooth)

    return strengths, avg_goals, home_adv, extra_param


def _simulate_table(
    played_df: pd.DataFrame,
    remaining: pd.DataFrame,
    strengths: dict[str, dict[str, float]],
    avg_goals: float,
    home_adv: float,
    team_home_advantages: dict[str, float],
    rating_method: str,
    extra_param: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return a simulated table based on remaining fixtures."""

    sims: list[dict] = []
    for _, row in remaining.iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        factor = team_home_advantages.get(ht, 1.0)
        mu_home = (
            avg_goals
            * strengths[ht]["attack"]
            * strengths[at]["defense"]
            * home_adv
            * factor
        )
        mu_away = avg_goals * strengths[at]["attack"] * strengths[ht]["defense"]
        if rating_method == "dixon_coles":
            hs, as_ = _dixon_coles_sample(mu_home, mu_away, extra_param, rng)
        elif rating_method == "neg_binom" and extra_param > 0:
            r = 1.0 / extra_param
            p_home = r / (r + mu_home)
            p_away = r / (r + mu_away)
            hs = rng.negative_binomial(r, p_home)
            as_ = rng.negative_binomial(r, p_away)
        else:
            hs = rng.poisson(mu_home)
            as_ = rng.poisson(mu_away)
        sims.append(
            {
                "date": row["date"],
                "home_team": ht,
                "away_team": at,
                "home_score": hs,
                "away_score": as_,
            }
        )
    all_matches = pd.concat([played_df, pd.DataFrame(sims)], ignore_index=True)
    return league_table(all_matches)


def simulate_chances(
    matches: pd.DataFrame,
    iterations: int = 1000,
    rating_method: str = "ratio",
    rng: np.random.Generator | None = None,
    elo_k: float = 20.0,
    home_field_advantage: float = 0.0,
    team_home_advantages: dict[str, float] | None = None,
    leader_history_paths: list[str | Path] | None = None,
    leader_history_weight: float = 0.5,
    smooth: float = 1.0,
) -> dict[str, float]:
    """Simulate remaining fixtures and return title probabilities.

    Parameters
    ----------
    matches : pd.DataFrame
        DataFrame containing all fixtures. Played games must have scores.
    iterations : int, default 1000
        Number of simulation runs.
    rating_method : str, default "ratio"
        Method used to estimate team strengths.
    rng : np.random.Generator | None, optional
        Random number generator to use. A new generator is created when ``None``.
    elo_k : float, default 20.0
        K factor when ``rating_method`` is ``"elo"``.
    home_field_advantage : float, default 0.0
        Rating bonus given to the home team when calculating Elo win
        probabilities. Only used when ``rating_method`` is ``"elo"``.
    team_home_advantages : dict[str, float] | None, optional
        Multiplicative home advantage factor for each team. When ``None``,
        factors are estimated from played matches.
    leader_history_paths : list[str | Path] | None, optional
        Season files used when ``rating_method`` is ``"leader_history"``.
    leader_history_weight : float, optional
        Influence of historic leader counts when ``rating_method`` is
        ``"leader_history"``.
    smooth : float, default 1.0
        Smoothing constant added to goals scored and conceded when estimating
        team strengths under the ``"ratio"`` methods.
    """
    if rng is None:
        rng = np.random.default_rng()

    if team_home_advantages is None:
        team_home_advantages = _estimate_team_home_advantages(matches)
    else:
        merged = _estimate_team_home_advantages(matches)
        merged.update(team_home_advantages)
        team_home_advantages = merged

    strengths, avg_goals, home_adv, extra_param = get_strengths(
        matches,
        rating_method,
        elo_k=elo_k,
        home_field_advantage=home_field_advantage,
        leader_history_paths=leader_history_paths,
        leader_history_weight=leader_history_weight,
        smooth=smooth,
    )
    teams = pd.unique(matches[['home_team', 'away_team']].values.ravel())
    champs = {t: 0 for t in teams}

    played_df = matches.dropna(subset=['home_score', 'away_score'])
    remaining = matches[matches['home_score'].isna() | matches['away_score'].isna()]

    for _ in range(iterations):
        table = _simulate_table(
            played_df,
            remaining,
            strengths,
            avg_goals,
            home_adv,
            team_home_advantages,
            rating_method,
            extra_param,
            rng,
        )
        champs[table.iloc[0]["team"]] += 1

    for t in champs:
        champs[t] = champs[t] / iterations
    return champs


def simulate_relegation_chances(
    matches: pd.DataFrame,
    iterations: int = 1000,
    rating_method: str = "ratio",
    rng: np.random.Generator | None = None,
    elo_k: float = 20.0,
    home_field_advantage: float = 0.0,
    team_home_advantages: dict[str, float] | None = None,
    leader_history_paths: list[str | Path] | None = None,
    leader_history_weight: float = 0.5,
    smooth: float = 1.0,
) -> dict[str, float]:
    """Simulate remaining fixtures and return relegation probabilities.

    The parameters mirror :func:`simulate_chances`. The returned values map
    each team to the probability of finishing in the bottom four positions.
    
    Parameters are the same as for :func:`simulate_chances`. ``smooth`` controls
    the constant added to goals scored and conceded when calculating attack and
    defence ratings under the ``"ratio"`` methods. ``home_field_advantage`` only
    affects the Elo method.
    """

    if rng is None:
        rng = np.random.default_rng()

    if team_home_advantages is None:
        team_home_advantages = _estimate_team_home_advantages(matches)
    else:
        merged = _estimate_team_home_advantages(matches)
        merged.update(team_home_advantages)
        team_home_advantages = merged

    strengths, avg_goals, home_adv, extra_param = get_strengths(
        matches,
        rating_method,
        elo_k=elo_k,
        home_field_advantage=home_field_advantage,
        leader_history_paths=leader_history_paths,
        leader_history_weight=leader_history_weight,
        smooth=smooth,
    )

    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    relegated = {t: 0 for t in teams}

    played_df = matches.dropna(subset=["home_score", "away_score"])
    remaining = matches[matches["home_score"].isna() | matches["away_score"].isna()]

    for _ in range(iterations):
        table = _simulate_table(
            played_df,
            remaining,
            strengths,
            avg_goals,
            home_adv,
            team_home_advantages,
            rating_method,
            extra_param,
            rng,
        )
        for team in table.tail(4)["team"]:
            relegated[team] += 1

    for t in relegated:
        relegated[t] = relegated[t] / iterations
    return relegated


def simulate_final_table(
    matches: pd.DataFrame,
    iterations: int = 1000,
    rating_method: str = "ratio",
    rng: np.random.Generator | None = None,
    elo_k: float = 20.0,
    home_field_advantage: float = 0.0,
    team_home_advantages: dict[str, float] | None = None,
    leader_history_paths: list[str | Path] | None = None,
    leader_history_weight: float = 0.5,
    smooth: float = 1.0,
) -> pd.DataFrame:
    """Project final league positions and points for each team.

    The parameters mirror :func:`simulate_chances`. The returned ``DataFrame``
    contains the average finishing position and point total of each club,
    sorted by expected position.
    ``smooth`` has the same meaning as in :func:`simulate_chances`.
    ``home_field_advantage`` only affects the Elo method.
    """

    if rng is None:
        rng = np.random.default_rng()

    if team_home_advantages is None:
        team_home_advantages = _estimate_team_home_advantages(matches)
    else:
        merged = _estimate_team_home_advantages(matches)
        merged.update(team_home_advantages)
        team_home_advantages = merged

    strengths, avg_goals, home_adv, extra_param = get_strengths(
        matches,
        rating_method,
        elo_k=elo_k,
        home_field_advantage=home_field_advantage,
        leader_history_paths=leader_history_paths,
        leader_history_weight=leader_history_weight,
        smooth=smooth,
    )

    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    pos_totals = {t: 0.0 for t in teams}
    points_totals = {t: 0.0 for t in teams}

    played_df = matches.dropna(subset=["home_score", "away_score"])
    remaining = matches[matches["home_score"].isna() | matches["away_score"].isna()]

    for _ in range(iterations):
        table = _simulate_table(
            played_df,
            remaining,
            strengths,
            avg_goals,
            home_adv,
            team_home_advantages,
            rating_method,
            extra_param,
            rng,
        )
        for idx, row in table.iterrows():
            pos_totals[row["team"]] += idx + 1
            points_totals[row["team"]] += row["points"]

    results = []
    for team in teams:
        results.append(
            {
                "team": team,
                "position": pos_totals[team] / iterations,
                "points": points_totals[team] / iterations,
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values("position").reset_index(drop=True)
    return df


def summary_table(
    matches: pd.DataFrame,
    iterations: int = 1000,
    rating_method: str = "ratio",
    rng: np.random.Generator | None = None,
    elo_k: float = 20.0,
    home_field_advantage: float = 0.0,
    team_home_advantages: dict[str, float] | None = None,
    leader_history_paths: list[str | Path] | None = None,
    leader_history_weight: float = 0.5,
    smooth: float = 1.0,
) -> pd.DataFrame:
    """Return combined projections for each team.

    The returned ``DataFrame`` contains one row per club with the expected
    final rank, projected point total rounded to an integer, title chance and
    relegation probability. The table is sorted by projected position.
    The ``smooth`` parameter is forwarded to the underlying simulation
    functions.
    ``home_field_advantage`` is forwarded to the Elo-based rating routine and
    has no effect for other methods.
    """

    chances = simulate_chances(
        matches,
        iterations=iterations,
        rating_method=rating_method,
        rng=rng,
        elo_k=elo_k,
        home_field_advantage=home_field_advantage,
        team_home_advantages=team_home_advantages,
        leader_history_paths=leader_history_paths,
        leader_history_weight=leader_history_weight,
        smooth=smooth,
    )
    relegation = simulate_relegation_chances(
        matches,
        iterations=iterations,
        rating_method=rating_method,
        rng=rng,
        elo_k=elo_k,
        home_field_advantage=home_field_advantage,
        team_home_advantages=team_home_advantages,
        leader_history_paths=leader_history_paths,
        leader_history_weight=leader_history_weight,
        smooth=smooth,
    )
    table = simulate_final_table(
        matches,
        iterations=iterations,
        rating_method=rating_method,
        rng=rng,
        elo_k=elo_k,
        home_field_advantage=home_field_advantage,
        team_home_advantages=team_home_advantages,
        leader_history_paths=leader_history_paths,
        leader_history_weight=leader_history_weight,
        smooth=smooth,
    )

    table = table.sort_values("position").reset_index(drop=True)
    table["position"] = range(1, len(table) + 1)
    table["points"] = table["points"].round().astype(int)
    table["title"] = table["team"].map(chances)
    table["relegation"] = table["team"].map(relegation)
    return table[["position", "team", "points", "title", "relegation"]]
