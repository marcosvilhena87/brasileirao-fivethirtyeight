from __future__ import annotations

import numpy as np
import pandas as pd
import re
from pathlib import Path
from collections.abc import Sequence
from scipy.optimize import minimize
from scipy.stats import poisson

# Default SPI coefficients derived from the 2023-2024 seasons
SPI_DEFAULT_INTERCEPT = -0.180149
SPI_DEFAULT_SLOPE = 0.228628


def _parse_date(date_str: str) -> pd.Timestamp:
    """Parse dates from multiple formats."""
    parts = date_str.split("/")
    year = parts[-1]
    if len(year) == 4:
        return pd.to_datetime(date_str, format="%d/%m/%Y")
    return pd.to_datetime(date_str, format="%m/%d/%y")


SCORE_PATTERN = re.compile(
    r"(\d+/\d+/\d+)\s+(.+?)\s+(\d+)-(\d+)\s+(.+?)\s*(?:\(ID:.*)?$"
)
NOSCORE_PATTERN = re.compile(
    r"(\d+/\d+/\d+)\s+(.+?)\s{2,}(.+?)\s*(?:\(ID:.*)?$"
)


def parse_matches(path: str | Path) -> pd.DataFrame:
    """Parse the fixture text file into a DataFrame.

    The data files are encoded in UTF-8 and may contain a byte order mark.
    Using ``utf-8-sig`` ensures both plain UTF-8 and BOM-prefixed files are
    accepted.
    """
    rows: list[dict] = []
    in_games = False
    with open(path, "r", encoding="utf-8-sig") as f:
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


def _head_to_head_points(
    matches: pd.DataFrame, teams: list[str]
) -> dict[str, int]:
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
        t: {
            "team": t,
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "gf": 0,
            "ga": 0,
        }
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

    # Maintain cumulative standings directly in a DataFrame so we don't need to
    # rebuild one from dictionaries on every loop iteration.
    stats_df = pd.DataFrame(
        0,
        index=pd.Index(teams, name="team"),
        columns=[
            "played",
            "wins",
            "draws",
            "losses",
            "gf",
            "ga",
            "gd",
            "points",
        ],
    )

    played_rows: list[dict] = []

    for _, row in matches.sort_values("date").iterrows():
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue

        ht = row["home_team"]
        at = row["away_team"]
        hs = int(row["home_score"])
        as_ = int(row["away_score"])

        played_rows.append(row.to_dict())

        stats_df.loc[ht, ["played", "gf", "ga", "gd"]] += [1, hs, as_, hs - as_]
        stats_df.loc[at, ["played", "gf", "ga", "gd"]] += [1, as_, hs, as_ - hs]

        if hs > as_:
            stats_df.loc[ht, ["wins", "points"]] += [1, 3]
            stats_df.loc[at, "losses"] += 1
        elif hs < as_:
            stats_df.loc[at, ["wins", "points"]] += [1, 3]
            stats_df.loc[ht, "losses"] += 1
        else:
            stats_df.loc[[ht, at], "draws"] += 1
            stats_df.loc[[ht, at], "points"] += 1

        current = stats_df.copy()
        current["head_to_head"] = 0
        played_df = pd.DataFrame(played_rows)

        for _, group in current.groupby(["points", "wins", "gd", "gf"]):
            if len(group) <= 1:
                continue
            teams_tied = group.index.tolist()
            h2h = _head_to_head_points(played_df, teams_tied)
            for t, val in h2h.items():
                current.loc[t, "head_to_head"] = val

        current = (
            current.reset_index()
            .sort_values(
                ["points", "wins", "gd", "gf", "head_to_head", "team"],
                ascending=[False, False, False, False, False, True],
            )
            .reset_index(drop=True)
        )

        leader_counts[current.iloc[0]["team"]] += 1

    return leader_counts


def _estimate_strengths(
    matches: pd.DataFrame,
    smooth: float = 1.0,
    decay_rate: float | None = None,
):
    """Return basic ratio strengths from ``matches``.

    When ``decay_rate`` is provided each match is weighted by
    ``exp(-decay_rate * days_since)`` where ``days_since`` is the number of days
    since the most recent fixture.  A rate of ``0`` (the default) applies equal
    weight to all games.
    """

    played = matches.dropna(subset=["home_score", "away_score"]).copy()

    if decay_rate and not played.empty:
        latest = played["date"].max()
        days = (latest - played["date"]).dt.days
        played["weight"] = np.exp(-decay_rate * days)
    else:
        played["weight"] = 1.0

    total_goals = (played["home_score"] * played["weight"]).sum() + (
        played["away_score"] * played["weight"]
    ).sum()
    total_games = played["weight"].sum()
    avg_goals = total_goals / total_games if total_games else 2.5
    away_sum = (played["away_score"] * played["weight"]).sum()
    home_sum = (played["home_score"] * played["weight"]).sum()
    home_adv = home_sum / away_sum if away_sum else 1.0

    teams = pd.unique(matches[['home_team', 'away_team']].values.ravel())
    strengths = {}
    for team in teams:
        home_mask = played.home_team == team
        away_mask = played.away_team == team

        gf = (
            (
                played.loc[home_mask, "home_score"]
                * played.loc[home_mask, "weight"]
            ).sum()
            + (
                played.loc[away_mask, "away_score"]
                * played.loc[away_mask, "weight"]
            ).sum()
        )
        ga = (
            (
                played.loc[home_mask, "away_score"]
                * played.loc[home_mask, "weight"]
            ).sum()
            + (
                played.loc[away_mask, "home_score"]
                * played.loc[away_mask, "weight"]
            ).sum()
        )
        gp = played.loc[home_mask | away_mask, "weight"].sum()
        if gp == 0:
            attack = defense = 1.0
        else:
            attack = ((gf + smooth) / (gp + smooth)) / avg_goals
            defense = ((ga + smooth) / (gp + smooth)) / avg_goals
        strengths[team] = {'attack': attack, 'defense': defense}
    return strengths, avg_goals, home_adv


def _estimate_team_home_advantages(
    matches: pd.DataFrame, factors: dict[str, float] | None = None
) -> dict[str, float]:
    """Return relative home advantage factors for each team.

    When ``factors`` is supplied the dictionary is updated in-place so the
    caller can maintain values across seasons.
    """
    played = matches.dropna(subset=["home_score", "away_score"])

    if played.empty:
        teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
        return {t: 1.0 for t in teams}
    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())

    total_home = played["home_score"].sum()
    total_away = played["away_score"].sum()
    baseline = total_home / total_away if total_away else 1.0

    results = {}
    for t in teams:
        home_games = played[played.home_team == t]
        away_games = played[played.away_team == t]
        if not len(home_games) or not len(away_games):
            results[t] = 1.0
            continue
        home_gpg = home_games["home_score"].mean()
        away_gpg = away_games["away_score"].mean()
        if away_gpg == 0 or np.isnan(home_gpg) or np.isnan(away_gpg):
            results[t] = 1.0
        else:
            results[t] = float((home_gpg / away_gpg) / baseline)

    if factors is not None:
        for t in results:
            factors[t] = results[t]
        for t in factors:
            factors.setdefault(t, 1.0)
        return factors
    return results


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


def load_market_values(
    path: str | Path = "data/Brasileirao2025A.csv",
) -> dict[str, float]:
    """Return team market values from ``path``.

    The CSV file is semicolon-delimited and may start with a UTF-8 BOM.
    Values are expected in millions of Euros.
    """
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    team_col = df.columns[0]
    value_col = df.columns[1]
    return df.set_index(team_col)[value_col].astype(float).to_dict()


def estimate_market_strengths(
    matches: pd.DataFrame,
    market_path: str | Path = "data/Brasileirao2025A.csv",
    smooth: float = 1.0,
    decay_rate: float | None = None,
) -> tuple[dict[str, dict[str, float]], float, float]:
    """Estimate strengths adjusted by team market values.

    Returns the attack and defence strengths for each team, the overall
    average goals per game and the baseline home advantage factor.
    """
    strengths, avg_goals, home_adv = _estimate_strengths(
        matches, smooth=smooth, decay_rate=decay_rate
    )

    market_values = load_market_values(market_path)
    if market_values:
        avg_value = float(np.mean(list(market_values.values()))) or 1.0
        for team, s in strengths.items():
            val = market_values.get(team)
            if val is None:
                continue
            factor = val / avg_value
            s["attack"] *= factor
            s["defense"] /= factor

    return strengths, avg_goals, home_adv


def update_spi_ratings(
    strengths: dict[str, dict[str, float]],
    home_team: str,
    away_team: str,
    home_goals: float,
    away_goals: float,
    avg_goals: float,
    home_adv: float,
    *,
    K: float = 0.05,
) -> tuple[float, float]:
    """Update SPI-style ratings after a single match.

    Parameters
    ----------
    strengths : dict[str, dict[str, float]]
        Current attack and defence multipliers for each team.
    home_team, away_team : str
        Competing teams.
    home_goals, away_goals : float
        Observed goals scored by the home and away side.
    avg_goals : float
        Baseline average goals per game.
    home_adv : float
        League-wide home advantage factor.
    K : float, default 0.05
        Update magnitude. Higher values produce larger rating changes.

    Returns
    -------
    tuple[float, float]
        Expected goals for the home and away team prior to the update.
    """

    mu_home = (
        avg_goals
        * strengths[home_team]["attack"]
        * strengths[away_team]["defense"]
        * home_adv
    )
    mu_away = (
        avg_goals * strengths[away_team]["attack"] * strengths[home_team]["defense"]
    )

    scale_home = np.exp(K * (home_goals - mu_home) / avg_goals)
    scale_away = np.exp(K * (away_goals - mu_away) / avg_goals)

    strengths[home_team]["attack"] *= scale_home
    strengths[away_team]["defense"] *= scale_home
    strengths[away_team]["attack"] *= scale_away
    strengths[home_team]["defense"] *= scale_away

    # keep ratings positive
    for t in (home_team, away_team):
        strengths[t]["attack"] = max(strengths[t]["attack"], 0.01)
        strengths[t]["defense"] = max(strengths[t]["defense"], 0.01)

    return mu_home, mu_away


def estimate_strengths_with_history(
    current_matches: pd.DataFrame | None = None,
    past_path: str | Path = "data/Brasileirao2024A.txt",
    past_weight: float = 0.5,
    smooth: float = 1.0,
    decay_rate: float | None = None,
) -> tuple[dict[str, dict[str, float]], float, float]:
    """Estimate strengths using current season matches and weighted history."""
    if current_matches is None:
        current_matches = parse_matches("data/Brasileirao2025A.txt")
    past_matches = parse_matches(past_path)
    if 0 < past_weight < 1:
        past_matches = (
            past_matches.sample(frac=past_weight, random_state=0)
            .reset_index(drop=True)
        )
    combined = pd.concat([current_matches, past_matches], ignore_index=True)
    return _estimate_strengths(combined, smooth=smooth, decay_rate=decay_rate)


def estimate_leader_history_strengths(
    current_matches: pd.DataFrame | None = None,
    past_paths: list[str | Path] | str | Path = "data/Brasileirao2024A.txt",
    weight: float = 0.5,
    smooth: float = 1.0,
    decay_rate: float | None = None,
) -> tuple[dict[str, dict[str, float]], float, float]:
    """Estimate strengths influenced by historical league leaders."""
    if current_matches is None:
        current_matches = parse_matches("data/Brasileirao2025A.txt")
    strengths, avg_goals, home_adv = _estimate_strengths(
        current_matches, smooth=smooth, decay_rate=decay_rate
    )

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

    if played.empty:
        return _estimate_strengths(matches)

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

    if played.empty:
        strengths, avg_goals, home_adv = _estimate_strengths(matches)
        dispersion = _estimate_dispersion(matches)
        return strengths, avg_goals, home_adv, dispersion

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

    if played.empty:
        return _estimate_strengths(matches)

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
        defense[1:] = params[n - 1:2 * (n - 1)]
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
    defense[1:] = params[n - 1:2 * (n - 1)]
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


def estimate_spi_strengths(
    matches: pd.DataFrame,
    market_path: str | Path = "data/Brasileirao2025A.csv",
    smooth: float = 1.0,
    decay_rate: float | None = None,
    logistic_decay: float | None = 0.007,
    match_weights: pd.Series | None = None,
    *,
    K: float = 0.05,
) -> tuple[dict[str, dict[str, float]], float, float, float, float]:
    """Estimate strengths with a logistic regression on match outcomes.

    The function first computes basic attack and defence factors using
    :func:`estimate_market_strengths`. It then derives the expected goal difference
    for each played match and fits a logistic regression of the home-win
    indicator on that value.  The fitted intercept and slope are returned and
    later used to transform expected goal differences into win/draw/loss
    probabilities when simulating matches.  The function returns five values:
    the strengths dictionary, average goals per game, baseline home advantage,
    intercept and slope.  The ``market_path`` parameter can be used to supply a
    custom CSV file with team market values. ``logistic_decay`` optionally
    applies exponential weighting to recent fixtures when fitting the logistic
    regression: a match ``d`` days before the latest carries weight
    ``exp(-logistic_decay * d)``. If omitted the decay defaults to ``0.007``.
    ``match_weights`` may directly provide a
    sequence of weights for the played matches when fitting the regression. If
    both ``logistic_decay`` and ``match_weights`` are given the resulting
    weights are multiplied.
    ``K`` controls the magnitude of rating updates after each played match.
    """

    strengths, avg_goals, home_adv = estimate_market_strengths(
        matches, market_path=market_path, smooth=smooth, decay_rate=decay_rate
    )

    played = matches.dropna(subset=["home_score", "away_score"]).sort_values("date")
    if played.empty:
        from .spi_coeffs import compute_spi_coeffs

        intercept, slope = compute_spi_coeffs()
        return (
            strengths,
            avg_goals,
            home_adv,
            intercept,
            slope,
        )

    diffs: list[float] = []
    outcomes: list[int] = []
    logistic_weights = None
    if logistic_decay is not None and not played.empty:
        latest = played["date"].max()
        days = (latest - played["date"]).dt.days
        logistic_weights = np.exp(-logistic_decay * days)
    for _, row in played.iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        mu_home, mu_away = update_spi_ratings(
            strengths,
            ht,
            at,
            float(row["home_score"]),
            float(row["away_score"]),
            avg_goals,
            home_adv,
            K=K,
        )
        diffs.append(mu_home - mu_away)
        outcomes.append(int(row["home_score"] > row["away_score"]))

    import statsmodels.api as sm

    exog = sm.add_constant(pd.Series(diffs, name="diff"))
    weights = None
    if logistic_weights is not None and match_weights is not None:
        weights = pd.Series(
            logistic_weights.values * match_weights.loc[played.index].values,
            index=exog.index,
        )
    elif logistic_weights is not None:
        weights = pd.Series(logistic_weights.values, index=exog.index)
    elif match_weights is not None:
        weights = pd.Series(match_weights.loc[played.index], index=exog.index)
    if weights is not None:
        model = sm.GLM(
            outcomes, exog, family=sm.families.Binomial(), freq_weights=weights
        ).fit(disp=False)
    else:
        model = sm.Logit(outcomes, exog).fit(disp=False)

    intercept = float(model.params["const"])
    slope = float(model.params["diff"])

    return strengths, avg_goals, home_adv, intercept, slope


def initial_spi_strengths(
    past_path: str | Path | Sequence[str | Path] = "data/Brasileirao2024A.txt",
    weight: float = 2 / 3,
    *,
    market_path: str | Path = "data/Brasileirao2024A.csv",
    smooth: float = 1.0,
    decay_rate: float | None = None,
    seasons: list[str] | None = None,
) -> tuple[dict[str, dict[str, float]], float, float, float, float]:
    """Return starting SPI strengths for a new season.

    ``past_path`` may be a single results file, year string or a sequence of
    such values. Each season is parsed with :func:`parse_matches` and basic
    strengths are computed using :func:`_estimate_strengths`. The seasons are
    blended with an exponential weight ``exp(-decay_rate * age)`` where
    ``age`` counts seasons back from the most recent. The weighted strengths are
    finally shrunk toward the league average using ``weight`` similar to
    FiveThirtyEight's approach:

    ``current = previous * weight + league_mean * (1 - weight)``.

    When ``seasons`` is provided the logistic regression coefficients are
    recalculated across those years using :func:`compute_spi_coeffs` and replace
    the values obtained from the past season(s).
    """

    if isinstance(past_path, Sequence) and not isinstance(past_path, (str, Path)):
        past_paths = list(past_path)
    else:
        past_paths = [past_path]

    resolved: list[tuple[Path, int]] = []
    for p in past_paths:
        if isinstance(p, Path):
            text = p.name
        else:
            text = str(p)
        if text.isdigit():
            year = int(text)
            path = Path(f"data/Brasileirao{year}A.txt")
        else:
            path = Path(text)
            m = re.search(r"(20\d{2})", path.stem)
            year = int(m.group(1)) if m else 0
        resolved.append((path, year))

    teams: set[str] = set()
    season_data: list[tuple[dict[str, dict[str, float]], float, float, float]] = []
    years: list[int] = []
    for path, year in resolved:
        matches = parse_matches(path)
        s, avg, ha = _estimate_strengths(matches, smooth=smooth, decay_rate=decay_rate)
        season_data.append((s, avg, ha, year))
        teams.update(s.keys())
        years.append(year)

    if years:
        max_year = max(years)
    else:
        max_year = 0

    weights = [
        float(np.exp(-decay_rate * (max_year - y))) if decay_rate else 1.0
        for _, _, _, y in season_data
    ]
    tot_w = sum(weights) or 1.0

    strengths: dict[str, dict[str, float]] = {
        t: {"attack": 0.0, "defense": 0.0} for t in teams
    }
    avg_goals = 0.0
    home_adv = 0.0
    for (s, avg, ha, _), w in zip(season_data, weights):
        avg_goals += w * avg
        home_adv += w * ha
        for t in teams:
            cur = s.get(t, {"attack": 1.0, "defense": 1.0})
            strengths[t]["attack"] += w * cur["attack"]
            strengths[t]["defense"] += w * cur["defense"]

    avg_goals /= tot_w
    home_adv /= tot_w
    for s in strengths.values():
        s["attack"] /= tot_w
        s["defense"] /= tot_w

    avg_attack = float(np.mean([s["attack"] for s in strengths.values()]))
    avg_defense = float(np.mean([s["defense"] for s in strengths.values()]))

    for s in strengths.values():
        s["attack"] = s["attack"] * weight + avg_attack * (1 - weight)
        s["defense"] = s["defense"] * weight + avg_defense * (1 - weight)

    if seasons is not None:
        from .spi_coeffs import compute_spi_coeffs

        intercept, slope = compute_spi_coeffs(
            seasons=seasons,
            market_path=market_path,
            smooth=smooth,
            decay_rate=decay_rate or 0.0,
        )
    else:
        # derive coefficients from the combined past seasons
        frames: list[pd.DataFrame] = []
        for (path, _year), w in zip(resolved, weights):
            df = parse_matches(path)
            df["weight"] = w
            frames.append(df)
        past_matches = pd.concat(frames, ignore_index=True)
        _, _, _, intercept, slope = estimate_spi_strengths(
            past_matches,
            market_path=market_path,
            smooth=smooth,
            decay_rate=decay_rate,
            match_weights=past_matches.pop("weight"),
        )

    return strengths, avg_goals, home_adv, intercept, slope


def initial_ratio_strengths(
    past_path: str | Path = "data/Brasileirao2024A.txt",
    weight: float = 2 / 3,
    *,
    smooth: float = 1.0,
    decay_rate: float | None = None,
) -> tuple[dict[str, dict[str, float]], float, float]:
    """Return starting ratio strengths for a new season.

    Ratings are derived from ``past_path`` and shrunk toward the league mean
    using ``weight`` similar to :func:`initial_spi_strengths`.
    """

    past_matches = parse_matches(past_path)
    strengths, avg_goals, home_adv = _estimate_strengths(
        past_matches, smooth=smooth, decay_rate=decay_rate
    )

    avg_attack = float(np.mean([s["attack"] for s in strengths.values()]))
    avg_defense = float(np.mean([s["defense"] for s in strengths.values()]))

    for s in strengths.values():
        s["attack"] = s["attack"] * weight + avg_attack * (1 - weight)
        s["defense"] = s["defense"] * weight + avg_defense * (1 - weight)

    return strengths, avg_goals, home_adv


def initial_points_strengths(
    past_path: str | Path = "data/Brasileirao2024A.txt",
    weight: float = 2 / 3,
    *,
    decay_rate: float | None = None,
) -> tuple[dict[str, dict[str, float]], float, float]:
    """Return starting strengths based on past season points.

    The final table of ``past_path`` is calculated via :func:`league_table`.
    Each club's point total is shrunk toward the league average using
    ``current = past_points * weight + mean_points * (1 - weight)``.  The
    resulting ratio to the mean determines the attack and defence factors.
    ``avg_goals`` and ``home_adv`` from the past season are returned as well.
    """

    past_matches = parse_matches(past_path)
    table = league_table(past_matches)
    _, avg_goals, home_adv = _estimate_strengths(
        past_matches, decay_rate=decay_rate
    )

    points = table.set_index("team")["points"].astype(float).to_dict()
    mean_points = float(np.mean(list(points.values()))) or 1.0

    strengths: dict[str, dict[str, float]] = {}
    for team, pts in points.items():
        current = pts * weight + mean_points * (1 - weight)
        ratio = current / mean_points if mean_points else 1.0
        strengths[team] = {"attack": ratio, "defense": 1.0 / ratio}

    return strengths, avg_goals, home_adv


def initial_points_market_strengths(
    past_path: str | Path = "data/Brasileirao2024A.txt",
    market_path: str | Path = "data/Brasileirao2024A.csv",
    *,
    points_weight: float = 2 / 3,
    market_weight: float = 1 / 3,
    decay_rate: float | None = None,
) -> tuple[dict[str, dict[str, float]], float, float]:
    """Return starting strengths from points and market values.

    Each team's rating is a weighted mix of its final points in ``past_path`` and
    its market value from ``market_path``. The weights default to two thirds for
    the point ratio and one third for the market value ratio following
    FiveThirtyEight's approach.
    """

    past_matches = parse_matches(past_path)
    table = league_table(past_matches)
    _, avg_goals, home_adv = _estimate_strengths(
        past_matches, decay_rate=decay_rate
    )

    points = table.set_index("team")["points"].astype(float).to_dict()
    mean_points = float(np.mean(list(points.values()))) or 1.0

    market_values = load_market_values(market_path)
    mean_market = float(np.mean(list(market_values.values()))) or 1.0

    teams = set(points) | set(market_values)
    strengths: dict[str, dict[str, float]] = {}
    for team in teams:
        p_ratio = (
            points.get(team, mean_points) / mean_points
            if mean_points
            else 1.0
        )
        m_ratio = (
            market_values.get(team, mean_market) / mean_market
            if mean_market
            else 1.0
        )
        ratio = points_weight * p_ratio + market_weight * m_ratio
        strengths[team] = {"attack": ratio, "defense": 1.0 / ratio}

    return strengths, avg_goals, home_adv


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


def _spi_probs(diff: float, coeffs: tuple[float, float]) -> tuple[float, float, float]:
    """Return win/draw/loss probabilities for a goal difference."""
    intercept, slope = coeffs
    home = 1.0 / (1.0 + np.exp(-(intercept + slope * diff)))
    away = 1.0 / (1.0 + np.exp(-(intercept - slope * diff)))
    draw = max(0.0, 1.0 - home - away)
    total = home + draw + away
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return home / total, draw / total, away / total


def get_strengths(
    matches: pd.DataFrame,
    rating_method: str,
    *,
    elo_k: float = 20.0,
    home_field_advantage: float = 0.0,
    leader_history_paths: list[str | Path] | None = None,
    leader_history_weight: float = 0.5,
    smooth: float = 1.0,
    market_path: str | Path = "data/Brasileirao2025A.csv",
    decay_rate: float | None = None,
    logistic_decay: float | None = 0.007,
    seasons: list[str] | None = None,
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

    ``market_path`` is only used when ``rating_method`` is ``"spi"`` or
    ``"initial_spi"`` and points to the CSV file with team market values.
    ``seasons`` may provide a list of past years for recalculating the SPI
    coefficients when ``rating_method`` is ``"spi"`` or ``"initial_spi"``.
    ``logistic_decay`` weighs recent results more heavily in the SPI logistic
    regression using ``exp(-logistic_decay * days_since_latest)``. When omitted
    the decay defaults to ``0.007``.
    """

    extra_param = 0.0
    if rating_method == "poisson":
        strengths, avg_goals, home_adv = estimate_poisson_strengths(matches)
    elif rating_method == "neg_binom":
        (
            strengths,
            avg_goals,
            home_adv,
            extra_param,
        ) = estimate_negative_binomial_strengths(matches)
    elif rating_method == "skellam":
        strengths, avg_goals, home_adv = estimate_skellam_strengths(matches)
    elif rating_method == "historic_ratio":
        strengths, avg_goals, home_adv = estimate_strengths_with_history(
            matches, smooth=smooth, decay_rate=decay_rate
        )
    elif rating_method == "elo":
        strengths, avg_goals, home_adv = estimate_elo_strengths(
            matches, K=elo_k, home_field_advantage=home_field_advantage
        )
    elif rating_method == "dixon_coles":
        (
            strengths,
            avg_goals,
            home_adv,
            extra_param,
        ) = estimate_dixon_coles_strengths(matches)
    elif rating_method == "spi":
        strengths, avg_goals, home_adv, intercept, slope = estimate_spi_strengths(
            matches,
            market_path=market_path,
            smooth=smooth,
            decay_rate=decay_rate,
            logistic_decay=logistic_decay,
        )
        if seasons is not None:
            from .spi_coeffs import compute_spi_coeffs

            intercept, slope = compute_spi_coeffs(
                seasons=seasons,
                market_path=market_path,
                smooth=smooth,
                decay_rate=decay_rate or 0.0,
            )
        extra_param = (intercept, slope)
    elif rating_method == "initial_spi":
        (
            strengths,
            avg_goals,
            home_adv,
            intercept,
            slope,
        ) = initial_spi_strengths(
            market_path=market_path,
            smooth=smooth,
            decay_rate=decay_rate,
            seasons=seasons,
        )
        extra_param = (intercept, slope)
    elif rating_method == "initial_ratio":
        strengths, avg_goals, home_adv = initial_ratio_strengths(
            smooth=smooth, decay_rate=decay_rate
        )
    elif rating_method == "initial_points":
        strengths, avg_goals, home_adv = initial_points_strengths(decay_rate=decay_rate)
    elif rating_method == "initial_points_market":
        strengths, avg_goals, home_adv = initial_points_market_strengths(
            market_path=market_path, decay_rate=decay_rate
        )
    elif rating_method == "leader_history":
        paths = leader_history_paths or ["data/Brasileirao2024A.txt"]
        strengths, avg_goals, home_adv = estimate_leader_history_strengths(
            matches,
            paths,
            weight=leader_history_weight,
            smooth=smooth,
            decay_rate=decay_rate,
        )
    else:
        strengths, avg_goals, home_adv = _estimate_strengths(
            matches, smooth=smooth, decay_rate=decay_rate
        )

    return strengths, avg_goals, home_adv, extra_param


def _simulate_table(
    played_df: pd.DataFrame,
    remaining: pd.DataFrame,
    strengths: dict[str, dict[str, float]],
    avg_goals: float,
    home_adv: float,
    team_home_advantages: dict[str, float],
    rating_method: str,
    extra_param: float | tuple[float, float],
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
        elif rating_method in {"spi", "initial_spi"} and isinstance(extra_param, tuple):
            probs = _spi_probs(mu_home - mu_away, extra_param)
            outcome = rng.choice(["H", "D", "A"], p=probs)
            for _ in range(25):
                hs = rng.poisson(mu_home)
                as_ = rng.poisson(mu_away)
                if (outcome == "H" and hs > as_) or (
                    outcome == "D" and hs == as_
                ) or (outcome == "A" and hs < as_):
                    break
            else:
                if outcome == "H" and hs <= as_:
                    hs = as_ + 1
                elif outcome == "A" and hs >= as_:
                    as_ = hs + 1
                elif outcome == "D":
                    as_ = hs
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
    rating_method: str = "spi",
    rng: np.random.Generator | None = None,
    elo_k: float = 20.0,
    home_field_advantage: float = 0.0,
    team_home_advantages: dict[str, float] | None = None,
    leader_history_paths: list[str | Path] | None = None,
    leader_history_weight: float = 0.5,
    smooth: float = 1.0,
    market_path: str | Path = "data/Brasileirao2025A.csv",
    decay_rate: float | None = None,
    logistic_decay: float | None = 0.007,
    seasons: list[str] | None = None,
) -> dict[str, float]:
    """Simulate remaining fixtures and return title probabilities.

    Parameters
    ----------
    matches : pd.DataFrame
        DataFrame containing all fixtures. Played games must have scores.
    iterations : int, default 1000
        Number of simulation runs.
    rating_method : str, default "spi"
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
    market_path : str | Path, default "data/Brasileirao2025A.csv"
        CSV file with team market values used by the ``"spi"`` or
        ``"initial_spi"`` rating method.
    decay_rate : float | None, optional
        Exponential decay factor applied to older matches when estimating
        strengths. ``None`` or ``0`` gives equal weight to all results.
    logistic_decay : float | None, default 0.007
        Weighting factor for the SPI logistic regression. A match ``d`` days
        before the latest fixture receives weight ``exp(-logistic_decay * d)``.
    seasons : list[str] | None, optional
        Seasons to recompute SPI coefficients for the ``"spi"`` and
        ``"initial_spi"`` rating methods.
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
        market_path=market_path,
        decay_rate=decay_rate,
        logistic_decay=logistic_decay,
        seasons=seasons,
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
    rating_method: str = "spi",
    rng: np.random.Generator | None = None,
    elo_k: float = 20.0,
    home_field_advantage: float = 0.0,
    team_home_advantages: dict[str, float] | None = None,
    leader_history_paths: list[str | Path] | None = None,
    leader_history_weight: float = 0.5,
    smooth: float = 1.0,
    market_path: str | Path = "data/Brasileirao2025A.csv",
    decay_rate: float | None = None,
    logistic_decay: float | None = 0.007,
    seasons: list[str] | None = None,
) -> dict[str, float]:
    """Simulate remaining fixtures and return relegation probabilities.

    The parameters mirror :func:`simulate_chances`. The returned values map
    each team to the probability of finishing in the bottom four positions.

    Parameters are the same as for :func:`simulate_chances`. ``smooth`` controls
    the constant added to goals scored and conceded when calculating attack and
    defence ratings under the ``"ratio"`` methods. ``home_field_advantage`` only
    affects the Elo method.
    market_path : str | Path, default "data/Brasileirao2025A.csv"
        CSV file with team market values used by the ``"spi"`` or
        ``"initial_spi"`` rating method.
    decay_rate : float | None, optional
        Exponential decay factor applied to older matches when estimating
        strengths. ``None`` or ``0`` gives equal weight to all results.
    logistic_decay : float | None, default 0.007
        Weighting factor for the SPI logistic regression. A match ``d`` days
        before the latest fixture receives weight ``exp(-logistic_decay * d)``.
    seasons : list[str] | None, optional
        Seasons to recompute SPI coefficients for the ``"spi"`` and
        ``"initial_spi"`` rating methods.
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
        market_path=market_path,
        decay_rate=decay_rate,
        logistic_decay=logistic_decay,
        seasons=seasons,
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
    rating_method: str = "spi",
    rng: np.random.Generator | None = None,
    elo_k: float = 20.0,
    home_field_advantage: float = 0.0,
    team_home_advantages: dict[str, float] | None = None,
    leader_history_paths: list[str | Path] | None = None,
    leader_history_weight: float = 0.5,
    smooth: float = 1.0,
    market_path: str | Path = "data/Brasileirao2025A.csv",
    decay_rate: float | None = None,
    logistic_decay: float | None = 0.007,
    seasons: list[str] | None = None,
) -> pd.DataFrame:
    """Project final league positions and points for each team.

    The parameters mirror :func:`simulate_chances`. The returned ``DataFrame``
    contains the average finishing position and point total of each club,
    sorted by expected position.
    ``smooth`` has the same meaning as in :func:`simulate_chances`.
    ``home_field_advantage`` only affects the Elo method.
    market_path : str | Path, default "data/Brasileirao2025A.csv"
        CSV file with team market values used by the ``"spi"`` or
        ``"initial_spi"`` rating method.
    decay_rate : float | None, optional
        Exponential decay factor applied to older matches when estimating
        strengths. ``None`` or ``0`` gives equal weight to all results.
    logistic_decay : float | None, default 0.007
        Weighting factor for the SPI logistic regression. A match ``d`` days
        before the latest fixture receives weight ``exp(-logistic_decay * d)``.
    seasons : list[str] | None, optional
        Seasons to recompute SPI coefficients for the ``"spi"`` and
        ``"initial_spi"`` rating methods.
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
        market_path=market_path,
        decay_rate=decay_rate,
        logistic_decay=logistic_decay,
        seasons=seasons,
    )

    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel())
    pos_totals = {t: 0.0 for t in teams}
    points_totals = {t: 0.0 for t in teams}
    wins_totals = {t: 0.0 for t in teams}
    gd_totals = {t: 0.0 for t in teams}
    gf_totals = {t: 0.0 for t in teams}

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
            team = row["team"]
            pos_totals[team] += idx + 1
            points_totals[team] += row["points"]
            wins_totals[team] += row["wins"]
            gd_totals[team] += row["gd"]
            gf_totals[team] += row["gf"]

    results = []
    for team in teams:
        results.append(
            {
                "team": team,
                "avg_position": pos_totals[team] / iterations,
                "points": points_totals[team] / iterations,
                "wins": wins_totals[team] / iterations,
                "gd": gd_totals[team] / iterations,
                "gf": gf_totals[team] / iterations,
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values(
        ["points", "wins", "gd", "gf", "team"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    df["position"] = range(1, len(df) + 1)
    return df[["team", "position", "points", "avg_position"]]


def summary_table(
    matches: pd.DataFrame,
    iterations: int = 1000,
    rating_method: str = "spi",
    rng: np.random.Generator | None = None,
    elo_k: float = 20.0,
    home_field_advantage: float = 0.0,
    team_home_advantages: dict[str, float] | None = None,
    leader_history_paths: list[str | Path] | None = None,
    leader_history_weight: float = 0.5,
    smooth: float = 1.0,
    market_path: str | Path = "data/Brasileirao2025A.csv",
    decay_rate: float | None = None,
    logistic_decay: float | None = 0.007,
) -> pd.DataFrame:
    """Return combined projections for each team.

    The returned ``DataFrame`` contains one row per club with the expected
    final rank, projected point total rounded to an integer, title chance and
    relegation probability. The table is sorted by projected position.
    The ``smooth`` parameter is forwarded to the underlying simulation
    functions.
    ``home_field_advantage`` is forwarded to the Elo-based rating routine and
    has no effect for other methods.
    market_path : str | Path, default "data/Brasileirao2025A.csv"
        CSV file with team market values used by the ``"spi"`` or
        ``"initial_spi"`` rating method.
    decay_rate : float | None, optional
        Exponential decay factor applied to older matches when estimating
        strengths. ``None`` or ``0`` gives equal weight to all results.
    logistic_decay : float | None, default 0.007
        Weighting factor for the SPI logistic regression. A match ``d`` days
        before the latest fixture receives weight ``exp(-logistic_decay * d)``.
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
        market_path=market_path,
        decay_rate=decay_rate,
        logistic_decay=logistic_decay,
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
        market_path=market_path,
        decay_rate=decay_rate,
        logistic_decay=logistic_decay,
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
        market_path=market_path,
        decay_rate=decay_rate,
        logistic_decay=logistic_decay,
    )

    table = table.sort_values("position").reset_index(drop=True)
    table["points"] = table["points"].round().astype(int)
    table["title"] = table["team"].map(chances)
    table["relegation"] = table["team"].map(relegation)
    return table[["position", "team", "points", "title", "relegation"]]
