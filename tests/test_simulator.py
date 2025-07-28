import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from brasileirao import (
    parse_matches,
    league_table,
    simulate_chances,
    simulate_final_table,
    estimate_spi_strengths,
    compute_spi_coeffs,
    initial_spi_strengths,
    SPI_DEFAULT_INTERCEPT,
    SPI_DEFAULT_SLOPE,
)
from brasileirao.simulator import (
    update_spi_ratings,
    estimate_market_strengths,
    compute_leader_stats,
)
from brasileirao.spi_coeffs import available_seasons


def test_parse_matches():
    df = parse_matches('data/Brasileirao2025A.txt')
    assert len(df) == 380
    assert {'home_team', 'away_team', 'home_score', 'away_score'}.issubset(df.columns)


def test_parse_matches_accepts_bom(tmp_path):
    data = Path('data/Brasileirao2025A.txt').read_bytes()
    bom_file = tmp_path / 'with_bom.txt'
    bom_file.write_bytes(b'\xef\xbb\xbf' + data)
    df = parse_matches(bom_file)
    assert len(df) == 380


def test_league_table_basic():
    df = parse_matches('data/Brasileirao2025A.txt')
    table = league_table(df)
    assert 'points' in table.columns


def test_simulate_chances_spi_repeatable():
    df = parse_matches('data/Brasileirao2025A.txt')
    rng = np.random.default_rng(123)
    first = simulate_chances(df, iterations=5, rng=rng)
    rng = np.random.default_rng(123)
    second = simulate_chances(df, iterations=5, rng=rng)
    assert first == second


def test_estimate_spi_strengths_returns_five_values():
    df = parse_matches('data/Brasileirao2025A.txt')
    result = estimate_spi_strengths(df)
    assert isinstance(result, tuple)
    assert len(result) == 5
    assert isinstance(result[-1], float)
    assert isinstance(result[-2], float)


def test_compute_spi_coeffs_empty_returns_defaults():
    intercept, slope = compute_spi_coeffs(seasons=[])
    assert intercept == SPI_DEFAULT_INTERCEPT
    assert slope == SPI_DEFAULT_SLOPE


def test_compute_spi_coeffs_filters_incomplete_seasons():
    seasons = available_seasons(only_complete=True)
    explicit = compute_spi_coeffs(seasons=seasons)
    default = compute_spi_coeffs()
    assert np.isclose(explicit[0], default[0])
    assert np.isclose(explicit[1], default[1])


def test_weighted_strengths_change():
    df = parse_matches("data/Brasileirao2025A.txt")
    from brasileirao.simulator import _estimate_strengths

    base, _, _ = _estimate_strengths(df)
    weighted, _, _ = _estimate_strengths(df, decay_rate=0.01)

    assert base.keys() == weighted.keys()
    diff = any(
        not np.isclose(base[t]["attack"], weighted[t]["attack"]) for t in base
    )
    assert diff


def test_update_spi_ratings_changes_values():
    strengths = {
        "A": {"attack": 1.0, "defense": 1.0},
        "B": {"attack": 1.0, "defense": 1.0},
    }
    mu_h, mu_a = update_spi_ratings(
        strengths,
        "A",
        "B",
        2,
        1,
        avg_goals=2.5,
        home_adv=1.0,
    )
    assert mu_h > 0 and mu_a > 0
    assert strengths["A"]["attack"] != 1.0 or strengths["B"]["defense"] != 1.0


def test_spi_sequential_updates_differ_from_static():
    df = parse_matches("data/Brasileirao2025A.txt")
    dynamic, _, _, _, _ = estimate_spi_strengths(df)
    static, _, _ = estimate_market_strengths(df)

    assert dynamic.keys() == static.keys()
    changed = any(
        not np.isclose(dynamic[t]["attack"], static[t]["attack"]) for t in dynamic
    )
    assert changed


def test_team_home_advantage_changes_results():
    df = parse_matches("data/Brasileirao2025A.txt")
    rng = np.random.default_rng(0)
    base = simulate_final_table(df, iterations=10, rng=rng)

    rng = np.random.default_rng(0)
    advantaged = simulate_final_table(
        df, iterations=10, rng=rng, team_home_advantages={"Palmeiras": 2.0}
    )

    base_pts = base.loc[base.team == "Palmeiras", "points"].iloc[0]
    adv_pts = advantaged.loc[advantaged.team == "Palmeiras", "points"].iloc[0]
    assert base_pts != adv_pts


def test_home_advantage_shrinkage_modifies_values():
    df = parse_matches("data/Brasileirao2025A.txt")
    from brasileirao.simulator import _estimate_team_home_advantages

    raw = _estimate_team_home_advantages(df, shrink_weight=1.0)
    shrunk = _estimate_team_home_advantages(df)

    assert raw.keys() == shrunk.keys()
    changed = any(not np.isclose(raw[t], shrunk[t]) for t in raw)
    assert changed


def test_spi_coeffs_decay_changes_values():
    seasons = ["2023", "2024"]
    no_decay = compute_spi_coeffs(seasons=seasons, decay_rate=0.0, logistic_decay=None)
    with_decay = compute_spi_coeffs(seasons=seasons, decay_rate=0.5, logistic_decay=None)
    assert not (
        np.isclose(no_decay[0], with_decay[0])
        and np.isclose(no_decay[1], with_decay[1])
    )


def test_compute_spi_coeffs_matches_manual_estimate():
    seasons = ["2023", "2024"]
    dfs = []
    weights = []
    max_year = max(int(s) for s in seasons)
    for season in seasons:
        df = parse_matches(f"data/Brasileirao{season}A.txt")
        dfs.append(df)
        age = max_year - int(season)
        w = np.exp(-0.5 * age)
        weights.append(pd.Series(w, index=df.index))
    combined = pd.concat(dfs, ignore_index=True)
    weight_series = pd.concat(weights, ignore_index=True)
    expected = estimate_spi_strengths(
        combined,
        market_path=f"data/Brasileirao{max_year}A.csv",
        match_weights=weight_series,
        smooth=1.0,
    )
    intercept_expected = expected[3]
    slope_expected = expected[4]
    intercept, slope = compute_spi_coeffs(seasons=seasons, decay_rate=0.5)
    assert np.isclose(intercept, intercept_expected)
    assert np.isclose(slope, slope_expected)


def test_initial_spi_strengths_with_seasons():
    seasons = ["2023", "2024"]
    expected = compute_spi_coeffs(
        seasons=seasons, decay_rate=0.0, market_path="data/Brasileirao2024A.csv"
    )
    default = initial_spi_strengths()
    overriden = initial_spi_strengths(seasons=seasons)

    assert not (
        np.isclose(default[3], expected[0]) and np.isclose(default[4], expected[1])
    )
    assert np.isclose(overriden[3], expected[0])
    assert np.isclose(overriden[4], expected[1])


def test_initial_spi_strengths_multiple_seasons_changes_output():
    single, _, _, _, _ = initial_spi_strengths(
        past_path="data/Brasileirao2024A.txt"
    )
    multi, _, _, _, _ = initial_spi_strengths(
        past_path=["data/Brasileirao2023A.txt", "data/Brasileirao2024A.txt"]
    )

    assert "Palmeiras" in single
    assert "Palmeiras" in multi
    assert not np.isclose(
        single["Palmeiras"]["attack"], multi["Palmeiras"]["attack"]
    )


def test_spi_coeffs_uses_season_market_files():
    seasons = ["2023", "2024"]
    default = compute_spi_coeffs(seasons=seasons)
    constant = compute_spi_coeffs(
        seasons=seasons, market_path="data/Brasileirao2025A.csv"
    )

    assert not (
        np.isclose(default[0], constant[0]) and np.isclose(default[1], constant[1])
    )


def test_spi_coeffs_accepts_market_mapping():
    seasons = ["2023", "2024"]
    mapping = {
        "2023": "data/Brasileirao2024A.csv",
        "2024": "data/Brasileirao2024A.csv",
    }
    mapped = compute_spi_coeffs(seasons=seasons, market_path=mapping)
    constant = compute_spi_coeffs(
        seasons=seasons, market_path="data/Brasileirao2024A.csv"
    )

    assert np.isclose(mapped[0], constant[0])
    assert np.isclose(mapped[1], constant[1])


def test_logistic_decay_changes_spi_coeffs():
    df = parse_matches("data/Brasileirao2025A.txt")
    base = estimate_spi_strengths(df)
    decayed = estimate_spi_strengths(df, logistic_decay=0.01)

    assert not (
        np.isclose(base[3], decayed[3]) and np.isclose(base[4], decayed[4])
    )


def test_compute_spi_coeffs_logistic_decay_changes_values():
    seasons = ["2023", "2024"]
    base = compute_spi_coeffs(seasons=seasons, logistic_decay=None)
    decayed = compute_spi_coeffs(seasons=seasons, logistic_decay=0.01)

    assert not (
        np.isclose(base[0], decayed[0]) and np.isclose(base[1], decayed[1])
    )


def test_logistic_and_match_weight_combination():
    df = parse_matches("data/Brasileirao2025A.txt")
    weights = pd.Series(np.linspace(1.0, 2.0, len(df)), index=df.index)

    logistic = estimate_spi_strengths(df, logistic_decay=0.01)
    manual = estimate_spi_strengths(df, logistic_decay=None, match_weights=weights)
    combined = estimate_spi_strengths(
        df, logistic_decay=0.01, match_weights=weights
    )

    assert not (
        np.isclose(combined[3], logistic[3])
        and np.isclose(combined[4], logistic[4])
    )
    assert not (
        np.isclose(combined[3], manual[3])
        and np.isclose(combined[4], manual[4])
    )


def test_compute_leader_stats_regression():
    df = parse_matches("data/Brasileirao2025A.txt")
    counts = compute_leader_stats(df)
    expected = {
        "Cruzeiro": 26,
        "Mirassol": 0,
        "Grêmio": 0,
        "Atlético-MG": 0,
        "Fortaleza": 7,
        "Fluminense": 1,
        "São Paulo": 0,
        "Sport": 0,
        "Juventude": 3,
        "Vitória": 0,
        "Flamengo": 51,
        "Internacional": 0,
        "Palmeiras": 56,
        "Botafogo": 0,
        "Vasco": 0,
        "Santos": 0,
        "Bahia": 0,
        "Corinthians": 9,
        "Bragantino": 3,
        "Ceará": 1,
    }
    assert counts == expected
