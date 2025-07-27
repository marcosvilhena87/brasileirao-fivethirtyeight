import os
import sys
import numpy as np
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
from brasileirao.simulator import update_spi_ratings, estimate_market_strengths


def test_parse_matches():
    df = parse_matches('data/Brasileirao2025A.txt')
    assert len(df) == 380
    assert {'home_team', 'away_team', 'home_score', 'away_score'}.issubset(df.columns)


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


def test_spi_coeffs_decay_changes_values():
    seasons = ["2023", "2024"]
    no_decay = compute_spi_coeffs(seasons=seasons, decay_rate=0.0)
    with_decay = compute_spi_coeffs(seasons=seasons, decay_rate=0.5)
    assert not (
        np.isclose(no_decay[0], with_decay[0])
        and np.isclose(no_decay[1], with_decay[1])
    )


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
