import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from brasileirao import (
    parse_matches,
    league_table,
    simulate_chances,
    estimate_spi_strengths,
    compute_spi_coeffs,
    SPI_DEFAULT_INTERCEPT,
    SPI_DEFAULT_SLOPE,
)


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
