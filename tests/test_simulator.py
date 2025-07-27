import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pandas as pd
from brasileirao import (
    parse_matches,
    league_table,
    simulate_chances,
    estimate_spi_strengths,
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
