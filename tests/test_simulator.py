import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import pandas as pd
import numpy as np
from brasileirao import parse_matches, league_table, simulate_chances
from brasileirao import simulator


def test_parse_matches():
    df = parse_matches('data/Brasileirao2025A.txt')
    assert len(df) == 380
    assert {'home_team', 'away_team', 'home_score', 'away_score'}.issubset(df.columns)


def test_league_table():
    df = parse_matches('data/Brasileirao2025A.txt')
    table = league_table(df)
    # after first rounds some teams have points
    assert 'points' in table.columns
    assert table['played'].max() > 0


def test_league_table_deterministic_sorting():
    data = [
        {
            'date': '2025-01-01',
            'home_team': 'Alpha',
            'away_team': 'Beta',
            'home_score': 1,
            'away_score': 0,
        },
        {
            'date': '2025-01-02',
            'home_team': 'Beta',
            'away_team': 'Gamma',
            'home_score': 1,
            'away_score': 0,
        },
        {
            'date': '2025-01-03',
            'home_team': 'Gamma',
            'away_team': 'Alpha',
            'home_score': 1,
            'away_score': 0,
        },
    ]
    df = pd.DataFrame(data)
    table = league_table(df)
    assert list(table.team) == sorted(table.team)


def test_simulate_chances():
    df = parse_matches('data/Brasileirao2025A.txt')
    chances = simulate_chances(df, iterations=10)
    assert abs(sum(chances.values()) - 1.0) < 1e-6


def test_simulate_chances_poisson():
    df = parse_matches('data/Brasileirao2025A.txt')
    chances = simulate_chances(df, iterations=10, rating_method="poisson")
    assert abs(sum(chances.values()) - 1.0) < 1e-6


def test_simulate_chances_seed_repeatability():
    df = parse_matches('data/Brasileirao2025A.txt')
    rng = np.random.default_rng(1234)
    chances1 = simulate_chances(df, iterations=5, rng=rng)
    rng = np.random.default_rng(1234)
    chances2 = simulate_chances(df, iterations=5, rng=rng)
    assert chances1 == chances2


def test_estimate_strengths():
    df = parse_matches('data/Brasileirao2025A.txt')
    strengths, _, _ = simulator._estimate_strengths(df)
    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())

    # every team from the matches should appear in the strengths dict
    assert set(teams) == set(strengths.keys())

    # all estimated attack and defense values must be positive
    assert all(v["attack"] > 0 and v["defense"] > 0 for v in strengths.values())


def test_estimate_strengths_zero_goals():
    data = [
        {"date": "2025-01-01", "home_team": "A", "away_team": "B", "home_score": 0, "away_score": 1},
        {"date": "2025-01-02", "home_team": "A", "away_team": "C", "home_score": 0, "away_score": 2},
        {"date": "2025-01-03", "home_team": "C", "away_team": "B", "home_score": 1, "away_score": 0},
    ]
    df = pd.DataFrame(data)
    strengths, _, _ = simulator._estimate_strengths(df)
    # team A scored zero goals, team C conceded zero goals
    assert strengths["A"]["attack"] > 0
    assert strengths["C"]["defense"] > 0


def test_estimate_strengths_with_history():
    df = parse_matches('data/Brasileirao2025A.txt')
    strengths, _, _ = simulator.estimate_strengths_with_history(df)
    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
    assert set(teams).issubset(set(strengths.keys()))


def test_estimate_skellam_strengths_deterministic():
    df = parse_matches('data/Brasileirao2025A.txt')
    first = simulator.estimate_skellam_strengths(df)
    second = simulator.estimate_skellam_strengths(df)
    assert first == second


def test_estimate_market_strengths_deterministic():
    df = parse_matches('data/Brasileirao2025A.txt')
    first = simulator.estimate_market_strengths(df)
    second = simulator.estimate_market_strengths(df)
    assert first == second


def test_simulate_chances_historic_ratio():
    df = parse_matches('data/Brasileirao2025A.txt')
    chances = simulate_chances(df, iterations=10, rating_method="historic_ratio")
    assert abs(sum(chances.values()) - 1.0) < 1e-6


def test_simulate_chances_elo_seed_repeatability():
    df = parse_matches('data/Brasileirao2025A.txt')
    rng = np.random.default_rng(42)
    chances1 = simulate_chances(
        df,
        iterations=5,
        rating_method="elo",
        rng=rng,
        elo_k=15.0,
    )
    rng = np.random.default_rng(42)
    chances2 = simulate_chances(
        df,
        iterations=5,
        rating_method="elo",
        rng=rng,
        elo_k=15.0,
    )
    assert chances1 == chances2
    assert abs(sum(chances1.values()) - 1.0) < 1e-6


def test_elo_k_value_changes_results():
    df = parse_matches('data/Brasileirao2025A.txt')
    rng = np.random.default_rng(99)
    chances_low = simulate_chances(
        df,
        iterations=5,
        rating_method="elo",
        rng=rng,
        elo_k=5.0,
    )
    rng = np.random.default_rng(99)
    chances_high = simulate_chances(
        df,
        iterations=5,
        rating_method="elo",
        rng=rng,
        elo_k=40.0,
    )
    assert chances_low != chances_high


def test_smooth_value_changes_results():
    df = parse_matches('data/Brasileirao2025A.txt')
    rng = np.random.default_rng(55)
    base = simulate_chances(
        df,
        iterations=5,
        rng=rng,
        smooth=0.5,
    )
    rng = np.random.default_rng(55)
    alt = simulate_chances(
        df,
        iterations=5,
        rng=rng,
        smooth=2.0,
    )
    assert base != alt


def test_simulate_chances_neg_binom_seed_repeatability():
    df = parse_matches('data/Brasileirao2025A.txt')
    rng = np.random.default_rng(7)
    chances1 = simulate_chances(
        df,
        iterations=5,
        rating_method="neg_binom",
        rng=rng,
    )
    rng = np.random.default_rng(7)
    chances2 = simulate_chances(
        df,
        iterations=5,
        rating_method="neg_binom",
        rng=rng,
    )
    assert chances1 == chances2
    assert abs(sum(chances1.values()) - 1.0) < 1e-6


def test_neg_binom_differs_from_poisson():
    df = parse_matches("data/Brasileirao2025A.txt")
    rng = np.random.default_rng(123)
    poisson_res = simulate_chances(df, iterations=50, rating_method="poisson", rng=rng)
    rng = np.random.default_rng(123)
    nb_res = simulate_chances(df, iterations=50, rating_method="neg_binom", rng=rng)
    assert poisson_res != nb_res


def test_estimate_negative_binomial_strengths_dispersion_used():
    df = parse_matches("data/Brasileirao2025A.txt")
    disp = simulator._estimate_dispersion(df)
    strengths, base_mu, home_adv, returned_disp = simulator.estimate_negative_binomial_strengths(df)
    assert np.isclose(disp, returned_disp)

    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    played = df.dropna(subset=["home_score", "away_score"])
    rows = []
    for _, row in played.iterrows():
        rows.append({"team": row["home_team"], "opponent": row["away_team"], "home": 1, "goals": row["home_score"]})
        rows.append({"team": row["away_team"], "opponent": row["home_team"], "home": 0, "goals": row["away_score"]})
    manual_df = pd.DataFrame(rows)
    manual_model = smf.glm(
        "goals ~ home + C(team) + C(opponent)",
        data=manual_df,
        family=sm.families.NegativeBinomial(alpha=disp),
    ).fit()
    manual_mu = float(np.exp(manual_model.params["Intercept"]))
    assert np.isclose(base_mu, manual_mu)


def test_simulate_chances_skellam_seed_repeatability():
    df = parse_matches('data/Brasileirao2025A.txt')
    rng = np.random.default_rng(66)
    chances1 = simulate_chances(
        df,
        iterations=5,
        rating_method="skellam",
        rng=rng,
    )
    rng = np.random.default_rng(66)
    chances2 = simulate_chances(
        df,
        iterations=5,
        rating_method="skellam",
        rng=rng,
    )
    assert chances1 == chances2
    assert abs(sum(chances1.values()) - 1.0) < 1e-6


def test_team_home_advantage_changes_results():
    df = parse_matches('data/Brasileirao2025A.txt')
    rng = np.random.default_rng(11)
    base = simulate_chances(df, iterations=5, rng=rng)
    rng = np.random.default_rng(11)
    custom = simulate_chances(
        df,
        iterations=5,
        rng=rng,
        team_home_advantages={"Bahia": 2.0},
    )
    assert base != custom


def test_home_field_advantage_changes_elo_results():
    df = parse_matches("data/Brasileirao2025A.txt")
    rng = np.random.default_rng(22)
    base = simulate_chances(
        df,
        iterations=5,
        rating_method="elo",
        rng=rng,
        elo_k=20.0,
    )
    rng = np.random.default_rng(22)
    adv = simulate_chances(
        df,
        iterations=5,
        rating_method="elo",
        rng=rng,
        elo_k=20.0,
        home_field_advantage=50.0,
    )
    assert base != adv


def test_simulate_chances_dixon_coles_seed_repeatability():
    df = parse_matches("data/Brasileirao2025A.txt")
    rng = np.random.default_rng(123)
    chances1 = simulate_chances(
        df,
        iterations=5,
        rating_method="dixon_coles",
        rng=rng,
    )
    rng = np.random.default_rng(123)
    chances2 = simulate_chances(
        df,
        iterations=5,
        rating_method="dixon_coles",
        rng=rng,
    )
    assert chances1 == chances2
    assert abs(sum(chances1.values()) - 1.0) < 1e-6


def test_simulate_chances_spi_seed_repeatability():
    df = parse_matches("data/Brasileirao2025A.txt")
    rng = np.random.default_rng(101)
    chances1 = simulate_chances(
        df,
        iterations=5,
        rating_method="spi",
        rng=rng,
    )
    rng = np.random.default_rng(101)
    chances2 = simulate_chances(
        df,
        iterations=5,
        rating_method="spi",
        rng=rng,
    )
    assert chances1 == chances2
    assert abs(sum(chances1.values()) - 1.0) < 1e-6


def test_estimate_spi_strengths_defaults():
    data = [
        {
            "date": "2025-01-01",
            "home_team": "A",
            "away_team": "B",
            "home_score": np.nan,
            "away_score": np.nan,
        }
    ]
    df = pd.DataFrame(data)
    _, _, _, intercept, slope = simulator.estimate_spi_strengths(df)
    assert intercept == simulator.SPI_DEFAULT_INTERCEPT
    assert slope == simulator.SPI_DEFAULT_SLOPE


def test_compute_leader_stats():
    data = [
        {
            "date": "2025-01-01",
            "home_team": "Alpha",
            "away_team": "Beta",
            "home_score": 1,
            "away_score": 0,
        },
        {
            "date": "2025-01-02",
            "home_team": "Alpha",
            "away_team": "Gamma",
            "home_score": 0,
            "away_score": 1,
        },
        {
            "date": "2025-01-03",
            "home_team": "Beta",
            "away_team": "Gamma",
            "home_score": 2,
            "away_score": 0,
        },
    ]
    df = pd.DataFrame(data)
    counts = simulator.compute_leader_stats(df)
    assert counts["Alpha"] == 1
    assert counts["Beta"] == 1
    assert counts["Gamma"] == 1


def _slow_leader_stats(df: pd.DataFrame) -> dict:
    """Naive reference implementation using league_table."""
    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
    counts = {t: 0 for t in teams}
    played: list[dict] = []
    for _, row in df.sort_values("date").iterrows():
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue
        played.append(row.to_dict())
        table = simulator.league_table(pd.DataFrame(played))
        counts[table.iloc[0]["team"]] += 1
    return counts


def test_compute_leader_stats_equivalence():
    df = parse_matches("data/Brasileirao2025A.txt")
    assert simulator.compute_leader_stats(df) == _slow_leader_stats(df)


def test_simulate_chances_leader_history_seed_repeatability():
    df = parse_matches("data/Brasileirao2025A.txt")
    rng = np.random.default_rng(55)
    chances1 = simulate_chances(
        df,
        iterations=5,
        rating_method="leader_history",
        rng=rng,
        leader_history_paths=["data/Brasileirao2024A.txt"],
        leader_history_weight=0.5,
    )
    rng = np.random.default_rng(55)
    chances2 = simulate_chances(
        df,
        iterations=5,
        rating_method="leader_history",
        rng=rng,
        leader_history_paths=["data/Brasileirao2024A.txt"],
        leader_history_weight=0.5,
    )
    assert chances1 == chances2
    assert abs(sum(chances1.values()) - 1.0) < 1e-6


def test_simulate_relegation_chances():
    df = parse_matches("data/Brasileirao2025A.txt")
    probs = simulator.simulate_relegation_chances(df, iterations=10)
    assert abs(sum(probs.values()) - 4.0) < 1e-6


def test_simulate_relegation_chances_seed_repeatability():
    df = parse_matches("data/Brasileirao2025A.txt")
    rng = np.random.default_rng(123)
    first = simulator.simulate_relegation_chances(df, iterations=5, rng=rng)
    rng = np.random.default_rng(123)
    second = simulator.simulate_relegation_chances(df, iterations=5, rng=rng)
    assert first == second


def test_simulate_final_table_deterministic():
    df = parse_matches("data/Brasileirao2025A.txt")
    rng = np.random.default_rng(1)
    table1 = simulator.simulate_final_table(df, iterations=5, rng=rng)
    rng = np.random.default_rng(1)
    table2 = simulator.simulate_final_table(df, iterations=5, rng=rng)
    pd.testing.assert_frame_equal(table1, table2)
    assert {"team", "position", "points"}.issubset(table1.columns)
    assert len(table1) == len(pd.unique(df[["home_team", "away_team"]].values.ravel()))


def test_summary_table_deterministic_columns():
    df = parse_matches("data/Brasileirao2025A.txt")
    rng = np.random.default_rng(123)
    table1 = simulator.summary_table(df, iterations=5, rng=rng)
    rng = np.random.default_rng(123)
    table2 = simulator.summary_table(df, iterations=5, rng=rng)
    pd.testing.assert_frame_equal(table1, table2)
    assert {"position", "team", "points", "title", "relegation"}.issubset(table1.columns)


def test_simulate_final_table_spi_seed_repeatability():
    df = parse_matches("data/Brasileirao2025A.txt")
    rng = np.random.default_rng(202)
    table1 = simulator.simulate_final_table(df, iterations=5, rating_method="spi", rng=rng)
    rng = np.random.default_rng(202)
    table2 = simulator.simulate_final_table(df, iterations=5, rating_method="spi", rng=rng)
    pd.testing.assert_frame_equal(table1, table2)


def test_summary_table_spi_seed_repeatability():
    df = parse_matches("data/Brasileirao2025A.txt")
    rng = np.random.default_rng(303)
    table1 = simulator.summary_table(df, iterations=5, rating_method="spi", rng=rng)
    rng = np.random.default_rng(303)
    table2 = simulator.summary_table(df, iterations=5, rating_method="spi", rng=rng)
    pd.testing.assert_frame_equal(table1, table2)


def test_simulate_chances_spi_custom_market_path_changes_results():
    df = parse_matches("data/Brasileirao2025A.txt")
    rng = np.random.default_rng(77)
    base = simulate_chances(df, iterations=20, rating_method="spi", rng=rng)
    rng = np.random.default_rng(77)
    custom = simulate_chances(
        df,
        iterations=20,
        rating_method="spi",
        rng=rng,
        market_path="data/Brasileirao2024A.csv",
    )
    assert base != custom


def test_summary_table_spi_custom_market_path_repeatability():
    df = parse_matches("data/Brasileirao2025A.txt")
    rng = np.random.default_rng(88)
    t1 = simulator.summary_table(
        df,
        iterations=5,
        rating_method="spi",
        rng=rng,
        market_path="data/Brasileirao2024A.csv",
    )
    rng = np.random.default_rng(88)
    t2 = simulator.summary_table(
        df,
        iterations=5,
        rating_method="spi",
        rng=rng,
        market_path="data/Brasileirao2024A.csv",
    )
    pd.testing.assert_frame_equal(t1, t2)


def test_league_table_goal_difference_tiebreak():
    data = [
        {"date": "2025-01-01", "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 2},
        {"date": "2025-01-02", "home_team": "A", "away_team": "C", "home_score": 1, "away_score": 0},
        {"date": "2025-01-03", "home_team": "C", "away_team": "A", "home_score": 0, "away_score": 1},
        {"date": "2025-01-04", "home_team": "B", "away_team": "C", "home_score": 3, "away_score": 0},
    ]
    df = pd.DataFrame(data)
    table = league_table(df)
    assert list(table.team[:2]) == ["B", "A"]


def test_league_table_goals_scored_tiebreak():
    data = [
        {"date": "2025-01-01", "home_team": "A", "away_team": "B", "home_score": 0, "away_score": 0},
        {"date": "2025-01-02", "home_team": "B", "away_team": "A", "home_score": 0, "away_score": 0},
        {"date": "2025-01-03", "home_team": "A", "away_team": "C", "home_score": 1, "away_score": 0},
        {"date": "2025-01-04", "home_team": "B", "away_team": "C", "home_score": 2, "away_score": 1},
    ]
    df = pd.DataFrame(data)
    table = league_table(df)
    assert list(table.team[:2]) == ["B", "A"]


def test_league_table_head_to_head_tiebreak():
    data = [
        {"date": "2025-01-01", "home_team": "A", "away_team": "B", "home_score": 0, "away_score": 0},
        {"date": "2025-01-02", "home_team": "B", "away_team": "A", "home_score": 1, "away_score": 0},
        {"date": "2025-01-03", "home_team": "A", "away_team": "C", "home_score": 1, "away_score": 0},
        {"date": "2025-01-04", "home_team": "B", "away_team": "C", "home_score": 0, "away_score": 1},
    ]
    df = pd.DataFrame(data)
    table = league_table(df)
    assert list(table.team[:2]) == ["B", "A"]


def test_initial_spi_strengths_weighting():
    past = parse_matches("data/Brasileirao2024A.txt")
    base, _, _, inter, slope = simulator.estimate_spi_strengths(
        past, market_path="data/Brasileirao2024A.csv"
    )
    weighted, _, _, inter2, slope2 = simulator.initial_spi_strengths(
        past_path="data/Brasileirao2024A.txt", weight=0.5, market_path="data/Brasileirao2024A.csv"
    )
    assert np.isclose(inter, inter2)
    assert np.isclose(slope, slope2)
    avg_attack = np.mean([v["attack"] for v in base.values()])
    for t in base:
        expect = base[t]["attack"] * 0.5 + avg_attack * 0.5
        assert np.isclose(weighted[t]["attack"], expect)
