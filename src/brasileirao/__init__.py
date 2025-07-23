"""Convenience exports for the simulator package."""

from .simulator import (
    league_table,
    parse_matches,
    simulate_chances,
    simulate_relegation_chances,
    simulate_final_table,
    summary_table,
    estimate_dixon_coles_strengths,
    compute_leader_stats,
    estimate_leader_history_strengths,
)

__all__ = [
    "parse_matches",
    "league_table",
    "simulate_chances",
    "simulate_relegation_chances",
    "simulate_final_table",
    "summary_table",
    "estimate_dixon_coles_strengths",
    "compute_leader_stats",
    "estimate_leader_history_strengths",
]

