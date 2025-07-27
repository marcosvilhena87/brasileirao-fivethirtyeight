"""Convenience exports for the simulator package."""

from .simulator import (
    league_table,
    parse_matches,
    simulate_chances,
    simulate_relegation_chances,
    simulate_final_table,
    summary_table,
    estimate_spi_strengths,
    initial_spi_strengths,
    SPI_DEFAULT_INTERCEPT,
    SPI_DEFAULT_SLOPE,
)
from .spi_coeffs import compute_spi_coeffs

__all__ = [
    "parse_matches",
    "league_table",
    "simulate_chances",
    "simulate_relegation_chances",
    "simulate_final_table",
    "summary_table",
    "estimate_spi_strengths",
    "initial_spi_strengths",
    "compute_spi_coeffs",
    "SPI_DEFAULT_INTERCEPT",
    "SPI_DEFAULT_SLOPE",
]
