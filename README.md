# Brasileirão Simulator

This project provides a simple simulator for the 2025 Brasileirão Série A season. It parses the fixtures provided in `data/Brasileirao2025A.txt`, builds a league table from played matches, and simulates the remaining games many times to estimate title and relegation probabilities.

## Usage

Install dependencies from `requirements.txt` and run the simulator:

```bash
pip install -r requirements.txt
python main.py --simulations 1000 --rating poisson
```

The `--rating` option accepts `ratio` (default), `historic_ratio`, `poisson`,
`neg_binom`, `skellam`, `dixon_coles`, `elo`, `spi`, or `leader_history` to choose how team
strengths are estimated. The `skellam` method fits a regression to goal
differences. The `historic_ratio` method
mixes results from the 2024 season with a lower weight. The `elo` method
updates team ratings over time using an Elo formula; the `simulate_chances`
function exposes an `elo_k` parameter for deterministic runs. Use the
`--elo-k` CLI option or the `elo_k` function parameter to adjust the update
factor (default `20.0`). Use the `--seed` option to set a random seed and
reproduce a specific simulation. You can also specify team-specific home
advantage multipliers by passing a dictionary to the `team_home_advantages`
argument of `simulate_chances`. The `leader_history` rating method adjusts
strengths based on how often teams led past seasons; configure its behaviour
with `--leader-history-paths` and `--leader-weight`. When using Elo ratings you
may set a base home field bonus in rating points via the `home_field_advantage`
function parameter or the `--elo-home-advantage` CLI option.
The `spi` rating mimics FiveThirtyEight's approach by fitting a logistic
regression of match results on the expected goal difference derived from the
basic attack and defence factors.

The script outputs the estimated chance of winning the title for each team. It then prints the probability of each side finishing in the bottom four and being relegated.
It also estimates the average final position and points of every club.

## Tie-break Rules

When building the league table teams are ordered using the official Série A
criteria:

1. Points
2. Number of wins
3. Goal difference
4. Goals scored
5. Points obtained in the games between the tied sides
6. Team name (alphabetical)

These rules are implemented in :func:`league_table` and therefore affect all
simulation utilities.

## Project Layout

- `data/` – raw fixtures and results.
- `src/brasileirao/simulator.py` – parsing, table calculation and simulation routines.
- `main.py` – command-line interface to run the simulation.
- `tests/` – basic unit tests.

The main functions can be imported directly from the package:

```python
from brasileirao import (
    parse_matches,
    league_table,
    simulate_chances,
    simulate_relegation_chances,
    simulate_final_table,
)
```
