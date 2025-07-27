# Brasileirão Simulator

This project provides a simple simulator for the 2025 Brasileirão Série A season. It parses the fixtures provided in `data/Brasileirao2025A.txt`, builds a league table from played matches, and simulates the remaining games many times to estimate title and relegation probabilities.

## Usage

Install dependencies from `requirements.txt` and run the simulator:

```bash
pip install -r requirements.txt
python main.py --simulations 1000
```

The simulator uses a FiveThirtyEight-style model (SPI) to estimate team
strengths. Use the `--market-path` option to specify an alternative CSV with
team market values. The `--seed` argument sets the random seed for reproducible
results.
The `spi` rating mimics FiveThirtyEight's approach by fitting a logistic
regression of match results on the expected goal difference derived from the
basic attack and defence factors. Before the regression, each team's attack and
defence are scaled by its market value from `data/Brasileirao2025A.csv`.  The
`estimate_spi_strengths` function accepts a ``market_path`` parameter to load a
different CSV file.  The fitted intercept and slope are used to convert an
expected goal difference into win/draw/loss probabilities during the
simulation. When no matches have been played the function returns default
coefficients of ``-0.180149`` and ``0.228628`` based on the 2023–2024 seasons.
The helper ``initial_spi_strengths`` can be used at the start of a season to
shrink each team's previous rating towards the league average following
``current = previous × weight + mean × (1 − weight)``.

``past_path`` may also be a sequence of season files or years.  Each season is
weighted by ``exp(-decay_rate * age)`` where ``age`` counts seasons back from
the most recent before being combined and shrunk to the league mean.  For
example::

    initial_spi_strengths(past_path=["2023", "2024"], decay_rate=0.5)

Passing a list of ``seasons`` to ``initial_spi_strengths`` replaces the logistic
regression coefficients with those produced by ``compute_spi_coeffs`` across the
specified years.  For instance::

    initial_spi_strengths(seasons=["2023", "2024"])

The ``compute_spi_coeffs`` helper scans the ``data/`` folder for past seasons
and recalculates the logistic regression intercept and slope.  Seasons can be
specified via the ``BRASILEIRAO_SEASONS`` environment variable or the
``--seasons`` argument of the ``spi_coeffs`` module.  ``--decay-rate`` applies
exponential weighting to older seasons using ``exp(-decay_rate * age)`` where
``age`` counts seasons back from the most recent.  When no historical files are
present the default coefficients above are returned.

To recompute these coefficients yourself run:

```bash
PYTHONPATH=src python -m brasileirao.spi_coeffs
```
By default all seasons in ``data/`` are used.  You may pass ``--seasons`` or set
``BRASILEIRAO_SEASONS`` to limit the years included.  ``--decay-rate`` controls
how quickly older seasons lose influence.

The script outputs the estimated chance of winning the title for each team. It
then prints the probability of each side finishing in the bottom four and being
relegated. It also estimates the average final position and points of every club.

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

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
