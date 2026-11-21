# makefun-gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bagusahmad03/makefun-gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bagusahmad03/makefun-gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An entry for make.fun's `gta6-trophy-rarity-forecast-01`: one program that
forecasts, for each of the 55 Grand Theft Auto VI trophies on PlayStation 5,
the percent of the players Exophase tracks who will have earned it at
2026-11-26T00:00:00Z, as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs the program as `python3 -P -s /bot/main.py` (Python 3.13,
standard library only, no network) and expects `<id>.out` in `/output` for
every `<id>.in` in `/input`.

## How it forecasts

A trophy's earn rate is earners divided by players tracked, and the two
sides move for different reasons: launch floods the tracker with new
players (diluting late trophies toward the floor) while early-story
earners pile up fast. So the program forecasts the two sides
separately, then divides at the target:

- **Players tracked** `N(t)`: a saturating curve `A + B·exp(-t/τ)` fitted
  to the history by grid search over τ.
- **Earners** `E(t)`: a saturating exponential
  `E_inf·(1 - exp(-max(0, t - delay)/τ))` per trophy, with (delay, τ)
  grid-searched around a group prior read off the trophy's description
  (Chapter 1 and early activities fast, later chapters slower, gold and
  platinum slowest), scored by the logit error of the fitted rate.
  Trophies with too little usable history fall back to persistence.
- **Medians**: `100 · Ê(target) / N̂(target)`, with story nesting
  enforced (a later chapter cannot sit above an earlier one, the
  platinum cannot sit above any other trophy). The always-owned trophy
  (100% in the last read and nearly every read) stays at 99.99.
- **Spread**: the random-walk term `0.5·√d` combined in quadrature with
  the fit's logit residual (floored at 0.35), so the bands stay honest
  about how far launch-week rates can move.

## Verification

- The starter runs clean on the practice inputs; this program run the
  same way produces valid outputs too: 55 rows each, quantiles
  non-decreasing, all in [0, 100], ~2.7 KB (limit 65,536), and different
  from the starter (5/55 rows differ on the flat practice histories —
  the rest fall back to persistence there, since a flat 90-minute
  window cannot identify a curve).
- Row id4 matches the spec example exactly on practice input
  ([0.8974, 1.012, 1.1, 1.1955, 1.3477]).
- Synthetic launch-week backtest (histories generated from saturating
  earners/tracked curves calibrated to the real Exophase snapshots of
  19–21 Nov, scored at 26 Nov with the real pinball loss): this fitted
  procedure scores **0.237** vs **0.310** for persistence and **0.299**
  for the log-time curve baseline, across three tracked-player
  scenarios (30k / 100k / 300k). The comparison is scored on the 46
  trophies whose curve had started by the calibration read; the other
  9 (platinum, Chapter 4, final mission and late "final mission for …"
  trophies) sit at the 0.01% display floor through the whole history,
  so no history can identify their curve and the simulator cannot
  define a fair truth for them.

## Files

- `main.py` — the forecaster, Python 3.13 standard library only
- `README.md` — this file
- `LICENSE` — MIT

---

<sub>Entered on [make.fun](https://make.fun/?ref=syht5d66&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
