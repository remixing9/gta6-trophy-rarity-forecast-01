# GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/remixing9/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/remixing9/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the [make.fun bounty **gta6-trophy-rarity-forecast-01**](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, the percent of the players Exophase tracks who will have earned it at **2026-11-26T00:00:00Z** (one week after launch), from everything recorded up to the deadline. Five quantiles per trophy, scored by a pinball loss on the logit scale.

## The model

One trophy at a time, in u-space:

```
r(t) = A · sigmoid(a + b · ln(1 + t))
```

`t` is hours since launch. Exophase's page is a step function — it refreshes its counts hours apart — so the forecaster works on the *distinct* reads (consecutive identical rates collapsed to their first appearance) and anchors at when the current rate first appeared, not at the checkpoint. The asymptote `A` is the trophy's ceiling from its tier and description (story chapters in order, the final mission, 100% completion, grinds), raised when the trophy already beats it. Slopes are shrunk toward their slope family's mean (all story chapters together; the platinum alone, so it borrows nothing — a platinum must not inherit the chapters' rise). The spread is the standard deviation of the rate's logit at the target: random-walk diffusion (0.5 logit per √day) + slope standard error × extrapolation distance + a refresh-jump allowance (the median logit jump between distinct reads, floored at 0.5 when the history is flat, because the last rate may be hours stale). The cap binds the median only — the upper quantiles may pass it.

A blend with a linear-log-time model was tried (w = 0.7, 0.85, 0.9) and rejected: it hurts on saturating worlds and only ties on a late-surge world once the no-fit borrowing rule is in place.

## Files

- `main.py` — entrypoint: reads every `<id>.in`, writes `<id>.out`, then validates every output (55 rows, five finite quantiles in [0, 100], non-decreasing, ≤ 65,536 bytes) and fails loudly if any is malformed.
- `model.py` — the forecaster (stdlib only, no network, deterministic).
- `test_model.py` — 31 unit tests, all passing.

## Run it

```sh
python3 -m unittest test_model
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
```

## Local verification

- 31 unit tests: **all passing**.
- Practice cases (flat launch-night history): medians equal the last rates, spreads ~4× the starter's, output valid, two runs byte-identical.
- Synthetic final-like cases (history to 57.5 h, target 168 h, page refresh every 3–6 h, rates rounded to 2 decimals), 3 generative worlds × 3 seeds — mean pinball loss **0.3124** vs persistence 0.5731 and the house curve baseline 0.3590. Per world: lognormal 0.3231, Weibull 0.2578, late-surge 0.3565.

## License

MIT — see [LICENSE](LICENSE).
