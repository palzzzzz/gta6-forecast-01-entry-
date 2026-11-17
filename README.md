# Launch-day Twitch audience forecaster

Entry for make.fun's **gta6-twitch-ccu-forecast-01**: forecast how many
people watch Grand Theft Auto VI live on Twitch in each hour of its
release day (2026-11-19, UTC) after the first, as five quantiles
(5%, 25%, 50%, 75%, 95%) per hour, scored by pinball loss on the log
scale.

## Run

```
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every
`*.in` case in the input directory and writes one `<id>.out` per case:
`{"hours": [{"hour": 2, "q": [...]}, ..., {"hour": 24, "q": [...]}]}`.
It is a pure function of the case file — no randomness, no clock — so
two runs on the same case write identical bytes. A case takes ~0.1 s,
far inside the 8 s limit.

## Method

The case carries its own training set: the *comparables*, ~40 past big
game launches with hourly Twitch category averages (data: SullyGnome,
as the kit's NOTICE.txt describes) spanning each launch's release day.

A launch day, measured in log space against its own first hour, has a
characteristic shape, and launches whose first hour was about the same
size share it — the first hour already encodes how much of the world
was unlocked and watching at 00:00 UTC under a rolling midnight-local
launch. So:

1. For every comparable, take its release day's log-ratios
   `r_h = ln(viewers_h / viewers_1)` for hours 2..24.
2. Weight each comparable by a Gaussian kernel on the log distance
   between its first hour and this case's first hour:
   `w = exp(-0.5 * (ln(v1_comp / v1_case) / 1.0) ** 2)`.
3. Per hour, read the five levels off the weighted empirical
   distribution of `r_h` (linear interpolation on the midpoint CDF)
   and scale back: `q_tau(h) = p1 * exp(Q_tau(r_h))`, where `p1` is
   this case's hour-1 print.

Fallbacks per hour, in order: unweighted pool if the weighted pool is
effectively empty, then a persistence row (flat at `p1`, log-normal
spread `0.15 * sqrt(h - 1)`). The target game's own pre-release history
entry among the comparables is never pooled — it is not a launch.

## Validation

Leave-one-out over the 37 real launches in the practice cases'
comparables (each launch in turn is the target, trained on the other
36, scored exactly as the contest scores):

|                        | mean pinball loss |
|------------------------|-------------------|
| this program           | **0.400**         |
| house baseline         | 0.930             |
| this program, targets with hour 1 ≥ 20k viewers (n=27) | **0.257** |
| house baseline, same subset | 0.321        |

At the scales closest to GTA VI's expected first hour the margin is
wide (ELDEN RING 0.105 vs 0.211, Cyberpunk 2077 0.144 vs 0.230,
Hogwarts Legacy 0.136 vs 0.288). Because the forecast is anchored
multiplicatively on hour 1 in log space, a target far larger than
every comparable is handled the same way: the kernel concentrates on
the largest launches and the ratios, not the levels, carry over.

## License

MIT (see LICENSE).
