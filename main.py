"""Launch-day Twitch audience forecaster (gta6-twitch-ccu-forecast-01).

Method
------
The case carries its own training set: the `comparables`, past big game
launches with hourly Twitch category averages from 14 days before their
release day to 3 days after it. A launch day, measured in log space against
its own first hour, has a characteristic shape: launches whose first hour
was about the same size tend to share that shape, because the first hour
already encodes how much of the world was unlocked and watching at 00:00
UTC (a rolling midnight-local launch lights New Zealand, Europe and the
UK before the US wakes).

So for every comparable we take the log-ratios

    r_h = ln(viewers_h / viewers_1)        for h = 2..24

of its release day, weight each comparable by a Gaussian kernel on the
log distance between its first hour and this case's first hour,

    w = exp(-0.5 * (ln(v1_comp / v1_case) / 1.25) ** 2),

and read the five wanted levels off the weighted empirical distribution
of r_h, hour by hour:

    q_tau(h) = p1 * exp(Q_tau(r_h))

where p1 is this case's hour-1 print. Everything is a pure function of
the case file: no randomness, no clock, no network, standard library
only, so two runs on the same case write the same bytes.

Fallbacks, in order, per hour: if an hour's weighted pool is effectively
empty, use the unweighted pool of that hour's ratios; if that is empty
too, use the house-style persistence row (flat at p1 with a log-normal
spread of 0.15 * sqrt(h - 1)). The target game's own pre-release history
entry among the comparables is never pooled: it is not a launch.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard-normal quantiles at LEVELS, for the persistence fallback row.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
FALLBACK_SIGMA = 0.15
# Kernel bandwidth on the log hour-1 distance between comparable and case.
BANDWIDTH = 1.25


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_day(hourly, day_start):
    """Map hour-of-day 1..24 -> mean viewers of that hour of the release day."""
    start = parse_ts(day_start)
    by_hour = {}
    for point in hourly:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        hour = int((parse_ts(point["from"]) - start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            by_hour.setdefault(hour, []).append(float(viewers))
    return {h: sum(v) / len(v) for h, v in by_hour.items()}


def weighted_quantile(pairs, tau):
    """Quantile of a weighted sample, linear on the midpoint CDF.

    `pairs` is a list of (value, weight). Each point stands for the CDF
    step centred on its own cumulative midpoint, which keeps the estimate
    smooth for the small pools (a few dozen launches) this case gives us.
    """
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    cum = 0.0
    prev_value, prev_cdf = ordered[0][0], 0.0
    for value, weight in ordered:
        cum += weight
        cdf = (cum - 0.5 * weight) / total
        if cdf >= tau:
            if cdf == prev_cdf:
                return value
            return prev_value + (value - prev_value) * (tau - prev_cdf) / (cdf - prev_cdf)
        prev_value, prev_cdf = value, cdf
    return ordered[-1][0]


def round2(x):
    """Round half up to 2 decimals; whole numbers serialise without a fraction."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def persistence_row(p1, hour):
    spread = FALLBACK_SIGMA * math.sqrt(hour - 1)
    return [p1 * math.exp(z * spread) for z in Z]


def forecast_case(case):
    first = next((k for k in case["known"] if k.get("hour") == 1), case["known"][0])
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)
    game_name = (case.get("game") or {}).get("name", "")

    # Build the pool: one (log-ratio curve, weight) per comparable launch.
    pool = []
    for comp in case.get("comparables") or []:
        if game_name and str(comp.get("name", "")).startswith(game_name):
            continue  # the target's own pre-release history is not a launch
        day = release_day(comp.get("hourly") or [], comp["day_start"])
        v1 = day.get(1)
        if not v1 or v1 <= 0:
            continue  # no first hour to anchor on: its ratios are meaningless
        weight = math.exp(-0.5 * ((math.log(v1) - log_p1) / BANDWIDTH) ** 2)
        ratios = {h: math.log(v / v1) for h, v in day.items() if h >= 2 and v > 0}
        pool.append((ratios, weight))

    rows = []
    for want in case["forecast"]:
        hour = want["hour"]
        weighted = [(r[hour], w) for r, w in pool if hour in r]
        total_weight = sum(w for _, w in weighted)
        if total_weight > 1e-9:
            quantiles = [p1 * math.exp(weighted_quantile(weighted, tau)) for tau in LEVELS]
        elif weighted:
            plain = [(r, 1.0) for r, _ in weighted]
            quantiles = [p1 * math.exp(weighted_quantile(plain, tau)) for tau in LEVELS]
        else:
            quantiles = persistence_row(p1, hour)
        # Guard the contract: positive, at most 1e9, non-decreasing.
        fixed = []
        for q in quantiles:
            q = min(1e9, max(1e-6, q))
            if fixed and q < fixed[-1]:
                q = fixed[-1]
            fixed.append(round2(q))
        rows.append({"hour": hour, "q": fixed})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
