"""GTA VI trophy rarity forecast: earners over tracked, saturating curves.

For every trophy the earn rate is earners / players-tracked, and the two
sides move for different reasons: new players flood the tracker (which
dilutes late trophies toward the floor) while early-story earners pile
up fast. So this program forecasts the two sides separately with
saturating exponentials fitted to the case history, then divides at the
target time:

  N(t)  = A + B * exp(-t / tau_n)              (players tracked)
  E(t)  = E_inf * (1 - exp(-max(0, t - d) / tau))   (earners, per trophy)

The (delay, tau) pair for each trophy is picked by grid search around a
group prior read off the trophy's description (Chapter 1 and early
activities fast, later chapters and the final mission slower, gold and
platinum slowest, side and grind trophies in between), scored by the
logit error of the fitted rate against the reads. A trophy with too
little usable history falls back to persistence (last read, random-walk
spread). After the fits, story nesting is enforced on the medians: a
later chapter cannot sit above an earlier one, and the platinum cannot
sit above any trophy. Quantiles put the median on the logit scale with
a spread that combines the random-walk term with the fit residual.

Runs as `python3 -P -s /bot/main.py`, standard library only, no
network. Reads every <id>.in in the input directory and writes the
matching <id>.out: {"trophies": [{"id", "q": [q05, q25, q50, q75,
q95]}]}, 55 rows, non-decreasing, each 0..100, at most 65,536 bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]
WALK_SIGMA = 0.5          # logit spread per sqrt-day, random-walk term
RESID_FLOOR = 0.35        # logit floor for trend uncertainty
RESID_CAP = 2.5           # logit cap on the residual term
LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc)


def parse_time(text):
    """ISO timestamp -> aware datetime (accepts a trailing Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_launch(moment):
    """Hours from the 2026-11-19T00:00Z launch, never below zero."""
    return max(0.0, (moment - LAUNCH).total_seconds() / 3600.0)


def clamp_pct(value):
    return min(99.99, max(0.01, value))


def to_logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def from_logit(value):
    return 100.0 / (1.0 + math.exp(-value))


def round_half_up_4(value):
    """Round to 4 decimals, half up; whole numbers come back as int."""
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 10000.0
    return int(out) if out.is_integer() else out


def fit_line_through_basis(xs, ys):
    """Least squares y ~ a + b*x; returns (a, b). Degenerate -> flat."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom < 1e-12:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return my - b * mx, b


def fit_tracked(reads):
    """Saturating fit for players tracked; returns a function N(t)."""
    times = [r["t"] for r in reads]
    counts = [float(r["players"]) for r in reads]
    best = None
    for tau in (6.0, 12.0, 18.0, 24.0, 36.0, 48.0, 72.0, 96.0, 144.0, 240.0):
        basis = [math.exp(-t / tau) for t in times]
        a, b = fit_line_through_basis(basis, counts)
        err = sum((c - (a + b * e)) ** 2 for c, e in zip(counts, basis))
        if best is None or err < best[0]:
            best = (err, tau, a, b)
    _, tau, a, b = best
    return lambda t: max(1.0, a + b * math.exp(-t / tau))


def group_prior(trophy):
    """(delay_h, tau_h) prior from the description and tier.

    Early story and early activities start fast; each later chapter
    starts later and stretches longer; gold/platinum are the slowest;
    side and grind trophies sit in the middle.
    """
    desc = trophy["description"].lower()
    if "chapter 1" in desc:
        return (2.0, 36.0)
    if "chapter 2" in desc:
        return (10.0, 48.0)
    if "chapter 3" in desc:
        return (24.0, 60.0)
    if "chapter 4" in desc or "final mission" in desc:
        return (48.0, 72.0)
    if trophy["tier"] in ("gold", "platinum"):
        return (72.0, 96.0)
    if "fishing" in desc or "breakfast run" in desc or "arcade" in desc:
        return (2.0, 36.0)
    return (18.0, 72.0)


def fit_earners(window, n_at, delay0, tau0):
    """Grid-search (delay, tau) for one trophy's earners curve.

    window: reads with (t, earners) in the fitting window. E_inf is
    solved by least squares for each grid point; the winner minimises
    the mean squared logit error of the fitted rate. Returns
    (delay, tau, e_inf, resid_rmse) or None when nothing fits.
    """
    delays = sorted({max(0.0, delay0 * f) for f in (0.5, 1.0, 2.0)})
    taus = sorted({tau0 * f for f in (0.6, 1.0, 1.7)})
    best = None
    for delay in delays:
        for tau in taus:
            fracs = [1.0 - math.exp(-max(0.0, r["t"] - delay) / tau)
                     for r in window]
            if fracs[-1] < 0.02:
                continue
            denom = sum(f * f for f in fracs)
            if denom < 1e-9:
                continue
            e_inf = sum(r["e"] * f for r, f in zip(window, fracs)) / denom
            sq = 0.0
            for r, f in zip(window, fracs):
                pred = 100.0 * e_inf * f / n_at(r["t"])
                sq += (to_logit(r["rate"]) - to_logit(pred)) ** 2
            loss = sq / len(window)
            if best is None or loss < best[0]:
                best = (loss, delay, tau, e_inf)
    if best is None:
        return None
    loss, delay, tau, e_inf = best
    return (delay, tau, e_inf, math.sqrt(loss))


def quantiles_for(median, spread):
    centre = to_logit(median)
    return [round_half_up_4(from_logit(centre + z * spread)) for z in Z_SCORES]


def persistence_row(rate, horizon_days):
    spread = WALK_SIGMA * math.sqrt(horizon_days)
    return quantiles_for(rate, spread)


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError(f"levels {case['levels']} are not {LEVELS}")
    history = case["history"]
    if not history:
        return {"trophies": [{"id": t["id"], "q": [0.01, 0.01, 0.01, 0.01, 0.01]}
                              for t in case["trophies"]]}
    horizon_days = case["horizon_minutes"] / 1440.0
    target_t = hours_since_launch(parse_time(case["target"]))
    reads = []
    for h in history:
        reads.append({
            "t": hours_since_launch(parse_time(h["at"])),
            "players": h["players_tracked"],
            "rates": h["rates"],
        })
    n_at = fit_tracked(reads)
    n_target = n_at(target_t)
    last_t = reads[-1]["t"]
    window_start = last_t - 36.0

    medians = []
    spreads = []
    for i, trophy in enumerate(case["trophies"]):
        last_rate = reads[-1]["rates"][i]
        # The always-owned trophy (about 100% in the last read and in
        # the large majority of reads) stays put: its earners are the
        # tracked players themselves, so dilution cannot move it.
        owned = [r["rates"][i] for r in reads if r["rates"][i] is not None]
        if (last_rate is not None and last_rate >= 99.99 and owned
                and sum(1 for v in owned if v >= 99.99) / len(owned) >= 0.8):
            medians.append(99.99)
            spreads.append(math.sqrt((WALK_SIGMA * math.sqrt(horizon_days)) ** 2
                                     + RESID_FLOOR ** 2))
            continue
        window = []
        for r in reads:
            rate = r["rates"][i]
            if r["t"] < window_start or rate is None or r["players"] <= 0:
                continue
            window.append({"t": r["t"], "rate": rate,
                           "e": rate / 100.0 * r["players"]})
        fitted = None
        if len(window) >= 3:
            delay0, tau0 = group_prior(trophy)
            fitted = fit_earners(window, n_at, delay0, tau0)
        if fitted is None:
            medians.append(clamp_pct(last_rate if last_rate is not None else 0.01))
            spreads.append(WALK_SIGMA * math.sqrt(horizon_days))
            continue
        delay, tau, e_inf, resid = fitted
        e_target = e_inf * (1.0 - math.exp(-max(0.0, target_t - delay) / tau))
        medians.append(clamp_pct(100.0 * e_target / n_target))
        u = min(RESID_CAP, max(RESID_FLOOR, resid))
        spreads.append(math.sqrt((WALK_SIGMA * math.sqrt(horizon_days)) ** 2 + u ** 2))

    # Story nesting: a later chapter cannot sit above an earlier one,
    # and the platinum cannot sit above any other trophy.
    def find(sub):
        for i, t in enumerate(case["trophies"]):
            if sub in t["description"].lower():
                return i
        return None

    chain = [find("complete chapter 1."), find("complete chapter 2."),
             find("complete chapter 3."), find("complete chapter 4.")]
    chain = [c for c in chain if c is not None]
    for earlier, later in zip(chain, chain[1:]):
        medians[later] = min(medians[later], medians[earlier])
    final_idx = None
    for i, t in enumerate(case["trophies"]):
        if t["description"].lower() == "complete the final mission.":
            final_idx = i
    if final_idx is not None and chain:
        medians[final_idx] = min(medians[final_idx], medians[chain[-1]])
    plat_idx = None
    for i, t in enumerate(case["trophies"]):
        if t["tier"] == "platinum":
            plat_idx = i
    if plat_idx is not None:
        others = [m for j, m in enumerate(medians) if j != plat_idx]
        medians[plat_idx] = min(medians[plat_idx], min(others))

    rows = []
    for trophy, median, spread in zip(case["trophies"], medians, spreads):
        rows.append({"id": trophy["id"],
                     "q": quantiles_for(median, spread)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
