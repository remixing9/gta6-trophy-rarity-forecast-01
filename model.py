"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, the
percent of the players Exophase tracks who will have earned it at the case's
target, as five quantiles a trophy, scored by a pinball loss on the logit
scale.

The model, per trophy
---------------------
Exophase's page is a step function: it refreshes its counts on its own
schedule (hours apart), so a case's history is long flat stretches and then
jumps. The forecaster therefore works on the *distinct* reads (consecutive
identical rates collapsed to their first appearance) and treats each trophy's
earn rate as a saturating curve in log-time:

    r(t) = A * sigmoid(u(t)),   u(t) = a + b * ln(1 + t)

where t is hours since launch, A the trophy's eventual (asymptote) rate, and
u the "progress logit". In u-space the curve is a straight line, so a
recency-weighted least-squares fit gives the slope; the forecast anchors at
the last observed rate and extrapolates the shrunk slope to the target, and
the asymptote A (a category prior, raised when the trophy already beats it)
keeps the forecast from running past what the trophy can reach. Slopes are
shrunk toward their slope family's mean slope, so a trophy with a flat or
noisy history borrows the typical growth of trophies like it -- except a
trophy whose history never moved borrows only its own family's slope, and
a family with no rising trophy assumes no growth (a platinum must not
inherit the story chapters' rise). The spread combines the starter's
random-walk diffusion, the slope's standard error times the extrapolation
distance, and a refresh-jump allowance (a flat history hides jumps, so the
spread never collapses to the random walk alone). The spread is the
standard deviation of the rate's logit at the target, the scale the scorer
uses, so the quantiles are the median's logit plus z times the spread; the
cap binds the median only, not the upper quantiles. A blend with a
linear-log-time model was tried (w = 0.7, 0.85, 0.9) and rejected: it hurts
on saturating worlds and only ties on a late-surge world once the no-fit
borrowing rule is in place, so the median is the saturating model's
extrapolation alone.

The entrypoint (main.py) reads every <id>.in in the input directory and
writes <id>.out to the output directory. Standard library only, no network,
deterministic: the same input bytes give the same output bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path: put it back
# first, so `import model` finds the model.py shipped beside main.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The persistence baseline's spread, logit units per square-root day.
BASELINE_SIGMA = 0.5
# A rate is clamped to [RATE_FLOOR, RATE_CEIL] percent before its logit
# (the scorer's clamp: the page prints 2 decimals, 0% and 100% have no logit).
RATE_FLOOR = 0.01
RATE_CEIL = 99.99
# Grand Theft Auto VI released 2026-11-19; t counts hours from its launch.
DEFAULT_LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc)
# Recency weight time constant for the trajectory fit, hours.
RECENCY_HOURS = 12.0
# Shrinkage strength: a trophy's slope counts as n/(n + SHRINK_K) its own.
SHRINK_K = 3.0
# The spread's floor and cap, logit units.
SPREAD_FLOOR = 0.45
SPREAD_CAP = 2.5
# A flat history hides refresh jumps, so the refresh allowance never drops
# below this, logit units; the observed jumps cap it.
REFRESH_FLOOR = 0.5
REFRESH_CAP = 1.0


# ------------------------------------------------------------------ basics

def clamp_rate(pct):
    """A rate clamped to the scorer's [0.01, 99.99] percent."""
    return min(RATE_CEIL, max(RATE_FLOOR, pct))


def logit(pct):
    """logit of a percent: ln(c / (100 - c)) of c clamped to [0.01, 99.99]."""
    c = clamp_rate(pct)
    return math.log(c / (100.0 - c))


def inv_logit(x):
    """Inverse logit, as a percent in (0, 100)."""
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def sigmoid(x):
    """The logistic function, as a fraction in (0, 1)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def norm_cdf(x):
    """The standard normal's CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p):
    """The standard normal's quantile.

    Acklam's rational approximation, then Newton steps on the CDF (which
    uses erf, accurate to machine precision) so the result is exact to
    ~1e-12, not just Acklam's ~3e-5.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf needs a probability in (0, 1), got {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    for _ in range(3):
        pdf = math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)
        x -= (norm_cdf(x) - p) / pdf
    return x


# ------------------------------------------------------------------ scoring

def pinball_loss(y, q, tau):
    """The contest's pinball loss on the logit scale for one item.

    2 * max(tau * d, (tau - 1) * d), d = logit(y) - logit(q). A quantile
    below the rate costs 2 * tau per logit unit it is short, one above it
    2 * (1 - tau) per unit it is over.
    """
    d = logit(y) - logit(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score_case(truth_rates, forecast):
    """Mean pinball loss over every trophy and quantile of one case.

    truth_rates is in the case's trophy order, as forecast's rows are.
    """
    total, n = 0.0, 0
    for y, row in zip(truth_rates, forecast["trophies"]):
        for tau, qv in zip(LEVELS, row["q"]):
            total += pinball_loss(y, qv, tau)
            n += 1
    return total / n if n else float("nan")


def baseline_row(rate, horizon_minutes):
    """The house persistence baseline's row for one trophy (the starter's).

    Every rate stays where the last read put it; the spread widens like a
    random walk's on the logit scale. Kept for tests and comparison.
    """
    spread = BASELINE_SIGMA * math.sqrt(horizon_minutes / 1440)
    lg = logit(rate if rate is not None else 0)
    return [round4(inv_logit(lg + z * spread)) for z in Z]


# ------------------------------------------------------------------ priors

def slope_family(trophy):
    """The trophy's family for slope borrowing: trophies that grow alike.

    Coarser than the (cap, prior) category: all story chapters share one
    family, so a chapter that has not started rising yet borrows the rise
    of the chapters before it; the platinum stands alone, so it borrows
    nothing.
    """
    tier = (trophy.get("tier") or "").lower()
    desc = trophy.get("description") or ""
    if tier == "platinum":
        return "platinum"
    if "Complete Chapter " in desc:
        return "chapter"
    if tier == "gold":
        return "gold"
    if tier == "silver":
        return "silver"
    if desc.startswith("Complete "):
        return "mission"
    return "activity"


def category_cap_prior(trophy):
    """(cap, prior): the trophy's ceiling and its eventual rate, in percent.

    The cap is the largest rate the trophy can plausibly reach in a week;
    the prior is the share of tracked players expected ever to earn it.
    Both come from the trophy's tier and what its description asks for:
    story chapters in order, the final mission, 100% completion, grinds and
    collections. They are priors, not answers: the fit moves the forecast,
    and a trophy already past its prior raises its own asymptote.
    """
    tier = (trophy.get("tier") or "").lower()
    desc = trophy.get("description") or ""
    name = trophy.get("name") or ""
    if tier == "platinum":
        return (1.0, 0.2)
    if "Chapter 1" in desc:
        return (85.0, 65.0)
    if "Chapter 2" in desc:
        return (70.0, 50.0)
    if "Chapter 3" in desc:
        return (60.0, 38.0)
    if "Chapter 4" in desc:
        return (50.0, 28.0)
    if tier == "gold":
        if "100% Completion" in desc:
            return (2.0, 0.5)
        return (30.0, 12.0)  # the final mission
    if "Try each type of activity once" in desc:
        return (60.0, 30.0)
    if tier == "silver":
        return (20.0, 8.0)
    if desc.startswith("Complete "):
        return (55.0, 25.0)
    return (45.0, 15.0)


# ------------------------------------------------------------------ series

def _parse_time(text):
    """An ISO timestamp with a Z suffix, as a timezone-aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_launch(text, launch=DEFAULT_LAUNCH):
    """Hours from the game's launch to a timestamp."""
    return (_parse_time(text) - launch).total_seconds() / 3600.0


def distinct_series(history, idx, launch=DEFAULT_LAUNCH):
    """The trophy's distinct reads: (hours since launch, rate).

    Exophase's page is a step function, so consecutive identical rates are
    one refresh: they collapse to their first appearance, which is the best
    estimate of when the refresh landed.
    """
    pts = []
    last = None
    for r in history:
        rate = r["rates"][idx]
        if rate is None:
            continue
        if last is not None and rate == last:
            continue
        pts.append((hours_since_launch(r["slot"], launch), float(rate)))
        last = rate
    return pts


def _wls_line(xs, ys, ws):
    """Weighted least squares of y on x. Returns (slope, intercept, se, resid).

    se is the slope's standard error from the weighted residual variance;
    resid is that variance's square root. With fewer than 3 points the
    residual is not identifiable, so se and resid come back as None.
    """
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return (None, None, None, None)
    xbar = sum(w * x for x, w in zip(xs, ws)) / sw
    ybar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 0:
        return (0.0, ybar, None, None)
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    if len(xs) < 3:
        return (slope, intercept, None, None)
    resid2 = sum(w * (y - (intercept + slope * x)) ** 2
                 for x, y, w in zip(xs, ys, ws))
    dof = max(sw - 2.0, 1.0)
    var = resid2 / dof
    se = math.sqrt(var / sxx) if sxx > 0 else None
    return (slope, intercept, se, math.sqrt(var))


# ------------------------------------------------------------------ forecast

def _asymptote(trophy, last_rate):
    """The trophy's asymptote: its prior, raised if it already beats it.

    The observed rate is a lower bound on what the trophy can reach, so an
    asymptote below it is impossible; but when the rate already beats the
    cap, the cap was wrong and the asymptote follows the observed rate (up
    to 100), not the cap.
    """
    cap, prior = category_cap_prior(trophy)
    asym = max(prior, last_rate * 1.05)
    if asym > cap:
        asym = min(last_rate * 1.05, 100.0)
    return max(asym, RATE_FLOOR * 2)


def _u_of(rate, asym):
    """The progress logit: logit of the rate as a fraction of its asymptote."""
    frac = min(max(rate / asym, RATE_FLOOR / asym), 0.999)
    return math.log(frac / (1.0 - frac))


def _fit_slope(series, asym):
    """The recency-weighted slope of u on ln(1 + t), with its std error."""
    if len(series) < 2:
        return (None, None)
    xs = [math.log1p(t) for t, _ in series]
    ys = [_u_of(r, asym) for _, r in series]
    t_last = series[-1][0]
    ws = [math.exp(-(t_last - t) / RECENCY_HOURS) for t, _ in series]
    slope, _, se, _ = _wls_line(xs, ys, ws)
    return (slope, se)


def _forecast_one(trophy, series, last_rate, x_target, horizon_days,
                  cat_slope, cat_se, cat_slope_nofit, refresh_jump):
    """One trophy's (median, spread): median in percent, spread in logit.

    The median is the saturating model's extrapolation: the rate follows
    r(t) = A*sigmoid(a + b*ln(1 + t)), fitted on distinct reads. A blend
    with a linear-log-time model was tried (w = 0.7, 0.85, 0.9) and
    rejected: it hurts on saturating worlds and only ties on a late-surge
    world once the no-fit borrowing rule below is in place. Slopes are
    shrunk toward their family's mean slope. A trophy whose history never
    moved borrows only its own family's slope (cat_slope_nofit): if its
    family has no rising trophy, it assumes no growth, rather than
    borrowing the whole field's growth -- a platinum must not inherit the
    story chapters' rise. The spread is the standard deviation of the
    rate's logit at the target (the scale the scorer uses), combining the
    random-walk diffusion, the slope's standard error times the
    extrapolation distance (converted from u-space to logit-rate units,
    which compresses near the asymptote, as it should: a rate near its
    ceiling cannot move much further), and the refresh allowance. The cap
    binds the median only: the upper quantiles may pass it, which is the
    honest statement when the cap itself is a guess.
    """
    cap, _ = category_cap_prior(trophy)
    asym = _asymptote(trophy, last_rate)

    if last_rate >= RATE_CEIL:
        # Already at 100%: it stays there, with a small spread for dilution
        # as new players sync who have not done the intro yet.
        return (99.9, 0.3)

    # The last distinct read's first appearance: the anchor time. The page
    # may have sat unrefreshed since, so the forecast extrapolates from
    # when the current value first appeared, not from the checkpoint.
    x_anchor = math.log1p(series[-1][0]) if series else x_target
    u_last = _u_of(last_rate, asym)

    slope, se = _fit_slope(series, asym)
    n_eff = len(series)
    if slope is None:
        b_star = cat_slope_nofit
        se_star = cat_se if cat_se is not None else 0.5
    else:
        w_self = n_eff / (n_eff + SHRINK_K)
        b_star = w_self * slope + (1.0 - w_self) * cat_slope
        se_star = (se if se is not None else 0.5) * w_self + 0.15

    u_star = u_last + b_star * (x_target - x_anchor)
    median = asym * sigmoid(u_star)
    median = min(max(median, RATE_FLOOR), RATE_CEIL)
    median = min(median, cap)

    # The spread is computed at the saturating median: d logit(rate) / d u
    # is measured on the saturating model's trajectory.
    conv = 100.0 * (1.0 - median / asym) / (100.0 - median) if median < asym else 0.0
    conv = min(max(conv, 0.0), 1.0)
    s_rw = BASELINE_SIGMA * math.sqrt(horizon_days)
    s_fit = min(se_star * (x_target - x_anchor) * conv, 1.5)
    s_ref = refresh_jump
    spread = math.sqrt(s_rw * s_rw + s_fit * s_fit + s_ref * s_ref)
    spread = min(max(spread, SPREAD_FLOOR), SPREAD_CAP)
    return (median, spread)


def forecast_case(case):
    """A case's forecast: {"trophies": [{"id", "q": [q05..q95]}, ...]}."""
    if case.get("levels", LEVELS) != LEVELS:
        raise ValueError(f"case asks for levels {case.get('levels')}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    release = (case.get("game") or {}).get("release_date") or "2026-11-19"
    launch = datetime.fromisoformat(release + "T00:00:00+00:00")
    x_target = math.log1p(max(hours_since_launch(case["target"], launch), 0.0))
    horizon_days = max(case["horizon_minutes"], 0) / 1440.0

    # Every trophy's distinct series and last rate, and the pooled refresh
    # jumps (the median absolute logit jump between distinct reads).
    series_list = []
    last_rates = []
    for i in range(len(trophies)):
        series_list.append(distinct_series(history, i, launch))
        rates = [r["rates"][i] for r in history if r["rates"][i] is not None]
        last_rates.append(float(rates[-1]) if rates else 0.0)
    jumps = []
    for series in series_list:
        for (_, r0), (_, r1) in zip(series, series[1:]):
            jumps.append(abs(logit(r1) - logit(r0)))
    jumps.sort()
    if jumps:
        refresh_jump = min(max(jumps[len(jumps) // 2], 0.3), REFRESH_CAP)
    else:
        refresh_jump = REFRESH_FLOOR

    # Family slopes for shrinkage: the inverse-variance mean u-space
    # slope of the trophies in the same slope family; the global mean as
    # the fallback.
    fits = []
    for k, trophy in enumerate(trophies):
        slope, se = _fit_slope(series_list[k], _asymptote(trophy, last_rates[k]))
        fits.append((k, slope, se))

    def _cat_means(fit_list):
        by_cat = {}
        for k, slope, se in fit_list:
            if slope is None:
                continue
            fam = slope_family(trophies[k])
            by_cat.setdefault(fam, []).append((slope, se))
        means = {}
        for fam, items in by_cat.items():
            num = sum(s / (se * se) for s, se in items if se)
            den = sum(1.0 / (se * se) for _, se in items if se)
            means[fam] = num / den if den else sum(s for s, _ in items) / len(items)
        return means

    cat_mean = _cat_means(fits)
    slopes = [s for _, s, _ in fits if s is not None]
    ses = [se for _, _, se in fits if se is not None]
    global_slope = sum(slopes) / len(slopes) if slopes else 0.0
    global_se = sorted(ses)[len(ses) // 2] if ses else 0.5

    rows = []
    for k, trophy in enumerate(trophies):
        fam = slope_family(trophy)
        cat_slope = cat_mean.get(fam, global_slope)
        # A trophy with no fitted slope borrows only its own family's
        # slope; a family with no rising trophy assumes no growth.
        cat_slope_nofit = cat_mean.get(fam, 0.0)
        median, spread = _forecast_one(
            trophy, series_list[k], last_rates[k], x_target,
            horizon_days, cat_slope, global_se, cat_slope_nofit, refresh_jump)
        q = [round4(inv_logit(logit(median) + z * spread)) for z in Z]
        # Non-decreasing, in [0, 100]: the spread is non-negative and the
        # z's increase, so this holds by construction; enforce it anyway.
        q = [min(max(v, 0), 100) for v in q]
        q = sorted(q)
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
