"""Paired comparison of two precisions graded on the same items.

Each item is graded once at each precision, so the evidence is the 2x2 table of right and wrong, and only the
items the two disagree on carry information: `lost` (the reference right, the candidate wrong) and `gained` (the
reverse). On Qwen2.5-14B at 24 GB, GPTQ against fp8 on 200 GSM8K problems lost 4 and gained none.

Two questions can be asked of that table, and they have different answers:

- Is there a difference? McNemar's exact test (`mcnemar_p`). 4 against 0 gives p = 0.125, "not significant",
  which says nothing about whether the loss is small: a significance test admits anything a small sample cannot
  convict.
- Is the loss within a tolerance? The lower end of a confidence interval for the difference in accuracy
  (`score_bounds`). Here it is about -5.0 points, so a 3-point tolerance cannot be certified, while a 6-point one
  can. A small sample refuses rather than admits, which is the direction a quality gate has to fail in.

The interval is Tango's score interval for a difference of paired proportions (Tango 1998, Statistics in
Medicine 17:891). The usual Wald interval is no use here: with nothing lost and nothing gained its width is
zero at any sample size, where the score interval on 300 items is still +/-1.3 points.

Proportions are fractions throughout (0.02 is two points); the difference is candidate minus reference.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

Z95 = 1.959963984540054  # two-sided 95%: each bound is a one-sided 97.5% bound
Z80_POWER = 0.8416212335729143  # the normal quantile for 80% power


def mcnemar_p(lost: int, gained: int) -> float:
    """Two-sided exact McNemar test: how lopsided the split is among problems only one side got right."""
    n = lost + gained
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(lost, gained) + 1)) / 2 ** n)


def restricted_lost_rate(d: float, lost: float, gained: float, n: float) -> float:
    """The maximum-likelihood probability of a lost item when the true difference in accuracy is fixed at `d`.

    With the lost probability p and the gained probability p + d, the likelihood of the table is
    p^lost (p + d)^gained (1 - 2p - d)^(n - lost - gained); setting its derivative to zero gives the quadratic
    2n p^2 + [d(2n + lost - gained) - (lost + gained)] p - lost d (1 - d) = 0, whose positive root this is.
    Counts may be fractional (expected counts, for `items_needed`)."""
    a = 2.0 * n
    b = d * (2.0 * n + lost - gained) - (lost + gained)
    c = -lost * d * (1.0 - d)
    disc = max(0.0, b * b - 4.0 * a * c)
    return max(0.0, (-b + math.sqrt(disc)) / (2.0 * a))


def score_z(d: float, lost: float, gained: float, n: float) -> float:
    """Tango's score statistic for the hypothesis that the difference in accuracy is `d`.

    Positive when the observed difference (gained - lost) / n lies above `d`. It falls as `d` rises, so each
    bound of the interval is where it crosses +/-1.96."""
    p = restricted_lost_rate(d, lost, gained, n)
    var = n * (2.0 * p + d * (1.0 - d))
    num = gained - lost - n * d
    if var <= 0.0:
        return math.copysign(math.inf, num) if num else 0.0
    return num / math.sqrt(var)


def _crossing(lo: float, hi: float, target: float, lost: float, gained: float, n: float) -> float:
    """The d in [lo, hi] where score_z(d) = target, by bisection (score_z decreases in d)."""
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if score_z(mid, lost, gained, n) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return (lo + hi) / 2.0


def score_bounds(lost: int, gained: int, n: int, z: float = Z95) -> Tuple[float, float]:
    """95% score interval (L, U) for the candidate's accuracy minus the reference's, from paired counts.

    L is where the score statistic falls to +z below the observed difference, U where it reaches -z above it.
    With no items at all nothing is known, and the interval is the whole range."""
    if n <= 0:
        return -1.0, 1.0
    if lost < 0 or gained < 0 or lost + gained > n:
        raise ValueError(f"impossible paired counts: {lost} lost and {gained} gained of {n}")
    observed = (gained - lost) / n
    lower = -1.0 if lost == n else _crossing(-1.0, observed, z, lost, gained, n)
    upper = 1.0 if gained == n else _crossing(observed, 1.0, -z, lost, gained, n)
    return lower, upper


def certifies(lost: int, gained: int, n: int, tolerance: float) -> bool:
    """Whether the evidence rules out a loss larger than `tolerance` (a fraction): the lower bound lies above it."""
    return n > 0 and score_bounds(lost, gained, n)[0] > -tolerance


def items_needed(lost: int, gained: int, n: int, tolerance: float, power_z: float = Z80_POWER) -> Optional[int]:
    """Roughly how many items it takes to certify `tolerance` with 80% probability, if the observed rates of
    lost and gained items are the true ones. None when the observed loss is already at or past the tolerance,
    since no number of items certifies that.

    This is the score-method sample size for a paired non-inferiority test (Nam 1997, Biometrics 53:1422):
    n = (1.96 s0 + 0.84 s1)^2 / (tolerance - loss)^2, with s1^2 = discordance - loss^2 the variance per item
    at the observed rates and s0^2 its value at the boundary, from the restricted estimate above. The simpler
    (1.96 + 0.84)^2 s1^2 / (tolerance - loss)^2 uses s1 for both and says zero items when nothing was
    discordant, which is wrong: identical answers on 100 items still leave a 3.7-point lower bound."""
    if n <= 0:
        return None
    p_lost, p_gained = lost / n, gained / n
    loss = p_lost - p_gained
    if loss >= tolerance:
        return None
    s1 = math.sqrt(max(0.0, p_lost + p_gained - loss * loss))
    p0 = restricted_lost_rate(-tolerance, p_lost, p_gained, 1.0)
    s0 = math.sqrt(max(0.0, 2.0 * p0 - tolerance * (1.0 + tolerance)))
    return max(1, math.ceil(((Z95 * s0 + power_z * s1) / (tolerance - loss)) ** 2))
