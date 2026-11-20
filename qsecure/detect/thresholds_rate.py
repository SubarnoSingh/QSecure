"""Derived thresholds for the rate and count statistics.

Examples
--------
>>> from qsecure.detect.thresholds_rate import matched_count_threshold
>>> from qsecure.protocol.params import DEFAULT_PARAMS
>>> from qsecure.protocol.verify import (
...     HONEST_ABORT_BUDGET, minimum_matched_count
... )
>>> low = matched_count_threshold(
...     DEFAULT_PARAMS.signing_length,
...     DEFAULT_PARAMS.match_probability,
...     eps=HONEST_ABORT_BUDGET,
...     side="lower",
...     method="chernoff",
... )
>>> low.critical_count, minimum_matched_count(DEFAULT_PARAMS) - 1
(36554, 36554)

>>> sharp = matched_count_threshold(
...     DEFAULT_PARAMS.signing_length,
...     DEFAULT_PARAMS.match_probability,
...     eps=HONEST_ABORT_BUDGET,
...     side="lower",
... )
>>> sharp.critical_count, sharp.inequality
(36951, 'exact binomial tail')
>>> sharp.false_positive_bound <= HONEST_ABORT_BUDGET
True

>>> [
...     matched_count_threshold(
...         192, 1 / 3, eps=e, side="upper"
...     ).critical_count
...     for e in (1e-3, 1e-6, 1e-9)
... ]
[86, 97, 106]

>>> import numpy as np
>>> from qsecure.detect.statistics import TranscriptStatistics
>>> from qsecure.detect.thresholds_rate import RateCountThresholds
>>> from qsecure.protocol.params import ProtocolParams
>>> from qsecure.protocol.session import QDSSession
>>> stats = TranscriptStatistics.from_transcript(
...     QDSSession(
...         ProtocolParams(key_length=192), rng=np.random.default_rng(7)
...     ).run(0)
... )
>>> family = RateCountThresholds.for_transcript(stats, eps=1e-9)
>>> verdict = family.evaluate(stats)
>>> verdict.flagged, verdict.fired
(False, ())
>>> f"{family.false_positive_bound:.3e}"
'4.804e-10'
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from qsecure.detect.statistics import CountStatistic, TranscriptStatistics
from qsecure.protocol.params import VERIFIERS, Party, ProtocolParams
from qsecure.protocol.verify import (
    HONEST_ABORT_BUDGET,
    minimum_matched_count,
    minimum_pooled_matched_count,
)

__all__ = [
    "EXACT_TRIALS_LIMIT",
    "INEQUALITIES",
    "RATE_COUNT_FREE_TESTS",
    "RATE_COUNT_ROSTER",
    "DerivedThreshold",
    "RateCountThresholds",
    "RateCountVerdict",
    "binomial_tail_bound",
    "critical_count",
    "declared_count_threshold",
    "dominance_noise_level",
    "floor_comparison",
    "forgery_separation_sigma",
    "matched_count_threshold",
    "mismatch_rate_threshold",
    "pooled_count_threshold",
]


INEQUALITIES: Final[tuple[str, ...]] = ("exact", "hoeffding", "chernoff")

#: Human-readable names, so a :class:`DerivedThreshold` prints the proof that
#: produced it rather than a slug.
_INEQUALITY_NAMES: Final[Mapping[str, str]] = {
    "exact": "exact binomial tail",
    "hoeffding": "Hoeffding",
    "chernoff": "multiplicative Chernoff",
}

EXACT_TRIALS_LIMIT: Final[int] = 4_000_000

RATE_COUNT_ROSTER: Final[tuple[str, ...]] = (
    "mismatch_rate:Bob",
    "mismatch_rate:Charlie",
    "matched_count_low:Bob",
    "matched_count_low:Charlie",
    "matched_count_high:Bob",
    "matched_count_high:Charlie",
    "declared_count_low:Bob",
    "declared_count_low:Charlie",
    "declared_count_high:Bob",
    "declared_count_high:Charlie",
    "pooled_count_low",
    "pooled_count_high",
)

RATE_COUNT_FREE_TESTS: Final[tuple[str, ...]] = ("declaration_gap",)

_SIDES: Final[frozenset[str]] = frozenset({"lower", "upper"})
_METHODS: Final[frozenset[str]] = frozenset(INEQUALITIES) | {"sharpest"}

_TAIL_SLACK: Final[float] = 1e-18


def _as_count(value: Any, name: str) -> int:
    """Coerce a non-negative integer count.

    Parameters
    ----------
    value : Any
        The candidate.
    name : str
        Field name, quoted in error messages.

    Returns
    -------
    int

    Raises
    ------
    TypeError
        If ``value`` is not an integer.
    ValueError
        If ``value`` is negative.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{name} must be an int, got {type(value).__name__}. Counts here "
            f"are numbers of positions, and a float count would silently make "
            f"an integer threshold non-integer."
        )
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def _as_probability(value: Any, name: str) -> float:
    """Coerce a probability in ``[0, 1]``.

    Parameters
    ----------
    value : Any
        The candidate.
    name : str
        Field name, quoted in error messages.

    Returns
    -------
    float

    Raises
    ------
    TypeError
        If ``value`` is not a real number.
    ValueError
        If ``value`` lies outside ``[0, 1]`` or is not finite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be a real number, got {type(value).__name__}."
        )
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(
            f"{name} must lie in [0, 1], got {number}. It is a probability, "
            f"not a percentage: pass 0.01, not 1."
        )
    return number


def _as_budget(value: Any, name: str = "eps") -> float:
    """Coerce a false-positive budget in the open interval ``(0, 1)``.

    Parameters
    ----------
    value : Any
        The candidate.
    name : str, optional
        Field name, quoted in error messages.

    Returns
    -------
    float

    Raises
    ------
    TypeError
        If ``value`` is not a real number.
    ValueError
        If ``value`` is not strictly inside ``(0, 1)``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be a real number, got {type(value).__name__}."
        )
    number = float(value)
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise ValueError(
            f"{name} must lie strictly inside (0, 1), got {number}. It is a "
            f"false-positive budget: a budget of 0 buys a detector that never "
            f"fires and a budget of 1 buys one that fires on everything, and "
            f"neither is a threshold."
        )
    return number


def _check_side(side: Any) -> str:
    """Validate the tail name.

    Parameters
    ----------
    side : Any
        ``'lower'`` or ``'upper'``.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If ``side`` names no tail.
    """
    if not isinstance(side, str) or side not in _SIDES:
        raise ValueError(
            f"side must be one of {sorted(_SIDES)}, got {side!r}. A one-sided "
            f"detector must name the tail it tests; a two-sided threshold "
            f"would spend half its budget on a tail nobody looks at."
        )
    return side


def _check_method(method: Any) -> str:
    """Validate the inequality selector.

    Parameters
    ----------
    method : Any
        One of :data:`INEQUALITIES` or ``'sharpest'``.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If ``method`` names no inequality.
    """
    if not isinstance(method, str) or method not in _METHODS:
        raise ValueError(
            f"method must be one of {sorted(_METHODS)}, got {method!r}. "
            f"'sharpest' consults every inequality at the same eps and keeps "
            f"the most sensitive threshold any of them proves; see D-5."
        )
    return method


def _log_pmf(successes: int, trials: int, probability: float) -> float:
    """Return ``log P(S = successes)`` for ``S ~ Binomial(trials, probability)``.

    Parameters
    ----------
    successes : int
        ``0 <= successes <= trials``.
    trials : int
        At least ``1``.
    probability : float
        Strictly inside ``(0, 1)``; the degenerate cases are handled by the
        callers, exactly.

    Returns
    -------
    float
    """
    return (
        math.lgamma(trials + 1.0)
        - math.lgamma(successes + 1.0)
        - math.lgamma(trials - successes + 1.0)
        + successes * math.log(probability)
        + (trials - successes) * math.log1p(-probability)
    )


def _upper_tail(count: int, trials: int, probability: float) -> float:
    """Return an upper bound on ``P(S >= count)``, tight to rounding.

    Parameters
    ----------
    count : int
        The tail's left endpoint, included.
    trials : int
        At least ``1``.
    probability : float
        Strictly inside ``(0, 1)``.

    Returns
    -------
    float
    """
    if count <= 0:
        return 1.0
    if count > trials:
        return 0.0
    odds = probability / (1.0 - probability)
    total = 0.0
    index = count
    while index <= trials:
        term = math.exp(_log_pmf(index, trials, probability))
        total += term
        if index < trials:
            ratio = ((trials - index) / (index + 1.0)) * odds
            if ratio < 1.0:
                remainder = term * ratio / (1.0 - ratio)
                if remainder <= _TAIL_SLACK * max(total, 5e-324):
                    total += remainder
                    break
        index += 1
    return min(1.0, total)


def _lower_tail(count: int, trials: int, probability: float) -> float:
    """Return an upper bound on ``P(S <= count)``, tight to rounding.

    Parameters
    ----------
    count : int
        The tail's right endpoint, included.
    trials : int
        At least ``1``.
    probability : float
        Strictly inside ``(0, 1)``.

    Returns
    -------
    float
    """
    if count < 0:
        return 0.0
    if count >= trials:
        return 1.0
    odds = (1.0 - probability) / probability
    total = 0.0
    index = count
    while index >= 0:
        term = math.exp(_log_pmf(index, trials, probability))
        total += term
        if index > 0:
            ratio = (index / (trials - index + 1.0)) * odds
            if ratio < 1.0:
                remainder = term * ratio / (1.0 - ratio)
                if remainder <= _TAIL_SLACK * max(total, 5e-324):
                    total += remainder
                    break
        index -= 1
    return min(1.0, total)


def binomial_tail_bound(
    count: int,
    trials: int,
    probability: float,
    *,
    side: str,
    method: str = "sharpest",
) -> float:
    """Bound the one-sided null probability of the observed count.

    Parameters
    ----------
    count : int
        ``0 <= count <= trials``.
    trials : int
        At least ``0``.
    probability : float
        The null's per-trial success probability, in ``[0, 1]``.
    side : {'lower', 'upper'}
        Keyword-only.
    method : {'sharpest', 'exact', 'hoeffding', 'chernoff'}, optional
        Keyword-only.

    Returns
    -------
    float
        A number in ``[0, 1]``, always an upper bound on the true tail.

    Raises
    ------
    TypeError
        If a count is not an integer or ``probability`` is not a real number.
    ValueError
        If ``count > trials``, ``probability`` is outside ``[0, 1]``, or
        ``side``/``method`` names nothing.

    Examples
    --------
    >>> from qsecure.detect.thresholds_rate import binomial_tail_bound
    >>> f"{binomial_tail_bound(0, 192, 1 / 3, side='lower'):.3e}"
    '1.551e-34'
    >>> f"{binomial_tail_bound(0, 192, 1/3, side='lower', method='chernoff'):.3e}"
    '1.266e-14'

    >>> counts = [
    ...     binomial_tail_bound(128, 192, 1 / 3, side="upper", method=m)
    ...     for m in ("exact", "hoeffding", "chernoff")
    ... ]
    >>> [f"{c:.2e}" for c in counts]
    ['4.39e-21', '2.95e-19', '5.43e-10']

    >>> binomial_tail_bound(1, 5000, 0.0, side="upper")
    0.0
    >>> binomial_tail_bound(0, 5000, 0.0, side="upper")
    1.0
    """
    total = _as_count(trials, "trials")
    hits = _as_count(count, "count")
    if hits > total:
        raise ValueError(
            f"count ({hits}) exceeds trials ({total}); a count above its own "
            f"denominator is a counting error upstream, not a deviation."
        )
    chance = _as_probability(probability, "probability")
    tail = _check_side(side)
    how = _check_method(method)

    if total == 0:
        return 1.0
    if chance == 0.0:
        if tail == "upper":
            return 0.0 if hits >= 1 else 1.0
        return 1.0
    if chance == 1.0:
        if tail == "lower":
            return 0.0 if hits <= total - 1 else 1.0
        return 1.0

    candidates = INEQUALITIES if how == "sharpest" else (how,)
    best = 1.0
    for name in candidates:
        if name == "exact" and how == "sharpest" and total > EXACT_TRIALS_LIMIT:
            continue
        best = min(best, _one_tail_bound(hits, total, chance, tail, name))
    return best


def _one_tail_bound(
    count: int, trials: int, probability: float, side: str, method: str
) -> float:
    """Apply one named inequality to one tail.

    Parameters
    ----------
    count : int
        The observation.
    trials : int
        At least ``1``.
    probability : float
        Strictly inside ``(0, 1)``.
    side : {'lower', 'upper'}
        Which tail.
    method : {'exact', 'hoeffding', 'chernoff'}
        Which inequality.

    Returns
    -------
    float
    """
    mean = trials * probability
    if method == "exact":
        if side == "lower":
            return _lower_tail(count, trials, probability)
        return _upper_tail(count, trials, probability)
    if method == "hoeffding":
        gap = mean - count if side == "lower" else count - mean
        if gap <= 0.0:
            return 1.0
        return min(1.0, math.exp(-2.0 * gap * gap / trials))
    deviation = abs(count - mean) / mean
    if side == "lower":
        if count >= mean:
            return 1.0
        if deviation > 1.0:
            return 0.0
        return min(1.0, math.exp(-deviation * deviation * mean / 2.0))
    if count <= mean:
        return 1.0
    return min(
        1.0, math.exp(-deviation * deviation * mean / (2.0 + deviation))
    )


@dataclass(frozen=True)
class _Critical:
    """One inverted threshold: the count, the proof and the realised bound."""

    count: int
    method: str
    bound: float
    vacuous: bool


def _chernoff_critical(
    trials: int, probability: float, eps: float, side: str
) -> int:
    """Invert the multiplicative Chernoff bound at ``eps``.

    Parameters
    ----------
    trials : int
        At least ``1``.
    probability : float
        Strictly inside ``(0, 1)``.
    eps : float
        Strictly inside ``(0, 1)``.
    side : {'lower', 'upper'}
        Which tail.

    Returns
    -------
    int
        ``-1`` for a lower tail the form cannot certify at this budget, where
        the derived detector is silent rather than clamped.
    """
    mean = trials * probability
    log_inverse = -math.log(eps)
    if side == "lower":
        slack = 2.0 * log_inverse
        if slack >= mean:
            return -1
        deviation = math.sqrt(slack / mean)
        return math.floor((1.0 - deviation) * mean)
    deviation = (
        log_inverse
        + math.sqrt(log_inverse * log_inverse + 8.0 * mean * log_inverse)
    ) / (2.0 * mean)
    return math.ceil((1.0 + deviation) * mean)


def _hoeffding_critical(
    trials: int, probability: float, eps: float, side: str
) -> int:
    """Invert the Hoeffding bound at ``eps``.

    Parameters
    ----------
    trials : int
        At least ``1``.
    probability : float
        Strictly inside ``(0, 1)``.
    eps : float
        Strictly inside ``(0, 1)``.
    side : {'lower', 'upper'}
        Which tail.

    Returns
    -------
    int
        May be negative for a lower tail, which the caller reads as "silent".
    """
    mean = trials * probability
    spread = math.sqrt(trials * -math.log(eps) / 2.0)
    if side == "lower":
        return math.floor(mean - spread)
    return math.ceil(mean + spread)


def _exact_critical(
    trials: int, probability: float, eps: float, side: str
) -> int:
    """Invert the exact binomial tail at ``eps``, by bisection.

    Parameters
    ----------
    trials : int
        At least ``1``.
    probability : float
        Strictly inside ``(0, 1)``.
    eps : float
        Strictly inside ``(0, 1)``.
    side : {'lower', 'upper'}
        Which tail.

    Returns
    -------
    int
    """
    if side == "lower":
        if _lower_tail(0, trials, probability) > eps:
            return -1
        low, high = 0, trials - 1
        while low < high:
            middle = (low + high + 1) // 2
            if _lower_tail(middle, trials, probability) <= eps:
                low = middle
            else:
                high = middle - 1
        return low
    low, high = 1, trials + 1
    while low < high:
        middle = (low + high) // 2
        if _upper_tail(middle, trials, probability) <= eps:
            high = middle
        else:
            low = middle + 1
    return low


def _invert(
    trials: int, probability: float, eps: float, side: str, method: str
) -> _Critical:
    """Return the sharpest critical count the requested proofs certify.

    Parameters
    ----------
    trials : int
        At least ``0``.
    probability : float
        In ``[0, 1]``.
    eps : float
        Strictly inside ``(0, 1)``.
    side : {'lower', 'upper'}
        Which tail.
    method : {'sharpest', 'exact', 'hoeffding', 'chernoff'}
        Which proofs to consult.

    Returns
    -------
    _Critical
    """
    named = _INEQUALITY_NAMES.get(method, method)
    silent_low = _Critical(-1, named, 0.0, True)
    silent_high = _Critical(trials + 1, named, 0.0, True)
    if trials == 0:
        empty = "no trials (the null has no denominator)"
        if side == "lower":
            return _Critical(-1, empty, 0.0, True)
        return _Critical(1, empty, 0.0, True)
    if probability == 0.0:
        if side == "upper":
            return _Critical(1, "point mass at 0", 0.0, False)
        return _Critical(-1, "point mass at 0", 0.0, True)
    if probability == 1.0:
        if side == "lower":
            return _Critical(trials - 1, "point mass at N", 0.0, False)
        return _Critical(trials + 1, "point mass at N", 0.0, True)

    candidates = INEQUALITIES if method == "sharpest" else (method,)
    # Start from the silent threshold on this side, so the loop only ever
    # improves on "never fires" and there is no None to guard afterwards.
    best = silent_low if side == "lower" else silent_high
    for name in candidates:
        if (
            name == "exact"
            and method == "sharpest"
            and trials > EXACT_TRIALS_LIMIT
        ):
            continue
        if name == "exact":
            found = _exact_critical(trials, probability, eps, side)
        elif name == "hoeffding":
            found = _hoeffding_critical(trials, probability, eps, side)
        else:
            found = _chernoff_critical(trials, probability, eps, side)
        if side == "lower":
            found = max(found, -1)
            if found <= best.count and not best.vacuous:
                continue
        else:
            found = min(found, trials + 1)
            if found >= best.count and not best.vacuous:
                continue
        vacuous = found < 0 if side == "lower" else found > trials
        if vacuous:
            continue
        best = _Critical(
            found,
            _INEQUALITY_NAMES[name],
            _one_tail_bound(found, trials, probability, side, name),
            False,
        )
    return best


def critical_count(
    trials: int,
    probability: float,
    *,
    eps: float,
    side: str,
    method: str = "sharpest",
) -> int:
    """Return the extreme count a false-positive budget of ``eps`` certifies.

    Parameters
    ----------
    trials : int
        The null's denominator, at least ``0``.
    probability : float
        The null's per-trial success probability, in ``[0, 1]``.
    eps : float
        Keyword-only false-positive budget, strictly inside ``(0, 1)``.
    side : {'lower', 'upper'}
        Keyword-only.
    method : {'sharpest', 'exact', 'hoeffding', 'chernoff'}, optional
        Keyword-only.

    Returns
    -------
    int
        ``-1`` on a lower tail, or ``trials + 1`` on an upper tail, when the
        requested proof certifies nothing at this budget.

    Raises
    ------
    TypeError
        If a count is not an integer or a probability is not a real number.
    ValueError
        If ``probability`` or ``eps`` is out of range, or ``side``/``method``
        names nothing.

    Examples
    --------
    >>> from qsecure.detect.thresholds_rate import critical_count
    >>> critical_count(192, 1 / 3, eps=1e-9, side="lower")
    26
    >>> critical_count(192, 1 / 3, eps=1e-9, side="upper")
    106

    >>> [
    ...     critical_count(192, 1 / 3, eps=e, side="lower")
    ...     for e in (1e-2, 1e-4, 1e-8, 1e-16)
    ... ]
    [48, 40, 29, 15]

    >>> critical_count(192, 1 / 3, eps=2.0**-64, side="lower", method="chernoff")
    -1
    >>> critical_count(192, 1 / 3, eps=2.0**-64, side="lower", method="exact")
    11
    """
    total = _as_count(trials, "trials")
    chance = _as_probability(probability, "probability")
    budget = _as_budget(eps)
    tail = _check_side(side)
    how = _check_method(method)
    return _invert(total, chance, budget, tail, how).count


@dataclass(frozen=True)
class DerivedThreshold:
    """One threshold, its null, the proof behind it and what it costs.

    Raises
    ------
    TypeError
        If a count is not an integer or a probability is not a real number.
    ValueError
        If ``side`` names no tail, ``null``/``derivation`` is empty, or
        :attr:`false_positive_bound` exceeds :attr:`budget` -- which would mean
        the derivation and the arithmetic disagree, and is refused at
        construction rather than reported downstream.

    Examples
    --------
    >>> from qsecure.detect.thresholds_rate import matched_count_threshold
    >>> threshold = matched_count_threshold(192, 1 / 3, eps=1e-9, side="upper")
    >>> threshold.critical_count, threshold.inequality
    (106, 'exact binomial tail')
    >>> f"{threshold.false_positive_bound:.3e}"
    '4.048e-10'
    >>> threshold.fires(128), threshold.fires(64)
    (True, False)
    >>> print(threshold.claim())
    matched_count_high [Bob]: fires when |M_R| >= 106 out of 192; under Binomial(192, 1/3), the honest law of a verifier's matched count, this fires with probability at most 4.048e-10, by the exact binomial tail at a budget of 1.000e-09.
    """

    name: str
    statistic: str
    null: str
    inequality: str
    side: str
    budget: float
    trials: int
    null_probability: float
    critical_count: int
    false_positive_bound: float
    derivation: str
    vacuous: bool

    def __post_init__(self) -> None:
        """Coerce the fields and refuse a bound that exceeds its own budget."""
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "statistic", str(self.statistic))
        object.__setattr__(self, "null", str(self.null))
        object.__setattr__(self, "inequality", str(self.inequality))
        object.__setattr__(self, "side", _check_side(self.side))
        object.__setattr__(self, "budget", _as_budget(self.budget, "budget"))
        object.__setattr__(self, "trials", _as_count(self.trials, "trials"))
        object.__setattr__(
            self,
            "null_probability",
            _as_probability(self.null_probability, "null_probability"),
        )
        if isinstance(self.critical_count, bool) or not isinstance(
            self.critical_count, int
        ):
            raise TypeError(
                f"{self.name}: critical_count must be an int, got "
                f"{type(self.critical_count).__name__}. Not coerced: a "
                f"threshold silently truncated from 5.7 to 5 is a different "
                f"rule with a different false-positive probability, and "
                f"nothing downstream would say so."
            )
        object.__setattr__(
            self,
            "false_positive_bound",
            _as_probability(
                self.false_positive_bound, "false_positive_bound"
            ),
        )
        object.__setattr__(self, "derivation", str(self.derivation))
        object.__setattr__(self, "vacuous", bool(self.vacuous))
        if not self.null or not self.derivation:
            raise ValueError(
                f"{self.name}: a threshold must carry both its null and its "
                f"derivation. a number with neither is a "
                f"constant somebody chose, and there is no way to tell it "
                f"apart from one afterwards."
            )
        if self.false_positive_bound > self.budget:
            raise ValueError(
                f"{self.name}: proven false-positive bound "
                f"{self.false_positive_bound!r} exceeds its budget "
                f"{self.budget!r}. The inversion and the forward bound "
                f"disagree, which is an arithmetic error in this module and "
                f"not a property of the run."
            )

    @property
    def critical_rate(self) -> float | None:
        """float or None: :attr:`critical_count` as a rate; ``None`` at no trials."""
        if self.trials == 0:
            return None
        return self.critical_count / self.trials

    @property
    def expected(self) -> float:
        """float: ``trials * null_probability``, the null's mean."""
        return self.trials * self.null_probability

    def fires(self, count: int) -> bool:
        """Return whether an observed count trips this threshold.

        Parameters
        ----------
        count : int
            The observation, non-negative.

        Returns
        -------
        bool
            ``count <= critical_count`` on a lower tail, ``count >=
            critical_count`` on an upper one.

        Raises
        ------
        TypeError
            If ``count`` is not an integer.
        ValueError
            If ``count`` is negative.
        """
        observed = _as_count(count, "count")
        if self.side == "lower":
            return observed <= self.critical_count
        return observed >= self.critical_count

    def claim(self) -> str:
        """Return the null-and-derivation sentence, as one line.

        Returns
        -------
        str
            *"... under the null this fires with probability at most X, and here is
            why"*, with the null, the bound and the inequality named.

        Examples
        --------
        >>> from qsecure.detect.thresholds_rate import (
        ...     matched_count_threshold, mismatch_rate_threshold
        ... )
        >>> print(mismatch_rate_threshold(59, eps=1e-9).claim())
        mismatch_rate [Bob]: fires when e_R >= 1 out of 59; under a point mass at 0 (e_R = 0 with probability one on a noiseless link), this fires with probability at most 0.000e+00, by the point mass at 0 at a budget of 1.000e-09.

        >>> print(
        ...     matched_count_threshold(
        ...         192, 1 / 3, eps=2.0**-64, side="lower", method="chernoff"
        ...     ).claim()
        ... )
        matched_count_low [Bob]: never fires -- no inequality certifies a threshold on Binomial(192, 1/3), the honest law of a verifier's matched count, at a budget of 5.421e-20, so this fires with probability exactly 0, and detects nothing. Unreachable critical count -1 out of 192.
        """
        base, _, party = self.name.partition(":")
        label = f"{base} [{party}]" if party else base
        if self.vacuous:
            return (
                f"{label}: never fires -- no inequality certifies a threshold "
                f"on {self.null}, at a budget of {self.budget:.3e}, so this "
                f"fires with probability exactly 0, and detects nothing. "
                f"Unreachable critical count {self.critical_count} out of "
                f"{self.trials}."
            )
        comparison = "<=" if self.side == "lower" else ">="
        return (
            f"{label}: fires when {self.statistic} "
            f"{comparison} {self.critical_count} out of {self.trials}"
            f"; under {self.null}, this fires with probability at most "
            f"{self.false_positive_bound:.3e}, by the {self.inequality} at a "
            f"budget of {self.budget:.3e}."
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
            Every field, plus :attr:`critical_rate`, :attr:`expected` and
            :meth:`claim`.
        """
        return {
            "name": self.name,
            "statistic": self.statistic,
            "null": self.null,
            "inequality": self.inequality,
            "side": self.side,
            "budget": self.budget,
            "trials": self.trials,
            "null_probability": self.null_probability,
            "critical_count": self.critical_count,
            "critical_rate": self.critical_rate,
            "expected": self.expected,
            "false_positive_bound": self.false_positive_bound,
            "derivation": self.derivation,
            "vacuous": self.vacuous,
            "claim": self.claim(),
        }


def _count_threshold(
    *,
    name: str,
    statistic: str,
    null: str,
    trials: int,
    probability: float,
    eps: float,
    side: str,
    method: str,
    extra: str = "",
) -> DerivedThreshold:
    """Build a :class:`DerivedThreshold` from one binomial inversion.

    Parameters
    ----------
    name : str
        Roster name.
    statistic : str
        What is compared, in words.
    null : str
        The distribution, written out.
    trials : int
        The null's denominator.
    probability : float
        The null's per-trial success probability.
    eps : float
        This threshold's share of the budget.
    side : {'lower', 'upper'}
        Which tail.
    method : str
        Which proofs to consult.
    extra : str, optional
        A sentence appended to the derivation, for a threshold with a
        provenance worth carrying (the pooled count's conservation argument,
        for instance).

    Returns
    -------
    DerivedThreshold
    """
    total = _as_count(trials, "trials")
    chance = _as_probability(probability, "probability")
    budget = _as_budget(eps)
    tail = _check_side(side)
    how = _check_method(method)
    found = _invert(total, chance, budget, tail, how)
    mean = total * chance
    reference = "D-1" if tail == "lower" else "D-2"
    algebra = (
        f"mu = {total} * {chance:.6f} = {mean:.4f}; "
        f"ln(1/eps) = {-math.log(budget):.4f}; "
        f"{reference} inverted by the {found.method} gives k = "
        f"{found.count}, at which the same argument bounds the honest firing "
        f"probability by {found.bound:.6e} <= eps = {budget:.6e}."
    )
    if extra:
        algebra = f"{algebra} {extra}"
    return DerivedThreshold(
        name=name,
        statistic=statistic,
        null=null,
        inequality=found.method,
        side=tail,
        budget=budget,
        trials=total,
        null_probability=chance,
        critical_count=found.count,
        false_positive_bound=found.bound,
        derivation=algebra,
        vacuous=found.vacuous,
    )


def matched_count_threshold(
    signing_length: int,
    match_probability: float,
    *,
    eps: float,
    side: str,
    method: str = "sharpest",
    party: Party | str | None = None,
) -> DerivedThreshold:
    """Derive a threshold on one verifier's matched count ``|M_R|``.

    Parameters
    ----------
    signing_length : int
        ``n``.
    match_probability : float
        ``1/|B|``.
    eps : float
        Keyword-only false-positive budget for **this** threshold.
    side : {'lower', 'upper'}
        Keyword-only.
    method : {'sharpest', 'exact', 'hoeffding', 'chernoff'}, optional
        Keyword-only.
    party : Party or str, optional
        Keyword-only.

    Returns
    -------
    DerivedThreshold

    Raises
    ------
    TypeError
        If ``signing_length`` is not an integer or a probability is not a real
        number.
    ValueError
        If a probability or ``eps`` is out of range, or ``side``/``method``
        names nothing.

    Examples
    --------
    >>> from qsecure.detect.thresholds_rate import matched_count_threshold
    >>> from qsecure.protocol.params import DEFAULT_PARAMS, ProtocolParams
    >>> from qsecure.protocol.verify import (
    ...     HONEST_ABORT_BUDGET, minimum_matched_count
    ... )
    >>> for params in (DEFAULT_PARAMS, ProtocolParams(key_length=600)):
    ...     derived = matched_count_threshold(
    ...         params.signing_length,
    ...         params.match_probability,
    ...         eps=HONEST_ABORT_BUDGET,
    ...         side="lower",
    ...         method="chernoff",
    ...     )
    ...     print(derived.critical_count, minimum_matched_count(params) - 1)
    36554 36554
    66 66

    >>> sharper = matched_count_threshold(
    ...     600, 1 / 3, eps=HONEST_ABORT_BUDGET, side="lower"
    ... )
    >>> sharper.critical_count, sharper.inequality
    (100, 'exact binomial tail')

    >>> high = matched_count_threshold(192, 1 / 3, eps=1e-9, side="upper")
    >>> high.critical_count, high.expected
    (106, 64.0)
    >>> high.fires(128)          # the forger's mean, 2n/3
    True
    >>> f"{high.false_positive_bound:.3e}"
    '4.048e-10'
    """
    label = "Bob" if party is None else str(party)
    suffix = "low" if _check_side(side) == "lower" else "high"
    chance = _as_probability(match_probability, "match_probability")
    total = _as_count(signing_length, "signing_length")
    return _count_threshold(
        name=f"matched_count_{suffix}:{label}",
        statistic="|M_R|",
        null=(
            f"Binomial({total}, {_fraction(chance)}), the honest law of a "
            f"verifier's matched count"
        ),
        trials=total,
        probability=chance,
        eps=eps,
        side=side,
        method=method,
        extra=(
            "The null is unconditional over the symmetrisation coins: a swap "
            "permutes an exchangeable pair and leaves the joint law alone. "
            "Conditionally on the records m_C = M - m_B exactly, which is a "
            "different law and is not reachable from a transcript."
        ),
    )


def declared_count_threshold(
    signing_length: int,
    match_probability: float,
    *,
    eps: float,
    side: str,
    method: str = "sharpest",
    party: Party | str | None = None,
) -> DerivedThreshold:
    """Derive a threshold on a Phase C' declared matched count.

    Parameters
    ----------
    signing_length : int
        ``n``, sifted.
    match_probability : float
        ``1/|B|``.
    eps : float
        Keyword-only budget for this threshold.
    side : {'lower', 'upper'}
        Keyword-only.
    method : {'sharpest', 'exact', 'hoeffding', 'chernoff'}, optional
        Keyword-only.
    party : Party or str, optional
        Keyword-only, names the roster entry.

    Returns
    -------
    DerivedThreshold

    Raises
    ------
    TypeError
        If ``signing_length`` is not an integer or a probability is not real.
    ValueError
        If a probability or ``eps`` is out of range, or ``side``/``method``
        names nothing.

    Examples
    --------
    >>> from qsecure.detect.thresholds_rate import declared_count_threshold
    >>> starve = declared_count_threshold(
    ...     192, 1 / 3, eps=1e-9, side="lower", party="Charlie"
    ... )
    >>> starve.name, starve.critical_count
    ('declared_count_low:Charlie', 26)
    >>> starve.fires(20), starve.fires(64)
    (True, False)
    """
    label = "Bob" if party is None else str(party)
    suffix = "low" if _check_side(side) == "lower" else "high"
    chance = _as_probability(match_probability, "match_probability")
    total = _as_count(signing_length, "signing_length")
    return _count_threshold(
        name=f"declared_count_{suffix}:{label}",
        statistic="the count declared in Phase C'",
        null=(
            f"Binomial({total}, {_fraction(chance)}), the honest law of the "
            f"count a recipient declares, which on an honest run is the count "
            f"he then scores"
        ),
        trials=total,
        probability=chance,
        eps=eps,
        side=side,
        method=method,
        extra=(
            "Same null and same inversion as the matched count; a separate "
            "roster entry because a starver moves the wire integer while "
            "leaving his own verdict count alone."
        ),
    )


def pooled_count_threshold(
    signing_length: int,
    match_probability: float,
    *,
    eps: float,
    side: str,
    method: str = "sharpest",
) -> DerivedThreshold:
    """Derive a threshold on the pooled matched count ``M = m_B + m_C``.

    Parameters
    ----------
    signing_length : int
        ``n`` per verifier, sifted.
    match_probability : float
        ``1/|B|``.
    eps : float
        Keyword-only budget for this threshold.
    side : {'lower', 'upper'}
        Keyword-only.
    method : {'sharpest', 'exact', 'hoeffding', 'chernoff'}, optional
        Keyword-only.

    Returns
    -------
    DerivedThreshold

    Raises
    ------
    TypeError
        If ``signing_length`` is not an integer or a probability is not real.
    ValueError
        If a probability or ``eps`` is out of range, or ``side``/``method``
        names nothing.

    Examples
    --------
    >>> from qsecure.detect.thresholds_rate import pooled_count_threshold
    >>> from qsecure.protocol.params import DEFAULT_PARAMS
    >>> from qsecure.protocol.verify import (
    ...     HONEST_ABORT_BUDGET, minimum_pooled_matched_count
    ... )
    >>> pooled = pooled_count_threshold(
    ...     DEFAULT_PARAMS.signing_length,
    ...     DEFAULT_PARAMS.match_probability,
    ...     eps=HONEST_ABORT_BUDGET,
    ...     side="lower",
    ...     method="chernoff",
    ... )
    >>> pooled.trials, pooled.critical_count
    (230400, 74189)
    >>> minimum_pooled_matched_count(DEFAULT_PARAMS) - 1
    74189

    >>> pooled_count_threshold(192, 1 / 3, eps=1e-9, side="lower").trials
    384
    """
    chance = _as_probability(match_probability, "match_probability")
    half = _as_count(signing_length, "signing_length")
    suffix = "low" if _check_side(side) == "lower" else "high"
    return _count_threshold(
        name=f"pooled_count_{suffix}",
        statistic="M = m_B + m_C",
        null=(
            f"Binomial({2 * half}, {_fraction(chance)}), exact by "
            f"conservation of the pooled count under the symmetrisation coins"
        ),
        trials=2 * half,
        probability=chance,
        eps=eps,
        side=side,
        method=method,
        extra=(
            "Exactness comes from conservation, not from independence: the "
            "coins re-assign a fixed multiset of 2n entries, so M is a sum of "
            "2n i.i.d. indicators. Given the records m_C = M - m_B with "
            "correlation -1, so convolving the marginals would prove nothing."
        ),
    )


def mismatch_rate_threshold(
    matched_count: int,
    *,
    eps: float,
    channel_error_rate: float = 0.0,
    method: str = "sharpest",
    party: Party | str | None = None,
) -> DerivedThreshold:
    """Derive a threshold on one verifier's mismatch rate ``r_R``.

    Parameters
    ----------
    matched_count : int
        ``|M_R|``, this verifier's realised matched-set size.
    eps : float
        Keyword-only budget for this threshold.
    channel_error_rate : float, optional
        Keyword-only ``p_e``, in ``[0, 1]``.
    method : {'sharpest', 'exact', 'hoeffding', 'chernoff'}, optional
        Keyword-only.
    party : Party or str, optional
        Keyword-only, names the roster entry.

    Returns
    -------
    DerivedThreshold
        Always an upper tail: a mismatch *deficit* is not an attack anyone has
        to detect, and a two-sided threshold would spend half its budget on it.

    Raises
    ------
    TypeError
        If ``matched_count`` is not an integer or a probability is not real.
    ValueError
        If a probability or ``eps`` is out of range, or ``method`` names
        nothing.

    Examples
    --------
    >>> from qsecure.detect.thresholds_rate import mismatch_rate_threshold
    >>> clean = mismatch_rate_threshold(38400, eps=1e-9)
    >>> clean.critical_count, clean.false_positive_bound
    (1, 0.0)
    >>> f"{clean.critical_rate:.3e}"
    '2.604e-05'

    >>> noisy = mismatch_rate_threshold(
    ...     38400, eps=1e-9, channel_error_rate=0.005
    ... )
    >>> noisy.critical_count, f"{noisy.critical_rate:.6f}"
    (281, '0.007318')

    >>> design = mismatch_rate_threshold(
    ...     38400, eps=1e-9, channel_error_rate=0.03125
    ... )
    >>> f"{design.critical_rate:.6f}", 1 / 64
    ('0.036745', 0.015625)
    """
    label = "Bob" if party is None else str(party)
    total = _as_count(matched_count, "matched_count")
    noise = _as_probability(channel_error_rate, "channel_error_rate")
    if noise == 0.0:
        null = (
            "a point mass at 0 (e_R = 0 with probability one on a noiseless "
            "link)"
        )
        extra = (
            "Degenerate null, answered exactly rather than bounded: every "
            "matched position reproduces the declared eigenvalue with "
            "probability one on a noiseless link, so P(e_R >= 1) = 0 and the "
            "false-positive probability is exactly zero -- under that null. "
            "The cost is that it is a claim about a noiseless link."
        )
    else:
        null = (
            f"Binomial({total}, {noise:.6g}) conditionally on |M_R| = "
            f"{total}, carried through by the tower rule"
        )
        extra = (
            "Conditioning, not fitting: the threshold is a function of the "
            "conditioning variable |M_R| and of eps, never of e_R, so "
            "P(fire) = E[P(e_R >= k(|M_R|) | |M_R|)] <= eps."
        )
    return _count_threshold(
        name=f"mismatch_rate:{label}",
        statistic="e_R",
        null=null,
        trials=total,
        probability=noise,
        eps=eps,
        side="upper",
        method=method,
        extra=extra,
    )


def _fraction(probability: float) -> str:
    """Render a match probability as a readable fraction where it is one.

    Parameters
    ----------
    probability : float
        In ``(0, 1]``.

    Returns
    -------
    str
        ``"1/3"`` rather than ``"0.3333333333333333"`` when the reciprocal is
        an integer to within floating-point rounding, so a printed null reads
        like the null it is.
    """
    if probability <= 0.0:
        return "0"
    reciprocal = 1.0 / probability
    rounded = round(reciprocal)
    if rounded >= 1 and abs(reciprocal - rounded) < 1e-9:
        return f"1/{rounded}"
    return f"{probability:.6g}"


def forgery_separation_sigma(
    signing_length: int,
    match_probability: float,
    *,
    pooled: bool = False,
) -> float:
    """Return the recipient forgery's separation, in honest standard deviations.

    Parameters
    ----------
    signing_length : int
        ``n``, sifted, at least ``1``.
    match_probability : float
        ``1/|B|``, strictly inside ``(0, 1)``.
    pooled : bool, optional
        Keyword-only.

    Returns
    -------
    float

    Raises
    ------
    TypeError
        If ``signing_length`` is not an integer or ``match_probability`` is not
        a real number.
    ValueError
        If ``signing_length`` is ``0`` or ``match_probability`` is not strictly
        inside ``(0, 1)``.

    Examples
    --------
    >>> from qsecure.detect.thresholds_rate import forgery_separation_sigma
    >>> from qsecure.protocol.params import DEFAULT_PARAMS
    >>> f"{forgery_separation_sigma(192, 1 / 3):.4f}"
    '9.7980'
    >>> forgery_separation_sigma(
    ...     DEFAULT_PARAMS.signing_length, DEFAULT_PARAMS.match_probability
    ... )
    240.0

    >>> import math
    >>> per = forgery_separation_sigma(192, 1 / 3)
    >>> both = forgery_separation_sigma(192, 1 / 3, pooled=True)
    >>> f"{both:.4f}", f"{per / both:.15f}" == f"{math.sqrt(2):.15f}"
    ('6.9282', True)
    """
    total = _as_count(signing_length, "signing_length")
    chance = _as_probability(match_probability, "match_probability")
    if total == 0:
        raise ValueError(
            "signing_length must be at least 1; a separation over no "
            "positions is not a small separation, it is undefined."
        )
    if not 0.0 < chance < 1.0:
        raise ValueError(
            f"match_probability must lie strictly inside (0, 1), got "
            f"{chance}. The degenerate alphabets have no forgery to separate."
        )
    shift = total * (1.0 + chance) / 2.0 - total * chance
    trials = 2 * total if pooled else total
    return shift / math.sqrt(trials * chance * (1.0 - chance))


def dominance_noise_level(
    matched_count: int,
    cut: float,
    *,
    eps: float,
    method: str = "sharpest",
    tolerance: float = 1e-12,
) -> float:
    """Return the link error rate at which the ``r_R`` detector stops adding.

    Parameters
    ----------
    matched_count : int
        ``|M_R|``, at least ``1``.
    cut : float
        The party's acceptance threshold, ``s_a`` or ``s_v``, in ``(0, 1)``.
    eps : float
        Keyword-only budget the threshold is derived at.
    method : {'sharpest', 'exact', 'hoeffding', 'chernoff'}, optional
        Keyword-only.
    tolerance : float, optional
        Keyword-only bisection width.

    Returns
    -------
    float
        In ``[0, cut]``.

    Raises
    ------
    TypeError
        If ``matched_count`` is not an integer or ``cut`` is not real.
    ValueError
        If ``matched_count`` is ``0``, or ``cut``/``eps``/``tolerance`` is out
        of range.

    Examples
    --------
    >>> from qsecure.detect.thresholds_rate import dominance_noise_level
    >>> f"{dominance_noise_level(38400, 1 / 64, eps=1e-9):.6f}"
    '0.012119'

    >>> dominance_noise_level(38400, 1 / 64, eps=1e-9) < 2 * (1 / 64)
    True

    >>> f"{dominance_noise_level(38400, 1 / 16, eps=1e-9):.6f}"
    '0.055355'
    """
    total = _as_count(matched_count, "matched_count")
    if total == 0:
        raise ValueError(
            "matched_count must be at least 1; a mismatch rate over an empty "
            "matched set is what verify() refuses to score at all."
        )
    limit = _as_probability(cut, "cut")
    budget = _as_budget(eps)
    how = _check_method(method)
    if not 0.0 < limit < 1.0:
        raise ValueError(
            f"cut must lie strictly inside (0, 1), got {limit}. It is an "
            f"acceptance threshold, s_a or s_v."
        )
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not 0.0 < float(tolerance) < 1.0
    ):
        raise ValueError(
            f"tolerance must lie strictly inside (0, 1), got {tolerance!r}."
        )

    def dominated(noise: float) -> bool:
        """Return whether the derived threshold still sits at or below the cut."""
        found = _invert(total, noise, budget, "upper", how)
        return found.count / total <= limit

    low, high = 0.0, limit
    if dominated(high):
        return high
    width = float(tolerance)
    while high - low > width:
        middle = (low + high) / 2.0
        if dominated(middle):
            low = middle
        else:
            high = middle
    return low


def floor_comparison(
    params: ProtocolParams, *, eps: float = HONEST_ABORT_BUDGET
) -> dict[str, int]:
    """Return the protocol's matched-count floors beside every derivation of them.

    Parameters
    ----------
    params : ProtocolParams
        The parameter set.
    eps : float, optional
        Keyword-only budget.

    Returns
    -------
    dict
        Keys ``"signing_length"``, ``"protocol_matched"``,
        ``"chernoff_matched"``, ``"hoeffding_matched"``, ``"exact_matched"``,
        ``"protocol_pooled"``, ``"chernoff_pooled"``, ``"hoeffding_pooled"``
        and ``"exact_pooled"``.

    Raises
    ------
    TypeError
        If ``params`` is not a
        :class:`~qsecure.protocol.params.ProtocolParams`.
    ValueError
        If ``eps`` is out of range.

    Examples
    --------
    >>> from qsecure.detect.thresholds_rate import floor_comparison
    >>> from qsecure.protocol.params import DEFAULT_PARAMS, ProtocolParams
    >>> row = floor_comparison(DEFAULT_PARAMS)
    >>> (row["protocol_matched"], row["chernoff_matched"],
    ...  row["hoeffding_matched"], row["exact_matched"])
    (36554, 36554, 36801, 36951)
    >>> row["protocol_pooled"], row["exact_pooled"]
    (74189, 74749)

    >>> short = floor_comparison(ProtocolParams(key_length=192))
    >>> short["protocol_matched"], short["exact_matched"]
    (0, 11)
    """
    if not isinstance(params, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(params).__name__}."
        )
    budget = _as_budget(eps)
    length = params.signing_length
    chance = params.match_probability
    row = {
        "signing_length": length,
        "protocol_matched": minimum_matched_count(params) - 1,
        "protocol_pooled": minimum_pooled_matched_count(params) - 1,
    }
    for name in INEQUALITIES:
        row[f"{name}_matched"] = _invert(
            length, chance, budget, "lower", name
        ).count
        row[f"{name}_pooled"] = _invert(
            2 * length, chance, budget, "lower", name
        ).count
    return row


@dataclass(frozen=True)
class RateCountVerdict:
    """What the rate-and-count family says about one run, and what it does not.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.detect.statistics import TranscriptStatistics
    >>> from qsecure.detect.thresholds_rate import RateCountThresholds
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> stats = TranscriptStatistics.from_transcript(
    ...     QDSSession(
    ...         ProtocolParams(key_length=192), rng=np.random.default_rng(7)
    ...     ).run(0)
    ... )
    >>> verdict = RateCountThresholds.for_transcript(
    ...     stats, eps=1e-9
    ... ).evaluate(stats)
    >>> verdict.flagged, verdict.not_scored
    (False, ())
    >>> verdict.grouping_key
    ('before-forwarding', True, True, False, True)
    """

    flagged: bool
    fired: tuple[str, ...]
    false_positive_bound: float
    observations: Mapping[str, int]
    not_scored: tuple[str, ...]
    grouping_key: tuple[Any, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
            ``"grouping_key"`` becomes a list, since JSON has no tuple, and keeps
            its order so an aggregator can group on it.
        """
        return {
            "flagged": self.flagged,
            "fired": list(self.fired),
            "false_positive_bound": self.false_positive_bound,
            "observations": dict(self.observations),
            "not_scored": list(self.not_scored),
            "grouping_key": list(self.grouping_key),
        }


@dataclass(frozen=True)
class RateCountThresholds:
    """The whole rate-and-count family at one budget, with a union bound.

    Examples
    --------
    >>> from qsecure.detect.thresholds_rate import (
    ...     RATE_COUNT_ROSTER, RateCountThresholds
    ... )
    >>> from qsecure.protocol.params import DEFAULT_PARAMS
    >>> family = RateCountThresholds.for_params(DEFAULT_PARAMS, eps=1e-9)
    >>> len(family.thresholds) == len(RATE_COUNT_ROSTER)
    True
    >>> f"{family.per_test_budget:.4e}"
    '8.3333e-11'
    >>> family.thresholds["matched_count_low:Bob"].critical_count
    37379
    >>> family.thresholds["pooled_count_high"].critical_count
    78249

    >>> f"{family.false_positive_bound:.4e}"
    '8.2524e-10'
    >>> family.false_positive_bound <= family.eps
    True
    """

    eps: float
    per_test_budget: float
    thresholds: Mapping[str, DerivedThreshold]
    channel_error_rate: float
    method: str
    signing_length: int
    match_probability: float

    @classmethod
    def for_params(
        cls,
        params: ProtocolParams,
        *,
        eps: float,
        channel_error_rate: float = 0.0,
        method: str = "sharpest",
        matched_counts: Mapping[str, int] | None = None,
    ) -> RateCountThresholds:
        """Derive the family from a parameter set alone, before any run.

        Parameters
        ----------
        params : ProtocolParams
            The **sifted** parameter set.
        eps : float
            Keyword-only family budget.
        channel_error_rate : float, optional
            Keyword-only ``p_e``, defaulting to the noiseless ``0.0``.
        method : {'sharpest', 'exact', 'hoeffding', 'chernoff'}, optional
            Keyword-only.
        matched_counts : mapping of str to int, optional
            Keyword-only per-party ``|M_R|``, when they are known.

        Returns
        -------
        RateCountThresholds

        Raises
        ------
        TypeError
            If ``params`` is not a
            :class:`~qsecure.protocol.params.ProtocolParams`.
        ValueError
            If a probability or ``eps`` is out of range, ``method`` names nothing,
            or ``matched_counts`` names a party who is not a verifier.
        """
        if not isinstance(params, ProtocolParams):
            raise TypeError(
                f"params must be a ProtocolParams, got "
                f"{type(params).__name__}."
            )
        family = _as_budget(eps)
        noise = _as_probability(channel_error_rate, "channel_error_rate")
        how = _check_method(method)
        length = params.signing_length
        chance = params.match_probability
        share = family / len(RATE_COUNT_ROSTER)
        counts = dict(matched_counts or {})
        strangers = set(counts) - {str(party) for party in VERIFIERS}
        if strangers:
            raise ValueError(
                f"matched_counts names {sorted(strangers)}, who are not "
                f"verifiers. Alice reaches no verdict and holds no matched "
                f"set, and a misspelled recipient would have silently fallen "
                f"back to the null's mean -- which is a threshold derived for "
                f"a run that never happened."
            )

        built: dict[str, DerivedThreshold] = {}
        for party in VERIFIERS:
            name = str(party)
            matched = _as_count(
                counts.get(name, int(length * chance)), "matched_count"
            )
            built[f"mismatch_rate:{name}"] = mismatch_rate_threshold(
                matched,
                eps=share,
                channel_error_rate=noise,
                method=how,
                party=name,
            )
            for side, suffix in (("lower", "low"), ("upper", "high")):
                built[f"matched_count_{suffix}:{name}"] = (
                    matched_count_threshold(
                        length,
                        chance,
                        eps=share,
                        side=side,
                        method=how,
                        party=name,
                    )
                )
                built[f"declared_count_{suffix}:{name}"] = (
                    declared_count_threshold(
                        length,
                        chance,
                        eps=share,
                        side=side,
                        method=how,
                        party=name,
                    )
                )
        for side, suffix in (("lower", "low"), ("upper", "high")):
            built[f"pooled_count_{suffix}"] = pooled_count_threshold(
                length, chance, eps=share, side=side, method=how
            )
        if set(built) != set(RATE_COUNT_ROSTER):
            missing = sorted(set(RATE_COUNT_ROSTER) - set(built))
            extra = sorted(set(built) - set(RATE_COUNT_ROSTER))
            raise ValueError(
                f"this constructor and RATE_COUNT_ROSTER disagree: "
                f"not built {missing}, not on the roster {extra}. The union "
                f"bound of D-4 divides eps by len(RATE_COUNT_ROSTER) and sums "
                f"over what was built, so the two must be the same set or the "
                f"family's proven bound is not the one it reports."
            )
        return cls(
            eps=family,
            per_test_budget=share,
            thresholds=built,
            channel_error_rate=noise,
            method=how,
            signing_length=length,
            match_probability=chance,
        )

    @classmethod
    def for_transcript(
        cls,
        stats: TranscriptStatistics,
        *,
        eps: float,
        channel_error_rate: float = 0.0,
        method: str = "sharpest",
    ) -> RateCountThresholds:
        """Derive the family for one run's shape.

        Parameters
        ----------
        stats : TranscriptStatistics
            Layer one's extraction of one JSON-round-tripped transcript.
        eps : float
            Keyword-only family budget.
        channel_error_rate : float, optional
            Keyword-only ``p_e``.
        method : {'sharpest', 'exact', 'hoeffding', 'chernoff'}, optional
            Keyword-only.

        Returns
        -------
        RateCountThresholds

        Raises
        ------
        TypeError
            If ``stats`` is not a
            :class:`~qsecure.detect.statistics.TranscriptStatistics`.
        ValueError
            If a probability or ``eps`` is out of range, or ``method`` names
            nothing.
        """
        if not isinstance(stats, TranscriptStatistics):
            raise TypeError(
                f"stats must be a TranscriptStatistics, got "
                f"{type(stats).__name__}. This layer reads layer one, which "
                f"is what keeps it on the far side of the JSON boundary."
            )
        counts = {
            name: verifier.matched.count
            for name, verifier in stats.verifiers.items()
        }
        return cls.for_params(
            stats.params,
            eps=eps,
            channel_error_rate=channel_error_rate,
            method=method,
            matched_counts=counts,
        )

    @property
    def false_positive_bound(self) -> float:
        """float: The family's proven honest-run firing probability."""
        return min(
            1.0,
            math.fsum(
                threshold.false_positive_bound
                for threshold in self.thresholds.values()
            ),
        )

    @property
    def bound_is_unconditional(self) -> bool:
        """bool: Whether :attr:`false_positive_bound` needs no conditioning."""
        return self.channel_error_rate == 0.0

    def threshold(self, name: str) -> DerivedThreshold:
        """Return one member by roster name.

        Parameters
        ----------
        name : str
            A member of :data:`RATE_COUNT_ROSTER`.

        Returns
        -------
        DerivedThreshold

        Raises
        ------
        KeyError
            If ``name`` is not on the roster.
        """
        if name not in self.thresholds:
            raise KeyError(
                f"{name!r} is not a member of this family. The roster is "
                f"fixed at {list(RATE_COUNT_ROSTER)}, plus the free "
                f"structural test(s) {list(RATE_COUNT_FREE_TESTS)}, which "
                f"carry no threshold object because their bound is exactly "
                f"zero."
            )
        return self.thresholds[name]

    def evaluate(self, stats: TranscriptStatistics) -> RateCountVerdict:
        """Apply the family to one run.

        Parameters
        ----------
        stats : TranscriptStatistics
            The run.

        Returns
        -------
        RateCountVerdict

        Raises
        ------
        TypeError
            If ``stats`` is not a
            :class:`~qsecure.detect.statistics.TranscriptStatistics`.
        ValueError
            If the run's sifted length or match probability differs from the one
            this family was derived at, or -- at a positive noise level -- if a
            verifier's matched count is not the one his mismatch threshold was
            conditioned on.

        Examples
        --------
        >>> import numpy as np
        >>> from qsecure.detect.statistics import TranscriptStatistics
        >>> from qsecure.detect.thresholds_rate import RateCountThresholds
        >>> from qsecure.protocol.params import ProtocolParams
        >>> from qsecure.protocol.session import QDSSession
        >>> stats = TranscriptStatistics.from_transcript(
        ...     QDSSession(
        ...         ProtocolParams(key_length=192), rng=np.random.default_rng(7)
        ...     ).run(0)
        ... )
        >>> family = RateCountThresholds.for_transcript(stats, eps=1e-9)
        >>> verdict = family.evaluate(stats)
        >>> verdict.flagged, sorted(verdict.observations)[:2]
        (False, ['declaration_gap', 'declared_count_high:Bob'])
        """
        if not isinstance(stats, TranscriptStatistics):
            raise TypeError(
                f"stats must be a TranscriptStatistics, got "
                f"{type(stats).__name__}."
            )
        if stats.key_length != self.signing_length:
            raise ValueError(
                f"this family was derived over n = {self.signing_length} and "
                f"the run has n = {stats.key_length}. A threshold applied to "
                f"a null it was not derived for carries no bound; rebuild the "
                f"family with for_transcript."
            )
        if stats.match_probability != self.match_probability:
            raise ValueError(
                f"this family was derived at p = {self.match_probability} and "
                f"the run has p = {stats.match_probability}. Different "
                f"alphabets are different nulls."
            )

        fired: list[str] = []
        observations: dict[str, int] = {}
        pooled = stats.pooled

        for party in VERIFIERS:
            name = str(party)
            if name not in stats.verifiers:
                continue
            verifier = stats.verifiers[name]
            self._check_conditioning(name, verifier.matched.count)
            self._apply(
                f"mismatch_rate:{name}",
                verifier.mismatch.count,
                fired,
                observations,
            )
            for suffix in ("low", "high"):
                self._apply(
                    f"matched_count_{suffix}:{name}",
                    verifier.matched.count,
                    fired,
                    observations,
                )
            declared = _declared_for(pooled, name)
            if declared is None:
                continue
            for suffix in ("low", "high"):
                self._apply(
                    f"declared_count_{suffix}:{name}",
                    declared.count,
                    fired,
                    observations,
                )

        if pooled.pooled is not None:
            for suffix in ("low", "high"):
                self._apply(
                    f"pooled_count_{suffix}",
                    pooled.pooled.count,
                    fired,
                    observations,
                )

        gap = pooled.declaration_gap
        if gap is not None:
            observations["declaration_gap"] = gap
            if gap != 0:
                fired.append("declaration_gap")

        not_scored = tuple(
            str(party)
            for party in VERIFIERS
            if str(party) not in stats.verifiers
        )
        return RateCountVerdict(
            flagged=bool(fired),
            fired=tuple(sorted(fired)),
            false_positive_bound=self.false_positive_bound,
            observations=observations,
            not_scored=not_scored,
            grouping_key=(
                stats.count_exchange_timing,
                stats.symmetrised,
                stats.counts_exchanged,
                stats.signer_saw_recipient_logs,
                stats.security_claim,
            ),
        )

    def _check_conditioning(self, party: str, matched: int) -> None:
        """Refuse to score a run whose matched count is not the one conditioned on.

        Parameters
        ----------
        party : str
            The verifier whose matched count is being checked.
        matched : int
            The run's realised ``|M_R|``.

        Raises
        ------
        ValueError
            If the family's mismatch threshold for this party was conditioned on a
            different matched count and the null is not the point mass.
        """
        if self.channel_error_rate == 0.0:
            return
        threshold = self.thresholds[f"mismatch_rate:{party}"]
        if threshold.trials == matched:
            return
        raise ValueError(
            f"the mismatch threshold for {party} was derived conditionally on "
            f"|M_R| = {threshold.trials} and this run has |M_R| = {matched}. "
            f"A threshold conditioned on one matched-set size bounds nothing "
            f"when applied to another, and the tower rule of D-3 is what "
            f"carries the bound through -- so this is refused rather than "
            f"scored. Build the family with "
            f"RateCountThresholds.for_transcript(stats, ...), which reads the "
            f"conditioning variable off the run it is going to score."
        )

    def _apply(
        self,
        name: str,
        observed: int,
        fired: list[str],
        observations: dict[str, int],
    ) -> None:
        """Evaluate one member and record what it saw.

        Parameters
        ----------
        name : str
            Roster name.
        observed : int
            The statistic's realised value.
        fired : list of str
            Accumulator, appended to in place.
        observations : dict
            Accumulator, written to in place.
        """
        threshold = self.thresholds[name]
        observations[name] = observed
        if threshold.fires(observed):
            fired.append(name)

    def claims(self) -> tuple[str, ...]:
        """Return every member's null-and-derivation sentence, in roster order.

        Returns
        -------
        tuple of str

        Examples
        --------
        >>> from qsecure.detect.thresholds_rate import RateCountThresholds
        >>> from qsecure.protocol.params import DEFAULT_PARAMS
        >>> family = RateCountThresholds.for_params(DEFAULT_PARAMS, eps=1e-9)
        >>> print(family.claims()[-1])
        pooled_count_high: fires when M = m_B + m_C >= 78249 out of 230400; under Binomial(230400, 1/3), exact by conservation of the pooled count under the symmetrisation coins, this fires with probability at most 8.195e-11, by the exact binomial tail at a budget of 8.333e-11.
        """
        return tuple(
            self.thresholds[name].claim() for name in RATE_COUNT_ROSTER
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the whole family.

        Returns
        -------
        dict
            Passes to :func:`json.dumps` unchanged.
        """
        return {
            "eps": self.eps,
            "per_test_budget": self.per_test_budget,
            "false_positive_bound": self.false_positive_bound,
            "bound_is_unconditional": self.bound_is_unconditional,
            "channel_error_rate": self.channel_error_rate,
            "method": self.method,
            "signing_length": self.signing_length,
            "match_probability": self.match_probability,
            "roster": list(RATE_COUNT_ROSTER),
            "free_tests": list(RATE_COUNT_FREE_TESTS),
            "thresholds": {
                name: threshold.to_dict()
                for name, threshold in self.thresholds.items()
            },
        }


def _declared_for(pooled: Any, party: str) -> CountStatistic | None:
    """Return one party's declared Phase C' count, or ``None``.

    Parameters
    ----------
    pooled : PooledStatistics
        Layer one's pooled block.
    party : str
        ``"Bob"`` or ``"Charlie"``.

    Returns
    -------
    CountStatistic or None
        ``None`` on a run whose recipients did not exchange counts, and for any
        party name the exchange does not carry.
    """
    if party == "Bob":
        return pooled.declared_bob
    if party == "Charlie":
        return pooled.declared_charlie
    return None
