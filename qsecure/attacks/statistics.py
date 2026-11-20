"""One binomial-interval implementation for the whole attack suite.

Examples
--------
>>> from qsecure.attacks.statistics import agrees_within, wilson_bounds
>>> low, high = wilson_bounds(0, 400)
>>> low, round(high, 5)
(0.0, 0.00951)

>>> verdict = agrees_within(8, 2000, 0.0042075)
>>> verdict.agrees, round(verdict.z_score, 3)
(True, -0.143)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

__all__ = [
    "Z_90",
    "Z_95",
    "Z_99",
    "AgreementVerdict",
    "agrees_within",
    "binomial_standard_error",
    "sigma_tolerance",
    "two_sided_z",
    "wilson_bounds",
]


Z_90: Final[float] = 1.6448536269514722
"""Two-sided 90% normal quantile, ``Phi^-1(0.95)``."""

Z_95: Final[float] = 1.959963984540054
"""Two-sided 95% normal quantile, ``Phi^-1(0.975)`` -- the suite's default."""

Z_99: Final[float] = 2.5758293035489004
"""Two-sided 99% normal quantile, ``Phi^-1(0.995)``."""

_DEFAULT_SIGMAS: Final[float] = 4.0
"""Default width of :func:`agrees_within`, in standard errors."""


def two_sided_z(confidence: float) -> float:
    """Return the two-sided normal quantile for a confidence level.

    Parameters
    ----------
    confidence : float
        Strictly inside ``(0, 1)``.

    Returns
    -------
    float
        The positive quantile.

    Examples
    --------
    >>> from qsecure.attacks.statistics import two_sided_z
    >>> round(two_sided_z(0.95), 6)
    1.959964
    >>> round(two_sided_z(0.99), 6)
    2.575829
    """
    if isinstance(confidence, bool) or not isinstance(
        confidence, (int, float, np.floating)
    ):
        raise TypeError(
            f"confidence must be a real number, got "
            f"{type(confidence).__name__}"
        )
    level = float(confidence)
    if not 0.0 < level < 1.0:
        raise ValueError(
            f"confidence must lie strictly inside (0, 1), got {level}. It is a "
            f"coverage probability, not a percentage: pass 0.95, not 95."
        )
    target = 1.0 - (1.0 - level) / 2.0
    low, high = 0.0, 40.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if 0.5 * (1.0 + math.erf(middle / math.sqrt(2.0))) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _as_count(value: Any, name: str) -> int:
    """Coerce a non-negative integer count, refusing bools and floats."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{name} must be an int, got {type(value).__name__}. Counts of "
            f"trials are integers; a float here is usually a rate that has "
            f"already been divided once."
        )
    count = int(value)
    if count < 0:
        raise ValueError(f"{name} must be non-negative, got {count}")
    return count


def _as_quantile(value: Any, name: str) -> float:
    """Coerce a positive finite normal quantile."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.floating, np.integer)
    ):
        raise TypeError(f"{name} must be a real number, got "
                        f"{type(value).__name__}")
    quantile = float(value)
    if not math.isfinite(quantile) or quantile <= 0.0:
        raise ValueError(
            f"{name} must be positive and finite, got {quantile}. It is a "
            f"normal quantile; Z_95 is the suite default."
        )
    return quantile


def wilson_bounds(
    successes: int, trials: int, *, z: float = Z_95
) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion.

    Parameters
    ----------
    successes : int
        ``0 <= successes <= trials``.
    trials : int
        At least ``1``.
    z : float, optional
        Normal quantile; defaults to :data:`Z_95`.

    Returns
    -------
    tuple of float
        ``(low, high)``, exactly ``0.0`` at zero successes and exactly ``1.0``
        at ``successes == trials``, otherwise clipped into
        ``[0, 1]``.

    Examples
    --------
    >>> from qsecure.attacks.statistics import wilson_bounds
    >>> low, high = wilson_bounds(0, 100)
    >>> low, round(high, 4)
    (0.0, 0.037)
    >>> low, high = wilson_bounds(100, 100)
    >>> round(low, 4), high
    (0.963, 1.0)
    >>> low, high = wilson_bounds(50, 100)
    >>> round(low, 4), round(high, 4)
    (0.4038, 0.5962)

    >>> wilson_bounds(0, 8000)[0] == 0.0
    True
    >>> wilson_bounds(400, 400)[1] == 1.0
    True
    """
    total = _as_count(trials, "trials")
    hits = _as_count(successes, "successes")
    if total < 1:
        raise ValueError(
            f"trials must be at least 1, got {total}. A rate measured over no "
            f"trials is not a wide interval, it is the absence of a "
            f"measurement; report the trial count instead."
        )
    if hits > total:
        raise ValueError(
            f"successes ({hits}) exceeds trials ({total}); a proportion above "
            f"one is a counting error upstream, not a rate."
        )
    quantile = _as_quantile(z, "z")

    n = float(total)
    rate = hits / n
    z2 = quantile * quantile
    denominator = 1.0 + z2 / n
    centre = (rate + z2 / (2.0 * n)) / denominator
    spread = (
        quantile
        * math.sqrt(rate * (1.0 - rate) / n + z2 / (4.0 * n * n))
        / denominator
    )
    low = 0.0 if hits == 0 else max(0.0, centre - spread)
    high = 1.0 if hits == total else min(1.0, centre + spread)
    return (low, high)


def binomial_standard_error(probability: float, trials: int) -> float:
    """Return ``sqrt(p(1-p)/n)``, the standard error of a measured proportion.

    Parameters
    ----------
    probability : float
        The proportion the error is computed *at*.
    trials : int
        At least ``1``.

    Returns
    -------
    float
        The standard error.

    Examples
    --------
    >>> from qsecure.attacks.statistics import binomial_standard_error
    >>> round(binomial_standard_error(0.5, 8000), 6)
    0.00559
    >>> round(binomial_standard_error(1 / 12, 32054), 6)
    0.001544
    """
    if isinstance(probability, bool) or not isinstance(
        probability, (int, float, np.floating)
    ):
        raise TypeError(
            f"probability must be a real number, got "
            f"{type(probability).__name__}"
        )
    p = float(probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must lie in [0, 1], got {p}")
    total = _as_count(trials, "trials")
    if total < 1:
        raise ValueError(f"trials must be at least 1, got {total}")
    return math.sqrt(p * (1.0 - p) / total)


def sigma_tolerance(
    probability: float, trials: int, *, sigmas: float = _DEFAULT_SIGMAS
) -> float:
    """Return the half-width of an agreement band, in units of the rate.

    Parameters
    ----------
    probability : float
        The analytic prediction, in ``[0, 1]``.
    trials : int
        At least ``1``.
    sigmas : float, optional
        Band half-width in standard errors; defaults to ``4.0``.

    Returns
    -------
    float
        The tolerance to compare ``abs(measured - predicted)`` against.

    Examples
    --------
    >>> from qsecure.attacks.statistics import sigma_tolerance
    >>> round(sigma_tolerance(0.5, 800), 5)
    0.07071
    >>> round(sigma_tolerance(1.0, 200), 5)
    0.02

    >>> round(sigma_tolerance(0.25, 2000) / sigma_tolerance(0.25, 4000), 6)
    1.414214
    """
    error = binomial_standard_error(probability, trials)
    span = _as_quantile(sigmas, "sigmas")
    total = int(trials)
    return max(span * error, span / total)


@dataclass(frozen=True)
class AgreementVerdict:
    """Whether one measured rate agrees with one analytic prediction.

    Parameters
    ----------
    successes : int
        Observed successes.
    trials : int
        Trials.
    predicted : float
        The analytic prediction the measurement is tested against.
    sigmas : float
        The band half-width in standard errors that was applied.

    Examples
    --------
    >>> from qsecure.attacks.statistics import AgreementVerdict
    >>> verdict = AgreementVerdict(279, 800, 0.345566, 4.0)
    >>> verdict.agrees
    True
    >>> round(verdict.measured, 5), round(verdict.z_score, 3)
    (0.34875, 0.189)
    """

    successes: int
    trials: int
    predicted: float
    sigmas: float

    @property
    def measured(self) -> float:
        """float: ``successes / trials``."""
        return self.successes / self.trials

    @property
    def tolerance(self) -> float:
        """float: The band half-width, from :func:`sigma_tolerance`."""
        return sigma_tolerance(
            self.predicted, self.trials, sigmas=self.sigmas
        )

    @property
    def standard_error(self) -> float:
        """float: ``sqrt(p(1-p)/n)`` at the *predicted* ``p``."""
        return binomial_standard_error(self.predicted, self.trials)

    @property
    def z_score(self) -> float:
        """float: Standard errors between measurement and prediction."""
        error = self.standard_error
        gap = self.measured - self.predicted
        if error == 0.0:
            return 0.0 if gap == 0.0 else math.copysign(math.inf, gap)
        return gap / error

    @property
    def agrees(self) -> bool:
        """bool: Whether the gap is inside :attr:`tolerance`."""
        return abs(self.measured - self.predicted) <= self.tolerance

    @property
    def interval(self) -> tuple[float, float]:
        """tuple of float: The measurement's own Wilson 95% interval."""
        return wilson_bounds(self.successes, self.trials)

    def summary(self) -> str:
        """Return a one-line verdict naming every number behind it.

        Returns
        -------
        str
            Measured rate with its count, the prediction, the gap in standard
            errors, and the band applied.

        Examples
        --------
        >>> from qsecure.attacks.statistics import AgreementVerdict
        >>> print(AgreementVerdict(8, 2000, 0.0042075, 4.0).summary())
        AGREES: 8/2000 = 0.004000 vs predicted 0.004208 (z = -0.14, band 4.0 se = 0.005790)
        """
        verdict = "AGREES" if self.agrees else "DISAGREES"
        return (
            f"{verdict}: {self.successes}/{self.trials} = "
            f"{self.measured:.6f} vs predicted {self.predicted:.6f} "
            f"(z = {self.z_score:.2f}, band {self.sigmas} se = "
            f"{self.tolerance:.6f})"
        )


def agrees_within(
    successes: int,
    trials: int,
    predicted: float,
    *,
    sigmas: float = _DEFAULT_SIGMAS,
) -> AgreementVerdict:
    """Test a measured rate against an analytic prediction at a computed band.

    Parameters
    ----------
    successes : int
        Observed successes, ``0 <= successes <= trials``.
    trials : int
        At least ``1``.
    predicted : float
        The analytic prediction, in ``[0, 1]``.
    sigmas : float, optional
        Band half-width in standard errors; defaults to ``4.0`` for the reason
        given on ``_DEFAULT_SIGMAS``.

    Returns
    -------
    AgreementVerdict
        The verdict with its evidence kept.

    Examples
    --------
    >>> from qsecure.attacks.statistics import agrees_within

    >>> print(agrees_within(8, 2000, 0.0042075).summary())
    AGREES: 8/2000 = 0.004000 vs predicted 0.004208 (z = -0.14, band 4.0 se = 0.005790)

    >>> wrong = agrees_within(2641, 32054, 1 / 3)
    >>> wrong.agrees, round(wrong.z_score, 1)
    (False, -95.3)
    """
    total = _as_count(trials, "trials")
    hits = _as_count(successes, "successes")
    if total < 1:
        raise ValueError(f"trials must be at least 1, got {total}")
    if hits > total:
        raise ValueError(
            f"successes ({hits}) exceeds trials ({total})"
        )
    if isinstance(predicted, bool) or not isinstance(
        predicted, (int, float, np.floating)
    ):
        raise TypeError(
            f"predicted must be a real number, got {type(predicted).__name__}"
        )
    probability = float(predicted)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"predicted must lie in [0, 1], got {probability}. It is a "
            f"probability from qsecure.protocol.analysis, not a count."
        )
    return AgreementVerdict(
        successes=hits,
        trials=total,
        predicted=probability,
        sigmas=_as_quantile(sigmas, "sigmas"),
    )
