"""The **channel** family of derived detection thresholds (QBER, CHSH,
fidelity, purity, concurrence).

Examples
--------
>>> from qsecure.detect.thresholds_channel import link_check_rounds
>>> from qsecure.protocol.params import ProtocolParams
>>> params = ProtocolParams(key_length=384, check_fraction=0.25)
>>> params.check_count
96
>>> shares = link_check_rounds(params)
>>> [(str(s.party), s.qber_rounds, s.chsh_rounds) for s in shares]
[('Bob', 24, 24), ('Charlie', 24, 24)]
>>> import math
>>> naive = math.sqrt(1.0 / 48)          # "the run drew 48 QBER rounds"
>>> real = math.sqrt(1.0 / 24)           # this link measured 24 of them
>>> f"{real / naive:.4f}"
'1.4142'

>>> from qsecure.detect.thresholds_channel import qber_threshold
>>> noiseless = qber_threshold(rounds=24, epsilon=1e-09)
>>> noiseless.critical_value, noiseless.false_positive_bound
(1.0, 0.0)
>>> noiseless.inequality
'point mass (exact)'
>>> qber_threshold(rounds=24, epsilon=0.5).critical_value
1.0

>>> for method in ("hoeffding", "chernoff", "exact"):
...     threshold = qber_threshold(
...         rounds=4114,
...         epsilon=1e-09,
...         tolerated_depolarising=1 / 32,
...         method=method,
...     )
...     print(
...         f"{method:10s} k = {threshold.critical_value:6.1f}  "
...         f"P(fire | honest) <= {threshold.false_positive_bound:.3e}"
...     )
hoeffding  k =  271.0  P(fire | honest) <= 9.503e-10
chernoff   k =  128.0  P(fire | honest) <= 6.757e-10
exact      k =  118.0  P(fire | honest) <= 8.776e-10

>>> from qsecure.detect.thresholds_channel import chsh_threshold
>>> from qsecure.protocol.checkrounds import CLASSICAL_CHSH_BOUND
>>> short = chsh_threshold(counts=(6, 6, 6, 6), epsilon=1e-09)
>>> f"{short.critical_value:.4f}", short.critical_value > CLASSICAL_CHSH_BOUND
('-2.4281', False)
>>> long = chsh_threshold(counts=(1028, 1029, 1029, 1029), epsilon=1e-09)
>>> f"{long.critical_value:.4f}", long.critical_value > CLASSICAL_CHSH_BOUND
('2.4270', True)

>>> from qsecure.detect.thresholds_channel import (
...     concurrence_threshold, purity_threshold, werner_purity,
... )
>>> ideal = purity_threshold(samples=48, epsilon=1e-09)
>>> ideal.statistic, ideal.false_positive_bound
('min_purity', 0.0)
>>> noisy = purity_threshold(
...     samples=48, epsilon=1e-09, tolerated_depolarising=0.1
... )
>>> noisy.statistic, f"{werner_purity(0.1):.4f}", f"{noisy.critical_value:.4f}"
('mean_purity', '0.8575', '0.5090')
>>> concurrence_threshold(samples=48, epsilon=1e-09).inequality
'point mass (exact)'

>>> import numpy as np
>>> from qsecure.detect.statistics import TranscriptStatistics
>>> from qsecure.detect.thresholds_channel import screen_link
>>> from qsecure.protocol.session import QDSSession
>>> stats = TranscriptStatistics.from_transcript(
...     QDSSession(
...         ProtocolParams(key_length=384, check_fraction=0.25),
...         rng=np.random.default_rng(7),
...     ).run(0)
... )
>>> screen = screen_link(stats.link("Bob", 0), epsilon=1e-09)
>>> screen.detected, screen.fired
(False, ())
>>> screen.cleared
('chsh', 'min_concurrence', 'min_fidelity', 'min_purity', 'qber_errors')
>>> f"{screen.false_positive_bound:.3e}"
'2.000e-10'

>>> from qsecure.detect.thresholds_channel import (
...     divide_budget, fidelity_threshold,
... )
>>> share = divide_budget(1e-09, 5)
>>> for rounds, seen, label in ((24, 48, "L = 384"), (4114, 8229, "CHECKED")):
...     errors = qber_threshold(
...         rounds=rounds, epsilon=share, tolerated_depolarising=0.05
...     )
...     mean = fidelity_threshold(
...         samples=seen, epsilon=share, tolerated_depolarising=0.05
...     )
...     print(
...         f"{label:8s} k = {errors.critical_value:5.0f} of {rounds:5d}"
...         f"   mean fidelity <= {mean.critical_value:.4f}"
...     )
L = 384  k =    10 of    24   mean fidelity <= 0.4802
CHECKED  k =   172 of  4114   mean fidelity <= 0.9257
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from qsecure.detect.statistics import IDEAL_TOLERANCE, LinkStatistics
from qsecure.protocol.analysis import depolarising_error_rate
from qsecure.protocol.checkrounds import (
    CHECK_CHSH_WEIGHT,
    CLASSICAL_CHSH_BOUND,
    IDEAL_CHSH,
    depolarising_chsh,
)
from qsecure.protocol.params import VERIFIERS, Party, ProtocolParams, _as_party

__all__ = [
    "CHANNEL_STATISTICS",
    "CONCURRENCE_RANGE",
    "FIDELITY_RANGE",
    "PURITY_RANGE",
    "ChannelScreen",
    "ChannelThreshold",
    "CheckRoundShare",
    "bounded_mean_threshold",
    "chsh_certificate_threshold",
    "chsh_threshold",
    "concurrence_threshold",
    "divide_budget",
    "fidelity_threshold",
    "link_check_rounds",
    "purity_threshold",
    "qber_threshold",
    "screen_link",
    "werner_concurrence",
    "werner_fidelity",
    "werner_purity",
]


CHANNEL_STATISTICS: Final[tuple[str, ...]] = (
    "qber_errors",
    "chsh",
    "fidelity",
    "purity",
    "concurrence",
)

#: ``[a, b]`` for the fidelity :math:`\langle\Phi^{+}|\rho|\Phi^{+}\rangle`,
#: which is an expectation of a projector and so lies in the unit interval.
FIDELITY_RANGE: Final[tuple[float, float]] = (0.0, 1.0)

PURITY_RANGE: Final[tuple[float, float]] = (0.25, 1.0)

#: ``[a, b]`` for concurrence, ``0`` on anything separable and ``1`` on a Bell
#: state.
CONCURRENCE_RANGE: Final[tuple[float, float]] = (0.0, 1.0)

#: The two directions a threshold can fire in. ``"upper"`` fires at or above
#: the critical value, ``"lower"`` at or below it.
_DIRECTIONS: Final[frozenset[str]] = frozenset({"lower", "upper"})

#: The inversions :func:`qber_threshold` will perform.
_QBER_METHODS: Final[frozenset[str]] = frozenset(
    {"exact", "chernoff", "hoeffding"}
)

_EXACT_METHOD_FLOOR: Final[float] = 1e-300


def _as_epsilon(value: Any, name: str = "epsilon") -> float:
    """Coerce a false-positive budget strictly inside ``(0, 1)``.

    Parameters
    ----------
    value : float
        The candidate.
    name : str, optional
        Argument name quoted in the error message.

    Returns
    -------
    float

    Raises
    ------
    TypeError
        If ``value`` is a boolean or not a real number.
    ValueError
        If ``value`` is not finite or lies outside the open unit interval.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be a real number, got {type(value).__name__}. It is "
            f"a probability budget, not a count and not a percentage."
        )
    budget = float(value)
    if not math.isfinite(budget) or not 0.0 < budget < 1.0:
        raise ValueError(
            f"{name} must lie strictly inside (0, 1), got {budget!r}. A "
            f"budget of 0 asks for a threshold that can never fire and a "
            f"budget of 1 for one that always does; neither is an operating "
            f"point. Pass 1e-09, not 9."
        )
    return budget


def _as_sample_size(value: Any, name: str) -> int:
    """Coerce a non-negative sample size, refusing booleans.

    Parameters
    ----------
    value : int
        The candidate.
    name : str
        Argument name quoted in the error message.

    Returns
    -------
    int

    Raises
    ------
    TypeError
        If ``value`` is a boolean or not an integer.
    ValueError
        If ``value`` is negative.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{name} must be an int, got {type(value).__name__}. It counts "
            f"rounds; a float here is usually a rate that has already been "
            f"divided once."
        )
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _as_strength(value: Any, name: str = "tolerated_depolarising") -> float:
    """Coerce a depolarising strength in ``[0, 1]``, refusing booleans.

    Parameters
    ----------
    value : float
        The candidate.
    name : str, optional
        Argument name quoted in the error message.

    Returns
    -------
    float

    Raises
    ------
    TypeError
        If ``value`` is a boolean or not a real number.
    ValueError
        If ``value`` is not finite or lies outside ``[0, 1]``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be a real number, got {type(value).__name__}"
        )
    strength = float(value)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError(
            f"{name} must be a finite number in [0, 1], got {strength!r}. It "
            f"is the weight of the maximally mixed component of the honest "
            f"resource, so it is a probability. It is a physical noise level "
            f"the deployment states, never a knob turned until a table looked "
            f"better."
        )
    return strength


def werner_fidelity(strength: float) -> float:
    """Return the fidelity of a Werner resource of depolarising strength ``p``.

    Parameters
    ----------
    strength : float
        ``p`` in ``[0, 1]``.

    Returns
    -------
    float
        The mean fidelity such a link's check rounds report.

    Raises
    ------
    TypeError
        If ``strength`` is a boolean or not a real number.
    ValueError
        If it is not finite or lies outside ``[0, 1]``.

    Examples
    --------
    >>> from qsecure.detect.thresholds_channel import werner_fidelity
    >>> werner_fidelity(0.0), werner_fidelity(1.0)
    (1.0, 0.25)
    >>> f"{werner_fidelity(1 / 32):.6f}"
    '0.976562'
    """
    return 1.0 - 0.75 * _as_strength(strength, "strength")


def werner_purity(strength: float) -> float:
    """Return the purity of a Werner resource of depolarising strength ``p``.

    Parameters
    ----------
    strength : float
        ``p`` in ``[0, 1]``.

    Returns
    -------
    float
        The mean purity such a link's check rounds report.

    Raises
    ------
    TypeError
        If ``strength`` is a boolean or not a real number.
    ValueError
        If it is not finite or lies outside ``[0, 1]``.

    Examples
    --------
    >>> from qsecure.detect.thresholds_channel import werner_purity
    >>> werner_purity(0.0), werner_purity(1.0)
    (1.0, 0.25)
    >>> f"{werner_purity(0.1):.6f}"
    '0.857500'
    """
    p = _as_strength(strength, "strength")
    return 1.0 - 1.5 * p + 0.75 * p * p


def werner_concurrence(strength: float) -> float:
    """Return the concurrence of a Werner resource of strength ``p``.

    Parameters
    ----------
    strength : float
        ``p`` in ``[0, 1]``.

    Returns
    -------
    float
        The mean concurrence such a link's check rounds report.

    Raises
    ------
    TypeError
        If ``strength`` is a boolean or not a real number.
    ValueError
        If it is not finite or lies outside ``[0, 1]``.

    Examples
    --------
    >>> from qsecure.detect.thresholds_channel import (
    ...     werner_concurrence, werner_fidelity,
    ... )
    >>> werner_concurrence(0.0), werner_concurrence(2 / 3)
    (1.0, 0.0)
    >>> f"{werner_concurrence(0.1):.6f}"
    '0.850000'
    >>> f"{2 * werner_fidelity(0.1) - 1:.6f}"
    '0.850000'
    """
    return max(0.0, 1.0 - 1.5 * _as_strength(strength, "strength"))


@dataclass(frozen=True)
class CheckRoundShare:
    """One link's share of a run's reserved check positions."""

    party: Party
    qber_rounds: int
    chsh_rounds: int

    @property
    def total(self) -> int:
        """int: Reserved positions this link measures, over both arms."""
        return self.qber_rounds + self.chsh_rounds

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
        """
        return {
            "party": str(self.party),
            "qber_rounds": self.qber_rounds,
            "chsh_rounds": self.chsh_rounds,
            "total": self.total,
        }


def _round_robin_share(total: int, offset: int, index: int, links: int) -> int:
    """Count the rounds a round-robin deal gives one link.

    Parameters
    ----------
    total : int
        Rounds in this arm.
    offset : int
        Value of the running counter when the arm starts.
    index : int
        Which link, ``0`` based.
    links : int
        How many links share the deal.

    Returns
    -------
    int
    """
    first = (index - offset) % links
    if first >= total:
        return 0
    return (total - first - 1) // links + 1


def link_check_rounds(
    params: ProtocolParams,
    *,
    chsh_weight: float = CHECK_CHSH_WEIGHT,
    parties: Sequence[Party | str] = VERIFIERS,
) -> tuple[CheckRoundShare, ...]:
    """Return each link's share of the reserved check rounds.

    Parameters
    ----------
    params : ProtocolParams
        The parameter set.
    chsh_weight : float, optional
        Keyword-only.
    parties : sequence of Party or str, optional
        Keyword-only.

    Returns
    -------
    tuple of CheckRoundShare
        One entry per link, in ``parties`` order.

    Raises
    ------
    TypeError
        If ``params`` is not a :class:`~qsecure.protocol.params.ProtocolParams`
        or ``chsh_weight`` is not a real number.
    ValueError
        If ``chsh_weight`` is outside ``[0, 1)``, or if ``parties`` is empty,
        repeats a link or names Alice.

    Examples
    --------
    >>> from qsecure.detect.thresholds_channel import link_check_rounds
    >>> from qsecure.protocol.params import CHECKED_PARAMS, ProtocolParams
    >>> shares = link_check_rounds(
    ...     ProtocolParams(key_length=120, check_fraction=0.125)
    ... )
    >>> [(str(s.party), s.qber_rounds, s.chsh_rounds) for s in shares]
    [('Bob', 4, 4), ('Charlie', 4, 3)]
    >>> sum(s.total for s in shares) == 15
    True

    >>> CHECKED_PARAMS.check_count
    16458
    >>> bob, charlie = link_check_rounds(CHECKED_PARAMS)
    >>> bob.qber_rounds, bob.chsh_rounds, bob.total
    (4114, 4115, 8229)
    >>> charlie.qber_rounds, charlie.chsh_rounds
    (4115, 4114)

    >>> [s.total for s in link_check_rounds(ProtocolParams(key_length=192))]
    [0, 0]
    """
    if not isinstance(params, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(params).__name__}"
        )
    if isinstance(chsh_weight, bool) or not isinstance(
        chsh_weight, (int, float)
    ):
        raise TypeError(
            f"chsh_weight must be a real number, got "
            f"{type(chsh_weight).__name__}"
        )
    weight = float(chsh_weight)
    if not math.isfinite(weight) or not 0.0 <= weight < 1.0:
        raise ValueError(
            f"chsh_weight must lie in [0, 1), got {weight!r}; it is the share "
            f"of check rounds spent on the Bell arm, and a weight of 1 would "
            f"leave the QBER arm empty."
        )
    resolved: list[Party] = []
    for entry in parties:
        party = _as_party(entry)
        if party is Party.ALICE:
            raise ValueError(
                "check rounds are dealt between the recipients; Alice "
                "measures every one of them and holds no link of her own."
            )
        if party in resolved:
            raise ValueError(f"parties repeats {party}")
        resolved.append(party)
    if not resolved:
        raise ValueError(
            "parties must name at least one link; an empty deal would report "
            "a sample size of zero for a run that measured something."
        )

    check_count = params.check_count
    chsh_total = int(math.floor(weight * check_count))
    qber_total = check_count - chsh_total
    links = len(resolved)
    return tuple(
        CheckRoundShare(
            party=party,
            qber_rounds=_round_robin_share(
                qber_total, chsh_total, index, links
            ),
            chsh_rounds=_round_robin_share(chsh_total, 0, index, links),
        )
        for index, party in enumerate(resolved)
    )


def divide_budget(epsilon: float, parts: int) -> float:
    """Split a false-positive budget over ``parts`` independent tests.

    Parameters
    ----------
    epsilon : float
        The budget for the whole collection, strictly inside ``(0, 1)``.
    parts : int
        How many tests share it, at least ``1``.

    Returns
    -------
    float
        The per-test budget.

    Raises
    ------
    TypeError
        If ``epsilon`` is not a real number or ``parts`` is not an integer.
    ValueError
        If ``epsilon`` is outside ``(0, 1)`` or ``parts`` is below ``1``.

    Examples
    --------
    >>> from qsecure.detect.thresholds_channel import divide_budget
    >>> divide_budget(1e-08, 4)
    2.5e-09
    >>> f"{divide_budget(0.05, 5):.4f}"
    '0.0100'
    """
    budget = _as_epsilon(epsilon)
    share = _as_sample_size(parts, "parts")
    if share < 1:
        raise ValueError(
            f"parts must be at least 1, got {share}. A budget divided over no "
            f"tests is not a wider budget, it is the absence of a test."
        )
    return budget / share


@dataclass(frozen=True)
class ChannelThreshold:
    """One derived critical value, with everything needed to audit it.

    Examples
    --------
    >>> from qsecure.detect.thresholds_channel import qber_threshold
    >>> threshold = qber_threshold(rounds=24, epsilon=1e-09)
    >>> threshold.statistic, threshold.direction, threshold.critical_value
    ('qber_errors', 'upper', 1.0)
    >>> threshold.fires(0), threshold.fires(1)
    (False, True)
    >>> print(threshold.summary())      # doctest: +NORMALIZE_WHITESPACE
    qber_errors >= 1.0 fires; under point mass at 0: E = 0 with probability
    one (a non-negative variable with zero mean) P(fire) <= 0.00e+00
    (budget 1.00e-09, point mass (exact), n = 24)
    """

    statistic: str
    direction: str
    critical_value: float
    epsilon: float
    false_positive_bound: float
    null: str
    inequality: str
    sample_size: int
    derivation: str
    is_alarm: bool = True
    reaches_its_statistic: bool = True

    def __post_init__(self) -> None:
        """Coerce the fields and refuse a bound that exceeds its own budget.

        Raises
        ------
        ValueError
            If ``direction`` names no side, if a required string is empty, or if
            ``false_positive_bound`` exceeds ``epsilon`` -- which would mean the
            derivation did not deliver what it was asked for, and is a bug in this
            module rather than a number to publish.
        """
        object.__setattr__(self, "statistic", str(self.statistic))
        object.__setattr__(self, "direction", str(self.direction))
        object.__setattr__(self, "critical_value", float(self.critical_value))
        object.__setattr__(self, "epsilon", _as_epsilon(self.epsilon))
        object.__setattr__(
            self, "false_positive_bound", float(self.false_positive_bound)
        )
        object.__setattr__(self, "null", str(self.null))
        object.__setattr__(self, "inequality", str(self.inequality))
        object.__setattr__(
            self, "sample_size", _as_sample_size(self.sample_size, "sample_size")
        )
        object.__setattr__(self, "derivation", str(self.derivation))
        object.__setattr__(self, "is_alarm", bool(self.is_alarm))
        object.__setattr__(
            self, "reaches_its_statistic", bool(self.reaches_its_statistic)
        )
        if self.direction not in _DIRECTIONS:
            raise ValueError(
                f"direction must be one of {sorted(_DIRECTIONS)}, got "
                f"{self.direction!r}"
            )
        if not math.isfinite(self.critical_value):
            raise ValueError(
                f"{self.statistic}: critical_value must be finite, got "
                f"{self.critical_value!r}"
            )
        if not 0.0 <= self.false_positive_bound <= 1.0:
            raise ValueError(
                f"{self.statistic}: false_positive_bound must be a "
                f"probability, got {self.false_positive_bound!r}"
            )
        if self.false_positive_bound > self.epsilon:
            raise ValueError(
                f"{self.statistic}: the derivation proved only "
                f"{self.false_positive_bound!r} against a budget of "
                f"{self.epsilon!r}. A threshold whose proven bound exceeds its "
                f"own budget has not been derived, it has been asserted; this "
                f"is a defect in the inversion, not a number to publish."
            )
        for field, text in (("null", self.null), ("derivation", self.derivation)):
            if not text:
                raise ValueError(
                    f"{self.statistic}: {field} must be a non-empty sentence. "
                    f"a threshold with no written-down null and "
                    f"no written-down algebra cannot be audited, and an empty "
                    f"string would let one ship anyway."
                )

    def fires(self, observed: float) -> bool:
        """Return whether an observation trips this threshold.

        Parameters
        ----------
        observed : float
            The measured statistic.

        Returns
        -------
        bool
            ``observed >= critical_value`` for ``direction="upper"``, ``observed <=
            critical_value`` for ``"lower"``.

        Raises
        ------
        TypeError
            If ``observed`` is ``None`` or is not a real number.
        ValueError
            If ``observed`` is not finite.

        Examples
        --------
        >>> from qsecure.detect.thresholds_channel import chsh_threshold
        >>> threshold = chsh_threshold(
        ...     counts=(1028, 1029, 1029, 1029), epsilon=1e-09
        ... )
        >>> threshold.fires(2.83), threshold.fires(2.0)
        (False, True)
        """
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise TypeError(
                f"{self.statistic}: observed must be a real number, got "
                f"{type(observed).__name__}. A missing observation is not a "
                f"passing one -- an unmonitored link has no value here, and "
                f"reading None as 'did not fire' would report a channel "
                f"nobody looked at as clean."
            )
        value = float(observed)
        if not math.isfinite(value):
            raise ValueError(
                f"{self.statistic}: observed must be finite, got {value!r}"
            )
        if self.direction == "upper":
            return value >= self.critical_value
        return value <= self.critical_value

    def summary(self) -> str:
        """Return a one-line account naming every number behind the threshold.

        Returns
        -------
        str
        """
        comparison = ">=" if self.direction == "upper" else "<="
        return (
            f"{self.statistic} {comparison} {self.critical_value} fires; "
            f"under {self.null} "
            f"P(fire) <= {self.false_positive_bound:.2e} "
            f"(budget {self.epsilon:.2e}, {self.inequality}, "
            f"n = {self.sample_size})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
        """
        return {
            "statistic": self.statistic,
            "direction": self.direction,
            "critical_value": self.critical_value,
            "epsilon": self.epsilon,
            "false_positive_bound": self.false_positive_bound,
            "null": self.null,
            "inequality": self.inequality,
            "sample_size": self.sample_size,
            "derivation": self.derivation,
            "is_alarm": self.is_alarm,
            "reaches_its_statistic": self.reaches_its_statistic,
        }


def _binomial_upper_tails(trials: int, probability: float) -> list[float]:
    """Return ``P(X >= k)`` for every ``k``, for ``X ~ Binomial(n, q)``.

    Parameters
    ----------
    trials : int
        ``n``, at least ``0``.
    probability : float
        ``q``, strictly inside ``(0, 1)``.

    Returns
    -------
    list of float
        Length ``n + 2``.
    """
    log_q = math.log(probability)
    log_one_minus_q = math.log1p(-probability)
    log_factorial_n = math.lgamma(trials + 1)
    tails = [0.0] * (trials + 2)
    accumulated = 0.0
    for hits in range(trials, -1, -1):
        log_mass = (
            log_factorial_n
            - math.lgamma(hits + 1)
            - math.lgamma(trials - hits + 1)
            + hits * log_q
            + (trials - hits) * log_one_minus_q
        )
        accumulated += math.exp(log_mass) if log_mass > -745.0 else 0.0
        tails[hits] = min(1.0, accumulated)
    return tails


def _qber_critical_hoeffding(
    rounds: int, rate: float, epsilon: float
) -> tuple[int, float]:
    """Invert the additive Hoeffding upper tail for a binomial count.

    Parameters
    ----------
    rounds : int
        ``n``.
    rate : float
        ``q0``, strictly positive.
    epsilon : float
        The budget.

    Returns
    -------
    tuple of (int, float)
        The critical count and the bound the inequality proves at it.
    """
    mean = rounds * rate
    spread = math.sqrt(rounds * -math.log(epsilon) / 2.0)
    critical = max(math.ceil(mean + spread), math.floor(mean) + 1)
    if critical > rounds:
        return rounds + 1, 0.0
    gap = critical - mean
    return critical, min(epsilon, math.exp(-2.0 * gap * gap / rounds))


def _qber_critical_chernoff(
    rounds: int, rate: float, epsilon: float
) -> tuple[int, float]:
    """Invert the multiplicative Chernoff upper tail for a binomial count.

    Parameters
    ----------
    rounds : int
        ``n``.
    rate : float
        ``q0``, strictly positive.
    epsilon : float
        The budget.

    Returns
    -------
    tuple of (int, float)
        The critical count and the bound the inequality proves at it.
    """
    mean = rounds * rate
    budget = -math.log(epsilon)
    deviation = (
        budget + math.sqrt(budget * budget + 8.0 * mean * budget)
    ) / (2.0 * mean)
    critical = max(math.ceil((1.0 + deviation) * mean), math.floor(mean) + 1)
    if critical > rounds:
        return rounds + 1, 0.0
    realised = (critical - mean) / mean
    return critical, min(
        epsilon, math.exp(-realised * realised * mean / (2.0 + realised))
    )


def _qber_critical_exact(
    rounds: int, rate: float, epsilon: float
) -> tuple[int, float]:
    """Invert the exact binomial upper tail.

    Parameters
    ----------
    rounds : int
        ``n``.
    rate : float
        ``q0``, strictly positive.
    epsilon : float
        The budget, at or above :data:`_EXACT_METHOD_FLOOR`.

    Returns
    -------
    tuple of (int, float)
        The critical count and the tail at it, which is the exact
        false-positive probability rather than a bound on it.
    """
    tails = _binomial_upper_tails(rounds, rate)
    for critical in range(1, rounds + 1):
        if tails[critical] <= epsilon:
            return critical, tails[critical]
    return rounds + 1, 0.0


def qber_threshold(
    *,
    rounds: int,
    epsilon: float,
    tolerated_depolarising: float = 0.0,
    method: str = "exact",
) -> ChannelThreshold:
    """Derive the check-round error count at which one link is called disturbed.

    Parameters
    ----------
    rounds : int
        Keyword-only.
    epsilon : float
        Keyword-only.
    tolerated_depolarising : float, optional
        Keyword-only.
    method : {'exact', 'chernoff', 'hoeffding'}, optional
        Keyword-only.

    Returns
    -------
    ChannelThreshold
        With ``statistic="qber_errors"`` and ``direction="upper"``: fire when
        the observed error count is at or above
        :attr:`~ChannelThreshold.critical_value`.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If ``epsilon`` is outside ``(0, 1)``, ``rounds`` is negative,
        ``tolerated_depolarising`` is outside ``[0, 1]``, ``method`` names no
        inversion, or ``method='exact'`` is asked for at a budget below
        :data:`_EXACT_METHOD_FLOOR`.

    Examples
    --------
    >>> from qsecure.detect.thresholds_channel import qber_threshold
    >>> threshold = qber_threshold(rounds=4114, epsilon=1e-09)
    >>> threshold.critical_value, threshold.false_positive_bound
    (1.0, 0.0)
    >>> threshold.inequality
    'point mass (exact)'

    >>> for method in ("hoeffding", "chernoff", "exact"):
    ...     value = qber_threshold(
    ...         rounds=4114,
    ...         epsilon=1e-09,
    ...         tolerated_depolarising=1 / 32,
    ...         method=method,
    ...     ).critical_value
    ...     print(f"{method:10s} {value:6.1f}")
    hoeffding   271.0
    chernoff    128.0
    exact       118.0

    >>> [
    ...     qber_threshold(
    ...         rounds=4114, epsilon=eps, tolerated_depolarising=1 / 32
    ...     ).critical_value
    ...     for eps in (1e-02, 1e-06, 1e-12)
    ... ]
    [84.0, 106.0, 129.0]

    >>> tiny = qber_threshold(
    ...     rounds=12, epsilon=1e-09, tolerated_depolarising=0.5
    ... )
    >>> tiny.critical_value, tiny.reaches_its_statistic
    (13.0, False)
    >>> tiny.false_positive_bound
    0.0
    """
    sample = _as_sample_size(rounds, "rounds")
    budget = _as_epsilon(epsilon)
    strength = _as_strength(tolerated_depolarising)
    if not isinstance(method, str) or method not in _QBER_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_QBER_METHODS)}, got {method!r}. "
            f"Each names an inequality applied to the same stated null; they "
            f"differ in tightness, not in what they claim."
        )
    rate = depolarising_error_rate(strength)

    if rate == 0.0 or sample == 0:
        return ChannelThreshold(
            statistic="qber_errors",
            direction="upper",
            critical_value=1.0,
            epsilon=budget,
            false_positive_bound=0.0,
            null=(
                "point mass at 0: E = 0 with probability one (a non-negative "
                "variable with zero mean)"
            ),
            inequality="point mass (exact)",
            sample_size=sample,
            derivation=(
                "E >= 0 and E[E] = n * q0 = 0, so E = 0 almost surely and "
                "P(E >= 1) = 0 exactly, at every eps"
            ),
            reaches_its_statistic=sample >= 1,
        )

    if method == "exact":
        if budget < _EXACT_METHOD_FLOOR:
            raise ValueError(
                f"method='exact' needs epsilon at or above "
                f"{_EXACT_METHOD_FLOOR!r}; got {budget!r}. Below that the "
                f"binomial tail is summed in arithmetic that underflows, so "
                f"the search would report a rounded-down 0.0 as a proven "
                f"bound. Use method='chernoff' or 'hoeffding', which are "
                f"closed forms and have no such floor."
            )
        critical, bound = _qber_critical_exact(sample, rate, budget)
        inequality = "exact binomial tail"
        derivation = (
            f"k = min{{k : sum_{{j>=k}} C({sample},j) q0^j (1-q0)^({sample}-j) "
            f"<= eps}} with q0 = {rate!r}"
        )
    elif method == "chernoff":
        critical, bound = _qber_critical_chernoff(sample, rate, budget)
        inequality = "multiplicative Chernoff upper tail"
        derivation = (
            "P(E >= (1+d) mu) <= exp(-d^2 mu / (2+d)); d = [ln(1/eps) + "
            "sqrt(ln(1/eps)^2 + 8 mu ln(1/eps))] / (2 mu); k = ceil((1+d) mu) "
            f"with mu = {sample} * {rate!r}"
        )
    else:
        critical, bound = _qber_critical_hoeffding(sample, rate, budget)
        inequality = "Hoeffding upper tail"
        derivation = (
            "P(E >= n q0 + s) <= exp(-2 s^2 / n); s = sqrt(n ln(1/eps) / 2); "
            f"k = ceil(n q0 + s) with n = {sample} and q0 = {rate!r}"
        )

    return ChannelThreshold(
        statistic="qber_errors",
        direction="upper",
        critical_value=float(critical),
        epsilon=budget,
        false_positive_bound=bound,
        null=f"Binomial({sample}, {rate!r})",
        inequality=inequality,
        sample_size=sample,
        derivation=derivation,
        reaches_its_statistic=critical <= sample,
    )


def _as_chsh_counts(counts: Any) -> tuple[int, int, int, int]:
    """Coerce the four CHSH cell counts, refusing an empty cell.

    Parameters
    ----------
    counts : sequence of int
        The four cell counts, in
        :attr:`~qsecure.protocol.checkrounds.ChshEstimate.counts` order.

    Returns
    -------
    tuple of int

    Raises
    ------
    TypeError
        If ``counts`` is not a sequence of integers.
    ValueError
        If it does not hold exactly four entries, or if any is ``0``.
    """
    if isinstance(counts, (str, bytes)) or not isinstance(counts, Sequence):
        raise TypeError(
            f"counts must be a sequence of four ints, got "
            f"{type(counts).__name__}"
        )
    values = tuple(_as_sample_size(entry, "counts entry") for entry in counts)
    if len(values) != 4:
        raise ValueError(
            f"a CHSH statistic has exactly four cells -- two settings per wing "
            f"-- got {len(values)}"
        )
    empty = [index for index, value in enumerate(values) if value == 0]
    if empty:
        raise ValueError(
            f"CHSH cell(s) {empty} hold no rounds, so their correlators are "
            f"undefined and the bounded-difference constant 2/n_c is "
            f"infinite. A threshold cannot be derived for a statistic that "
            f"does not exist; the caller should read "
            f"LinkStatistics.chsh_unavailable and record the statistic as "
            f"unavailable rather than as clean."
        )
    return values  # type: ignore[return-value]


def _chsh_half_width(counts: Sequence[int], epsilon: float) -> float:
    """Return the bounded-difference half-width ``t(eps)`` for ``S``.

    Parameters
    ----------
    counts : sequence of int
        The four cell counts, all positive.
    epsilon : float
        The budget.

    Returns
    -------
    float
        ``sqrt(2 ln(1/eps) sum_c 1/n_c)``.
    """
    reciprocal = sum(1.0 / count for count in counts)
    return math.sqrt(2.0 * -math.log(epsilon) * reciprocal)


def chsh_threshold(
    *,
    counts: Sequence[int],
    epsilon: float,
    tolerated_depolarising: float = 0.0,
) -> ChannelThreshold:
    """Derive the ``S`` below which one link's Bell test is called disturbed.

    Parameters
    ----------
    counts : sequence of int
        Keyword-only.
    epsilon : float
        Keyword-only.
    tolerated_depolarising : float, optional
        Keyword-only.

    Returns
    -------
    ChannelThreshold
        With ``statistic="chsh"`` and ``direction="lower"``: fire when the
        observed ``S`` is at or below :attr:`~ChannelThreshold.critical_value`.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If ``counts`` is not four positive integers, ``epsilon`` is outside
        ``(0, 1)``, or ``tolerated_depolarising`` is outside ``[0, 1]``.

    Examples
    --------
    >>> import math
    >>> from qsecure.protocol.checkrounds import CHECK_CONFIDENCE
    >>> alpha = 1.0 - CHECK_CONFIDENCE            # 0.01, two-sided
    >>> summed = 4 * math.sqrt(8 * math.log(8 / alpha))     # x 1/sqrt(N)
    >>> direct = math.sqrt(32 * -math.log(alpha / 2))       # x 1/sqrt(N)
    >>> f"{summed:.2f}", f"{direct:.2f}", f"{summed / direct:.2f}"
    ('29.25', '13.02', '2.25')
    >>> f"{(summed / direct) ** 2:.2f}"
    '5.05'

    >>> from qsecure.detect.thresholds_channel import chsh_threshold
    >>> counts = (1028, 1029, 1029, 1029)
    >>> for eps in (1e-03, 1e-06, 1e-09):
    ...     threshold = chsh_threshold(counts=counts, epsilon=eps)
    ...     print(f"{eps:.0e}  S <= {threshold.critical_value:.4f}")
    1e-03  S <= 2.5967
    1e-06  S <= 2.5007
    1e-09  S <= 2.4270

    >>> import math
    >>> from qsecure.protocol.checkrounds import IDEAL_CHSH, normal_quantile
    >>> rounds = sum(counts)
    >>> threshold = chsh_threshold(counts=counts, epsilon=1e-09)
    >>> gap = IDEAL_CHSH - threshold.critical_value
    >>> f"{gap / math.sqrt(8 / rounds):.4f}"
    '9.1046'
    >>> f"{2 * math.sqrt(-math.log(1e-09)):.4f}"
    '9.1046'

    >>> f"{normal_quantile(1 - 1e-09):.4f}"
    '5.9978'
    >>> f"{2 * math.sqrt(-math.log(1e-09)) / normal_quantile(1 - 1e-09):.4f}"
    '1.5180'

    >>> from qsecure.protocol.checkrounds import CLASSICAL_CHSH_BOUND
    >>> short = chsh_threshold(counts=(6, 6, 6, 6), epsilon=1e-09)
    >>> f"{short.critical_value:.4f}"
    '-2.4281'
    >>> short.critical_value > CLASSICAL_CHSH_BOUND, short.reaches_its_statistic
    (False, True)

    >>> noisy = chsh_threshold(
    ...     counts=counts, epsilon=1e-09, tolerated_depolarising=0.1
    ... )
    >>> f"{noisy.critical_value:.4f}"
    '2.1441'
    """
    cells = _as_chsh_counts(counts)
    budget = _as_epsilon(epsilon)
    strength = _as_strength(tolerated_depolarising)
    rounds = sum(cells)
    null_mean = depolarising_chsh(strength)
    half_width = _chsh_half_width(cells, budget)
    critical = null_mean - half_width
    return ChannelThreshold(
        statistic="chsh",
        direction="lower",
        critical_value=critical,
        epsilon=budget,
        false_positive_bound=budget,
        null=(
            f"E[S] = {null_mean!r} for a Werner resource of strength "
            f"{strength!r}, over {rounds} independent rounds in cells {cells}"
        ),
        inequality="McDiarmid bounded differences (one-sided)",
        sample_size=rounds,
        derivation=(
            "flipping one round of cell c moves S by 2/n_c, so sum c_i^2 = "
            "4 sum_c 1/n_c and P(S <= S0 - t) <= exp(-t^2 / (2 sum_c 1/n_c)); "
            f"t = sqrt(2 ln(1/eps) sum_c 1/n_c) = {half_width!r}"
        ),
        reaches_its_statistic=critical >= -4.0,
    )


def chsh_certificate_threshold(
    *, counts: Sequence[int], epsilon: float
) -> ChannelThreshold:
    """Derive the ``S`` above which entanglement on arrival is *certified*.

    Parameters
    ----------
    counts : sequence of int
        Keyword-only.
    epsilon : float
        Keyword-only.

    Returns
    -------
    ChannelThreshold
        With ``statistic="chsh"``, ``direction="upper"`` and
        ``is_alarm=False``.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If ``counts`` is not four positive integers or ``epsilon`` is outside
        ``(0, 1)``.

    Examples
    --------
    >>> import math
    >>> from qsecure.detect.thresholds_channel import chsh_certificate_threshold
    >>> certificate = chsh_certificate_threshold(
    ...     counts=(1028, 1029, 1029, 1029), epsilon=1e-09
    ... )
    >>> f"{certificate.critical_value:.4f}", certificate.is_alarm
    ('2.4014', False)
    >>> certificate.reaches_its_statistic
    True

    >>> from qsecure.protocol.checkrounds import (
    ...     CLASSICAL_CHSH_BOUND, IDEAL_CHSH, normal_quantile,
    ... )
    >>> gap = IDEAL_CHSH - CLASSICAL_CHSH_BOUND
    >>> math.ceil(32 * -math.log(1e-09) / gap ** 2)
    967
    >>> math.ceil(8 * normal_quantile(1 - 1e-09) ** 2 / gap ** 2)
    420

    >>> short = chsh_certificate_threshold(counts=(6, 6, 6, 6), epsilon=1e-09)
    >>> f"{short.critical_value:.4f}", short.reaches_its_statistic
    ('7.2565', False)
    """
    cells = _as_chsh_counts(counts)
    budget = _as_epsilon(epsilon)
    rounds = sum(cells)
    half_width = _chsh_half_width(cells, budget)
    critical = CLASSICAL_CHSH_BOUND + half_width
    return ChannelThreshold(
        statistic="chsh",
        direction="upper",
        critical_value=critical,
        epsilon=budget,
        # Exactly the budget, and for the same reason chsh_threshold gives.
        false_positive_bound=budget,
        null=(
            f"S_true <= {CLASSICAL_CHSH_BOUND!r}, the classical bound, over "
            f"{rounds} independent rounds in cells {cells}"
        ),
        inequality="McDiarmid bounded differences (one-sided)",
        sample_size=rounds,
        derivation=(
            "P(S >= S_true + t) <= exp(-t^2 / (2 sum_c 1/n_c)); with "
            f"S_true <= 2 the certificate fires above 2 + t, t = {half_width!r}"
        ),
        is_alarm=False,
        reaches_its_statistic=critical < IDEAL_CHSH,
    )


def bounded_mean_threshold(
    *,
    samples: int,
    epsilon: float,
    honest_mean: float,
    span: tuple[float, float],
    statistic: str,
    quantity: str,
    tolerance: float = IDEAL_TOLERANCE,
) -> ChannelThreshold:
    """Derive a lower threshold on the mean of a bounded per-round summary.

    Parameters
    ----------
    samples : int
        Keyword-only.
    epsilon : float
        Keyword-only.
    honest_mean : float
        Keyword-only.
    span : tuple of float
        Keyword-only.
    statistic : str
        Keyword-only.
    quantity : str
        Keyword-only.
    tolerance : float, optional
        Keyword-only.

    Returns
    -------
    ChannelThreshold
        With ``direction="lower"``.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If ``span`` is not an ordered pair, ``honest_mean`` lies outside it,
        ``honest_mean`` sits on the bottom of the span -- where the point-mass
        argument would run the other way and a *lower* threshold is meaningless
        -- ``tolerance`` is not positive and finite, or ``samples`` is
        negative.

    Examples
    --------
    >>> from qsecure.detect.thresholds_channel import bounded_mean_threshold
    >>> ideal = bounded_mean_threshold(
    ...     samples=48,
    ...     epsilon=1e-09,
    ...     honest_mean=1.0,
    ...     span=(0.0, 1.0),
    ...     statistic="min_fidelity",
    ...     quantity="fidelity",
    ... )
    >>> ideal.critical_value, ideal.false_positive_bound
    (0.999999999, 0.0)
    >>> noisy = bounded_mean_threshold(
    ...     samples=48,
    ...     epsilon=1e-09,
    ...     honest_mean=0.925,
    ...     span=(0.0, 1.0),
    ...     statistic="mean_fidelity",
    ...     quantity="fidelity",
    ... )
    >>> f"{noisy.critical_value:.6f}", noisy.inequality
    ('0.460385', 'Hoeffding (bounded means, one-sided)')
    """
    size = _as_sample_size(samples, "samples")
    budget = _as_epsilon(epsilon)
    if isinstance(span, (str, bytes)) or not isinstance(span, Sequence):
        raise TypeError(
            f"span must be a pair of floats, got {type(span).__name__}"
        )
    if len(span) != 2:
        raise ValueError(
            f"span must hold exactly two endpoints, got {len(span)}"
        )
    low, high = (float(span[0]), float(span[1]))
    if not (math.isfinite(low) and math.isfinite(high)) or not low < high:
        raise ValueError(
            f"span must be a finite ordered pair a < b, got ({low!r}, {high!r})"
        )
    if isinstance(honest_mean, bool) or not isinstance(
        honest_mean, (int, float)
    ):
        raise TypeError(
            f"honest_mean must be a real number, got "
            f"{type(honest_mean).__name__}"
        )
    mean = float(honest_mean)
    if not math.isfinite(mean) or not low <= mean <= high:
        raise ValueError(
            f"honest_mean must lie inside span [{low!r}, {high!r}], got "
            f"{mean!r}"
        )
    if mean == low:
        raise ValueError(
            f"honest_mean sits on the bottom of span ({low!r}), where the "
            f"quantity is already as low as it can go and a lower-tail "
            f"threshold detects nothing. A concurrence null of 0 is what a "
            f"separable honest resource looks like; there is no deviation "
            f"below it to bound."
        )
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise TypeError(
            f"tolerance must be a real number, got {type(tolerance).__name__}"
        )
    slack = float(tolerance)
    if not math.isfinite(slack) or slack <= 0.0:
        raise ValueError(
            f"tolerance must be finite and positive, got {slack!r}. It absorbs "
            f"the floating-point dust of a 4x4 eigendecomposition; zero would "
            f"turn that dust into a false alarm."
        )

    if mean == high:
        return ChannelThreshold(
            statistic=statistic,
            direction="lower",
            critical_value=high - slack,
            epsilon=budget,
            false_positive_bound=0.0,
            null=(
                f"point mass at {high!r}: the honest {quantity} is bounded "
                f"above by {high!r} and has mean {high!r}, which forces every "
                f"round onto it"
            ),
            inequality="point mass (exact)",
            sample_size=size,
            derivation=(
                f"V <= {high!r} and E[V] = {high!r} give E[{high!r} - V] = 0 "
                f"with {high!r} - V >= 0, so V = {high!r} almost surely; the "
                f"mean and the minimum both sit on it and "
                f"P(V <= {high!r} - tol) = 0 exactly, at every eps"
            ),
            reaches_its_statistic=size >= 1,
        )

    span_width = high - low
    half_width = (
        span_width * math.sqrt(-math.log(budget) / (2.0 * size))
        if size
        else math.inf
    )
    if not math.isfinite(half_width):
        return ChannelThreshold(
            statistic=statistic,
            direction="lower",
            critical_value=low - 1.0,
            epsilon=budget,
            false_positive_bound=0.0,
            null=(
                f"i.i.d. {quantity} in [{low!r}, {high!r}] with mean {mean!r}, "
                f"over 0 rounds -- no sample, so no statistic"
            ),
            inequality="Hoeffding (bounded means, one-sided)",
            sample_size=0,
            derivation=(
                "t(eps) = (b - a) sqrt(ln(1/eps) / (2n)) is unbounded at "
                "n = 0; the threshold is placed one unit below the range's "
                "floor, where no observation can reach it"
            ),
            reaches_its_statistic=False,
        )
    critical = mean - half_width
    return ChannelThreshold(
        statistic=statistic,
        direction="lower",
        critical_value=critical,
        epsilon=budget,
        false_positive_bound=budget,
        null=(
            f"i.i.d. {quantity} in [{low!r}, {high!r}] with mean {mean!r}, "
            f"over {size} monitored rounds"
        ),
        inequality="Hoeffding (bounded means, one-sided)",
        sample_size=size,
        derivation=(
            "P(mean <= mu0 - t) <= exp(-2 n t^2 / (b - a)^2); "
            f"t = (b - a) sqrt(ln(1/eps) / (2n)) = {half_width!r}"
        ),
        reaches_its_statistic=critical >= low,
    )


def _resource_threshold(
    *,
    samples: int,
    epsilon: float,
    tolerance: float,
    honest: float,
    span: tuple[float, float],
    quantity: str,
) -> ChannelThreshold:
    """Build one resource threshold, naming the field it should be read from.

    Parameters
    ----------
    samples : int
        Monitored rounds on this link.
    epsilon : float
        The budget.
    tolerance : float
        Point-mass slack.
    honest : float
        The Werner mean at ``p0``, which is where the null is stated and is the
        only thing the derivation needs ``p0`` for.
    span : tuple of float
        The quantity's range.
    quantity : str
        ``"fidelity"``, ``"purity"`` or ``"concurrence"``.

    Returns
    -------
    ChannelThreshold
    """
    field = "min" if honest == span[1] else "mean"
    return bounded_mean_threshold(
        samples=samples,
        epsilon=epsilon,
        honest_mean=honest,
        span=span,
        statistic=f"{field}_{quantity}",
        quantity=quantity,
        tolerance=tolerance,
    )


def fidelity_threshold(
    *,
    samples: int,
    epsilon: float,
    tolerated_depolarising: float = 0.0,
    tolerance: float = IDEAL_TOLERANCE,
) -> ChannelThreshold:
    """Derive the fidelity below which one link's resource is called disturbed.

    Parameters
    ----------
    samples : int
        Keyword-only.
    epsilon : float
        Keyword-only.
    tolerated_depolarising : float, optional
        Keyword-only.
    tolerance : float, optional
        Keyword-only.

    Returns
    -------
    ChannelThreshold
        Named ``"min_fidelity"`` under the point-mass null and
        ``"mean_fidelity"`` otherwise, because only the mean has a law the
        stated mean implies.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> from qsecure.detect.thresholds_channel import fidelity_threshold
    >>> ideal = fidelity_threshold(samples=48, epsilon=1e-09)
    >>> ideal.statistic, ideal.false_positive_bound
    ('min_fidelity', 0.0)
    >>> noisy = fidelity_threshold(
    ...     samples=48, epsilon=1e-09, tolerated_depolarising=1 / 32
    ... )
    >>> noisy.statistic, f"{noisy.critical_value:.6f}"
    ('mean_fidelity', '0.511947')
    """
    return _resource_threshold(
        samples=samples,
        epsilon=epsilon,
        tolerance=tolerance,
        honest=werner_fidelity(tolerated_depolarising),
        span=FIDELITY_RANGE,
        quantity="fidelity",
    )


def purity_threshold(
    *,
    samples: int,
    epsilon: float,
    tolerated_depolarising: float = 0.0,
    tolerance: float = IDEAL_TOLERANCE,
) -> ChannelThreshold:
    """Derive the purity below which one link's resource is called disturbed.

    Parameters
    ----------
    samples : int
        Keyword-only.
    epsilon : float
        Keyword-only.
    tolerated_depolarising : float, optional
        Keyword-only.
    tolerance : float, optional
        Keyword-only.

    Returns
    -------
    ChannelThreshold
        Named ``"min_purity"`` under the point-mass null and ``"mean_purity"``
        otherwise.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> from qsecure.detect.thresholds_channel import purity_threshold
    >>> ideal = purity_threshold(samples=48, epsilon=1e-09)
    >>> ideal.statistic, ideal.critical_value, ideal.false_positive_bound
    ('min_purity', 0.999999999, 0.0)
    >>> noisy = purity_threshold(
    ...     samples=48, epsilon=1e-09, tolerated_depolarising=0.1
    ... )
    >>> noisy.statistic, f"{noisy.critical_value:.6f}"
    ('mean_purity', '0.509039')

    >>> from qsecure.detect.thresholds_channel import bounded_mean_threshold
    >>> wide = bounded_mean_threshold(
    ...     samples=48,
    ...     epsilon=1e-09,
    ...     honest_mean=0.8575,
    ...     span=(0.0, 1.0),
    ...     statistic="mean_purity",
    ...     quantity="purity",
    ... )
    >>> f"{0.8575 - wide.critical_value:.6f}"
    '0.464615'
    >>> f"{0.8575 - noisy.critical_value:.6f}"
    '0.348461'
    """
    return _resource_threshold(
        samples=samples,
        epsilon=epsilon,
        tolerance=tolerance,
        honest=werner_purity(tolerated_depolarising),
        span=PURITY_RANGE,
        quantity="purity",
    )


def concurrence_threshold(
    *,
    samples: int,
    epsilon: float,
    tolerated_depolarising: float = 0.0,
    tolerance: float = IDEAL_TOLERANCE,
) -> ChannelThreshold:
    """Derive the concurrence below which one link's pair is called broken.

    Parameters
    ----------
    samples : int
        Keyword-only.
    epsilon : float
        Keyword-only.
    tolerated_depolarising : float, optional
        Keyword-only.
    tolerance : float, optional
        Keyword-only.

    Returns
    -------
    ChannelThreshold
        Named ``"min_concurrence"`` under the point-mass null and
        ``"mean_concurrence"`` otherwise.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If an argument is out of range, or if ``tolerated_depolarising`` is at
        or above ``2/3``, where the honest null is already zero.

    Examples
    --------
    >>> from qsecure.detect.thresholds_channel import concurrence_threshold
    >>> ideal = concurrence_threshold(samples=48, epsilon=1e-09)
    >>> ideal.statistic, ideal.false_positive_bound
    ('min_concurrence', 0.0)
    >>> noisy = concurrence_threshold(
    ...     samples=48, epsilon=1e-09, tolerated_depolarising=0.1
    ... )
    >>> noisy.statistic, f"{noisy.critical_value:.6f}"
    ('mean_concurrence', '0.385385')

    >>> concurrence_threshold(
    ...     samples=48, epsilon=1e-09, tolerated_depolarising=0.7
    ... )                                        # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: honest_mean sits on the bottom of span (0.0), where the...
    """
    return _resource_threshold(
        samples=samples,
        epsilon=epsilon,
        tolerance=tolerance,
        honest=werner_concurrence(tolerated_depolarising),
        span=CONCURRENCE_RANGE,
        quantity="concurrence",
    )


@dataclass(frozen=True)
class ChannelScreen:
    """The channel family applied to one link, with its union bound."""

    party: Party
    message_bit: int
    epsilon: float
    tolerated_depolarising: float
    thresholds: Mapping[str, ChannelThreshold]
    observed: Mapping[str, float]
    fired: tuple[str, ...]
    cleared: tuple[str, ...]
    unavailable: Mapping[str, str]
    false_positive_bound: float

    @property
    def detected(self) -> bool:
        """bool: Whether any threshold fired."""
        return bool(self.fired)

    @property
    def evaluated(self) -> int:
        """int: How many thresholds were evaluated, the union bound's ``m``."""
        return len(self.thresholds)

    def summary(self) -> str:
        """Return a one-line account of the screen.

        Returns
        -------
        str
        """
        verdict = ",".join(self.fired) if self.fired else "nothing fired"
        return (
            f"{self.party}/{self.message_bit}: {verdict} "
            f"({self.evaluated} evaluated, {len(self.unavailable)} "
            f"unavailable, P(any fire | honest) <= "
            f"{self.false_positive_bound:.2e} at eps = {self.epsilon:.2e})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
        """
        return {
            "party": str(self.party),
            "message_bit": self.message_bit,
            "epsilon": self.epsilon,
            "tolerated_depolarising": self.tolerated_depolarising,
            "thresholds": {
                name: threshold.to_dict()
                for name, threshold in self.thresholds.items()
            },
            "observed": dict(self.observed),
            "fired": list(self.fired),
            "cleared": list(self.cleared),
            "unavailable": dict(self.unavailable),
            "false_positive_bound": self.false_positive_bound,
            "detected": self.detected,
        }


def screen_link(
    link: LinkStatistics,
    *,
    epsilon: float,
    tolerated_depolarising: float = 0.0,
    tolerance: float = IDEAL_TOLERANCE,
    qber_method: str = "exact",
) -> ChannelScreen:
    """Apply the whole channel family to one link at one budget.

    Parameters
    ----------
    link : LinkStatistics
        One link's statistics, from a JSON-round-tripped transcript.
    epsilon : float
        Keyword-only.
    tolerated_depolarising : float, optional
        Keyword-only.
    tolerance : float, optional
        Keyword-only.
    qber_method : {'exact', 'chernoff', 'hoeffding'}, optional
        Keyword-only.

    Returns
    -------
    ChannelScreen

    Raises
    ------
    TypeError
        If ``link`` is not a
        :class:`~qsecure.detect.statistics.LinkStatistics`, or an argument has
        the wrong type.
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.detect.statistics import TranscriptStatistics
    >>> from qsecure.detect.thresholds_channel import screen_link
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> stats = TranscriptStatistics.from_transcript(
    ...     QDSSession(
    ...         ProtocolParams(key_length=384, check_fraction=0.25),
    ...         rng=np.random.default_rng(7),
    ...     ).run(0)
    ... )
    >>> screen = screen_link(stats.link("Charlie", 1), epsilon=1e-09)
    >>> screen.detected, screen.evaluated, dict(screen.unavailable)
    (False, 5, {})
    >>> print(screen.summary())         # doctest: +NORMALIZE_WHITESPACE
    Charlie/1: nothing fired (5 evaluated, 0 unavailable,
    P(any fire | honest) <= 2.00e-10 at eps = 1.00e-09)

    >>> plain = TranscriptStatistics.from_transcript(
    ...     QDSSession(
    ...         ProtocolParams(key_length=192), rng=np.random.default_rng(7)
    ...     ).run(0)
    ... )
    >>> plain.links
    {}
    """
    if not isinstance(link, LinkStatistics):
        raise TypeError(
            f"link must be a LinkStatistics, got {type(link).__name__}. The "
            f"detector reads a JSON-round-tripped transcript through "
            f"TranscriptStatistics and nothing else."
        )
    budget = _as_epsilon(epsilon)
    strength = _as_strength(tolerated_depolarising)

    unavailable: dict[str, str] = {}
    observed: dict[str, float] = {}

    resource = link.resource
    chsh_counts = None if link.chsh is None else link.chsh.counts
    evaluable: list[tuple[str, float]] = []

    if link.errors.trials == 0:
        unavailable["qber_errors"] = "this link published no QBER check rounds"
    else:
        evaluable.append(("qber_errors", float(link.errors.count)))

    if link.chsh is None or chsh_counts is None:
        unavailable["chsh"] = (
            link.chsh_unavailable or "this link published no CHSH estimate"
        )
    else:
        evaluable.append(("chsh", float(link.chsh.statistic)))

    prefix = "min" if strength == 0.0 else "mean"
    separable = werner_concurrence(strength) == 0.0
    for quantity in ("fidelity", "purity", "concurrence"):
        value = getattr(resource, f"{prefix}_{quantity}")
        if resource.samples == 0 or value is None:
            unavailable[quantity] = (
                "this link's channel seam was never called, so it has no "
                "resource summary -- an unmonitored link is not a clean one"
            )
        elif quantity == "concurrence" and separable:
            # Decided from the stated noise level alone, never from the
            # observation, so the budget split stays independent of the data.
            unavailable[quantity] = (
                f"a Werner resource at the stated strength "
                f"{strength!r} is separable, so the honest null is already "
                f"zero and there is no lower tail to bound"
            )
        else:
            evaluable.append((quantity, float(value)))

    if not evaluable:
        return ChannelScreen(
            party=link.party,
            message_bit=link.message_bit,
            epsilon=budget,
            tolerated_depolarising=strength,
            thresholds={},
            observed={},
            fired=(),
            cleared=(),
            unavailable=dict(unavailable),
            false_positive_bound=0.0,
        )

    share = divide_budget(budget, len(evaluable))
    thresholds: dict[str, ChannelThreshold] = {}
    for name, value in evaluable:
        if name == "qber_errors":
            threshold = qber_threshold(
                rounds=link.errors.trials,
                epsilon=share,
                tolerated_depolarising=strength,
                method=qber_method,
            )
        elif name == "chsh" and chsh_counts is not None:
            threshold = chsh_threshold(
                counts=chsh_counts,
                epsilon=share,
                tolerated_depolarising=strength,
            )
        else:
            builder = {
                "fidelity": fidelity_threshold,
                "purity": purity_threshold,
                "concurrence": concurrence_threshold,
            }[name]
            threshold = builder(
                samples=resource.samples,
                epsilon=share,
                tolerated_depolarising=strength,
                tolerance=tolerance,
            )
        if not threshold.is_alarm:  # pragma: no cover - none of the five is
            raise ValueError(
                f"{threshold.statistic}: a screen evaluates alarms only. "
                f"Failing to satisfy a certificate is not a detection."
            )
        thresholds[threshold.statistic] = threshold
        observed[threshold.statistic] = value

    fired = tuple(
        sorted(
            name
            for name, threshold in thresholds.items()
            if threshold.fires(observed[name])
        )
    )
    cleared = tuple(sorted(set(thresholds) - set(fired)))
    return ChannelScreen(
        party=link.party,
        message_bit=link.message_bit,
        epsilon=budget,
        tolerated_depolarising=strength,
        thresholds=thresholds,
        observed=observed,
        fired=fired,
        cleared=cleared,
        unavailable=dict(unavailable),
        false_positive_bound=min(
            1.0,
            math.fsum(
                threshold.false_positive_bound
                for threshold in thresholds.values()
            ),
        ),
    )
