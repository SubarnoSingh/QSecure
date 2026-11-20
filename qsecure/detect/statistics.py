"""Statistics a detector may read from a JSON transcript, each with its
honest-run null.

Examples
--------
>>> import numpy as np
>>> from qsecure.detect.statistics import TranscriptStatistics
>>> from qsecure.protocol.params import ProtocolParams
>>> from qsecure.protocol.session import QDSSession
>>> transcript = QDSSession(
...     ProtocolParams(key_length=192), rng=np.random.default_rng(7)
... ).run(0)
>>> stats = TranscriptStatistics.from_transcript(transcript)
>>> stats.key_length, stats.match_probability
(192, 0.3333333333333333)
>>> bob = stats.verifier("Bob")
>>> bob.matched.count, bob.mismatch.count, bob.mismatch_rate
(69, 0, 0.0)
>>> f"{bob.matched.z_score:.4f}"
'0.7655'

>>> stats.pooled.pooled.count, f"{stats.pooled.pooled.z_score:.4f}"
(137, '0.9743')

>>> stats.aborts.total, stats.replay.total_refusals, stats.replay.spent_rounds
(0, 0, 2)
>>> stats.transferable, stats.repudiated
(True, False)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from qsecure.protocol.checkrounds import (
    CHECK_CONFIDENCE,
    IDEAL_CHSH,
    CheckLog,
    CheckRole,
    ChshEstimate,
    Interval,
    QberEstimate,
    QberObservation,
    estimate_chsh,
    estimate_qber,
    normal_quantile,
)
from qsecure.protocol.params import Party, ProtocolParams
from qsecure.protocol.records import RecipientRecord
from qsecure.protocol.session import ChannelSample, SessionTranscript
from qsecure.protocol.signature import Signature
from qsecure.protocol.verify import (
    HONEST_ABORT_BUDGET,
    AbortReason,
)
from qsecure.protocol.verify import (
    enforced_repudiation_bound as _enforced_repudiation_bound,
)
from qsecure.protocol.verify import (
    VerificationResult,
    matched_positions,
    minimum_matched_count,
    minimum_pooled_matched_count,
    mismatch_positions,
)

__all__ = [
    "EVIDENCE_ABORT_REASONS",
    "IDEAL_TOLERANCE",
    "STRUCTURAL_ABORT_REASONS",
    "AbortStatistics",
    "CountStatistic",
    "DeclarationStatistics",
    "LinkStatistics",
    "PooledStatistics",
    "RecordStatistics",
    "ReplayStatistics",
    "ResourceStatistics",
    "TranscriptStatistics",
    "VerifierStatistics",
    "chernoff_deviation_bound",
    "evidence_abort_probability_bound",
    "floor_shortfall_bound",
    "pooled_check_qber",
    "why_wings_agree_is_absent",
    "wilson_interval",
]


STRUCTURAL_ABORT_REASONS: Final[frozenset[AbortReason]] = frozenset(
    {
        AbortReason.SESSION_MISMATCH,
        AbortReason.RECORD_ALREADY_VERIFIED,
        AbortReason.COUNT_OF_UNRECORDED_PROVENANCE,
        AbortReason.COUNTS_FROM_TWO_DECLARATIONS,
        AbortReason.UNAUTHORISED_VERIFIER,
    }
)

EVIDENCE_ABORT_REASONS: Final[frozenset[AbortReason]] = frozenset(
    {
        AbortReason.EMPTY_MATCHED_SET,
        AbortReason.BELOW_FLOOR,
        AbortReason.POOLED_BELOW_FLOOR,
        AbortReason.COUNTERPART_BELOW_FLOOR,
    }
)

IDEAL_TOLERANCE: Final[float] = 1e-9


def why_wings_agree_is_absent() -> str:
    """Return the reason ``wings_agree`` is not a statistic in this module.

    Returns
    -------
    str
        One paragraph, suitable for pasting into a report that has to explain
        why an obvious-looking signal is missing.

    Examples
    --------
    >>> from qsecure.detect.statistics import why_wings_agree_is_absent
    >>> "no detection power" in why_wings_agree_is_absent()
    True
    """
    return (
        "wings_agree is excluded from the statistics layer on "
        "purpose. No trace-preserving map on one half of a maximally "
        "entangled pair can change only that half's marginal, so every "
        "channel adversary in the threat model leaves it True on every "
        "round: it has no detection power against that family, as "
        "measured. Read per-link check-round QBER, fidelity and concurrence "
        "instead."
    )


def _as_count(value: Any, name: str) -> int:
    """Coerce a non-negative integer count, refusing booleans.

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
            f"{name} must be an int, got {type(value).__name__}. Counts of "
            f"trials are integers; a float here is usually a rate that has "
            f"already been divided once."
        )
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _as_probability(value: Any, name: str) -> float:
    """Coerce a probability in the closed unit interval, refusing booleans.

    Parameters
    ----------
    value : float
        The candidate.
    name : str
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
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"{name} must be a finite probability in [0, 1], got "
            f"{probability!r}. It is the null's parameter, not a count."
        )
    return probability


def wilson_interval(
    successes: int, trials: int, *, confidence: float = CHECK_CONFIDENCE
) -> Interval:
    """Return the Wilson score interval for a binomial proportion.

    Parameters
    ----------
    successes : int
        ``0 <= successes <= trials``.
    trials : int
        At least ``1``.
    confidence : float, optional
        Keyword-only two-sided coverage, strictly inside ``(0, 1)``.

    Returns
    -------
    Interval
        The protocol's own interval type, tagged ``method="wilson"``.

    Raises
    ------
    TypeError
        If a count is not an integer or ``confidence`` is not a real number.
    ValueError
        If ``trials < 1``, ``successes`` is outside ``[0, trials]``, or
        ``confidence`` is not strictly inside ``(0, 1)``.

    Examples
    --------
    >>> from qsecure.detect.statistics import wilson_interval
    >>> interval = wilson_interval(0, 400)
    >>> interval.low, f"{interval.high:.6f}"
    (0.0, '0.016317')
    >>> wilson_interval(400, 400).high
    1.0

    >>> from qsecure.core.paulis import PauliBasis
    >>> from qsecure.protocol.checkrounds import QberObservation, estimate_qber
    >>> sample = [
    ...     QberObservation(i, PauliBasis.Z, 1, 1 if i < 7 else -1)
    ...     for i in range(50)
    ... ]
    >>> theirs = estimate_qber(sample)
    >>> theirs.errors, theirs.rounds
    (43, 50)
    >>> ours = wilson_interval(theirs.errors, theirs.rounds)
    >>> (ours.low, ours.high) == (theirs.interval.low, theirs.interval.high)
    True
    """
    total = _as_count(trials, "trials")
    hits = _as_count(successes, "successes")
    if total < 1:
        raise ValueError(
            f"trials must be at least 1, got {total}. A proportion measured "
            f"over no trials is not a wide interval, it is the absence of a "
            f"measurement; report the trial count instead."
        )
    if hits > total:
        raise ValueError(
            f"successes ({hits}) exceeds trials ({total}); a proportion above "
            f"one is a counting error upstream, not a rate."
        )
    level = _as_probability(confidence, "confidence")
    if not 0.0 < level < 1.0:
        raise ValueError(
            f"confidence must lie strictly inside (0, 1), got {level}. It is "
            f"a coverage probability, not a percentage: pass 0.99, not 99."
        )

    z = normal_quantile(0.5 + level / 2.0)
    z_squared = z * z
    denominator = total + z_squared
    centre = (hits + z_squared / 2.0) / denominator
    spread = (
        z
        / denominator
        * math.sqrt(hits * (total - hits) / total + z_squared / 4.0)
    )
    low = 0.0 if hits == 0 else max(0.0, centre - spread)
    high = 1.0 if hits == total else min(1.0, centre + spread)
    return Interval(low=low, high=high, confidence=level, method="wilson")


#: The three sides :func:`chernoff_deviation_bound` will bound.
_SIDES: Final[frozenset[str]] = frozenset({"two-sided", "lower", "upper"})

_LOG_FLOOR_BUDGET: Final[float] = math.log(1.0 / HONEST_ABORT_BUDGET)


def evidence_abort_probability_bound(
    params: ProtocolParams, *, counts_exchanged: bool = True
) -> float:
    """Bound P(an honest run records an evidence abort), as a function of ``n``.

    Parameters
    ----------
    params : ProtocolParams
        The run's **sifted** parameter set -- ``TranscriptStatistics.params``,
        never the nominal one.
    counts_exchanged : bool, optional
        Keyword-only.

    Returns
    -------
    float
        The bound, in ``[0, 1]``.

    Examples
    --------
    >>> from qsecure.detect.statistics import evidence_abort_probability_bound
    >>> from qsecure.protocol.params import ProtocolParams
    >>> for length in (24, 96, 136, 137, 273, 115200):
    ...     bound = evidence_abort_probability_bound(
    ...         ProtocolParams(key_length=length))
    ...     print(f"{length:6d}  {bound:.4e}")
        24  1.1881e-04
        96  2.4904e-17
       136  2.2523e-24
       137  5.4212e-20
       273  1.6263e-19
    115200  1.6263e-19

    >>> from qsecure.protocol.params import DEFAULT_PARAMS
    >>> evidence_abort_probability_bound(DEFAULT_PARAMS) == 3 * 2.0 ** -64
    True
    >>> evidence_abort_probability_bound(
    ...     DEFAULT_PARAMS, counts_exchanged=False) == 2 * 2.0 ** -64
    True
    """
    length = int(params.key_length)
    probability = float(params.match_probability)
    bound = 2.0 * floor_shortfall_bound(
        length, probability, minimum_matched_count(params)
    )
    if counts_exchanged:
        bound += floor_shortfall_bound(
            2 * length, probability, minimum_pooled_matched_count(params)
        )
    return min(1.0, bound)


def floor_shortfall_bound(trials: int, probability: float, floor: int) -> float:
    """Bound ``P(Binomial(trials, probability) < floor)`` in its two exact
    regimes.

    Parameters
    ----------
    trials : int
        ``n`` for a per-verifier floor, ``2n`` for the pooled one.
    probability : float
        ``p = 1/|B|``.
    floor : int
        The floor.

    Returns
    -------
    float
        The bound, in ``[0, 1]``.

    Examples
    --------
    >>> from qsecure.detect.statistics import floor_shortfall_bound
    >>> f"{floor_shortfall_bound(115200, 1 / 3, 36555):.4e}"
    '5.4210e-20'
    >>> f"{floor_shortfall_bound(267, 1 / 3, 1):.4e}"
    '9.6302e-48'
    >>> floor_shortfall_bound(24, 1 / 3, 3)
    1.0

    >>> floor_shortfall_bound(115200, 1 / 3, 38400)
    1.0
    """
    candidates: list[float] = []
    if floor <= 1:
        candidates.append((1.0 - probability) ** trials)
    mean = trials * probability
    if 2.0 * _LOG_FLOOR_BUDGET < mean:
        spread = math.sqrt(2.0 * _LOG_FLOOR_BUDGET / mean)
        if floor - 1 <= (1.0 - spread) * mean:
            candidates.append(HONEST_ABORT_BUDGET)
    if not candidates:
        return 1.0
    return min(candidates)


def chernoff_deviation_bound(
    count: int, trials: int, probability: float, *, side: str = "two-sided"
) -> float:
    """Bound the null probability of a deviation at least as large as observed.

    Parameters
    ----------
    count : int
        The observed count, ``0 <= count <= trials``.
    trials : int
        Trials, at least ``0``.
    probability : float
        The null's success probability, in ``[0, 1]``.
    side : {'two-sided', 'lower', 'upper'}, optional
        Keyword-only.

    Returns
    -------
    float
        A number in ``[0, 1]``.

    Raises
    ------
    TypeError
        If a count is not an integer or ``probability`` is not a real number.
    ValueError
        If a count is negative, ``count > trials``, ``probability`` is outside
        ``[0, 1]``, or ``side`` names no tail.

    Examples
    --------
    >>> from qsecure.detect.statistics import chernoff_deviation_bound
    >>> f"{chernoff_deviation_bound(64, 192, 1 / 3):.4f}"      # on the mean
    '1.0000'
    >>> f"{chernoff_deviation_bound(32, 192, 1 / 3):.6f}"      # half the mean
    '0.001997'

    >>> f"{chernoff_deviation_bound(0, 192, 1 / 3):.3e}"
    '5.433e-10'
    >>> f"{chernoff_deviation_bound(0, 192, 1 / 3, side='lower'):.3e}"
    '1.266e-14'
    >>> chernoff_deviation_bound(0, 192, 1 / 3, side="upper")
    1.0

    >>> chernoff_deviation_bound(0, 5000, 0.0)
    1.0
    >>> chernoff_deviation_bound(1, 5000, 0.0)
    0.0
    """
    total = _as_count(trials, "trials")
    hits = _as_count(count, "count")
    if hits > total:
        raise ValueError(
            f"count ({hits}) exceeds trials ({total}); a count above its own "
            f"denominator is a counting error upstream, not a deviation."
        )
    p = _as_probability(probability, "probability")
    if not isinstance(side, str) or side not in _SIDES:
        raise ValueError(
            f"side must be one of {sorted(_SIDES)}, got {side!r}. A one-sided "
            f"detector should ask for the tail it tests; a two-sided bound "
            f"charges it for a tail it never looks at."
        )

    if p == 0.0:
        if hits == 0:
            return 1.0
        return 0.0 if side in ("two-sided", "upper") else 1.0
    if p == 1.0:
        if hits == total:
            return 1.0
        return 0.0 if side in ("two-sided", "lower") else 1.0
    mean = total * p
    if mean == 0.0:
        return 1.0
    deviation = abs(hits - mean)
    if deviation == 0.0:
        return 1.0
    relative = deviation / mean
    lower = (
        math.exp(-relative * relative * mean / 2.0) if relative <= 1.0 else 0.0
    )
    upper = math.exp(-relative * relative * mean / (2.0 + relative))
    if side == "lower":
        return 1.0 if hits > mean else min(1.0, lower)
    if side == "upper":
        return 1.0 if hits < mean else min(1.0, upper)
    return min(1.0, lower + upper)


@dataclass(frozen=True)
class CountStatistic:
    """One count, the null it is read against, and what that null permits.

    Parameters
    ----------
    name : str
        What was counted, for reports.
    count : int
        The observation, ``0 <= count <= trials``.
    trials : int
        The denominator the null is stated over.
    null_probability : float
        The per-trial success probability under the honest-run null, in ``[0,
        1]``.
    null : str
        One sentence naming the distribution, written out so that a table
        rendered from these objects carries its own derivation.

    Raises
    ------
    TypeError
        If ``count`` or ``trials`` is not an integer, or ``null_probability``
        is not a real number, or ``name``/``null`` is not a string.
    ValueError
        If a count is negative, ``count > trials``, ``null_probability`` is
        outside ``[0, 1]``, or ``null`` is empty.

    Examples
    --------
    >>> from qsecure.detect.statistics import CountStatistic
    >>> matched = CountStatistic(
    ...     name="matched_count",
    ...     count=59,
    ...     trials=192,
    ...     null_probability=1 / 3,
    ...     null="Binomial(192, 1/3)",
    ... )
    >>> matched.expected, f"{matched.z_score:.4f}"
    (64.0, '-0.7655')
    >>> f"{matched.tail_bound:.4f}", f"{matched.lower_tail_bound:.4f}"
    ('1.0000', '0.8226')

    >>> clean = CountStatistic("mismatches", 0, 59, 0.0, "point mass at 0")
    >>> clean.z_score, clean.tail_bound
    (0.0, 1.0)
    >>> dirty = CountStatistic("mismatches", 3, 59, 0.0, "point mass at 0")
    >>> dirty.z_score, dirty.tail_bound
    (inf, 0.0)
    """

    name: str
    count: int
    trials: int
    null_probability: float
    null: str

    def __post_init__(self) -> None:
        """Coerce the counts and check the observation fits its denominator."""
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "count", _as_count(self.count, "count"))
        object.__setattr__(self, "trials", _as_count(self.trials, "trials"))
        object.__setattr__(
            self,
            "null_probability",
            _as_probability(self.null_probability, "null_probability"),
        )
        object.__setattr__(self, "null", str(self.null))
        if self.count > self.trials:
            raise ValueError(
                f"{self.name}: count ({self.count}) exceeds trials "
                f"({self.trials}). A statistic whose observation is larger "
                f"than its own denominator is a wiring error, and reporting a "
                f"rate above one would hide it."
            )
        if not self.null:
            raise ValueError(
                f"{self.name}: null must be a non-empty sentence naming the "
                f"distribution this count is read against. a "
                f"statistic with no written-down null cannot carry a derived "
                f"threshold, and an empty string would let one be attached "
                f"anyway."
            )

    @property
    def rate(self) -> float | None:
        """float or None: ``count / trials``; ``None`` when ``trials == 0``."""
        if self.trials == 0:
            return None
        return self.count / self.trials

    @property
    def expected(self) -> float:
        """float: ``trials * null_probability``, the null's mean."""
        return self.trials * self.null_probability

    @property
    def variance(self) -> float:
        """float: ``trials * p * (1 - p)``, the null's variance."""
        return self.trials * self.null_probability * (1.0 - self.null_probability)

    @property
    def standard_deviation(self) -> float:
        """float: The square root of :attr:`variance`."""
        return math.sqrt(self.variance)

    @property
    def z_score(self) -> float:
        """float: Standard deviations between the observation and the null mean."""
        deviation = self.count - self.expected
        spread = self.standard_deviation
        if spread == 0.0:
            return 0.0 if deviation == 0.0 else math.copysign(math.inf, deviation)
        return deviation / spread

    @property
    def tail_bound(self) -> float:
        """float: Upper bound on the null probability of a deviation this large."""
        return chernoff_deviation_bound(
            self.count, self.trials, self.null_probability
        )

    @property
    def lower_tail_bound(self) -> float:
        """float: Upper bound on ``P(S <= count)`` under the null."""
        return chernoff_deviation_bound(
            self.count, self.trials, self.null_probability, side="lower"
        )

    @property
    def upper_tail_bound(self) -> float:
        """float: Upper bound on ``P(S >= count)`` under the null."""
        return chernoff_deviation_bound(
            self.count, self.trials, self.null_probability, side="upper"
        )

    def interval(
        self, *, confidence: float = CHECK_CONFIDENCE
    ) -> Interval | None:
        """Return the Wilson interval on the underlying rate.

        Parameters
        ----------
        confidence : float, optional
            Keyword-only two-sided coverage.

        Returns
        -------
        Interval or None
            ``None`` when ``trials == 0``, for the reason :attr:`rate` gives.

        Examples
        --------
        >>> from qsecure.detect.statistics import CountStatistic
        >>> stat = CountStatistic("e", 0, 59, 0.0, "point mass at 0")
        >>> f"{stat.interval().high:.6f}"
        '0.101088'
        """
        if self.trials == 0:
            return None
        return wilson_interval(self.count, self.trials, confidence=confidence)

    def summary(self) -> str:
        """Return a one-line account naming every number behind it.

        Returns
        -------
        str

        Examples
        --------
        >>> from qsecure.detect.statistics import CountStatistic
        >>> print(
        ...     CountStatistic(
        ...         "matched_count", 59, 192, 1 / 3, "Binomial(192, 1/3)"
        ...     ).summary()
        ... )
        matched_count: 59/192 = 0.307292 vs null Binomial(192, 1/3) mean 64.000000 (z = -0.77, P(|dev| >= this) <= 1.00e+00)
        """
        rate = "undefined" if self.rate is None else f"{self.rate:.6f}"
        return (
            f"{self.name}: {self.count}/{self.trials} = {rate} vs null "
            f"{self.null} mean {self.expected:.6f} "
            f"(z = {self.z_score:.2f}, "
            f"P(|dev| >= this) <= {self.tail_bound:.2e})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
            Keys ``"name"``, ``"count"``, ``"trials"``, ``"null_probability"``,
            ``"null"``, ``"rate"``, ``"expected"``, ``"z_score"`` and
            ``"tail_bound"``.
        """
        z = self.z_score
        return {
            "name": self.name,
            "count": self.count,
            "trials": self.trials,
            "null_probability": self.null_probability,
            "null": self.null,
            "rate": self.rate,
            "expected": self.expected,
            "z_score": None if math.isinf(z) else z,
            "tail_bound": self.tail_bound,
            "lower_tail_bound": self.lower_tail_bound,
            "upper_tail_bound": self.upper_tail_bound,
        }


@dataclass(frozen=True)
class VerifierStatistics:
    """One verifier's verdict reduced to counts, each against its own null.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.detect.statistics import TranscriptStatistics
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> stats = TranscriptStatistics.from_transcript(
    ...     QDSSession(
    ...         ProtocolParams(key_length=192), rng=np.random.default_rng(7)
    ...     ).run(0)
    ... )
    >>> charlie = stats.verifier("Charlie")
    >>> charlie.accepted, charlie.matched.count, charlie.mismatch.count
    (True, 68, 0)
    >>> charlie.margin == charlie.threshold
    True
    >>> charlie.consistent, charlie.recomputed
    (True, True)
    >>> f"{charlie.unmatched_agreement.z_score:.4f}"
    '-0.3592'
    """

    party: Party
    accepted: bool
    threshold: float
    margin: float
    matched: CountStatistic
    mismatch: CountStatistic
    unmatched_agreement: CountStatistic
    recomputed: bool
    reported_matched: int
    reported_mismatches: int
    consistent: bool

    @property
    def mismatch_rate(self) -> float | None:
        """float or None: ``r_R = e_R / |M_R|``, the quantity the cut applies to."""
        return self.mismatch.rate

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
        """
        return {
            "party": str(self.party),
            "accepted": self.accepted,
            "threshold": self.threshold,
            "margin": self.margin,
            "mismatch_rate": self.mismatch_rate,
            "matched": self.matched.to_dict(),
            "mismatch": self.mismatch.to_dict(),
            "unmatched_agreement": self.unmatched_agreement.to_dict(),
            "recomputed": self.recomputed,
            "reported_matched": self.reported_matched,
            "reported_mismatches": self.reported_mismatches,
            "consistent": self.consistent,
        }


@dataclass(frozen=True)
class PooledStatistics:
    """The evidence base of the whole run, and what Phase C' put on the wire.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.detect.statistics import TranscriptStatistics
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> stats = TranscriptStatistics.from_transcript(
    ...     QDSSession(
    ...         ProtocolParams(key_length=192), rng=np.random.default_rng(7)
    ...     ).run(0)
    ... )
    >>> pooled = stats.pooled
    >>> pooled.counts_exchanged, pooled.declaration_gap
    (True, 0)
    >>> pooled.pooled.count, pooled.pooled.trials
    (137, 384)
    >>> f"{pooled.declared_bob.z_score:.4f}"
    '0.7655'
    """

    pooled: CountStatistic | None
    declared_bob: CountStatistic | None
    declared_charlie: CountStatistic | None
    declared_pooled: CountStatistic | None
    counts_exchanged: bool
    minimum_matched: int | None
    minimum_pooled: int | None
    meets_pooled_floor: bool | None
    meets_every_floor: bool | None
    declaration_digest_present: bool
    matched_floor_bound: float

    @property
    def declaration_gap(self) -> int | None:
        """int or None: Declared pooled total minus the verdicts' pooled total."""
        if self.declared_pooled is None or self.pooled is None:
            return None
        return self.declared_pooled.count - self.pooled.count

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
        """
        return {
            "pooled": None if self.pooled is None else self.pooled.to_dict(),
            "declared_bob": (
                None if self.declared_bob is None else self.declared_bob.to_dict()
            ),
            "declared_charlie": (
                None
                if self.declared_charlie is None
                else self.declared_charlie.to_dict()
            ),
            "declared_pooled": (
                None
                if self.declared_pooled is None
                else self.declared_pooled.to_dict()
            ),
            "counts_exchanged": self.counts_exchanged,
            "minimum_matched": self.minimum_matched,
            "minimum_pooled": self.minimum_pooled,
            "meets_pooled_floor": self.meets_pooled_floor,
            "meets_every_floor": self.meets_every_floor,
            "declaration_digest_present": self.declaration_digest_present,
            "matched_floor_bound": self.matched_floor_bound,
            "declaration_gap": self.declaration_gap,
        }


@dataclass(frozen=True)
class ResourceStatistics:
    """What the entanglement resource looked like on one link's check rounds."""

    samples: int
    qber_samples: int
    chsh_samples: int
    mean_fidelity: float | None
    min_fidelity: float | None
    mean_purity: float | None
    min_purity: float | None
    mean_concurrence: float | None
    min_concurrence: float | None
    mean_alice_purity: float | None
    mean_recipient_purity: float | None
    ideal_samples: CountStatistic
    extra_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
        """
        return {
            "samples": self.samples,
            "qber_samples": self.qber_samples,
            "chsh_samples": self.chsh_samples,
            "mean_fidelity": self.mean_fidelity,
            "min_fidelity": self.min_fidelity,
            "mean_purity": self.mean_purity,
            "min_purity": self.min_purity,
            "mean_concurrence": self.mean_concurrence,
            "min_concurrence": self.min_concurrence,
            "mean_alice_purity": self.mean_alice_purity,
            "mean_recipient_purity": self.mean_recipient_purity,
            "ideal_samples": self.ideal_samples.to_dict(),
            "extra_keys": list(self.extra_keys),
        }


@dataclass(frozen=True)
class LinkStatistics:
    """One recipient's link on one message bit, estimated from its own sample.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.detect.statistics import TranscriptStatistics
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> stats = TranscriptStatistics.from_transcript(
    ...     QDSSession(
    ...         ProtocolParams(key_length=192, check_fraction=0.25),
    ...         rng=np.random.default_rng(7),
    ...     ).run(0)
    ... )
    >>> link = stats.link("Bob", 0)
    >>> link.errors.count, link.errors.trials
    (0, 12)
    >>> f"{link.qber.interval.high:.6f}"
    '0.356047'
    >>> link.resource.samples, link.resource.mean_fidelity
    (24, 1.0)
    >>> round(link.resource.mean_alice_purity, 9)
    0.5

    >>> link.qber.interval.method, link.qber_bound.method
    ('wilson', 'hoeffding')
    >>> link.qber_bound.high > link.qber.interval.high
    True
    >>> link.chsh_bound.method, link.chsh.interval.method
    ('hoeffding', 'normal')
    """

    party: Party
    message_bit: int
    errors: CountStatistic
    qber: QberEstimate | None
    qber_bound: Interval | None
    chsh: ChshEstimate | None
    chsh_bound: Interval | None
    chsh_unavailable: str | None
    resource: ResourceStatistics

    @property
    def qber_rounds(self) -> int:
        """int: How many QBER rounds this link published."""
        return self.errors.trials

    @property
    def chsh_statistic(self) -> float | None:
        """float or None: ``S``, or ``None`` when the arm produced no estimate."""
        return None if self.chsh is None else self.chsh.statistic

    @property
    def chsh_z(self) -> float | None:
        """float or None: ``S - 2 sqrt(2)`` in units of the ideal null's sd."""
        if self.chsh is None:
            return None
        variance = sum(1.0 / (2.0 * count) for count in self.chsh.counts)
        deviation = self.chsh.statistic - IDEAL_CHSH
        if variance == 0.0:
            return 0.0 if deviation == 0.0 else math.copysign(
                math.inf, deviation
            )
        return deviation / math.sqrt(variance)

    @property
    def violates_classical_bound(self) -> bool | None:
        """bool or None: Whether the ``S`` interval sits strictly above ``2``."""
        if self.chsh is None:
            return None
        return self.chsh.violates_classical_bound

    @property
    def consistent_with_ideal(self) -> bool | None:
        """bool or None: Whether the ``S`` interval covers ``2 sqrt(2)``."""
        if self.chsh is None:
            return None
        return self.chsh.consistent_with_ideal

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
        """
        return {
            "party": str(self.party),
            "message_bit": self.message_bit,
            "errors": self.errors.to_dict(),
            "qber": None if self.qber is None else self.qber.to_dict(),
            "qber_bound": (
                None if self.qber_bound is None else self.qber_bound.to_dict()
            ),
            "chsh": None if self.chsh is None else self.chsh.to_dict(),
            "chsh_bound": (
                None if self.chsh_bound is None else self.chsh_bound.to_dict()
            ),
            "chsh_z": self.chsh_z,
            "chsh_unavailable": self.chsh_unavailable,
            "violates_classical_bound": self.violates_classical_bound,
            "consistent_with_ideal": self.consistent_with_ideal,
            "resource": self.resource.to_dict(),
        }


def pooled_check_qber(
    logs: Iterable[CheckLog], *, pool: bool, confidence: float = CHECK_CONFIDENCE
) -> QberEstimate:
    """Estimate one QBER over several links' check logs, if you really mean to.

    Parameters
    ----------
    logs : iterable of CheckLog
        The logs to pool.
    pool : bool
        Keyword-only and mandatory.
    confidence : float, optional
        Keyword-only two-sided level, defaulting to
        :data:`~qsecure.protocol.checkrounds.CHECK_CONFIDENCE`.

    Returns
    -------
    QberEstimate
        Over every QBER observation in every supplied log.

    Raises
    ------
    ValueError
        If ``pool`` is not ``True``, or if the pooled sample is empty -- a rate
        needs a denominator.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.detect.statistics import pooled_check_qber
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession, SessionTranscript
    >>> transcript = SessionTranscript.from_json(
    ...     QDSSession(
    ...         ProtocolParams(key_length=192, check_fraction=0.25),
    ...         rng=np.random.default_rng(7),
    ...     ).run(0).to_json()
    ... )
    >>> estimate = pooled_check_qber(transcript.check_logs, pool=True)
    >>> estimate.errors, estimate.rounds
    (0, 48)

    >>> pooled_check_qber(transcript.check_logs, pool=False)
    Traceback (most recent call last):
        ...
    ValueError: pooling two links' check logs is a deliberate choice, ...
    """
    if pool is not True:
        raise ValueError(
            "pooling two links' check logs is a deliberate choice, so pass "
            "pool=True to make it. Per-link QBER is the only statistic that "
            "both detects a party-targeted channel attack and attributes it; "
            "a pooled rate is the average of two channels and detects "
            "neither. Use TranscriptStatistics.link(party, message_bit) for "
            "the default."
        )
    observations: list[QberObservation] = []
    for log in logs:
        observations.extend(log.qber)
    if not observations:
        raise ValueError(
            "the pooled sample holds no QBER rounds, so there is no rate to "
            "estimate: a rate needs a denominator. This is what a run with "
            "check_fraction = 0 looks like, and the honest report is that it "
            "published no channel statistics."
        )
    return estimate_qber(observations, confidence=confidence)


@dataclass(frozen=True)
class AbortStatistics:
    """Refusals to score, counted by reason -- and never mixed with verdicts.

    Examples
    --------
    >>> from qsecure.detect.statistics import AbortStatistics
    >>> empty = AbortStatistics.empty()
    >>> empty.total, empty.structural, empty.evidence
    (0, 0, 0)
    >>> empty.honest_bound is None
    True

    >>> from qsecure.protocol.params import ProtocolParams
    >>> for length in (24, 96, 273, 115200):
    ...     bound = AbortStatistics.empty(
    ...         ProtocolParams(key_length=length)
    ...     ).honest_bound
    ...     print(f"n = {length:6d}   {bound:.4e}")
    n =     24   1.1881e-04
    n =     96   2.4904e-17
    n =    273   1.6263e-19
    n = 115200   1.6263e-19
    """

    total: int
    by_reason: Mapping[str, int]
    by_party: Mapping[str, str]
    structural: int
    evidence: int
    honest_bound: float | None
    shortfalls: Mapping[str, int]

    @classmethod
    def empty(cls, params: ProtocolParams | None = None) -> AbortStatistics:
        """Return the statistics of a run in which nobody refused to score.

        Parameters
        ----------
        params : ProtocolParams or None, optional
            The run's **sifted** parameter set, used only to state
            :attr:`honest_bound`.

        Returns
        -------
        AbortStatistics
        """
        return cls(
            total=0,
            by_reason={},
            by_party={},
            structural=0,
            evidence=0,
            honest_bound=(
                None
                if params is None
                else evidence_abort_probability_bound(params)
            ),
            shortfalls={},
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
        """
        return {
            "total": self.total,
            "by_reason": dict(self.by_reason),
            "by_party": dict(self.by_party),
            "structural": self.structural,
            "evidence": self.evidence,
            "honest_bound": self.honest_bound,
            "shortfalls": dict(self.shortfalls),
        }


@dataclass(frozen=True)
class ReplayStatistics:
    """The replay defence's own record: refusals counted, rounds spent.

    Examples
    --------
    >>> from qsecure.detect.statistics import ReplayStatistics
    >>> quiet = ReplayStatistics(0, {}, 2, 1, 1, ("Bob", "Charlie"))
    >>> quiet.total_refusals, quiet.spent_rounds
    (0, 2)
    """

    total_refusals: int
    refusals_by_party: Mapping[str, int]
    spent_rounds: int
    distinct_sessions: int
    distinct_message_bits: int
    parties_with_spent_rounds: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
        """
        return {
            "total_refusals": self.total_refusals,
            "refusals_by_party": dict(self.refusals_by_party),
            "spent_rounds": self.spent_rounds,
            "distinct_sessions": self.distinct_sessions,
            "distinct_message_bits": self.distinct_message_bits,
            "parties_with_spent_rounds": list(self.parties_with_spent_rounds),
        }


@dataclass(frozen=True)
class DeclarationStatistics:
    """The declared key read as a random string, and its binding.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.detect.statistics import TranscriptStatistics
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> stats = TranscriptStatistics.from_transcript(
    ...     QDSSession(
    ...         ProtocolParams(key_length=192), rng=np.random.default_rng(7)
    ...     ).run(0)
    ... )
    >>> declaration = stats.declarations["Bob"]
    >>> declaration.length, declaration.bound, declaration.off_alphabet
    (192, True, 0)
    >>> sorted(declaration.basis_counts)
    ['X', 'Y', 'Z']
    >>> sum(stat.count for stat in declaration.basis_counts.values())
    192
    """

    party: str
    length: int
    basis_counts: Mapping[str, CountStatistic]
    off_alphabet: int
    positive_eigenvalues: CountStatistic
    bound: bool
    session_id: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
        """
        return {
            "party": self.party,
            "length": self.length,
            "basis_counts": {
                name: stat.to_dict() for name, stat in self.basis_counts.items()
            },
            "off_alphabet": self.off_alphabet,
            "positive_eigenvalues": self.positive_eigenvalues.to_dict(),
            "bound": self.bound,
            "session_id": self.session_id,
        }


@dataclass(frozen=True)
class RecordStatistics:
    """One recipient's log for one message bit, read as a random string."""

    party: Party
    message_bit: int
    length: int
    symmetrised: bool
    session_id: str | None
    basis_counts: Mapping[str, CountStatistic]
    off_alphabet: int
    positive_eigenvalues: CountStatistic
    signed_bit: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
        """
        return {
            "party": str(self.party),
            "message_bit": self.message_bit,
            "length": self.length,
            "symmetrised": self.symmetrised,
            "session_id": self.session_id,
            "basis_counts": {
                name: stat.to_dict() for name, stat in self.basis_counts.items()
            },
            "off_alphabet": self.off_alphabet,
            "positive_eigenvalues": self.positive_eigenvalues.to_dict(),
            "signed_bit": self.signed_bit,
        }


def _basis_statistics(
    bases: Sequence[Any], params: ProtocolParams, prefix: str
) -> tuple[dict[str, CountStatistic], int]:
    """Count basis symbols against the uniform-alphabet null.

    Parameters
    ----------
    bases : sequence
        The declared or recorded bases, in key order.
    params : ProtocolParams
        Supplies the alphabet and hence the null's ``1/|B|``.
    prefix : str
        Prepended to each statistic's name.

    Returns
    -------
    tuple
        ``(statistics_by_symbol, off_alphabet_count)``.
    """
    alphabet = [str(basis) for basis in params.bases]
    length = len(bases)
    probability = params.match_probability
    observed = [str(basis) for basis in bases]
    counts = {symbol: observed.count(symbol) for symbol in alphabet}
    off = length - sum(counts.values())
    null = f"Binomial({length}, 1/{len(alphabet)})"
    return (
        {
            symbol: CountStatistic(
                name=f"{prefix}.basis[{symbol}]",
                count=count,
                trials=length,
                null_probability=probability,
                null=null,
            )
            for symbol, count in counts.items()
        },
        off,
    )


def _eigenvalue_statistic(
    eigenvalues: Sequence[int], name: str
) -> CountStatistic:
    """Count ``+1`` outcomes against the fair-coin null.

    Parameters
    ----------
    eigenvalues : sequence of int
        ``+1``/``-1`` outcomes.
    name : str
        The statistic's name.

    Returns
    -------
    CountStatistic
    """
    length = len(eigenvalues)
    return CountStatistic(
        name=name,
        count=sum(1 for value in eigenvalues if value == 1),
        trials=length,
        null_probability=0.5,
        null=f"Binomial({length}, 1/2)",
    )


def _resource_statistics(
    samples: Sequence[ChannelSample],
) -> ResourceStatistics:
    """Summarise one link's channel samples.

    Parameters
    ----------
    samples : sequence of ChannelSample
        Possibly empty.

    Returns
    -------
    ResourceStatistics
    """
    total = len(samples)
    if total == 0:
        zero = CountStatistic(
            name="ideal_samples",
            count=0,
            trials=0,
            null_probability=1.0,
            null="point mass at every ideal value (no samples observed)",
        )
        return ResourceStatistics(
            samples=0,
            qber_samples=0,
            chsh_samples=0,
            mean_fidelity=None,
            min_fidelity=None,
            mean_purity=None,
            min_purity=None,
            mean_concurrence=None,
            min_concurrence=None,
            mean_alice_purity=None,
            mean_recipient_purity=None,
            ideal_samples=zero,
            extra_keys=(),
        )

    def mean(values: Iterable[float]) -> float:
        collected = list(values)
        return sum(collected) / len(collected)

    ideal = sum(
        1
        for sample in samples
        if sample.is_ideal
        and abs(sample.alice_purity - 0.5) <= IDEAL_TOLERANCE
        and abs(sample.recipient_purity - 0.5) <= IDEAL_TOLERANCE
    )
    extra_keys: set[str] = set()
    for sample in samples:
        extra_keys.update(str(key) for key in sample.extra)
    return ResourceStatistics(
        samples=total,
        qber_samples=sum(
            1 for sample in samples if sample.role is CheckRole.QBER
        ),
        chsh_samples=sum(
            1 for sample in samples if sample.role is CheckRole.CHSH
        ),
        mean_fidelity=mean(sample.fidelity for sample in samples),
        min_fidelity=min(sample.fidelity for sample in samples),
        mean_purity=mean(sample.purity for sample in samples),
        min_purity=min(sample.purity for sample in samples),
        mean_concurrence=mean(sample.concurrence for sample in samples),
        min_concurrence=min(sample.concurrence for sample in samples),
        mean_alice_purity=mean(sample.alice_purity for sample in samples),
        mean_recipient_purity=mean(
            sample.recipient_purity for sample in samples
        ),
        ideal_samples=CountStatistic(
            name="ideal_samples",
            count=ideal,
            trials=total,
            null_probability=1.0,
            null=(
                "point mass: an ideal Phi+ resource has fidelity, purity and "
                "concurrence 1 and both wing purities 1/2"
            ),
        ),
        extra_keys=tuple(sorted(extra_keys)),
    )


def _link_statistics(
    log: CheckLog, samples: Sequence[ChannelSample]
) -> LinkStatistics:
    """Build one link's statistics from its published log and samples.

    Parameters
    ----------
    log : CheckLog
        The published observations for one ``(party, message_bit)``.
    samples : sequence of ChannelSample
        The resource summaries for the same rounds.

    Returns
    -------
    LinkStatistics
    """
    qber_rounds = len(log.qber)
    errors = CountStatistic(
        name=f"qber_errors[{log.party.value},{log.message_bit}]",
        count=sum(1 for entry in log.qber if entry.is_error),
        trials=qber_rounds,
        null_probability=0.0,
        null=(
            "point mass at 0 on a noiseless link; "
            f"Binomial({qber_rounds}, p/2) under depolarising strength p"
        ),
    )
    qber = estimate_qber(log.qber) if qber_rounds else None
    qber_bound = (
        estimate_qber(log.qber, method="hoeffding").interval
        if qber_rounds
        else None
    )

    chsh: ChshEstimate | None = None
    chsh_bound: Interval | None = None
    unavailable: str | None = None
    if not log.chsh:
        unavailable = "this link published no CHSH rounds"
    else:
        cells = [0, 0, 0, 0]
        for entry in log.chsh:
            cells[entry.cell] += 1
        empty = [index for index, count in enumerate(cells) if count == 0]
        if empty:
            unavailable = (
                f"CHSH cell(s) {empty} hold no rounds out of "
                f"{len(log.chsh)}, so their correlators are undefined; "
                f"treating an empty cell as 0 would report a two-setting "
                f"experiment as a Bell test"
            )
        else:
            chsh = estimate_chsh(log.chsh)
            chsh_bound = estimate_chsh(log.chsh, method="hoeffding").interval

    return LinkStatistics(
        party=log.party,
        message_bit=log.message_bit,
        errors=errors,
        qber=qber,
        qber_bound=qber_bound,
        chsh=chsh,
        chsh_bound=chsh_bound,
        chsh_unavailable=unavailable,
        resource=_resource_statistics(samples),
    )


def _matched_null(trials: int, alphabet: int) -> str:
    """Return the written-out matched-count null.

    Parameters
    ----------
    trials : int
        The number of Bernoulli trials the null is stated over.
    alphabet : int
        ``|B|``.

    Returns
    -------
    str
    """
    return f"Binomial({trials}, 1/{alphabet})"


def _verifier_statistics(
    result: VerificationResult,
    record: RecipientRecord | None,
    declaration: Signature,
    scored: ProtocolParams,
) -> VerifierStatistics:
    """Build one verifier's statistics, recomputing the counts where possible.

    Parameters
    ----------
    result : VerificationResult
        The verdict as recorded.
    record : RecipientRecord or None
        This verifier's log for the signed bit, if the transcript carries one.
    declaration : Signature
        The declaration *this* verifier scored -- which is not always the one
        Bob scored.
    scored : ProtocolParams
        The sifted parameter set, supplying ``n`` and the alphabet.

    Returns
    -------
    VerifierStatistics
    """
    signing_length = scored.key_length
    alphabet = len(scored.bases)
    recomputed = (
        record is not None
        and len(record) == signing_length
        and declaration.length == signing_length
    )
    matched_count = result.matched_count
    mismatch_count = result.mismatches
    unmatched_agreements = 0
    unmatched_trials = 0

    if recomputed:
        matched = matched_positions(declaration, record)
        matched_count = len(matched)
        mismatch_count = len(mismatch_positions(declaration, record))
        matched_set = set(matched)
        declared = declaration.eigenvalues
        measured = record.eigenvalues
        unmatched_trials = signing_length - matched_count
        unmatched_agreements = sum(
            1
            for index in range(signing_length)
            if index not in matched_set and measured[index] == declared[index]
        )

    consistent = (
        matched_count == result.matched_count
        and mismatch_count == result.mismatches
    )
    return VerifierStatistics(
        party=result.party,
        accepted=result.accepted,
        threshold=result.threshold,
        margin=result.margin,
        matched=CountStatistic(
            name=f"matched_count[{result.party.value}]",
            count=matched_count,
            trials=signing_length,
            null_probability=scored.match_probability,
            null=_matched_null(signing_length, alphabet),
        ),
        mismatch=CountStatistic(
            name=f"mismatches[{result.party.value}]",
            count=mismatch_count,
            trials=matched_count,
            null_probability=0.0,
            null=(
                "point mass at 0 on a noiseless honest link; "
                f"Binomial({matched_count}, p/2) under depolarising strength p"
            ),
        ),
        unmatched_agreement=CountStatistic(
            name=f"unmatched_agreements[{result.party.value}]",
            count=unmatched_agreements,
            trials=unmatched_trials,
            null_probability=0.5,
            null=f"Binomial({unmatched_trials}, 1/2)",
        ),
        recomputed=recomputed,
        reported_matched=result.matched_count,
        reported_mismatches=result.mismatches,
        consistent=consistent,
    )


@dataclass(frozen=True)
class TranscriptStatistics:
    """Everything a detector may read, and nothing else.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.detect.statistics import TranscriptStatistics
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> transcript = QDSSession(
    ...     ProtocolParams(key_length=192), rng=np.random.default_rng(7)
    ... ).run(0)
    >>> stats = TranscriptStatistics.from_transcript(transcript)

    >>> stats == TranscriptStatistics.from_json(transcript.to_json())
    True

    >>> import json
    >>> json.loads(json.dumps(stats.to_dict()))["message_bit"]
    0
    >>> stats.security_claim, stats.has_check_rounds
    (True, False)
    """

    params: ProtocolParams
    nominal_key_length: int
    message_bit: int
    verifiers: Mapping[str, VerifierStatistics]
    aborts: AbortStatistics
    pooled: PooledStatistics
    replay: ReplayStatistics
    links: Mapping[tuple[str, int], LinkStatistics]
    declarations: Mapping[str, DeclarationStatistics]
    records: Mapping[tuple[str, int], RecordStatistics]
    symmetrised: bool
    counts_exchanged: bool
    channel_monitored: bool
    signer_saw_recipient_logs: bool
    count_exchange_timing: str
    forwarding_altered_signature: bool
    session_coherent: bool
    is_complete: bool
    aborted: bool
    transferable: bool
    repudiated: bool
    repudiation_guarantee: float | None
    enforced_repudiation_bound: float
    run_id: str | None
    has_check_rounds: bool
    security_claim: bool = True

    @classmethod
    def from_json(cls, text: str) -> TranscriptStatistics:
        """Extract the statistics from JSON transcript text.

        Parameters
        ----------
        text : str
            :meth:`~qsecure.protocol.session.SessionTranscript.to_json` output.

        Returns
        -------
        TranscriptStatistics

        Raises
        ------
        TypeError
            If ``text`` is not a string.
        ValueError
            Propagated from
            :meth:`~qsecure.protocol.session.SessionTranscript.from_json` if the
            document does not describe a self-consistent run.
        """
        if not isinstance(text, str):
            raise TypeError(
                f"text must be JSON transcript text, got "
                f"{type(text).__name__}. Pass transcript.to_json(), or use "
                f"TranscriptStatistics.from_transcript(transcript), which "
                f"does the round trip for you."
            )
        return cls._extract(SessionTranscript.from_json(text))

    @classmethod
    def from_transcript(
        cls, transcript: SessionTranscript
    ) -> TranscriptStatistics:
        """Extract the statistics from a transcript object, via JSON.

        Parameters
        ----------
        transcript : SessionTranscript
            The run to read.

        Returns
        -------
        TranscriptStatistics

        Raises
        ------
        TypeError
            If ``transcript`` is not a
            :class:`~qsecure.protocol.session.SessionTranscript`.
        """
        if not isinstance(transcript, SessionTranscript):
            raise TypeError(
                f"transcript must be a SessionTranscript, got "
                f"{type(transcript).__name__}. The round trip through JSON is "
                f"what makes a measured detection rate mean something, so a "
                f"stand-in that cannot be serialised is refused rather than "
                f"read directly."
            )
        return cls.from_json(transcript.to_json())

    @classmethod
    def _extract(cls, transcript: SessionTranscript) -> TranscriptStatistics:
        """Do the extraction, from an already round-tripped transcript.

        Parameters
        ----------
        transcript : SessionTranscript
            Restored from JSON by :meth:`from_json`.

        Returns
        -------
        TranscriptStatistics
        """
        scored = transcript.params.sifted()
        records_for_bit = transcript.records_for(transcript.message_bit)
        verifiers: dict[str, VerifierStatistics] = {}
        for result in transcript.results:
            verifiers[result.party.value] = _verifier_statistics(
                result=result,
                record=records_for_bit.get(result.party),
                declaration=transcript.signature_for(result.party),
                scored=scored,
            )

        by_reason: dict[str, int] = {}
        by_party: dict[str, str] = {}
        shortfalls: dict[str, int] = {}
        structural = 0
        evidence = 0
        for abort in transcript.aborts:
            reason = str(abort.reason)
            by_reason[reason] = by_reason.get(reason, 0) + 1
            by_party[abort.party.value] = reason
            shortfalls[abort.party.value] = abort.shortfall
            if abort.reason in STRUCTURAL_ABORT_REASONS:
                structural += 1
            else:
                evidence += 1
        aborts = AbortStatistics(
            total=len(transcript.aborts),
            by_reason=by_reason,
            by_party=by_party,
            structural=structural,
            evidence=evidence,
            honest_bound=evidence_abort_probability_bound(
                scored, counts_exchanged=transcript.counts_exchanged
            ),
            shortfalls=shortfalls,
        )

        pooled = cls._pooled_statistics(transcript, scored)
        replay = cls._replay_statistics(transcript)

        samples_by_link: dict[tuple[Party, int], list[ChannelSample]] = {}
        for sample in transcript.channel:
            samples_by_link.setdefault(
                (sample.party, sample.message_bit), []
            ).append(sample)
        links = {
            (log.party.value, log.message_bit): _link_statistics(
                log, samples_by_link.get((log.party, log.message_bit), [])
            )
            for log in transcript.check_logs
        }

        declarations: dict[str, DeclarationStatistics] = {}
        scoring = [Party.BOB]
        if transcript.forwarding_altered_signature:
            scoring.append(Party.CHARLIE)
        for party in scoring:
            declarations[party.value] = cls._declaration_statistics(
                transcript.signature_for(party), scored, party.value
            )

        records: dict[tuple[str, int], RecordStatistics] = {}
        for record in transcript.records:
            counts, off = _basis_statistics(
                record.bases,
                scored,
                f"record[{record.party.value},{record.message_bit}]",
            )
            records[(record.party.value, record.message_bit)] = RecordStatistics(
                party=record.party,
                message_bit=record.message_bit,
                length=len(record),
                symmetrised=record.symmetrised,
                session_id=record.session_id,
                basis_counts=counts,
                off_alphabet=off,
                positive_eigenvalues=_eigenvalue_statistic(
                    record.eigenvalues,
                    f"record[{record.party.value},"
                    f"{record.message_bit}].positive",
                ),
                signed_bit=record.message_bit == transcript.message_bit,
            )

        floor = minimum_matched_count(scored)
        pooled_floor = minimum_pooled_matched_count(scored)
        return cls(
            params=scored,
            nominal_key_length=transcript.params.key_length,
            message_bit=transcript.message_bit,
            verifiers=verifiers,
            aborts=aborts,
            pooled=pooled,
            replay=replay,
            links=links,
            declarations=declarations,
            records=records,
            symmetrised=transcript.symmetrised,
            counts_exchanged=transcript.counts_exchanged,
            channel_monitored=transcript.channel_monitored,
            signer_saw_recipient_logs=transcript.signer_saw_recipient_logs,
            count_exchange_timing=transcript.count_exchange_timing,
            forwarding_altered_signature=(
                transcript.forwarding_altered_signature
            ),
            session_coherent=transcript.session_coherent,
            is_complete=transcript.is_complete,
            aborted=transcript.aborted,
            transferable=transcript.transferable,
            repudiated=transcript.repudiated,
            repudiation_guarantee=transcript.repudiation_guarantee,
            enforced_repudiation_bound=_enforced_repudiation_bound(scored),
            run_id=transcript.run_id,
            has_check_rounds=bool(transcript.check_logs),
            security_claim=floor > 1 or pooled_floor > 1,
        )

    @staticmethod
    def _pooled_statistics(
        transcript: SessionTranscript, scored: ProtocolParams
    ) -> PooledStatistics:
        """Build the pooled-evidence statistics.

        Parameters
        ----------
        transcript : SessionTranscript
            The round-tripped run.
        scored : ProtocolParams
            The sifted parameter set, supplying ``n`` and the alphabet.

        Returns
        -------
        PooledStatistics
        """
        signing_length = scored.key_length
        probability = scored.match_probability
        alphabet = len(scored.bases)
        one = _matched_null(signing_length, alphabet)
        both = _matched_null(2 * signing_length, alphabet)

        pooled_count = transcript.pooled_matched_count
        pooled_stat = (
            None
            if pooled_count is None
            else CountStatistic(
                name="pooled_matched_count",
                count=pooled_count,
                trials=2 * signing_length,
                null_probability=probability,
                null=both,
            )
        )
        exchange = transcript.pooled
        declared_bob = declared_charlie = declared_pooled = None
        if exchange is not None:
            declared_bob = CountStatistic(
                name="declared_matched_count[Bob]",
                count=exchange.bob_count,
                trials=signing_length,
                null_probability=probability,
                null=one,
            )
            declared_charlie = CountStatistic(
                name="declared_matched_count[Charlie]",
                count=exchange.charlie_count,
                trials=signing_length,
                null_probability=probability,
                null=one,
            )
            declared_pooled = CountStatistic(
                name="declared_pooled_matched_count",
                count=exchange.pooled,
                trials=2 * signing_length,
                null_probability=probability,
                null=both,
            )
        return PooledStatistics(
            pooled=pooled_stat,
            declared_bob=declared_bob,
            declared_charlie=declared_charlie,
            declared_pooled=declared_pooled,
            counts_exchanged=exchange is not None,
            minimum_matched=(
                None if exchange is None else exchange.minimum_matched
            ),
            minimum_pooled=(
                None if exchange is None else exchange.minimum_pooled
            ),
            meets_pooled_floor=(
                None if exchange is None else exchange.meets_pooled_floor
            ),
            meets_every_floor=(
                None if exchange is None else exchange.meets_every_floor
            ),
            declaration_digest_present=(
                exchange is not None and bool(exchange.declaration_digest)
            ),
            matched_floor_bound=HONEST_ABORT_BUDGET,
        )

    @staticmethod
    def _replay_statistics(transcript: SessionTranscript) -> ReplayStatistics:
        """Build the replay statistics.

        Parameters
        ----------
        transcript : SessionTranscript

        Returns
        -------
        ReplayStatistics
        """
        refusals = {party: count for party, count in transcript.replay_refusals}
        return ReplayStatistics(
            total_refusals=sum(refusals.values()),
            refusals_by_party=refusals,
            spent_rounds=len(transcript.spent_rounds),
            distinct_sessions=len(
                {session for _, session, _ in transcript.spent_rounds}
            ),
            distinct_message_bits=len(
                {bit for _, _, bit in transcript.spent_rounds}
            ),
            parties_with_spent_rounds=tuple(
                sorted({party for party, _, _ in transcript.spent_rounds})
            ),
        )

    @staticmethod
    def _declaration_statistics(
        declaration: Signature, scored: ProtocolParams, party: str
    ) -> DeclarationStatistics:
        """Build one declaration's frequency statistics.

        Parameters
        ----------
        declaration : Signature
        scored : ProtocolParams
            The sifted parameter set, supplying the alphabet.
        party : str
            The verifier who scored this declaration.

        Returns
        -------
        DeclarationStatistics
        """
        counts, off = _basis_statistics(
            declaration.bases, scored, f"declaration[{party}]"
        )
        return DeclarationStatistics(
            party=party,
            length=declaration.length,
            basis_counts=counts,
            off_alphabet=off,
            positive_eigenvalues=_eigenvalue_statistic(
                declaration.eigenvalues, f"declaration[{party}].positive"
            ),
            bound=declaration.session_id is not None,
            session_id=declaration.session_id,
        )

    @property
    def key_length(self) -> int:
        """int: ``n``, the sifted key length every null here is stated over."""
        return self.params.key_length

    @property
    def match_probability(self) -> float:
        """float: ``1/|B|``, the per-position match probability."""
        return self.params.match_probability

    def verifier(self, party: Party | str) -> VerifierStatistics:
        """Return one verifier's statistics.

        Parameters
        ----------
        party : Party or str
            Bob or Charlie.

        Returns
        -------
        VerifierStatistics

        Raises
        ------
        KeyError
            If that party reached no verdict in this run.
        """
        name = str(party)
        if name not in self.verifiers:
            refusal = self.aborts.by_party.get(name)
            because = (
                f" He was asked and refused to score: {refusal}."
                if refusal is not None
                else " He reached no verdict; read aborts and is_complete."
            )
            raise KeyError(
                f"{name} has no verdict statistics in this run.{because} This "
                f"is not a rejection: no decision was made, and counting it as "
                f"one would report a detection that never happened."
            )
        return self.verifiers[name]

    def link(self, party: Party | str, message_bit: int) -> LinkStatistics:
        """Return one link's check-round statistics, for one message bit.

        Parameters
        ----------
        party : Party or str
            The recipient whose link is wanted.
        message_bit : int
            ``0`` or ``1``.

        Returns
        -------
        LinkStatistics

        Raises
        ------
        KeyError
            If the run published no check log for that link and bit, which is every
            run of a parameter set with ``check_fraction = 0``.
        """
        key = (str(party), int(message_bit))
        if key not in self.links:
            raise KeyError(
                f"no check log for {key[0]} on message bit {key[1]}. A run "
                f"with check_fraction = 0 publishes none, which is the honest "
                f"representation of 'this run published no channel "
                f"statistics'; read has_check_rounds before asking."
            )
        return self.links[key]

    def record(self, party: Party | str, message_bit: int) -> RecordStatistics:
        """Return one recipient's log statistics, for one message bit.

        Parameters
        ----------
        party : Party or str
        message_bit : int

        Returns
        -------
        RecordStatistics

        Raises
        ------
        KeyError
            If the transcript carries no such record.
        """
        return self.records[(str(party), int(message_bit))]

    def summary(self) -> str:
        """Return a short account of every statistic that has a verdict behind it.

        Returns
        -------
        str
            One line per verifier, then the pooled line, then one line each for
            aborts and replay.

        Examples
        --------
        >>> import numpy as np
        >>> from qsecure.detect.statistics import TranscriptStatistics
        >>> from qsecure.protocol.params import ProtocolParams
        >>> from qsecure.protocol.session import QDSSession
        >>> stats = TranscriptStatistics.from_transcript(
        ...     QDSSession(
        ...         ProtocolParams(key_length=192), rng=np.random.default_rng(7)
        ...     ).run(0)
        ... )
        >>> print(stats.summary().splitlines()[0])
        Bob: accepted, r_R = 0.000000 vs threshold 0.015625, |M_R| = 69 (z = 0.77)
        >>> print(stats.summary().splitlines()[-1])
        replay: 0 refusal(s), 2 spent round(s) over 1 session(s)
        """
        lines: list[str] = []
        for name in sorted(self.verifiers):
            stat = self.verifiers[name]
            rate = stat.mismatch_rate
            shown = "undefined" if rate is None else f"{rate:.6f}"
            lines.append(
                f"{name}: {'accepted' if stat.accepted else 'rejected'}, "
                f"r_R = {shown} vs threshold {stat.threshold:.6f}, "
                f"|M_R| = {stat.matched.count} "
                f"(z = {stat.matched.z_score:.2f})"
            )
        if self.pooled.pooled is not None:
            lines.append(f"pooled: {self.pooled.pooled.summary()}")
        else:
            lines.append("pooled: no pooled matched count (not a full run)")
        lines.append(
            f"aborts: {self.aborts.total} refusal(s), "
            f"{self.aborts.structural} structural, "
            f"{self.aborts.evidence} evidence"
        )
        lines.append(
            f"replay: {self.replay.total_refusals} refusal(s), "
            f"{self.replay.spent_rounds} spent round(s) over "
            f"{self.replay.distinct_sessions} session(s)"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of every statistic extracted.

        Returns
        -------
        dict
            Passes to :func:`json.dumps` unchanged.
        """
        return {
            "params": self.params.to_dict(),
            "nominal_key_length": self.nominal_key_length,
            "key_length": self.key_length,
            "match_probability": self.match_probability,
            "message_bit": self.message_bit,
            "verifiers": {
                name: stat.to_dict() for name, stat in self.verifiers.items()
            },
            "aborts": self.aborts.to_dict(),
            "pooled": self.pooled.to_dict(),
            "replay": self.replay.to_dict(),
            "links": {
                f"{party}|{bit}": stat.to_dict()
                for (party, bit), stat in self.links.items()
            },
            "declarations": {
                name: stat.to_dict() for name, stat in self.declarations.items()
            },
            "records": {
                f"{party}|{bit}": stat.to_dict()
                for (party, bit), stat in self.records.items()
            },
            "symmetrised": self.symmetrised,
            "counts_exchanged": self.counts_exchanged,
            "channel_monitored": self.channel_monitored,
            "signer_saw_recipient_logs": self.signer_saw_recipient_logs,
            "count_exchange_timing": self.count_exchange_timing,
            "forwarding_altered_signature": self.forwarding_altered_signature,
            "session_coherent": self.session_coherent,
            "is_complete": self.is_complete,
            "aborted": self.aborted,
            "transferable": self.transferable,
            "repudiated": self.repudiated,
            "repudiation_guarantee": self.repudiation_guarantee,
            "enforced_repudiation_bound": self.enforced_repudiation_bound,
            "run_id": self.run_id,
            "has_check_rounds": self.has_check_rounds,
            "security_claim": self.security_claim,
        }
