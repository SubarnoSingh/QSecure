"""Derived thresholds for the **structural** signals (aborts, replay refusals,
run shape).

Examples
--------
>>> import numpy as np
>>> from qsecure.detect.statistics import TranscriptStatistics
>>> from qsecure.detect.thresholds_structural import structural_report
>>> from qsecure.protocol.params import ProtocolParams
>>> from qsecure.protocol.session import QDSSession
>>> stats = TranscriptStatistics.from_transcript(
...     QDSSession(
...         ProtocolParams(key_length=384), rng=np.random.default_rng(7)
...     ).run(0)
... )
>>> report = structural_report(stats, eps=1e-9)
>>> report.alarm_raised, len(report.alarms)
(False, 0)

>>> f"{report.false_positive_bound:.4e}"
'1.6263e-19'
>>> report.outcomes["Bob"], report.outcomes["Charlie"]
(<RunOutcome.ACCEPTED: 'accepted'>, <RunOutcome.ACCEPTED: 'accepted'>)

>>> for threshold in report.thresholds:
...     print(f"{threshold.check:20s} {threshold.false_positive_bound:.4e}")
structural-abort     0.0000e+00
evidence-abort       1.6263e-19
replay-refusal       0.0000e+00
run-shape            0.0000e+00
"""

from __future__ import annotations

import enum
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from qsecure.detect.statistics import (
    EVIDENCE_ABORT_REASONS,
    STRUCTURAL_ABORT_REASONS,
    TranscriptStatistics,
    chernoff_deviation_bound,
    floor_shortfall_bound,
)
from qsecure.protocol.analysis import matched_shortfall_probability
from qsecure.protocol.params import DEFAULT_BASES, Party, ProtocolParams
from qsecure.protocol.verify import (
    HONEST_ABORT_BUDGET,
    AbortReason,
    minimum_matched_count,
    minimum_pooled_matched_count,
)

__all__ = [
    "FLOOR_BUDGET",
    "STRUCTURAL_CHECKS",
    "AbortAttribution",
    "NoVerdictCount",
    "OutcomeTally",
    "RunOutcome",
    "StructuralAlarm",
    "StructuralCheck",
    "StructuralReport",
    "StructuralThreshold",
    "abort_reason_bound",
    "attribute_aborts",
    "chernoff_lower_tail_count",
    "evidence_abort_bound",
    "evidence_abort_threshold",
    "minimum_sifted_length",
    "outcome_of",
    "outcomes",
    "point_mass_threshold",
    "replay_refusal_threshold",
    "run_shape_threshold",
    "run_shape_violations",
    "shortfall_threshold",
    "structural_abort_threshold",
    "structural_report",
    "tally_outcomes",
]


FLOOR_BUDGET: Final[float] = HONEST_ABORT_BUDGET

#: ``ln(1/eps0)``, the constant every floor derivation in the protocol is
#: written in terms of. ``44.3614...``.
_LOG_FLOOR_BUDGET: Final[float] = -math.log(HONEST_ABORT_BUDGET)

_SEARCH_CEILING: Final[int] = 400


class StructuralCheck(enum.StrEnum):
    """Which structural check a threshold, alarm or withholding is about.

    Examples
    --------
    >>> from qsecure.detect.thresholds_structural import StructuralCheck
    >>> StructuralCheck.EVIDENCE_ABORT
    <StructuralCheck.EVIDENCE_ABORT: 'evidence-abort'>
    >>> f"{StructuralCheck.RUN_SHAPE}"
    'run-shape'
    """

    STRUCTURAL_ABORT = "structural-abort"
    EVIDENCE_ABORT = "evidence-abort"
    REPLAY_REFUSAL = "replay-refusal"
    RUN_SHAPE = "run-shape"


STRUCTURAL_CHECKS: Final[tuple[StructuralCheck, ...]] = (
    StructuralCheck.STRUCTURAL_ABORT,
    StructuralCheck.EVIDENCE_ABORT,
    StructuralCheck.REPLAY_REFUSAL,
    StructuralCheck.RUN_SHAPE,
)


class RunOutcome(enum.StrEnum):
    """What one party did, with **no truth value** so it cannot be misread.

    Examples
    --------
    >>> from qsecure.detect.thresholds_structural import RunOutcome
    >>> RunOutcome.REFUSED == "refused-to-score"
    True
    >>> sorted(RunOutcome)[0]
    <RunOutcome.ACCEPTED: 'accepted'>

    >>> if RunOutcome.REFUSED:
    ...     print("counted as a rejection")
    Traceback (most recent call last):
        ...
    TypeError: RunOutcome has no truth value: 'refused-to-score' is neither an
    acceptance nor a rejection, and a no-verdict counted as one reports a
    detection that never happened. Compare against the member you mean, e.g.
    `outcome is RunOutcome.REJECTED`.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REFUSED = "refused-to-score"
    NOT_ASKED = "not-asked"

    def __bool__(self) -> bool:
        """Refuse to have a truth value.

        Returns
        -------
        bool
            Never returns.

        Raises
        ------
        TypeError
            Always.
        """
        raise TypeError(
            f"RunOutcome has no truth value: {self.value!r} is neither an\n"
            f"acceptance nor a rejection, and a no-verdict counted as one "
            f"reports a\ndetection that never happened. Compare against the "
            f"member you mean, e.g.\n`outcome is RunOutcome.REJECTED`."
        )


_NO_VERDICT_OUTCOMES: Final[frozenset[RunOutcome]] = frozenset(
    {RunOutcome.REFUSED, RunOutcome.NOT_ASKED}
)


class NoVerdictCount(int):
    """A count of no-verdicts that refuses to be added to anything else.

    Parameters
    ----------
    value : int
        The count.

    Examples
    --------
    >>> from qsecure.detect.thresholds_structural import NoVerdictCount
    >>> refusals = NoVerdictCount(2)
    >>> refusals, refusals == 2, int(refusals) + 1
    (2, True, 3)
    >>> refusals + NoVerdictCount(3)
    5

    >>> refusals + 1
    Traceback (most recent call last):
        ...
    TypeError: a no-verdict count may only be added to another no-verdict
    count. A refusal is not a rejection, and summing the two reports a
    detection that never happened -- use NoVerdictCount.total(...) for a
    deliberate total of refusals, or int(...) if you really mean plain
    arithmetic.
    >>> sum([NoVerdictCount(1), NoVerdictCount(1)])
    Traceback (most recent call last):
        ...
    TypeError: a no-verdict count may only be added to another no-verdict
    count. A refusal is not a rejection, and summing the two reports a
    detection that never happened -- use NoVerdictCount.total(...) for a
    deliberate total of refusals, or int(...) if you really mean plain
    arithmetic.
    >>> NoVerdictCount.total([NoVerdictCount(1), NoVerdictCount(1)])
    2
    """

    __slots__ = ()

    def __add__(self, other: Any) -> NoVerdictCount:
        """Add another :class:`NoVerdictCount`, and nothing else.

        Parameters
        ----------
        other : NoVerdictCount
            The other count.

        Returns
        -------
        NoVerdictCount

        Raises
        ------
        TypeError
            If ``other`` is not a :class:`NoVerdictCount`.
        """
        if isinstance(other, NoVerdictCount):
            return NoVerdictCount(int(self) + int(other))
        raise TypeError(
            "a no-verdict count may only be added to another no-verdict\n"
            "count. A refusal is not a rejection, and summing the two reports "
            "a\ndetection that never happened -- use NoVerdictCount.total(...) "
            "for a\ndeliberate total of refusals, or int(...) if you really "
            "mean plain\narithmetic."
        )

    __radd__ = __add__

    @classmethod
    def total(cls, counts: Iterable[NoVerdictCount]) -> NoVerdictCount:
        """Return the deliberate total of several no-verdict counts.

        Parameters
        ----------
        counts : iterable of NoVerdictCount
            The counts to total.

        Returns
        -------
        NoVerdictCount

        Raises
        ------
        TypeError
            If any element is not a :class:`NoVerdictCount`.
        """
        running = cls(0)
        for count in counts:
            running = running + count
        return running


def outcome_of(stats: TranscriptStatistics, party: Party | str) -> RunOutcome:
    """Return what one party did, as an outcome that cannot be read as a bool.

    Parameters
    ----------
    stats : TranscriptStatistics
        The run.
    party : Party or str
        Bob or Charlie.

    Returns
    -------
    RunOutcome

    Raises
    ------
    TypeError
        If ``stats`` is not a
        :class:`~qsecure.detect.statistics.TranscriptStatistics`.
    ValueError
        If ``party`` is Alice or names no party.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.detect.statistics import TranscriptStatistics
    >>> from qsecure.detect.thresholds_structural import outcome_of
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> stats = TranscriptStatistics.from_transcript(
    ...     QDSSession(
    ...         ProtocolParams(key_length=384), rng=np.random.default_rng(7)
    ...     ).run(0)
    ... )
    >>> outcome_of(stats, "Bob")
    <RunOutcome.ACCEPTED: 'accepted'>
    >>> outcome_of(stats, "Alice")
    Traceback (most recent call last):
        ...
    ValueError: Alice reaches no verdict: she signs, holds no record and has no
    acceptance threshold. Ask about Party.BOB or Party.CHARLIE.
    """
    resolved = _as_verifier(party)
    if not isinstance(stats, TranscriptStatistics):
        raise TypeError(
            f"stats must be a TranscriptStatistics, got "
            f"{type(stats).__name__}. Build one with "
            f"TranscriptStatistics.from_transcript(transcript); the round trip "
            f"through JSON is what makes a detection rate mean something."
        )
    if resolved in stats.verifiers:
        verdict = stats.verifiers[resolved]
        return RunOutcome.ACCEPTED if verdict.accepted else RunOutcome.REJECTED
    if resolved in stats.aborts.by_party:
        return RunOutcome.REFUSED
    return RunOutcome.NOT_ASKED


def outcomes(stats: TranscriptStatistics) -> dict[str, RunOutcome]:
    """Return both verifiers' outcomes, keyed by party name.

    Parameters
    ----------
    stats : TranscriptStatistics
        The run.

    Returns
    -------
    dict of str to RunOutcome
        Always exactly two entries, ``"Bob"`` and ``"Charlie"``, because a
        party who was never asked has an outcome too and leaving him out is how
        he gets counted as something else later.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.detect.statistics import TranscriptStatistics
    >>> from qsecure.detect.thresholds_structural import outcomes
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> stats = TranscriptStatistics.from_transcript(
    ...     QDSSession(
    ...         ProtocolParams(key_length=384), rng=np.random.default_rng(7)
    ...     ).run(0)
    ... )
    >>> sorted(outcomes(stats).items())
    [('Bob', <RunOutcome.ACCEPTED: 'accepted'>), ('Charlie', <RunOutcome.ACCEPTED: 'accepted'>)]
    """
    return {
        party.value: outcome_of(stats, party)
        for party in (Party.BOB, Party.CHARLIE)
    }


@dataclass(frozen=True)
class OutcomeTally:
    """Verdicts and no-verdicts counted apart, with no field that sums them.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.detect.statistics import TranscriptStatistics
    >>> from qsecure.detect.thresholds_structural import tally_outcomes
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> stats = TranscriptStatistics.from_transcript(
    ...     QDSSession(
    ...         ProtocolParams(key_length=384), rng=np.random.default_rng(7)
    ...     ).run(0)
    ... )
    >>> tally = tally_outcomes(stats)
    >>> tally.accepted, tally.rejected, tally.refused, tally.not_asked
    (2, 0, 0, 0)
    >>> tally.merge(tally).accepted
    4

    >>> tally.rejected + tally.refused
    Traceback (most recent call last):
        ...
    TypeError: a no-verdict count may only be added to another no-verdict
    count. A refusal is not a rejection, and summing the two reports a
    detection that never happened -- use NoVerdictCount.total(...) for a
    deliberate total of refusals, or int(...) if you really mean plain
    arithmetic.
    """

    accepted: int
    rejected: int
    refused: NoVerdictCount
    not_asked: NoVerdictCount
    count_exchange_timing: str
    runs: int = 1

    def __post_init__(self) -> None:
        """Coerce the two no-verdict counts and refuse a negative tally.

        Raises
        ------
        TypeError
            If a count is not an integer or the timing is not a string.
        ValueError
            If a count is negative.
        """
        for name in ("accepted", "rejected", "refused", "not_asked", "runs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an int, got {type(value).__name__}"
                )
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if not isinstance(self.count_exchange_timing, str):
            raise TypeError(
                f"count_exchange_timing must be a string, got "
                f"{type(self.count_exchange_timing).__name__}. It is the "
                f"grouping key, and a tally that does not know its own group "
                f"can be merged with the wrong one."
            )
        object.__setattr__(self, "refused", NoVerdictCount(self.refused))
        object.__setattr__(self, "not_asked", NoVerdictCount(self.not_asked))

    @property
    def verdicts(self) -> int:
        """int: Acceptances plus rejections."""
        return self.accepted + self.rejected

    @property
    def no_verdicts(self) -> NoVerdictCount:
        """NoVerdictCount: Refusals plus never-askeds, kept off the verdicts."""
        return self.refused + self.not_asked

    def merge(self, other: OutcomeTally) -> OutcomeTally:
        """Return the tally of both, refusing to pool two count orderings.

        Parameters
        ----------
        other : OutcomeTally
            The tally to fold in.

        Returns
        -------
        OutcomeTally

        Raises
        ------
        TypeError
            If ``other`` is not an :class:`OutcomeTally`.
        ValueError
            If the two were produced under different ``count_exchange_timing``
            values.

        Examples
        --------
        >>> from qsecure.detect.thresholds_structural import (
        ...     NoVerdictCount, OutcomeTally
        ... )
        >>> zero = NoVerdictCount(0)
        >>> before = OutcomeTally(2, 0, zero, zero, "before-forwarding")
        >>> after = OutcomeTally(0, 2, zero, zero, "after-forwarding")
        >>> before.merge(after)
        Traceback (most recent call last):
            ...
        ValueError: refusing to merge tallies from two count orderings:
        'before-forwarding' and 'after-forwarding'. The two give different
        answers to the same attack, so a merged tally would average a forgery
        rate with a denial-of-service rate. Group by count_exchange_timing and
        report the groups.
        """
        if not isinstance(other, OutcomeTally):
            raise TypeError(
                f"other must be an OutcomeTally, got {type(other).__name__}"
            )
        if self.count_exchange_timing != other.count_exchange_timing:
            raise ValueError(
                f"refusing to merge tallies from two count orderings:\n"
                f"{self.count_exchange_timing!r} and "
                f"{other.count_exchange_timing!r}. The two give different\n"
                f"answers to the same attack, so a merged tally would average "
                f"a forgery\nrate with a denial-of-service rate. Group by "
                f"count_exchange_timing and\nreport the groups."
            )
        return OutcomeTally(
            accepted=self.accepted + other.accepted,
            rejected=self.rejected + other.rejected,
            refused=self.refused + other.refused,
            not_asked=self.not_asked + other.not_asked,
            count_exchange_timing=self.count_exchange_timing,
            runs=self.runs + other.runs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
            The four counts stay four counts; nothing here sums two of them.
        """
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "refused": int(self.refused),
            "not_asked": int(self.not_asked),
            "count_exchange_timing": self.count_exchange_timing,
            "runs": self.runs,
        }


def tally_outcomes(stats: TranscriptStatistics) -> OutcomeTally:
    """Tally one run's two outcomes, keeping refusals off the verdicts.

    Parameters
    ----------
    stats : TranscriptStatistics
        The run.

    Returns
    -------
    OutcomeTally
        With ``runs == 1`` and the run's own ``count_exchange_timing``, so that
        merging two tallies across orderings raises instead of averaging.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.detect.statistics import TranscriptStatistics
    >>> from qsecure.detect.thresholds_structural import tally_outcomes
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> stats = TranscriptStatistics.from_transcript(
    ...     QDSSession(
    ...         ProtocolParams(key_length=384), rng=np.random.default_rng(7)
    ...     ).run(0)
    ... )
    >>> tally_outcomes(stats).to_dict()["count_exchange_timing"]
    'before-forwarding'
    """
    seen = outcomes(stats)
    counts = {outcome: 0 for outcome in RunOutcome}
    for outcome in seen.values():
        counts[outcome] += 1
    return OutcomeTally(
        accepted=counts[RunOutcome.ACCEPTED],
        rejected=counts[RunOutcome.REJECTED],
        refused=NoVerdictCount(counts[RunOutcome.REFUSED]),
        not_asked=NoVerdictCount(counts[RunOutcome.NOT_ASKED]),
        count_exchange_timing=stats.count_exchange_timing,
        runs=1,
    )


def _as_budget(value: Any, name: str = "eps") -> float:
    """Coerce a false-positive budget in ``(0, 1)``.

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
        If ``value`` is not finite or is outside ``(0, 1)``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be a real number, got {type(value).__name__}. It is "
            f"a false-positive budget, not a count."
        )
    budget = float(value)
    if not math.isfinite(budget) or not 0.0 < budget < 1.0:
        raise ValueError(
            f"{name} must be a finite budget strictly inside (0, 1), got "
            f"{budget!r}. Sweep it to build a ROC curve; each value is an "
            f"operating point that has to be derivable, and 0 and 1 are not."
        )
    return budget


def _as_verifier(party: Party | str) -> str:
    """Coerce a verifier's name, refusing Alice.

    Parameters
    ----------
    party : Party or str
        Bob or Charlie.

    Returns
    -------
    str
        ``"Bob"`` or ``"Charlie"``.

    Raises
    ------
    ValueError
        If ``party`` is Alice or names no party.
    """
    name = str(party)
    if name == Party.ALICE.value:
        raise ValueError(
            "Alice reaches no verdict: she signs, holds no record and has no\n"
            "acceptance threshold. Ask about Party.BOB or Party.CHARLIE."
        )
    if name not in (Party.BOB.value, Party.CHARLIE.value):
        raise ValueError(
            f"{name!r} names no party in this protocol; expected "
            f"{Party.BOB.value!r} or {Party.CHARLIE.value!r}."
        )
    return name


def _counterpart(party: str) -> str:
    """Return the other verifier's name.

    Parameters
    ----------
    party : str
        ``"Bob"`` or ``"Charlie"``.

    Returns
    -------
    str
    """
    return (
        Party.CHARLIE.value if party == Party.BOB.value else Party.BOB.value
    )


def chernoff_lower_tail_count(
    trials: int, probability: float, eps: float
) -> int | None:
    """Return the largest count whose lower tail is inside a budget.

    Parameters
    ----------
    trials : int
        Trials, at least ``0``.
    probability : float
        The null's success probability, in ``[0, 1]``.
    eps : float
        The false-positive budget, in ``(0, 1)``.

    Returns
    -------
    int or None
        The largest admissible count, or ``None`` when the Chernoff form is
        vacuous at this budget -- ``c(eps) < 0``, i.e. ``mu < 2 ln(1/eps)`` --
        which is the honest answer that this sample is too small to spend that
        budget on.

    Raises
    ------
    TypeError
        If ``trials`` is not an integer or a probability is not a real number.
    ValueError
        If ``trials`` is negative, ``probability`` is outside ``[0, 1]``, or
        ``eps`` is outside ``(0, 1)``.

    Examples
    --------
    >>> from qsecure.detect.thresholds_structural import (
    ...     chernoff_lower_tail_count
    ... )
    >>> chernoff_lower_tail_count(115200, 1 / 3, 2.0**-64)
    36554
    >>> chernoff_lower_tail_count(115200, 1 / 3, 1e-9)
    37138

    >>> from qsecure.protocol.params import DEFAULT_PARAMS
    >>> from qsecure.protocol.verify import minimum_matched_count
    >>> minimum_matched_count(DEFAULT_PARAMS)
    36555

    >>> chernoff_lower_tail_count(24, 1 / 3, 1e-9) is None
    True
    """
    total = _as_count(trials, "trials")
    p = _as_unit(probability, "probability")
    budget = _as_budget(eps)
    mean = total * p
    if mean <= 0.0:
        return None
    spread = math.sqrt(2.0 * mean * -math.log(budget))
    if spread > mean:
        return None
    return int(math.floor(mean - spread))


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
            f"{name} must be an int, got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _as_unit(value: Any, name: str) -> float:
    """Coerce a probability in ``[0, 1]``, refusing booleans.

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
            f"{probability!r}"
        )
    return probability


def _as_params(value: Any) -> ProtocolParams:
    """Coerce a parameter set.

    Parameters
    ----------
    value : ProtocolParams
        The candidate.

    Returns
    -------
    ProtocolParams

    Raises
    ------
    TypeError
        If ``value`` is not a :class:`~qsecure.protocol.params.ProtocolParams`.
    """
    if not isinstance(value, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(value).__name__}. "
            f"Pass the sifted set -- TranscriptStatistics.params already is "
            f"one -- because every null here is stated over the positions that "
            f"carried key, not over the nominal L."
        )
    return value


def _floor_bound(
    trials: int,
    probability: float,
    floor: int,
    bases: tuple[Any, ...],
    *,
    exact: bool,
) -> float:
    """Bound the honest probability of falling under one matched-count floor.

    Parameters
    ----------
    trials : int
        ``n`` for a per-verifier floor, ``2n`` for the pooled one.
    probability : float
        ``p = 1/|B|``.
    floor : int
        The floor as the protocol derives it.
    bases : tuple
        The alphabet, carried so the exact branch states the same null the
        closed-form one does rather than the shipped three-basis default.
    exact : bool
        Keyword-only at every public call site.

    Returns
    -------
    float
        A probability in ``[0, 1]``.
    """
    if exact:
        return matched_shortfall_probability(
            ProtocolParams(key_length=trials, bases=bases),
            minimum_matched=max(1, floor),
        )
    return floor_shortfall_bound(trials, probability, floor)


def evidence_abort_bound(
    params: ProtocolParams, *, counts_exchanged: bool = True, exact: bool = False
) -> float:
    """Bound the honest-run probability that any evidence abort occurs.

    Parameters
    ----------
    params : ProtocolParams
        The **sifted** parameter set -- ``params.key_length`` is ``n``.
    counts_exchanged : bool, optional
        Keyword-only.
    exact : bool, optional
        Keyword-only.

    Returns
    -------
    float
        A probability in ``[0, 1]``.

    Raises
    ------
    TypeError
        If ``params`` is not a :class:`~qsecure.protocol.params.ProtocolParams`
        or a flag is not a :class:`bool`.

    Examples
    --------
    >>> from qsecure.detect.thresholds_structural import evidence_abort_bound
    >>> from qsecure.protocol.params import DEFAULT_PARAMS, ProtocolParams
    >>> f"{evidence_abort_bound(DEFAULT_PARAMS):.4e}"
    '1.6263e-19'
    >>> f"{evidence_abort_bound(DEFAULT_PARAMS, exact=True):.4e}"
    '8.0154e-31'

    >>> f"{evidence_abort_bound(DEFAULT_PARAMS, counts_exchanged=False):.4e}"
    '1.0842e-19'

    >>> for length in (24, 96, 136, 137, 272, 273):
    ...     bound = evidence_abort_bound(ProtocolParams(key_length=length))
    ...     print(f"{length:4d}  {bound:.4e}")
      24  1.1881e-04
      96  2.4904e-17
     136  2.2523e-24
     137  5.4212e-20
     272  5.4210e-20
     273  1.6263e-19
    """
    scored = _as_params(params)
    if not isinstance(counts_exchanged, bool):
        raise TypeError(
            f"counts_exchanged must be a bool, got "
            f"{type(counts_exchanged).__name__}"
        )
    if not isinstance(exact, bool):
        raise TypeError(f"exact must be a bool, got {type(exact).__name__}")
    trials = scored.key_length
    probability = scored.match_probability
    own = _floor_bound(
        trials,
        probability,
        minimum_matched_count(scored),
        scored.bases,
        exact=exact,
    )
    total = 2.0 * own
    if counts_exchanged:
        total += _floor_bound(
            2 * trials,
            probability,
            minimum_pooled_matched_count(scored),
            scored.bases,
            exact=exact,
        )
    return min(1.0, total)


def abort_reason_bound(
    params: ProtocolParams,
    reason: AbortReason | str,
    *,
    exact: bool = False,
) -> float:
    """Bound the honest-run probability of **one** abort reason.

    Parameters
    ----------
    params : ProtocolParams
        The **sifted** parameter set.
    reason : AbortReason or str
        The reason to bound.
    exact : bool, optional
        Keyword-only.

    Returns
    -------
    float
        A probability in ``[0, 1]``.

    Raises
    ------
    TypeError
        If ``params`` is not a :class:`~qsecure.protocol.params.ProtocolParams`
        or ``exact`` is not a :class:`bool`.
    ValueError
        If ``reason`` names no :class:`~qsecure.protocol.verify.AbortReason`,
        or one with no attribution entry here.

    Examples
    --------
    >>> from qsecure.detect.thresholds_structural import abort_reason_bound
    >>> from qsecure.protocol.params import DEFAULT_PARAMS
    >>> from qsecure.protocol.verify import AbortReason
    >>> for reason in AbortReason:
    ...     bound = abort_reason_bound(DEFAULT_PARAMS, reason)
    ...     print(f"{reason.value:44s} {bound:.4e}")
    session-identifier-mismatch                  0.0000e+00
    records-already-verified                     0.0000e+00
    empty-matched-set                            0.0000e+00
    matched-count-below-floor                    5.4210e-20
    counterpart-count-of-unrecorded-provenance   0.0000e+00
    counts-from-two-declarations                 0.0000e+00
    pooled-matched-count-below-floor             5.4210e-20
    counterpart-matched-count-below-floor        5.4210e-20
    unauthorised-verifier                        0.0000e+00

    >>> from qsecure.protocol.params import ProtocolParams
    >>> f"{abort_reason_bound(ProtocolParams(key_length=96), AbortReason.EMPTY_MATCHED_SET):.4e}"
    '1.2452e-17'
    >>> f"{(2 / 3) ** 96:.4e}"
    '1.2452e-17'
    """
    scored = _as_params(params)
    if not isinstance(exact, bool):
        raise TypeError(f"exact must be a bool, got {type(exact).__name__}")
    resolved = _as_reason(str(reason))
    if resolved in STRUCTURAL_ABORT_REASONS:
        return 0.0
    probability = scored.match_probability
    if resolved is AbortReason.EMPTY_MATCHED_SET:
        return (1.0 - probability) ** scored.key_length
    if resolved is AbortReason.POOLED_BELOW_FLOOR:
        return _floor_bound(
            2 * scored.key_length,
            probability,
            minimum_pooled_matched_count(scored),
            scored.bases,
            exact=exact,
        )
    return _floor_bound(
        scored.key_length,
        probability,
        minimum_matched_count(scored),
        scored.bases,
        exact=exact,
    )


def minimum_sifted_length(
    eps: float, *, alphabet: int = 3, counts_exchanged: bool = True
) -> int | None:
    """Return the shortest sifted key at which an evidence abort fits a budget.

    Parameters
    ----------
    eps : float
        The false-positive budget, in ``(0, 1)``.
    alphabet : int, optional
        Keyword-only.
    counts_exchanged : bool, optional
        Keyword-only.

    Returns
    -------
    int or None
        ``None`` when the budget is below what the floors themselves spend --
        ``3 * 2**-64`` with the exchange, ``2 * 2**-64`` without -- because no
        key length reaches it and firing anyway would be a claim the derivation
        does not support.

    Raises
    ------
    TypeError
        If ``eps`` is not a real number, ``alphabet`` is not an integer or
        ``counts_exchanged`` is not a :class:`bool`.
    ValueError
        If ``eps`` is outside ``(0, 1)`` or ``alphabet < 2``.

    Examples
    --------
    >>> from qsecure.detect.thresholds_structural import minimum_sifted_length
    >>> for budget in (1e-3, 1e-6, 1e-9, 1e-15, 1e-18):
    ...     print(f"{budget:.0e}  {minimum_sifted_length(budget)}")
    1e-03  19
    1e-06  36
    1e-09  53
    1e-15  87
    1e-18  104

    >>> minimum_sifted_length(1e-19) is None
    True
    >>> f"{3 * 2.0**-64:.4e}"
    '1.6263e-19'
    >>> minimum_sifted_length(1.7e-19)
    109

    >>> minimum_sifted_length(1.2e-19, counts_exchanged=False)
    110
    >>> minimum_sifted_length(1.2e-19) is None
    True
    """
    budget = _as_budget(eps)
    if isinstance(alphabet, bool) or not isinstance(alphabet, int):
        raise TypeError(
            f"alphabet must be an int, got {type(alphabet).__name__}"
        )
    if alphabet < 2:
        raise ValueError(
            f"alphabet must be at least 2, got {alphabet}; one basis is not a "
            f"basis choice and the match probability would be 1."
        )
    if not isinstance(counts_exchanged, bool):
        raise TypeError(
            f"counts_exchanged must be a bool, got "
            f"{type(counts_exchanged).__name__}"
        )

    bases = _bases_of_size(alphabet)
    values = [
        evidence_abort_bound(
            ProtocolParams(key_length=length, bases=bases),
            counts_exchanged=counts_exchanged,
        )
        for length in range(1, _SEARCH_CEILING + 1)
    ]
    asymptote = values[-1]
    if asymptote > budget:
        return None
    answer: int | None = None
    running = asymptote
    for index in range(_SEARCH_CEILING - 1, -1, -1):
        running = max(running, values[index])
        if running <= budget:
            answer = index + 1
    return answer


def _bases_of_size(alphabet: int) -> tuple[Any, ...]:
    """Return an alphabet of the requested size, from the shipped Pauli bases.

    Parameters
    ----------
    alphabet : int
        ``2`` or ``3``.

    Returns
    -------
    tuple
        A prefix of :data:`~qsecure.protocol.params.DEFAULT_BASES`.

    Raises
    ------
    ValueError
        If ``alphabet`` exceeds the number of Pauli bases there are.
    """
    if alphabet > len(DEFAULT_BASES):
        raise ValueError(
            f"alphabet must be at most {len(DEFAULT_BASES)}, got {alphabet}: "
            f"there are three mutually unbiased Pauli bases and no fourth."
        )
    return DEFAULT_BASES[:alphabet]


@dataclass(frozen=True)
class StructuralThreshold:
    """One derived operating point, with everything needed to audit it.

    Examples
    --------
    >>> from qsecure.detect.thresholds_structural import (
    ...     structural_abort_threshold
    ... )
    >>> threshold = structural_abort_threshold(1e-9)
    >>> threshold.fires_at, threshold.false_positive_bound
    (1, 0.0)
    >>> threshold.admissible, threshold.fires(0), threshold.fires(1)
    (True, False, True)
    >>> threshold.inequality
    'none -- point mass at 0'
    """

    check: StructuralCheck
    statistic: str
    null: str
    inequality: str
    budget: float
    fires_at: int | None
    false_positive_bound: float
    detail: str
    derivation: str

    @property
    def admissible(self) -> bool:
        """bool: Whether an operating point exists inside :attr:`budget`."""
        return self.fires_at is not None

    def fires(self, count: int) -> bool:
        """Whether an observed count trips this threshold.

        Parameters
        ----------
        count : int
            The observed statistic.

        Returns
        -------
        bool
            ``False`` for every count when the threshold is not admissible: a
            withheld check does not fire, it abstains, and the difference is
            visible on :attr:`admissible`.

        Raises
        ------
        TypeError
            If ``count`` is not an integer.
        ValueError
            If ``count`` is negative.
        """
        observed = _as_count(count, "count")
        if self.fires_at is None:
            return False
        return observed >= self.fires_at

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
        """
        return {
            "check": str(self.check),
            "statistic": self.statistic,
            "null": self.null,
            "inequality": self.inequality,
            "budget": self.budget,
            "fires_at": self.fires_at,
            "false_positive_bound": self.false_positive_bound,
            "admissible": self.admissible,
            "detail": self.detail,
            "derivation": self.derivation,
        }


def point_mass_threshold(
    check: StructuralCheck,
    eps: float,
    *,
    statistic: str,
    null: str,
    detail: str,
    derivation: str,
) -> StructuralThreshold:
    """Return the derived threshold for a statistic whose null is a point mass.

    Parameters
    ----------
    check : StructuralCheck
        Which check this is.
    eps : float
        The budget, in ``(0, 1)``.
    statistic, null, detail, derivation : str
        Keyword-only.

    Returns
    -------
    StructuralThreshold
        With ``fires_at = 1`` and ``false_positive_bound = 0.0`` at every
        budget.

    Raises
    ------
    TypeError
        If ``check`` is not a :class:`StructuralCheck`, ``eps`` is not a real
        number, or a text field is not a string.
    ValueError
        If ``eps`` is outside ``(0, 1)``.

    Examples
    --------
    >>> from qsecure.detect.thresholds_structural import (
    ...     StructuralCheck, point_mass_threshold
    ... )
    >>> threshold = point_mass_threshold(
    ...     StructuralCheck.REPLAY_REFUSAL,
    ...     1e-30,
    ...     statistic="replay.total_refusals",
    ...     null="point mass at 0",
    ...     detail="an honest run asks each verifier once",
    ...     derivation="structural-null",
    ... )
    >>> threshold.fires_at, threshold.false_positive_bound
    (1, 0.0)
    """
    if not isinstance(check, StructuralCheck):
        raise TypeError(
            f"check must be a StructuralCheck, got {type(check).__name__}"
        )
    budget = _as_budget(eps)
    for name, value in (
        ("statistic", statistic),
        ("null", null),
        ("detail", detail),
        ("derivation", derivation),
    ):
        if not isinstance(value, str):
            raise TypeError(
                f"{name} must be a string, got {type(value).__name__}"
            )
    return StructuralThreshold(
        check=check,
        statistic=statistic,
        null=null,
        inequality="none -- point mass at 0",
        budget=budget,
        fires_at=1,
        false_positive_bound=0.0,
        detail=detail,
        derivation=derivation,
    )


def structural_abort_threshold(eps: float) -> StructuralThreshold:
    """Return the derived threshold on structural aborts.

    Parameters
    ----------
    eps : float
        The budget, in ``(0, 1)``.

    Returns
    -------
    StructuralThreshold
        ``fires_at = 1``, ``false_positive_bound = 0.0``.

    Raises
    ------
    TypeError
        If ``eps`` is not a real number.
    ValueError
        If ``eps`` is outside ``(0, 1)``.

    Examples
    --------
    >>> from qsecure.detect.thresholds_structural import (
    ...     structural_abort_threshold
    ... )
    >>> loose = structural_abort_threshold(0.5)
    >>> tight = structural_abort_threshold(1e-300)
    >>> (loose.fires_at, loose.false_positive_bound) == (
    ...     tight.fires_at, tight.false_positive_bound
    ... )
    True
    >>> loose.false_positive_bound
    0.0
    """
    return point_mass_threshold(
        StructuralCheck.STRUCTURAL_ABORT,
        eps,
        statistic="aborts.structural",
        null=(
            "point mass at 0: each of the five structural reasons is decided "
            "without a random draw, four by an equality test on data the "
            "honest protocol fixes and the fifth by set membership on an "
            "authorised set the caller supplies"
        ),
        detail=(
            "a session mismatch, a re-presented round, a count with no "
            "provenance and two counts from two declarations cannot occur on "
            "an honest run at any key length, and neither can a verifier "
            "outside the authorised set once that set names the recipients "
            "the signer distributed to"
        ),
        derivation="qsecure.detect.thresholds_structural, structural-null",
    )


def replay_refusal_threshold(eps: float) -> StructuralThreshold:
    """Return the derived threshold on replay refusals.

    Parameters
    ----------
    eps : float
        The budget, in ``(0, 1)``.

    Returns
    -------
    StructuralThreshold
        ``fires_at = 1``, ``false_positive_bound = 0.0``.

    Raises
    ------
    TypeError
        If ``eps`` is not a real number.
    ValueError
        If ``eps`` is outside ``(0, 1)``.

    Examples
    --------
    >>> from qsecure.detect.thresholds_structural import (
    ...     replay_refusal_threshold
    ... )
    >>> threshold = replay_refusal_threshold(1e-9)
    >>> threshold.fires_at, threshold.false_positive_bound
    (1, 0.0)
    >>> "one direction" in threshold.detail
    True
    """
    return point_mass_threshold(
        StructuralCheck.REPLAY_REFUSAL,
        eps,
        statistic="replay.total_refusals",
        null=(
            "point mass at 0: an honest run asks each verifier once per "
            "(session, bit) and the ledger check is set membership"
        ),
        detail=(
            "a non-zero refusal count is a replay outright; zero is not "
            "evidence of none, because the ledger denial-of-service route "
            "burns a round through a rejection, which is a verdict -- sound "
            "in one direction only, and there is no denominator for a rate"
        ),
        derivation="qsecure.detect.thresholds_structural, structural-null",
    )


def run_shape_threshold(eps: float) -> StructuralThreshold:
    """Return the derived threshold on run-shape violations.

    Parameters
    ----------
    eps : float
        The budget, in ``(0, 1)``.

    Returns
    -------
    StructuralThreshold
        ``fires_at = 1``, ``false_positive_bound = 0.0``.

    Raises
    ------
    TypeError
        If ``eps`` is not a real number.
    ValueError
        If ``eps`` is outside ``(0, 1)``.

    Examples
    --------
    >>> from qsecure.detect.thresholds_structural import run_shape_threshold
    >>> run_shape_threshold(1e-9).false_positive_bound
    0.0
    """
    return point_mass_threshold(
        StructuralCheck.RUN_SHAPE,
        eps,
        statistic="run_shape_violations(stats)",
        null=(
            "point mass at 0: six equalities the shipped session satisfies by "
            "construction, none of them a function of any random draw"
        ),
        detail=(
            "a violation says the transcript did not come from the shipped "
            "session -- a provenance finding about the file, not a detection "
            "of any adversary in the threat model"
        ),
        derivation="qsecure.detect.thresholds_structural, structural-null",
    )


def evidence_abort_threshold(
    params: ProtocolParams,
    eps: float,
    *,
    counts_exchanged: bool = True,
    exact: bool = False,
) -> StructuralThreshold:
    """Return the derived threshold on evidence aborts.

    Parameters
    ----------
    params : ProtocolParams
        The **sifted** parameter set.
    eps : float
        The budget, in ``(0, 1)``.
    counts_exchanged : bool, optional
        Keyword-only.
    exact : bool, optional
        Keyword-only.

    Returns
    -------
    StructuralThreshold
        ``fires_at = 1`` and ``false_positive_bound`` the union bound, or
        ``fires_at = None`` with the same bound recorded, so a report can say
        *why* the check was withheld.

    Raises
    ------
    TypeError
        If ``params`` is not a
        :class:`~qsecure.protocol.params.ProtocolParams`, ``eps`` is not a real
        number, or a flag is not a :class:`bool`.
    ValueError
        If ``eps`` is outside ``(0, 1)``.

    Examples
    --------
    >>> from qsecure.detect.thresholds_structural import (
    ...     evidence_abort_threshold
    ... )
    >>> from qsecure.protocol.params import DEFAULT_PARAMS, ProtocolParams
    >>> threshold = evidence_abort_threshold(DEFAULT_PARAMS, 1e-9)
    >>> threshold.fires_at, f"{threshold.false_positive_bound:.4e}"
    (1, '1.6263e-19')

    >>> tight = evidence_abort_threshold(DEFAULT_PARAMS, 1e-20)
    >>> tight.admissible, tight.fires(2)
    (False, False)

    >>> demo = evidence_abort_threshold(ProtocolParams(key_length=24), 1e-9)
    >>> demo.admissible, f"{demo.false_positive_bound:.4e}"
    (False, '1.1881e-04')
    """
    scored = _as_params(params)
    budget = _as_budget(eps)
    bound = evidence_abort_bound(
        scored, counts_exchanged=counts_exchanged, exact=exact
    )
    admissible = bound <= budget
    checks = "three" if counts_exchanged else "two"
    return StructuralThreshold(
        check=StructuralCheck.EVIDENCE_ABORT,
        statistic="aborts.evidence",
        null=(
            f"union of {checks} binomial lower tails: "
            f"m_B, m_C ~ Binomial(n, p) and M ~ Binomial(2n, p) with "
            f"n = {scored.key_length}, p = 1/{len(scored.bases)}"
        ),
        inequality=(
            "multiplicative Chernoff lower tail at eps0 = 2**-64, exact point "
            "mass where a floor degenerates, union bound over the "
            f"{checks} floor events"
        )
        if not exact
        else (
            "exact binomial lower tail per floor, union bound over the "
            f"{checks} floor events"
        ),
        budget=budget,
        fires_at=1 if admissible else None,
        false_positive_bound=bound,
        detail=(
            "an evidence abort on an honest run needs one of the three floor "
            "events; the floors were derived from that same tail at 2**-64, so "
            "the abort itself is the threshold"
        )
        if admissible
        else (
            f"withheld: the proven bound {bound:.3e} exceeds the budget "
            f"{budget:.3e}, so no operating point on this statistic is "
            f"derivable at this key length -- lengthen the key or loosen the "
            f"budget, do not lower the bar"
        ),
        derivation="qsecure.detect.thresholds_structural, evidence-null",
    )


def shortfall_threshold(
    params: ProtocolParams, eps: float, *, pooled: bool = False
) -> StructuralThreshold:
    """Return the derived threshold on how far below a floor a run fell.

    Parameters
    ----------
    params : ProtocolParams
        The **sifted** parameter set.
    eps : float
        The budget, in ``(0, 1)``.
    pooled : bool, optional
        Keyword-only.

    Returns
    -------
    StructuralThreshold
        ``fires_at`` is the smallest shortfall inside the budget, or ``None``
        where the sample is too small to spend it.

    Raises
    ------
    TypeError
        If ``params`` is not a
        :class:`~qsecure.protocol.params.ProtocolParams`, ``eps`` is not a real
        number, or ``pooled`` is not a :class:`bool`.
    ValueError
        If ``eps`` is outside ``(0, 1)``.

    Examples
    --------
    >>> from qsecure.detect.thresholds_structural import shortfall_threshold
    >>> from qsecure.protocol.params import DEFAULT_PARAMS
    >>> for budget in (1e-3, 1e-9, 2.0**-64, 1e-25, 1e-40):
    ...     threshold = shortfall_threshold(DEFAULT_PARAMS, budget)
    ...     print(f"{budget:.2e}  {threshold.fires_at}")
    1.00e-03  1
    1.00e-09  1
    5.42e-20  1
    1.00e-25  258
    1.00e-40  815

    >>> threshold = shortfall_threshold(DEFAULT_PARAMS, 1e-3)
    >>> f"{threshold.false_positive_bound:.4e}", f"{2.0**-64:.4e}"
    ('5.3677e-20', '5.4210e-20')
    """
    scored = _as_params(params)
    budget = _as_budget(eps)
    if not isinstance(pooled, bool):
        raise TypeError(f"pooled must be a bool, got {type(pooled).__name__}")
    trials = 2 * scored.key_length if pooled else scored.key_length
    probability = scored.match_probability
    floor = (
        minimum_pooled_matched_count(scored)
        if pooled
        else minimum_matched_count(scored)
    )
    largest = chernoff_lower_tail_count(trials, probability, budget)
    if largest is None:
        fires_at: int | None = None
        bound = 1.0
    else:
        fires_at = max(1, floor - largest)
        bound = chernoff_deviation_bound(
            max(0, floor - fires_at), trials, probability, side="lower"
        )
    which = "pooled" if pooled else "per-verifier"
    return StructuralThreshold(
        check=StructuralCheck.EVIDENCE_ABORT,
        statistic=f"aborts.shortfalls[party] ({which} floor)",
        null=(
            f"the count behind the shortfall is Binomial({trials}, "
            f"1/{len(scored.bases)}); "
            f"{{shortfall >= s}} = {{count <= {floor} - s}}"
        ),
        inequality="multiplicative Chernoff lower tail, inverted",
        budget=budget,
        fires_at=fires_at,
        false_positive_bound=bound,
        detail=(
            f"the {which} floor is {floor}; at every budget at or above "
            f"2**-64 this collapses to 1, because the floor already spends "
            f"that budget and the abort is itself the threshold"
        ),
        derivation="qsecure.detect.thresholds_structural, shortfall",
    )


def run_shape_violations(stats: TranscriptStatistics) -> tuple[str, ...]:
    """Return the run-shape equalities this transcript breaks.

    Parameters
    ----------
    stats : TranscriptStatistics
        The run.

    Returns
    -------
    tuple of str
        One line per violation, empty on a well-formed transcript.

    Raises
    ------
    TypeError
        If ``stats`` is not a
        :class:`~qsecure.detect.statistics.TranscriptStatistics`.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.detect.statistics import TranscriptStatistics
    >>> from qsecure.detect.thresholds_structural import run_shape_violations
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> stats = TranscriptStatistics.from_transcript(
    ...     QDSSession(
    ...         ProtocolParams(key_length=384), rng=np.random.default_rng(7)
    ...     ).run(0)
    ... )
    >>> run_shape_violations(stats)
    ()
    """
    if not isinstance(stats, TranscriptStatistics):
        raise TypeError(
            f"stats must be a TranscriptStatistics, got "
            f"{type(stats).__name__}"
        )
    replay = stats.replay
    verdicts = set(stats.verifiers)
    refusals = set(stats.aborts.by_party)
    found: list[str] = []
    if replay.spent_rounds > 2:
        found.append(
            f"spent_rounds is {replay.spent_rounds}; one ledger entry per "
            f"verifier per (session, bit) caps it at 2"
        )
    if replay.spent_rounds < len(verdicts):
        found.append(
            f"spent_rounds is {replay.spent_rounds} against "
            f"{len(verdicts)} verdict(s); reaching a verdict spends the round"
        )
    if replay.distinct_sessions > 1:
        found.append(
            f"the ledgers name {replay.distinct_sessions} sessions; one run "
            f"builds one fresh ledger per verifier"
        )
    if replay.distinct_message_bits > 1:
        found.append(
            f"the ledgers name {replay.distinct_message_bits} message bits; "
            f"one run signs one bit"
        )
    strangers = sorted(
        party
        for party in replay.parties_with_spent_rounds
        if party not in (Party.BOB.value, Party.CHARLIE.value)
    )
    if strangers:
        found.append(
            f"{', '.join(strangers)} hold(s) a spent round without being a "
            f"verifier"
        )
    both = sorted(verdicts & refusals)
    if both:
        found.append(
            f"{', '.join(both)} appear(s) among both the verdicts and the "
            f"refusals; each verifier holds exactly one current outcome"
        )
    return tuple(sorted(found))


_SHORT_PARTY: Final[Mapping[AbortReason, str]] = {
    AbortReason.EMPTY_MATCHED_SET: "self",
    AbortReason.BELOW_FLOOR: "self",
    AbortReason.COUNTERPART_BELOW_FLOOR: "counterpart",
    AbortReason.POOLED_BELOW_FLOOR: "pair",
    AbortReason.SESSION_MISMATCH: "none",
    AbortReason.RECORD_ALREADY_VERIFIED: "none",
    AbortReason.COUNT_OF_UNRECORDED_PROVENANCE: "none",
    AbortReason.COUNTS_FROM_TWO_DECLARATIONS: "none",
    AbortReason.UNAUTHORISED_VERIFIER: "none",
}

_FORCED_BY: Final[Mapping[AbortReason, str]] = {
    AbortReason.EMPTY_MATCHED_SET: (
        "a signer who chooses the declaration against this verifier's log; "
        "also an honest run at a key length too short for the floors"
    ),
    AbortReason.BELOW_FLOOR: (
        "a signer who chooses the declaration against this verifier's log"
    ),
    AbortReason.POOLED_BELOW_FLOOR: (
        "a signer aiming the pooled total low -- the split-coin route the "
        "pooled floor closes"
    ),
    AbortReason.COUNTERPART_BELOW_FLOOR: (
        "the counterpart himself, with one integer and probability 1: any "
        "declaration at or below the denial headroom"
    ),
    AbortReason.SESSION_MISMATCH: (
        "anyone who can pair a declaration with another round's record"
    ),
    AbortReason.RECORD_ALREADY_VERIFIED: (
        "anyone who can make this verifier reach a verdict twice, including "
        "the ledger denial-of-service route on the forwarding hop"
    ),
    AbortReason.COUNT_OF_UNRECORDED_PROVENANCE: (
        "the counterpart, by declining to say which declaration he counted"
    ),
    AbortReason.COUNTS_FROM_TWO_DECLARATIONS: (
        "the forwarding hop, with probability 1: a substituted declaration "
        "makes the two counts describe two declarations"
    ),
    AbortReason.UNAUTHORISED_VERIFIER: (
        "nobody in the threat model: the authorised set is the caller's, the "
        "check is off unless he names one, and a party holding a leaked record "
        "passes it under its owner's name"
    ),
}


@dataclass(frozen=True)
class AbortAttribution:
    """One refusal, read for what it says about who made the evidence thin.

    Examples
    --------
    >>> from qsecure.detect.thresholds_structural import AbortAttribution
    >>> AbortAttribution(
    ...     party="Bob",
    ...     reason="counterpart-matched-count-below-floor",
    ...     group="evidence",
    ...     short_party="Charlie",
    ...     shortfall=9,
    ...     false_positive_bound=1.2e-21,
    ...     forced_by="the counterpart himself",
    ... ).short_party
    'Charlie'
    """

    party: str
    reason: str
    group: str
    short_party: str | None
    shortfall: int
    false_positive_bound: float
    forced_by: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
        """
        return {
            "party": self.party,
            "reason": self.reason,
            "group": self.group,
            "short_party": self.short_party,
            "shortfall": self.shortfall,
            "false_positive_bound": self.false_positive_bound,
            "forced_by": self.forced_by,
        }


def attribute_aborts(
    stats: TranscriptStatistics,
) -> tuple[AbortAttribution, ...]:
    """Read every refusal in a run for whose evidence was short.

    Parameters
    ----------
    stats : TranscriptStatistics
        The run.

    Returns
    -------
    tuple of AbortAttribution
        One per refusing party, sorted by party name.

    Raises
    ------
    TypeError
        If ``stats`` is not a
        :class:`~qsecure.detect.statistics.TranscriptStatistics`.
    ValueError
        If the transcript names an abort reason this module has no entry for,
        which means a reason was added upstream without deciding where it
        belongs.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.starvation import CountStarver
    >>> from qsecure.detect.statistics import TranscriptStatistics
    >>> from qsecure.detect.thresholds_structural import attribute_aborts
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> from qsecure.protocol.verify import MatchedSetTooSmall
    >>> session = QDSSession(
    ...     ProtocolParams(key_length=384),
    ...     rng=np.random.default_rng(700000),
    ...     count_exchange=CountStarver(rng=np.random.default_rng(9001)),
    ... )
    >>> try:
    ...     transcript = session.run(0)
    ... except MatchedSetTooSmall:
    ...     transcript = session.transcript()
    >>> found = attribute_aborts(
    ...     TranscriptStatistics.from_transcript(transcript)
    ... )
    >>> found[0].party, found[0].short_party, found[0].shortfall
    ('Bob', 'Charlie', 9)
    >>> found[0].group, f"{found[0].false_positive_bound:.3e}"
    ('evidence', '3.667e-23')
    """
    if not isinstance(stats, TranscriptStatistics):
        raise TypeError(
            f"stats must be a TranscriptStatistics, got "
            f"{type(stats).__name__}"
        )
    scored = stats.params
    probability = scored.match_probability
    floor = minimum_matched_count(scored)
    pooled_floor = minimum_pooled_matched_count(scored)
    found: list[AbortAttribution] = []
    for party in sorted(stats.aborts.by_party):
        raw = stats.aborts.by_party[party]
        reason = _as_reason(raw)
        shortfall = int(stats.aborts.shortfalls.get(party, 0))
        role = _SHORT_PARTY[reason]
        if role == "self":
            short_party: str | None = party
            trials, base = scored.key_length, floor
        elif role == "counterpart":
            short_party = _counterpart(party)
            trials, base = scored.key_length, floor
        elif role == "pair":
            short_party = None
            trials, base = 2 * scored.key_length, pooled_floor
        else:
            short_party = None
            trials, base = 0, 0
        if reason in STRUCTURAL_ABORT_REASONS:
            group, bound = "structural", 0.0
        else:
            group = "evidence"
            bound = chernoff_deviation_bound(
                max(0, base - shortfall), trials, probability, side="lower"
            )
        found.append(
            AbortAttribution(
                party=party,
                reason=str(reason),
                group=group,
                short_party=short_party,
                shortfall=shortfall,
                false_positive_bound=bound,
                forced_by=_FORCED_BY[reason],
            )
        )
    return tuple(found)


def _as_reason(value: str) -> AbortReason:
    """Coerce a recorded abort reason, refusing an unknown one.

    Parameters
    ----------
    value : str
        The reason as the transcript recorded it.

    Returns
    -------
    AbortReason

    Raises
    ------
    ValueError
        If the value names no known reason, or names one this module has no
        attribution entry for.
    """
    try:
        reason = AbortReason(value)
    except ValueError as unknown:
        raise ValueError(
            f"{value!r} names no AbortReason. A transcript carrying one was "
            f"not produced by this protocol."
        ) from unknown
    if reason not in _SHORT_PARTY or reason not in _FORCED_BY:
        raise ValueError(
            f"{reason!r} has no attribution entry in "
            f"qsecure.detect.thresholds_structural. A reason added upstream "
            f"must be placed deliberately: the two abort groups carry "
            f"different provable bounds and must not be summed."
        )
    if reason not in STRUCTURAL_ABORT_REASONS | EVIDENCE_ABORT_REASONS:
        raise ValueError(
            f"{reason!r} belongs to neither abort group. The two carry "
            f"different provable bounds -- exactly zero against 2**-64 -- so a "
            f"reason in neither cannot be reported without inventing one."
        )
    return reason


@dataclass(frozen=True)
class StructuralAlarm:
    """One check that fired, and what firing it costs in false positives.

    Examples
    --------
    >>> from qsecure.detect.thresholds_structural import (
    ...     StructuralAlarm, structural_abort_threshold
    ... )
    >>> alarm = StructuralAlarm(
    ...     check=structural_abort_threshold(1e-9).check,
    ...     count=1,
    ...     threshold=structural_abort_threshold(1e-9),
    ...     detail="Charlie: counts-from-two-declarations",
    ... )
    >>> alarm.false_positive_bound
    0.0
    """

    check: StructuralCheck
    count: int
    threshold: StructuralThreshold
    detail: str

    @property
    def false_positive_bound(self) -> float:
        """float: The proven bound carried by the threshold that fired."""
        return self.threshold.false_positive_bound

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
        """
        return {
            "check": str(self.check),
            "count": self.count,
            "detail": self.detail,
            "false_positive_bound": self.false_positive_bound,
            "threshold": self.threshold.to_dict(),
        }


@dataclass(frozen=True)
class StructuralReport:
    """Everything the structural family has to say about one run.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.detect.statistics import TranscriptStatistics
    >>> from qsecure.detect.thresholds_structural import structural_report
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> stats = TranscriptStatistics.from_transcript(
    ...     QDSSession(
    ...         ProtocolParams(key_length=384), rng=np.random.default_rng(7)
    ...     ).run(0)
    ... )
    >>> report = structural_report(stats, eps=1e-9)
    >>> report.alarm_raised, report.withheld
    (False, ())
    >>> print(report.summary())
    structural: no alarm at eps = 1.000e-09 (false positives <= 1.6263e-19)
    outcomes: Bob accepted, Charlie accepted
    """

    budget: float
    thresholds: tuple[StructuralThreshold, ...]
    alarms: tuple[StructuralAlarm, ...]
    withheld: tuple[StructuralCheck, ...]
    false_positive_bound: float
    outcomes: Mapping[str, RunOutcome]
    attributions: tuple[AbortAttribution, ...]
    count_exchange_timing: str
    key_length: int

    @property
    def alarm_raised(self) -> bool:
        """bool: Whether any structural check fired."""
        return bool(self.alarms)

    @property
    def detection_alarms(self) -> tuple[StructuralAlarm, ...]:
        """tuple of StructuralAlarm: The alarms that are about an adversary."""
        return tuple(
            alarm
            for alarm in self.alarms
            if alarm.check is not StructuralCheck.RUN_SHAPE
        )

    def summary(self) -> str:
        """Return a short account, with refusals on their own line.

        Returns
        -------
        str
            One line for the family's verdict, one per alarm, then the outcomes.
        """
        lines: list[str] = []
        if self.alarms:
            lines.append(
                f"structural: {len(self.alarms)} alarm(s) at eps = "
                f"{self.budget:.3e} "
                f"(false positives <= {self.false_positive_bound:.4e})"
            )
            lines.extend(
                f"  {alarm.check}: {alarm.detail} "
                f"[<= {alarm.false_positive_bound:.4e}]"
                for alarm in self.alarms
            )
        else:
            lines.append(
                f"structural: no alarm at eps = {self.budget:.3e} "
                f"(false positives <= {self.false_positive_bound:.4e})"
            )
        if self.withheld:
            lines.append(
                "withheld (no derivable operating point at this budget): "
                + ", ".join(str(check) for check in sorted(self.withheld))
            )
        lines.append(
            "outcomes: "
            + ", ".join(
                f"{party} {self.outcomes[party].value}"
                for party in sorted(self.outcomes)
            )
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
        """
        return {
            "budget": self.budget,
            "thresholds": [
                threshold.to_dict() for threshold in self.thresholds
            ],
            "alarms": [alarm.to_dict() for alarm in self.alarms],
            "withheld": [str(check) for check in self.withheld],
            "false_positive_bound": self.false_positive_bound,
            "outcomes": {
                party: outcome.value
                for party, outcome in self.outcomes.items()
            },
            "attributions": [
                attribution.to_dict() for attribution in self.attributions
            ],
            "count_exchange_timing": self.count_exchange_timing,
            "key_length": self.key_length,
            "alarm_raised": self.alarm_raised,
            "detections": len(self.detection_alarms),
        }


def structural_report(
    stats: TranscriptStatistics, *, eps: float, exact: bool = False
) -> StructuralReport:
    """Apply the whole structural family to one run at one budget.

    Parameters
    ----------
    stats : TranscriptStatistics
        The run, extracted from a JSON round-tripped transcript.
    eps : float
        Keyword-only.
    exact : bool, optional
        Keyword-only.

    Returns
    -------
    StructuralReport

    Raises
    ------
    TypeError
        If ``stats`` is not a
        :class:`~qsecure.detect.statistics.TranscriptStatistics`, ``eps`` is
        not a real number, or ``exact`` is not a :class:`bool`.
    ValueError
        If ``eps`` is outside ``(0, 1)``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.forgery import RecipientForger
    >>> from qsecure.detect.statistics import TranscriptStatistics
    >>> from qsecure.detect.thresholds_structural import structural_report
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> from qsecure.protocol.verify import MatchedSetTooSmall
    >>> session = QDSSession(
    ...     ProtocolParams(key_length=384),
    ...     rng=np.random.default_rng(800000),
    ...     forwarder=RecipientForger(rng=np.random.default_rng(4242)),
    ... )
    >>> try:
    ...     transcript = session.run(0)
    ... except MatchedSetTooSmall:
    ...     transcript = session.transcript()
    >>> report = structural_report(
    ...     TranscriptStatistics.from_transcript(transcript), eps=1e-9
    ... )
    >>> print(report.summary())
    structural: 1 alarm(s) at eps = 1.000e-09 (false positives <= 1.6263e-19)
      structural-abort: Charlie refused: counts-from-two-declarations [<= 0.0000e+00]
    outcomes: Bob accepted, Charlie refused-to-score

    >>> report.outcomes["Charlie"] is RunOutcome.REFUSED
    True
    >>> report.outcomes["Charlie"] is RunOutcome.REJECTED
    False
    """
    if not isinstance(stats, TranscriptStatistics):
        raise TypeError(
            f"stats must be a TranscriptStatistics, got "
            f"{type(stats).__name__}. Build one with "
            f"TranscriptStatistics.from_transcript(transcript)."
        )
    budget = _as_budget(eps)
    if not isinstance(exact, bool):
        raise TypeError(f"exact must be a bool, got {type(exact).__name__}")

    thresholds = {
        StructuralCheck.STRUCTURAL_ABORT: structural_abort_threshold(budget),
        StructuralCheck.EVIDENCE_ABORT: evidence_abort_threshold(
            stats.params,
            budget,
            counts_exchanged=stats.counts_exchanged,
            exact=exact,
        ),
        StructuralCheck.REPLAY_REFUSAL: replay_refusal_threshold(budget),
        StructuralCheck.RUN_SHAPE: run_shape_threshold(budget),
    }
    shape = run_shape_violations(stats)
    counts = {
        StructuralCheck.STRUCTURAL_ABORT: stats.aborts.structural,
        StructuralCheck.EVIDENCE_ABORT: stats.aborts.evidence,
        StructuralCheck.REPLAY_REFUSAL: stats.replay.total_refusals,
        StructuralCheck.RUN_SHAPE: len(shape),
    }
    attributions = attribute_aborts(stats)
    details = {
        StructuralCheck.STRUCTURAL_ABORT: "; ".join(
            f"{item.party} refused: {item.reason}"
            for item in attributions
            if item.group == "structural"
        ),
        StructuralCheck.EVIDENCE_ABORT: "; ".join(
            f"{item.party} refused: {item.reason}"
            + (
                f", {item.short_party} short by {item.shortfall}"
                if item.short_party is not None
                else f", short by {item.shortfall}"
            )
            for item in attributions
            if item.group == "evidence"
        ),
        StructuralCheck.REPLAY_REFUSAL: "; ".join(
            f"{party} asked to re-decide a spent round {count} time(s)"
            for party, count in sorted(stats.replay.refusals_by_party.items())
        ),
        StructuralCheck.RUN_SHAPE: "; ".join(shape),
    }

    alarms: list[StructuralAlarm] = []
    withheld: list[StructuralCheck] = []
    bound = 0.0
    for check in STRUCTURAL_CHECKS:
        threshold = thresholds[check]
        if not threshold.admissible:
            withheld.append(check)
            continue
        bound += threshold.false_positive_bound
        if threshold.fires(counts[check]):
            alarms.append(
                StructuralAlarm(
                    check=check,
                    count=counts[check],
                    threshold=threshold,
                    detail=details[check] or "(no detail recorded)",
                )
            )
    return StructuralReport(
        budget=budget,
        thresholds=tuple(thresholds[check] for check in STRUCTURAL_CHECKS),
        alarms=tuple(alarms),
        withheld=tuple(withheld),
        false_positive_bound=min(1.0, bound),
        outcomes=outcomes(stats),
        attributions=attributions,
        count_exchange_timing=stats.count_exchange_timing,
        key_length=stats.key_length,
    )
