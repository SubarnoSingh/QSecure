"""Composite detection rule: split a false-positive budget across the three
threshold families, recombine by a union bound, and attribute what fired.

Examples
--------
>>> import numpy as np
>>> from qsecure.detect.detector import detect
>>> from qsecure.protocol.params import ProtocolParams
>>> from qsecure.protocol.session import QDSSession
>>> transcript = QDSSession(
...     ProtocolParams(key_length=384, check_fraction=0.25),
...     rng=np.random.default_rng(7),
... ).run(0)
>>> result = detect(transcript, eps=1e-9)
>>> result.detected, result.signals
(False, ())
>>> f"{result.false_positive_bound:.4e}"
'3.3964e-10'

>>> f"{result.slack_factor:.3f}"
'2.944'

>>> import math
>>> from qsecure.detect.statistics import TranscriptStatistics
>>> from qsecure.detect.thresholds_channel import screen_link
>>> from qsecure.detect.thresholds_rate import RateCountThresholds
>>> from qsecure.detect.thresholds_structural import structural_report
>>> stats = TranscriptStatistics.from_transcript(transcript)
>>> split = result.budget
>>> family = RateCountThresholds.for_transcript(stats, eps=split.rate)
>>> f"{family.per_test_budget:.4e}"
'4.1667e-11'
>>> family.per_test_budget == split.rate / 12 and split.rate < 1e-9 / 2
True
>>> f"{family.evaluate(stats).false_positive_bound:.4e}"
'2.3964e-10'
>>> f"{structural_report(stats, eps=split.structural).false_positive_bound:.4e}"
'1.6263e-19'
>>> screens = [
...     screen_link(link, epsilon=split.per_link)
...     for link in stats.links.values()
... ]
>>> f"{math.fsum(s.false_positive_bound for s in screens):.4e}", len(screens)
('1.0000e-10', 4)

>>> naive = math.fsum(
...     (
...         RateCountThresholds.for_transcript(stats, eps=1e-9)
...         .evaluate(stats)
...         .false_positive_bound,
...         structural_report(stats, eps=1e-9).false_positive_bound,
...         *(
...             screen_link(link, epsilon=1e-9).false_positive_bound
...             for link in stats.links.values()
...         ),
...     )
... )
>>> f"{naive:.4e}", naive > 1e-9
('1.3442e-09', True)

>>> bare = detect(
...     QDSSession(
...         ProtocolParams(key_length=192), rng=np.random.default_rng(7)
...     ).run(0),
...     eps=1e-9,
... )
>>> f"{bare.false_positive_bound:.4e}", f"{bare.slack_factor:.2f}"
('2.6499e-10', '3.77')

>>> result.named
(<Hypothesis.HONEST: 'honest'>,)
>>> result.attribution(Hypothesis.IMPERSONATION_FULL).status
<Support.UNDETECTABLE: 'undetectable-by-construction'>
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from qsecure.detect.statistics import TranscriptStatistics
from qsecure.detect.thresholds_channel import (
    ChannelScreen,
    divide_budget,
    screen_link,
)
from qsecure.detect.thresholds_rate import (
    RATE_COUNT_FREE_TESTS,
    RateCountThresholds,
    RateCountVerdict,
)
from qsecure.detect.thresholds_structural import (
    RunOutcome,
    StructuralCheck,
    StructuralReport,
    evidence_abort_bound,
    structural_report,
)
from qsecure.protocol.params import ProtocolParams
from qsecure.protocol.session import SessionTranscript
from qsecure.protocol.verify import AbortReason

__all__ = [
    "DETECTOR_FAMILIES",
    "HYPOTHESES",
    "HYPOTHESIS_TABLE",
    "LINK_ROSTER",
    "SIGNATURE_SUBSTITUTION_GROUP",
    "Attribution",
    "Detection",
    "FamilyBudget",
    "Hypothesis",
    "HypothesisPredicate",
    "Signal",
    "SignalKind",
    "Support",
    "detect",
    "family_budget",
]


DETECTOR_FAMILIES: Final[tuple[str, ...]] = ("rate", "structural", "channel")

LINK_ROSTER: Final[tuple[tuple[str, int], ...]] = (
    ("Bob", 0),
    ("Bob", 1),
    ("Charlie", 0),
    ("Charlie", 1),
)

_BOUND_SLACK: Final[float] = 1e-12


class SignalKind(enum.StrEnum):
    """What a fired threshold says, in the vocabulary the table is written in.

    Examples
    --------
    >>> from qsecure.detect.detector import SignalKind
    >>> SignalKind.COUNT_HIGH
    <SignalKind.COUNT_HIGH: 'count-high'>
    >>> f"{SignalKind.LEDGER}"
    'ledger'
    """

    MISMATCH = "mismatch"
    COUNT_HIGH = "count-high"
    COUNT_LOW = "count-low"
    EVIDENCE_SHORTFALL = "evidence-shortfall"
    DECLARATION_CONFLICT = "declaration-conflict"
    FORWARDING_TAMPER = "forwarding-tamper"
    LEDGER = "ledger"
    CHANNEL = "channel"
    FILE_SHAPE = "file-shape"


_NOISE_REACHABLE: Final[frozenset[SignalKind]] = frozenset(
    {SignalKind.MISMATCH, SignalKind.CHANNEL}
)

#: The kind that is not about an adversary at all.
_NOT_A_DETECTION: Final[frozenset[SignalKind]] = frozenset(
    {SignalKind.FILE_SHAPE}
)


class Hypothesis(enum.StrEnum):
    """One position in the threat model, plus the honest one.

    Examples
    --------
    >>> from qsecure.detect.detector import Hypothesis
    >>> Hypothesis.RECIPIENT_FORGERY
    <Hypothesis.RECIPIENT_FORGERY: 'recipient-forgery'>
    >>> f"{Hypothesis.IMPERSONATION_FULL}"
    'impersonation-full'
    """

    HONEST = "honest"
    OUTSIDE_FORGERY = "outside-forgery"
    IMPERSONATION_SIGNING = "impersonation-signing-seam"
    IMPERSONATION_DISTRIBUTION = "impersonation-distribution-seam"
    IMPERSONATION_FULL = "impersonation-full"
    RECIPIENT_FORGERY = "recipient-forgery"
    COUNT_STARVATION = "count-starvation"
    REPLAY = "replay"
    CHANNEL_MANIPULATION = "channel-manipulation"


HYPOTHESES: Final[tuple[Hypothesis, ...]] = (
    Hypothesis.HONEST,
    Hypothesis.OUTSIDE_FORGERY,
    Hypothesis.IMPERSONATION_SIGNING,
    Hypothesis.IMPERSONATION_DISTRIBUTION,
    Hypothesis.IMPERSONATION_FULL,
    Hypothesis.RECIPIENT_FORGERY,
    Hypothesis.COUNT_STARVATION,
    Hypothesis.REPLAY,
    Hypothesis.CHANNEL_MANIPULATION,
)

SIGNATURE_SUBSTITUTION_GROUP: Final[tuple[Hypothesis, ...]] = (
    Hypothesis.OUTSIDE_FORGERY,
    Hypothesis.IMPERSONATION_SIGNING,
    Hypothesis.IMPERSONATION_DISTRIBUTION,
)


class Support(enum.StrEnum):
    """What the observed signals say about one hypothesis.

    Examples
    --------
    >>> from qsecure.detect.detector import Support
    >>> Support.EXCLUDED
    <Support.EXCLUDED: 'excluded'>
    >>> f"{Support.UNDETECTABLE}"
    'undetectable-by-construction'
    """

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    EXCLUDED = "excluded"
    UNDETECTABLE = "undetectable-by-construction"


@dataclass(frozen=True)
class HypothesisPredicate:
    """One row of the deduction table: what a position produces, and what not.

    Raises
    ------
    ValueError
        If ``predicts`` and ``leaves_intact`` overlap -- a kind cannot be both
        produced and left alone -- if ``requires`` is not a subset of
        ``predicts``, if ``leaves_intact`` names a kind a noisy honest link
        reaches on a row that is not the null, or if ``mechanism`` is empty.

    Examples
    --------
    >>> from qsecure.detect.detector import HYPOTHESIS_TABLE, Hypothesis
    >>> row = HYPOTHESIS_TABLE[Hypothesis.CHANNEL_MANIPULATION]
    >>> sorted(str(kind) for kind in row.predicts)
    ['channel', 'mismatch']
    >>> 'count-high' in {str(kind) for kind in row.leaves_intact}
    True
    """

    predicts: frozenset[SignalKind]
    leaves_intact: frozenset[SignalKind]
    mechanism: str
    requires: frozenset[SignalKind] = frozenset()
    detectable: bool = True
    is_null: bool = False

    def __post_init__(self) -> None:
        """Refuse a row that contradicts itself or excludes on noise."""
        object.__setattr__(self, "predicts", frozenset(self.predicts))
        object.__setattr__(
            self, "leaves_intact", frozenset(self.leaves_intact)
        )
        object.__setattr__(self, "requires", frozenset(self.requires))
        object.__setattr__(self, "mechanism", str(self.mechanism))
        object.__setattr__(self, "detectable", bool(self.detectable))
        object.__setattr__(self, "is_null", bool(self.is_null))
        both = self.predicts & self.leaves_intact
        if both:
            raise ValueError(
                f"a hypothesis cannot both produce and leave intact "
                f"{sorted(str(kind) for kind in both)}: the first says the "
                f"signal is evidence for it and the second says the signal "
                f"excludes it."
            )
        unpredicted = self.requires - self.predicts
        if unpredicted:
            raise ValueError(
                f"requires must be a subset of predicts, and "
                f"{sorted(str(kind) for kind in unpredicted)} is not: a kind "
                f"a position produces with probability one is a kind it "
                f"produces."
            )
        noisy = self.leaves_intact & _NOISE_REACHABLE
        if noisy and not self.is_null:
            raise ValueError(
                f"{sorted(str(kind) for kind in noisy)} cannot be in a "
                f"leaves_intact set on a row that is not the null: a "
                f"merely noisy honest link produces them, so excluding an "
                f"adversary hypothesis on one would be excluding it on the "
                f"channel's noise level. The null row itself may -- that "
                f"exclusion is the detection. See C-5, and finding F6 of the "
                f"rate family."
            )
        if not self.mechanism:
            raise ValueError(
                "a row of the deduction table must carry the mechanism it was "
                "read off, or it is a rule somebody chose."
            )


_CLASSICAL_KINDS: Final[frozenset[SignalKind]] = frozenset(
    {
        SignalKind.COUNT_HIGH,
        SignalKind.COUNT_LOW,
        SignalKind.EVIDENCE_SHORTFALL,
        SignalKind.DECLARATION_CONFLICT,
        SignalKind.FORWARDING_TAMPER,
        SignalKind.LEDGER,
    }
)

_SUBSTITUTION_ROW: Final[HypothesisPredicate] = HypothesisPredicate(
    predicts=frozenset({SignalKind.MISMATCH}),
    requires=frozenset({SignalKind.MISMATCH}),
    leaves_intact=_CLASSICAL_KINDS,
    mechanism=(
        "a declaration drawn independently of the recipients' records leaves "
        "the matched count's law exactly Binomial(n, 1/|B|) and touches no "
        "wire integer, no forwarding hop and no ledger; only the eigenvalues "
        "at positions that do match are disturbed, and each of those survives "
        "with probability 1/2 "
        "(qsecure.protocol.analysis.FORGER_MATCHED_MISMATCH_PROBABILITY), so "
        "the mismatch count is zero with probability 2**-|M_R| and the "
        "signal is required rather than merely predicted"
    ),
)

HYPOTHESIS_TABLE: Final[Mapping[Hypothesis, HypothesisPredicate]] = {
    Hypothesis.HONEST: HypothesisPredicate(
        predicts=frozenset(),
        leaves_intact=frozenset(SignalKind) - _NOT_A_DETECTION,
        is_null=True,
        mechanism=(
            "the honest run is the run every null in this package describes, "
            "so every threshold fires on it with at most its own proven "
            "bound; this is the one row that may be excluded on a mismatch or "
            "a channel signal, because for the null itself those are the "
            "departure the threshold was built to bound rather than the "
            "channel's noise level"
        ),
    ),
    Hypothesis.OUTSIDE_FORGERY: _SUBSTITUTION_ROW,
    Hypothesis.IMPERSONATION_SIGNING: _SUBSTITUTION_ROW,
    Hypothesis.IMPERSONATION_DISTRIBUTION: _SUBSTITUTION_ROW,
    Hypothesis.IMPERSONATION_FULL: HypothesisPredicate(
        predicts=frozenset(),
        leaves_intact=frozenset(),
        detectable=False,
        mechanism=(
            "Mallory runs the protocol correctly with a key pair of her own, "
            "so every statistic on the transcript is drawn from the honest "
            "law; assumption (AUTH) puts this position out of model rather "
            "than pretending a detector reaches it"
        ),
    ),
    Hypothesis.RECIPIENT_FORGERY: HypothesisPredicate(
        predicts=frozenset(
            {
                SignalKind.COUNT_HIGH,
                SignalKind.FORWARDING_TAMPER,
                SignalKind.MISMATCH,
            }
        ),
        requires=frozenset({SignalKind.COUNT_HIGH, SignalKind.MISMATCH}),
        leaves_intact=frozenset({SignalKind.LEDGER}),
        mechanism=(
            "he forwards a declaration built from his own raw log, which "
            "after Phase A' is roughly half of what the receiving verifier "
            "holds, so that verifier's matched count moves from n/3 to 2n/3 "
            "-- sqrt(n/2) standard deviations of the null, 9.80 at n = 192 "
            "and 240 at DEFAULT_PARAMS, with the derived upper-tail threshold "
            "strictly between the two means, which is what makes COUNT_HIGH a "
            "requirement and not a hope; his forwarded declaration is not "
            "Alice's, so the pair's counts name two declarations; he presents "
            "it into a fresh round, so no ledger entry is spent twice"
        ),
    ),
    Hypothesis.COUNT_STARVATION: HypothesisPredicate(
        predicts=frozenset(
            {
                SignalKind.COUNT_LOW,
                SignalKind.EVIDENCE_SHORTFALL,
                SignalKind.DECLARATION_CONFLICT,
            }
        ),
        leaves_intact=frozenset(
            {
                SignalKind.COUNT_HIGH,
                SignalKind.FORWARDING_TAMPER,
                SignalKind.LEDGER,
            }
        ),
        mechanism=(
            "he understates one integer on the Phase C' wire and nothing "
            "else: his own scored matched count is untouched, so the two part "
            "company, and the lie is downward by construction -- an upper "
            "tail, a second declaration on the forwarding hop and a spent "
            "round are all three out of a count-exchange seam's reach"
        ),
    ),
    Hypothesis.REPLAY: HypothesisPredicate(
        predicts=frozenset(
            {
                SignalKind.LEDGER,
                SignalKind.FORWARDING_TAMPER,
                SignalKind.MISMATCH,
            }
        ),
        leaves_intact=frozenset(
            {
                SignalKind.COUNT_HIGH,
                SignalKind.COUNT_LOW,
                SignalKind.EVIDENCE_SHORTFALL,
            }
        ),
        mechanism=(
            "he re-presents a captured declaration, which the consumed-records "
            "ledger and the session binding both record, and which is not the "
            "declaration Alice sent this run; that declaration was drawn "
            "independently of the receiving verifier's current record, so "
            "every count null is left exactly where it was while the "
            "eigenvalues at matched positions disagree. LEDGER is predicted "
            "and NOT required: the ledger check is sound in one direction "
            "only -- a replay that burns its round through a rejection leaves "
            "the refusal count at zero (finding 4 of the structural family)"
        ),
    ),
    Hypothesis.CHANNEL_MANIPULATION: HypothesisPredicate(
        predicts=frozenset({SignalKind.CHANNEL, SignalKind.MISMATCH}),
        leaves_intact=_CLASSICAL_KINDS,
        mechanism=(
            "the matched set is decided by basis agreement, and a "
            "trace-preserving map on the entanglement resource acts on the "
            "teleported eigenstate; it cannot move a basis, so it cannot move "
            "a count, a floor, a declaration or a bookkeeping equality. "
            "CHANNEL is predicted and NOT required: a weak enough deviation "
            "clears a short check sample, and this layer bounds false "
            "positives and never false negatives"
        ),
    ),
}


def _as_budget(value: Any, name: str = "eps") -> float:
    """Coerce a false-positive budget strictly inside ``(0, 1)``.

    Parameters
    ----------
    value : Any
        The candidate.
    name : str, optional
        Name to quote in an error.

    Returns
    -------
    float

    Raises
    ------
    TypeError
        If ``value`` is not a real number.
    ValueError
        If ``value`` is not finite or lies outside ``(0, 1)``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be a real number in (0, 1), got "
            f"{type(value).__name__}. It is a false-positive budget, not a "
            f"threshold: sweep it to build a ROC curve."
        )
    budget = float(value)
    if not math.isfinite(budget) or not 0.0 < budget < 1.0:
        raise ValueError(
            f"{name} must lie strictly inside (0, 1), got {budget!r}. A "
            f"budget of 0 admits no test and a budget of 1 admits every one."
        )
    return budget


def _as_probability(value: Any, name: str) -> float:
    """Coerce a probability in ``[0, 1]``.

    Parameters
    ----------
    value : Any
        The candidate.
    name : str
        Name to quote in an error.

    Returns
    -------
    float

    Raises
    ------
    TypeError
        If ``value`` is not a real number.
    ValueError
        If ``value`` is not finite or lies outside ``[0, 1]``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be a real number in [0, 1], got "
            f"{type(value).__name__}"
        )
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"{name} must lie in [0, 1], got {probability!r}"
        )
    return probability


@dataclass(frozen=True)
class FamilyBudget:
    """How ``eps`` was divided over the three families, with the algebra on it.

    Raises
    ------
    ValueError
        If the three shares do not sum to :attr:`eps`.

    Examples
    --------
    >>> from qsecure.detect.detector import family_budget
    >>> from qsecure.protocol.params import DEFAULT_PARAMS
    >>> split = family_budget(DEFAULT_PARAMS, eps=1e-9)
    >>> f"{split.rate:.4e}", f"{split.structural:.4e}"
    ('5.0000e-10', '1.6263e-19')
    >>> split.rate + split.structural + split.channel == split.eps
    True
    >>> f"{split.per_link:.4e}"
    '1.2500e-10'

    >>> tight = family_budget(DEFAULT_PARAMS, eps=1e-19)
    >>> f"{tight.structural:.4e}", tight.structural < tight.evidence_bound
    ('3.3333e-20', True)
    """

    eps: float
    rate: float
    structural: float
    channel: float
    evidence_bound: float
    derivation: str

    def __post_init__(self) -> None:
        """Coerce the shares and refuse a split that does not sum to ``eps``."""
        object.__setattr__(self, "eps", _as_budget(self.eps))
        for field in ("rate", "structural", "channel"):
            try:
                object.__setattr__(
                    self, field, _as_budget(getattr(self, field), field)
                )
            except ValueError as exc:
                raise ValueError(
                    f"eps={self.eps!r} is too small to divide into family "
                    f"shares: the {field} share came out as "
                    f"{getattr(self, field)!r}, which is outside (0, 1). "
                    f"The smallest budget this project quotes is 2**-64, "
                    f"about 5.42e-20, and the split stays representable to "
                    f"roughly 1e-310; below that a share underflows and no "
                    f"bound can be proven from it. Sweep no lower rather "
                    f"than reading this as a defect in the split."
                ) from exc
        object.__setattr__(
            self,
            "evidence_bound",
            _as_probability(self.evidence_bound, "evidence_bound"),
        )
        object.__setattr__(self, "derivation", str(self.derivation))
        total = math.fsum((self.rate, self.structural, self.channel))
        if not math.isclose(total, self.eps, rel_tol=_BOUND_SLACK):
            raise ValueError(
                f"eps={self.eps!r}: the shares sum to {total!r} instead. "
                f"The union bound is only a bound on eps when the "
                f"shares sum to it, so a split that does not is refused here "
                f"rather than reported as if it did. Below roughly 1e-313 "
                f"the shares carry subnormal rounding dust and stop summing "
                f"to the budget, which is a limit of binary floating point "
                f"and not a statement about the detector -- refusing is the "
                f"safe direction, and the region is far below any usable "
                f"budget."
            )

    @property
    def per_link(self) -> float:
        """float: One screen's budget, ``channel / len(LINK_ROSTER)``."""
        return divide_budget(self.channel, len(LINK_ROSTER))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
        """
        return {
            "eps": self.eps,
            "rate": self.rate,
            "structural": self.structural,
            "channel": self.channel,
            "per_link": self.per_link,
            "evidence_bound": self.evidence_bound,
            "link_roster": [list(link) for link in LINK_ROSTER],
            "derivation": self.derivation,
        }


def family_budget(
    params: ProtocolParams, *, eps: float, counts_exchanged: bool = True
) -> FamilyBudget:
    """Derive the split of ``eps`` over the three families.

    Parameters
    ----------
    params : ProtocolParams
        The **sifted** parameter set.
    eps : float
        Keyword-only.
    counts_exchanged : bool, optional
        Keyword-only.

    Returns
    -------
    FamilyBudget

    Raises
    ------
    TypeError
        If ``params`` is not a
        :class:`~qsecure.protocol.params.ProtocolParams`, ``eps`` is not a real
        number, or ``counts_exchanged`` is not a :class:`bool`.
    ValueError
        If ``eps`` lies outside ``(0, 1)``.

    Examples
    --------
    >>> from qsecure.detect.detector import family_budget
    >>> from qsecure.protocol.params import ProtocolParams
    >>> split = family_budget(ProtocolParams(key_length=384), eps=1e-9)
    >>> f"{split.structural:.4e}"
    '1.6263e-19'
    >>> f"{split.rate:.6e}", f"{split.channel:.6e}"
    ('5.000000e-10', '5.000000e-10')

    >>> all(
    ...     family_budget(ProtocolParams(key_length=384), eps=e).rate
    ...     >= e / 3
    ...     for e in (1e-30, 1e-19, 1e-9, 1e-3, 0.5)
    ... )
    True

    >>> pooled_off = family_budget(
    ...     ProtocolParams(key_length=384), eps=1e-9, counts_exchanged=False
    ... )
    >>> f"{pooled_off.structural:.4e}"
    '1.0842e-19'
    """
    if not isinstance(params, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(params).__name__}. "
            f"Pass TranscriptStatistics.params, which is already sifted."
        )
    if not isinstance(counts_exchanged, bool):
        raise TypeError(
            f"counts_exchanged must be a bool, got "
            f"{type(counts_exchanged).__name__}"
        )
    budget = _as_budget(eps)
    evidence = evidence_abort_bound(params, counts_exchanged=counts_exchanged)
    even = budget / len(DETECTOR_FAMILIES)
    if evidence >= even:
        structural = even
        rest = even
    else:
        structural = evidence
        rest = (budget - structural) / 2.0
    return FamilyBudget(
        eps=budget,
        rate=rest,
        structural=structural,
        channel=rest,
        evidence_bound=evidence,
        derivation=(
            f"eps_struct = min(B_evid, eps/3) = min({evidence:.4e}, "
            f"{even:.4e}) = {structural:.4e}; "
            f"eps_rate = eps_chan = (eps - eps_struct)/2 = {rest:.4e}; "
            f"sum = eps = {budget:.4e}. The structural family's proven bound "
            f"is B_evid(n) whatever its share, so a larger share buys it "
            f"nothing; the cap at eps/3 keeps the split from ever being "
            f"worse than the even one."
        ),
    )


@dataclass(frozen=True)
class Signal:
    """One threshold that fired, with the proof that let it.

    Examples
    --------
    >>> from qsecure.detect.detector import Signal, SignalKind
    >>> signal = Signal(
    ...     name="rate:matched_count_high:Charlie",
    ...     family="rate",
    ...     kind=SignalKind.COUNT_HIGH,
    ...     statistic="|M_R|",
    ...     observed=128.0,
    ...     critical=108.0,
    ...     false_positive_bound=6.2367e-11,
    ...     claim="...",
    ... )
    >>> signal.is_detection, f"{signal.false_positive_bound:.3e}"
    (True, '6.237e-11')
    """

    name: str
    family: str
    kind: SignalKind
    statistic: str
    observed: float | None
    critical: float | None
    false_positive_bound: float
    claim: str
    is_detection: bool = True

    def __post_init__(self) -> None:
        """Coerce the fields and refuse a family that is not on the roster."""
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "family", str(self.family))
        object.__setattr__(self, "kind", SignalKind(self.kind))
        object.__setattr__(self, "statistic", str(self.statistic))
        if self.observed is not None:
            object.__setattr__(self, "observed", float(self.observed))
        if self.critical is not None:
            object.__setattr__(self, "critical", float(self.critical))
        object.__setattr__(
            self,
            "false_positive_bound",
            _as_probability(self.false_positive_bound, "false_positive_bound"),
        )
        object.__setattr__(self, "claim", str(self.claim))
        object.__setattr__(self, "is_detection", bool(self.is_detection))
        if self.family not in DETECTOR_FAMILIES:
            raise ValueError(
                f"family must be one of {list(DETECTOR_FAMILIES)}, got "
                f"{self.family!r}. A family off the roster would be summed "
                f"into the composite bound without a share of the budget."
            )
        if (self.kind in _NOT_A_DETECTION) == self.is_detection:
            raise ValueError(
                f"{self.name}: kind {str(self.kind)!r} and is_detection="
                f"{self.is_detection} disagree. Only a file-shape signal is "
                f"not a detection, and it never is one (C-7)."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
        """
        return {
            "name": self.name,
            "family": self.family,
            "kind": str(self.kind),
            "statistic": self.statistic,
            "observed": self.observed,
            "critical": self.critical,
            "false_positive_bound": self.false_positive_bound,
            "is_detection": self.is_detection,
            "claim": self.claim,
        }


@dataclass(frozen=True)
class Attribution:
    """What the observed signals say about one hypothesis, and how surely.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.detect.detector import Hypothesis, detect
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> result = detect(
    ...     QDSSession(
    ...         ProtocolParams(key_length=192), rng=np.random.default_rng(7)
    ...     ).run(0),
    ...     eps=1e-9,
    ... )
    >>> result.attribution(Hypothesis.HONEST).status
    <Support.SUPPORTED: 'supported'>
    >>> result.attribution(Hypothesis.REPLAY).status
    <Support.UNSUPPORTED: 'unsupported'>
    """

    hypothesis: Hypothesis
    status: Support
    supporting: tuple[str, ...]
    excluding: tuple[str, ...]
    missing_requirements: tuple[SignalKind, ...]
    false_positive_bound: float | None
    indistinguishable_from: tuple[Hypothesis, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view.

        Returns
        -------
        dict
        """
        return {
            "hypothesis": str(self.hypothesis),
            "status": str(self.status),
            "supporting": list(self.supporting),
            "excluding": list(self.excluding),
            "missing_requirements": [
                str(kind) for kind in self.missing_requirements
            ],
            "false_positive_bound": self.false_positive_bound,
            "indistinguishable_from": [
                str(other) for other in self.indistinguishable_from
            ],
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class Detection:
    """The composite verdict on one run: what fired, what it costs, what it says.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.detect.detector import detect
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> result = detect(
    ...     QDSSession(
    ...         ProtocolParams(key_length=192), rng=np.random.default_rng(7)
    ...     ).run(0),
    ...     eps=1e-9,
    ... )
    >>> result.detected, result.evidence_bound
    (False, None)
    >>> f"{result.false_positive_bound:.4e}"
    '2.6499e-10'
    >>> for line in result.summary().splitlines()[:2]:
    ...     print(line)
    detect: nothing fired at eps = 1.000e-09
      P(any signal | honest) <= 2.6499e-10, a factor of 3.77 inside the budget
    >>> print(result.summary().splitlines()[2])
      consistent with: honest
    >>> print(result.summary().splitlines()[-1])
      outcomes: Bob accepted, Charlie accepted
    """

    eps: float
    detected: bool
    signals: tuple[Signal, ...]
    false_positive_bound: float
    bound_is_unconditional: bool
    evidence_bound: float | None
    budget: FamilyBudget
    attributions: tuple[Attribution, ...]
    outcomes: Mapping[str, RunOutcome]
    not_scored: tuple[str, ...]
    withheld: tuple[str, ...]
    grouping_key: tuple[Any, ...]
    requirements_enforced: bool
    security_claim: bool
    key_length: int
    channel_error_rate: float

    @property
    def null_is_noiseless(self) -> bool:
        """bool: whether the mismatch members were scored against a noiseless link.

        Examples
        --------
        >>> from qsecure.detect import detect
        >>> from qsecure.protocol.params import ProtocolParams
        >>> from qsecure.protocol.session import QDSSession
        >>> import numpy as np
        >>> t = QDSSession(ProtocolParams(key_length=96),
        ...                rng=np.random.default_rng(4)).run(0)
        >>> detect(t, eps=1e-9).null_is_noiseless
        True
        >>> detect(t, eps=1e-9, channel_error_rate=0.01).null_is_noiseless
        False
        """
        return self.channel_error_rate == 0.0

    @property
    def slack_factor(self) -> float:
        """float: ``eps`` over what the composite actually proves."""
        if self.false_positive_bound == 0.0:
            return math.inf
        return self.eps / self.false_positive_bound

    @property
    def detection_signals(self) -> tuple[Signal, ...]:
        """tuple of Signal: The signals that are about an adversary."""
        return tuple(signal for signal in self.signals if signal.is_detection)

    @property
    def kinds(self) -> tuple[SignalKind, ...]:
        """tuple of SignalKind: Which kinds fired, sorted and deduplicated."""
        return tuple(
            sorted({signal.kind for signal in self.detection_signals})
        )

    @property
    def named(self) -> tuple[Hypothesis, ...]:
        """tuple of Hypothesis: The hypotheses the evidence supports."""
        return tuple(
            attribution.hypothesis
            for attribution in self.attributions
            if attribution.status is Support.SUPPORTED
        )

    @property
    def excluded(self) -> tuple[Hypothesis, ...]:
        """tuple of Hypothesis: The hypotheses this run's evidence rules out."""
        return tuple(
            attribution.hypothesis
            for attribution in self.attributions
            if attribution.status is Support.EXCLUDED
        )

    def attribution(self, hypothesis: Hypothesis | str) -> Attribution:
        """Return this run's attribution for one hypothesis.

        Parameters
        ----------
        hypothesis : Hypothesis or str
            A member of :class:`Hypothesis`, or its value.

        Returns
        -------
        Attribution

        Raises
        ------
        KeyError
            If ``hypothesis`` names nothing on :data:`HYPOTHESES`.
        """
        try:
            wanted = Hypothesis(hypothesis)
        except ValueError as error:
            raise KeyError(
                f"{hypothesis!r} is not a hypothesis. The roster is "
                f"{[str(item) for item in HYPOTHESES]}."
            ) from error
        for attribution in self.attributions:
            if attribution.hypothesis is wanted:
                return attribution
        raise KeyError(  # pragma: no cover - HYPOTHESES is complete
            f"{wanted!r} has no attribution on this Detection."
        )

    def signal(self, name: str) -> Signal:
        """Return one fired signal by name.

        Parameters
        ----------
        name : str
            The qualified signal name.

        Returns
        -------
        Signal

        Raises
        ------
        KeyError
            If no signal of that name fired on this run.
        """
        for item in self.signals:
            if item.name == name:
                return item
        raise KeyError(
            f"{name!r} did not fire on this run. What did: "
            f"{[item.name for item in self.signals]}."
        )

    def summary(self) -> str:
        """Return the whole verdict as a few readable lines.

        Returns
        -------
        str
            The headline, the proven bound and its slack, one line per fired
            signal, what the evidence is consistent with, what it rules out and how
            surely, what it cannot separate, and the outcomes -- which are never
            folded into the detection.

        Examples
        --------
        >>> import numpy as np
        >>> from qsecure.attacks.starvation import CountStarver
        >>> from qsecure.detect.detector import detect
        >>> from qsecure.protocol.params import ProtocolParams
        >>> from qsecure.protocol.session import QDSSession
        >>> from qsecure.protocol.verify import MatchedSetTooSmall
        >>> session = QDSSession(
        ...     ProtocolParams(key_length=384),
        ...     rng=np.random.default_rng(500001),
        ...     count_exchange=CountStarver(rng=np.random.default_rng(9001)),
        ... )
        >>> try:
        ...     transcript = session.run(0)
        ... except MatchedSetTooSmall:
        ...     transcript = session.transcript()
        >>> lines = detect(transcript, eps=1e-9).summary().splitlines()
        >>> print(lines[0])
        detect: 2 signal(s) at eps = 1.000e-09
        >>> print(lines[4])
          consistent with: count-starvation
        >>> print(lines[-1])
          outcomes: Bob refused-to-score, Charlie accepted

        >>> from qsecure.detect.thresholds_structural import RunOutcome
        >>> detect(transcript, eps=1e-9).outcomes["Bob"] is RunOutcome.REFUSED
        True
        """
        lines: list[str] = []
        if self.detected:
            lines.append(
                f"detect: {len(self.detection_signals)} signal(s) at eps = "
                f"{self.eps:.3e}"
            )
        else:
            lines.append(f"detect: nothing fired at eps = {self.eps:.3e}")
        lines.append(
            f"  P(any signal | honest) <= "
            f"{self.false_positive_bound:.4e}, a factor of "
            f"{self.slack_factor:.2f} inside the budget"
        )
        for signal in self.signals:
            shown = "n/a" if signal.observed is None else f"{signal.observed:g}"
            mark = "" if signal.is_detection else "  [not a detection, C-7]"
            lines.append(
                f"  {signal.name}: {signal.statistic} = {shown} "
                f"[<= {signal.false_positive_bound:.4e}]{mark}"
            )
        named = self.named
        if named:
            lines.append(
                "  consistent with: "
                + ", ".join(str(item) for item in named)
            )
        else:
            lines.append(
                "  consistent with: nothing in the model -- every in-model "
                "position is excluded by some signal, and no survivor is "
                "invented to fill the row"
            )
        ruled_out = self.excluded
        if ruled_out:
            worst = max(
                self.attribution(item).false_positive_bound or 0.0
                for item in ruled_out
            )
            lines.append(
                "  ruled out: "
                + ", ".join(str(item) for item in ruled_out)
                + f" (each wrongly with probability at most {worst:.4e})"
            )
        if named and not self.requirements_enforced:
            lines.append(
                "  requirements waived: a verifier reached no verdict, so a "
                "signal a position produces with probability one may never "
                "have been looked for"
            )
        group = {
            other
            for item in named
            for other in self.attribution(item).indistinguishable_from
        }
        if group:
            lines.append(
                "  not separable: "
                + ", ".join(sorted(str(other) for other in group))
                + " present identically from a transcript"
            )
        lines.append(
            "  not separable: impersonation-full is undetectable by "
            "construction (AUTH)"
        )
        if self.withheld:
            lines.append("  withheld: " + ", ".join(self.withheld))
        if self.not_scored:
            lines.append(
                "  reached no verdict: " + ", ".join(self.not_scored)
            )
        lines.append(
            "  outcomes: "
            + ", ".join(
                f"{party} {self.outcomes[party].value}"
                for party in sorted(self.outcomes)
            )
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the whole verdict.

        Returns
        -------
        dict
            Passes to :func:`json.dumps` unchanged.
        """
        return {
            "eps": self.eps,
            "detected": self.detected,
            "signals": [signal.to_dict() for signal in self.signals],
            "false_positive_bound": self.false_positive_bound,
            "bound_is_unconditional": self.bound_is_unconditional,
            "channel_error_rate": self.channel_error_rate,
            "null_is_noiseless": self.null_is_noiseless,
            "evidence_bound": self.evidence_bound,
            "slack_factor": self.slack_factor,
            "budget": self.budget.to_dict(),
            "attributions": [
                attribution.to_dict() for attribution in self.attributions
            ],
            "named": [str(item) for item in self.named],
            "kinds": [str(kind) for kind in self.kinds],
            "outcomes": {
                party: outcome.value
                for party, outcome in self.outcomes.items()
            },
            "not_scored": list(self.not_scored),
            "withheld": list(self.withheld),
            "grouping_key": list(self.grouping_key),
            "requirements_enforced": self.requirements_enforced,
            "security_claim": self.security_claim,
            "key_length": self.key_length,
            "summary": self.summary(),
        }


_RATE_KINDS: Final[Mapping[str, SignalKind]] = {
    "mismatch_rate": SignalKind.MISMATCH,
    "matched_count_low": SignalKind.COUNT_LOW,
    "matched_count_high": SignalKind.COUNT_HIGH,
    "declared_count_low": SignalKind.COUNT_LOW,
    "declared_count_high": SignalKind.COUNT_HIGH,
    "pooled_count_low": SignalKind.COUNT_LOW,
    "pooled_count_high": SignalKind.COUNT_HIGH,
    "declaration_gap": SignalKind.DECLARATION_CONFLICT,
}

_REASON_KINDS: Final[Mapping[AbortReason, SignalKind]] = {
    AbortReason.SESSION_MISMATCH: SignalKind.LEDGER,
    AbortReason.RECORD_ALREADY_VERIFIED: SignalKind.LEDGER,
    AbortReason.COUNT_OF_UNRECORDED_PROVENANCE: (
        SignalKind.DECLARATION_CONFLICT
    ),
    AbortReason.COUNTS_FROM_TWO_DECLARATIONS: SignalKind.FORWARDING_TAMPER,
    AbortReason.EMPTY_MATCHED_SET: SignalKind.EVIDENCE_SHORTFALL,
    AbortReason.BELOW_FLOOR: SignalKind.EVIDENCE_SHORTFALL,
    AbortReason.POOLED_BELOW_FLOOR: SignalKind.EVIDENCE_SHORTFALL,
    AbortReason.COUNTERPART_BELOW_FLOOR: SignalKind.EVIDENCE_SHORTFALL,
    AbortReason.UNAUTHORISED_VERIFIER: SignalKind.LEDGER,
}


def _rate_signals(
    family: RateCountThresholds, verdict: RateCountVerdict
) -> list[Signal]:
    """Turn the rate family's verdict into signals.

    Parameters
    ----------
    family : RateCountThresholds
        The family that was evaluated, for the thresholds behind each name.
    verdict : RateCountVerdict
        What it saw.

    Returns
    -------
    list of Signal
    """
    signals: list[Signal] = []
    for name in verdict.fired:
        base = name.partition(":")[0]
        observed = verdict.observations.get(name)
        if name in RATE_COUNT_FREE_TESTS:
            signals.append(
                Signal(
                    name=f"rate:{name}",
                    family="rate",
                    kind=_RATE_KINDS[base],
                    statistic="declared pooled count - scored pooled count",
                    observed=observed,
                    critical=0.0,
                    false_positive_bound=0.0,
                    claim=(
                        "declaration_gap: fires when the two Phase C' wire "
                        "integers do not sum to the count the two verdicts "
                        "scored; under every null this fires with probability "
                        "exactly 0, because it compares two integers that are "
                        "equal by construction on an honest run."
                    ),
                )
            )
            continue
        threshold = family.threshold(name)
        signals.append(
            Signal(
                name=f"rate:{name}",
                family="rate",
                kind=_RATE_KINDS[base],
                statistic=threshold.statistic,
                observed=observed,
                critical=float(threshold.critical_count),
                false_positive_bound=threshold.false_positive_bound,
                claim=threshold.claim(),
            )
        )
    return signals


def _structural_signals(report: StructuralReport) -> list[Signal]:
    """Turn the structural report's alarms into signals, one per refusal reason.

    Parameters
    ----------
    report : StructuralReport
        The structural family's verdict on this run.

    Returns
    -------
    list of Signal
    """
    signals: list[Signal] = []
    reasons: dict[StructuralCheck, list[str]] = {}
    for attribution in report.attributions:
        check = (
            StructuralCheck.STRUCTURAL_ABORT
            if attribution.group == "structural"
            else StructuralCheck.EVIDENCE_ABORT
        )
        reasons.setdefault(check, []).append(
            f"{attribution.party}:{attribution.reason}"
        )
    for alarm in report.alarms:
        threshold = alarm.threshold
        claim = (
            f"{alarm.check}: fires when {threshold.statistic} >= "
            f"{threshold.fires_at}; under {threshold.null}, this fires with "
            f"probability at most {threshold.false_positive_bound:.4e}, by "
            f"{threshold.inequality}."
        )
        if alarm.check in (
            StructuralCheck.STRUCTURAL_ABORT,
            StructuralCheck.EVIDENCE_ABORT,
        ):
            tags = sorted(reasons.get(alarm.check, ()))
            if tags:
                for tag in tags:
                    party, _, reason = tag.partition(":")
                    signals.append(
                        Signal(
                            name=(
                                f"structural:{alarm.check}:{party}:{reason}"
                            ),
                            family="structural",
                            kind=_REASON_KINDS[AbortReason(reason)],
                            statistic=f"{party} refused: {reason}",
                            observed=None,
                            critical=float(threshold.fires_at or 1),
                            false_positive_bound=(
                                threshold.false_positive_bound
                            ),
                            claim=claim,
                        )
                    )
                continue
            signals.append(
                Signal(
                    name=f"structural:{alarm.check}",
                    family="structural",
                    kind=(
                        SignalKind.EVIDENCE_SHORTFALL
                        if alarm.check is StructuralCheck.EVIDENCE_ABORT
                        else SignalKind.DECLARATION_CONFLICT
                    ),
                    statistic=alarm.detail or threshold.statistic,
                    observed=float(alarm.count),
                    critical=float(threshold.fires_at or 1),
                    false_positive_bound=threshold.false_positive_bound,
                    claim=claim,
                )
            )
            continue
        kind = (
            SignalKind.FILE_SHAPE
            if alarm.check is StructuralCheck.RUN_SHAPE
            else SignalKind.LEDGER
        )
        signals.append(
            Signal(
                name=f"structural:{alarm.check}",
                family="structural",
                kind=kind,
                statistic=alarm.detail or threshold.statistic,
                observed=float(alarm.count),
                critical=float(threshold.fires_at or 1),
                false_positive_bound=threshold.false_positive_bound,
                claim=claim,
                is_detection=kind is not SignalKind.FILE_SHAPE,
            )
        )
    return signals


def _channel_signals(
    screens: Mapping[tuple[str, int], ChannelScreen],
) -> list[Signal]:
    """Turn the per-link channel screens into signals.

    Parameters
    ----------
    screens : mapping
        One :class:`~qsecure.detect.thresholds_channel.ChannelScreen` per link
        the run published, keyed by ``(party, message bit)``.

    Returns
    -------
    list of Signal
    """
    signals: list[Signal] = []
    for (party, bit), screen in sorted(screens.items()):
        for name in screen.fired:
            threshold = screen.thresholds[name]
            signals.append(
                Signal(
                    name=f"channel:{party}/{bit}:{name}",
                    family="channel",
                    kind=SignalKind.CHANNEL,
                    statistic=threshold.statistic,
                    observed=screen.observed[name],
                    critical=threshold.critical_value,
                    false_positive_bound=threshold.false_positive_bound,
                    claim=threshold.summary(),
                )
            )
    return signals


def _requirement_kinds(
    rate_verdict: RateCountVerdict,
    report: StructuralReport,
    screens: Mapping[tuple[str, int], ChannelScreen],
) -> frozenset[SignalKind]:
    """Return the kinds a ``requires`` clause may be checked against.

    Parameters
    ----------
    rate_verdict : RateCountVerdict
        The rate family's verdict, whose ``observations`` name exactly the
        members that were evaluated.
    report : StructuralReport
        The structural family's report, whose thresholds say which checks had
        an admissible operating point.
    screens : mapping
        The channel screens that were built.

    Returns
    -------
    frozenset of SignalKind
    """
    if rate_verdict.not_scored:
        return frozenset()
    kinds: set[SignalKind] = set()
    for name in rate_verdict.observations:
        kinds.add(_RATE_KINDS[name.partition(":")[0]])
    for threshold in report.thresholds:
        if not threshold.admissible:
            continue
        if threshold.check is StructuralCheck.EVIDENCE_ABORT:
            kinds.add(SignalKind.EVIDENCE_SHORTFALL)
        elif threshold.check is StructuralCheck.STRUCTURAL_ABORT:
            kinds.update(
                (
                    SignalKind.LEDGER,
                    SignalKind.DECLARATION_CONFLICT,
                    SignalKind.FORWARDING_TAMPER,
                )
            )
        elif threshold.check is StructuralCheck.REPLAY_REFUSAL:
            kinds.add(SignalKind.LEDGER)
    if any(screen.thresholds for screen in screens.values()):
        kinds.add(SignalKind.CHANNEL)
    return frozenset(kinds)


def _attribute(
    signals: Mapping[SignalKind, list[Signal]],
    evaluated: frozenset[SignalKind],
) -> tuple[Attribution, ...]:
    """Apply the hypothesis deduction table to the fired kinds.

    Parameters
    ----------
    signals : mapping of SignalKind to list of Signal
        The detection signals that fired, grouped by kind.
    evaluated : frozenset of SignalKind
        Kinds for which at least one threshold was actually applied on this
        run, and against which a row's ``requires`` is checked.

    Returns
    -------
    tuple of Attribution
        One per member of :data:`HYPOTHESES`, in that order.
    """
    built: list[Attribution] = []
    for hypothesis in HYPOTHESES:
        row = HYPOTHESIS_TABLE[hypothesis]
        supporting = tuple(
            sorted(
                signal.name
                for kind in row.predicts
                for signal in signals.get(kind, ())
            )
        )
        excluding = tuple(
            sorted(
                signal.name
                for kind in row.leaves_intact
                for signal in signals.get(kind, ())
            )
        )
        missing = tuple(
            sorted(
                kind
                for kind in row.requires
                if kind in evaluated and not signals.get(kind)
            )
        )
        group = (
            tuple(
                other
                for other in SIGNATURE_SUBSTITUTION_GROUP
                if other is not hypothesis
            )
            if hypothesis in SIGNATURE_SUBSTITUTION_GROUP
            else ()
        )

        if not row.detectable:
            status = Support.UNDETECTABLE
            bound: float | None = None
            note = (
                "never named and never ruled out: assumption (AUTH) puts this "
                "position out of model, and a hypothesis silently missing "
                "from a table reads as one that was excluded"
            )
        elif excluding:
            status = Support.EXCLUDED
            bound = min(
                signal.false_positive_bound
                for kind in row.leaves_intact
                for signal in signals.get(kind, ())
            )
            note = (
                f"excluded by {list(excluding)}, whose nulls this position "
                f"leaves intact; the exclusion is mistaken with probability "
                f"at most {bound:.4e}"
            )
        elif missing:
            status = Support.UNSUPPORTED
            bound = None
            note = (
                f"not supported: this position produces "
                f"{[str(kind) for kind in missing]} with probability "
                f"essentially one, and on this run that was evaluated and did "
                f"not fire. Withheld rather than refuted -- a signal that did "
                f"not fire carries no bound, only a false-negative statement "
                f"nothing here derives"
            )
        elif supporting:
            status = Support.SUPPORTED
            bound = min(
                signal.false_positive_bound
                for kind in row.predicts
                for signal in signals.get(kind, ())
            )
            note = (
                f"supported by {list(supporting)}; an honest run trips all of "
                f"that evidence with probability at most {bound:.4e}, which "
                f"is a bound against the honest null and not a bound on "
                f"having named the wrong adversary (C-6)"
            )
        elif row.is_null:
            # Reached only when nothing fired at all: the null row leaves every
            # detection kind intact, so any signal takes the branch above.
            status = Support.SUPPORTED
            bound = None
            note = (
                "nothing fired, so nothing departs from the nulls; the "
                "detector's own false-positive bound is the number that says "
                "how much that is worth"
            )
        else:
            status = Support.UNSUPPORTED
            bound = None
            note = (
                "nothing fired that this position predicts, and nothing that "
                "excludes it -- an absence of evidence, which is not evidence "
                "of absence and is not reported as one"
            )
        built.append(
            Attribution(
                hypothesis=hypothesis,
                status=status,
                supporting=supporting,
                excluding=excluding,
                missing_requirements=missing,
                false_positive_bound=bound,
                indistinguishable_from=group,
                rationale=f"{row.mechanism}. {note}.",
            )
        )
    return tuple(built)


def _as_statistics(source: Any) -> TranscriptStatistics:
    """Coerce the detector's one input, keeping the JSON boundary intact.

    Parameters
    ----------
    source : SessionTranscript or TranscriptStatistics or str
        A transcript, its JSON text, or layer one's extraction of one.

    Returns
    -------
    TranscriptStatistics

    Raises
    ------
    TypeError
        If ``source`` is none of the three.
    """
    if isinstance(source, TranscriptStatistics):
        return source
    if isinstance(source, SessionTranscript):
        return TranscriptStatistics.from_transcript(source)
    if isinstance(source, str):
        return TranscriptStatistics.from_json(source)
    raise TypeError(
        f"transcript must be a SessionTranscript, its JSON text, or a "
        f"TranscriptStatistics, got {type(source).__name__}. The detector "
        f"reads a JSON round-tripped transcript and nothing else -- not the "
        f"session, not the adversary, not any harness state."
    )


def detect(
    transcript: SessionTranscript | TranscriptStatistics | str,
    *,
    eps: float,
    channel_error_rate: float = 0.0,
    tolerated_depolarising: float = 0.0,
    method: str = "sharpest",
    qber_method: str = "exact",
) -> Detection:
    """Score one run against every derived threshold, at one budget.

    Parameters
    ----------
    transcript : SessionTranscript or str or TranscriptStatistics
        The run.
    eps : float
        Keyword-only.
    channel_error_rate : float, optional
        Keyword-only.
    tolerated_depolarising : float, optional
        Keyword-only.
    method : {'sharpest', 'exact', 'hoeffding', 'chernoff'}, optional
        Keyword-only.
    qber_method : {'exact', 'chernoff', 'hoeffding'}, optional
        Keyword-only.

    Returns
    -------
    Detection

    Raises
    ------
    TypeError
        If ``transcript`` is not one of the three accepted forms, or an
        argument has the wrong type.
    ValueError
        If ``eps`` lies outside ``(0, 1)``, a noise level lies outside ``[0,
        1]``, or -- and this one would be a bug in this module rather than a
        property of the run -- the composite's proven bound exceeds its own
        budget.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.detect.detector import detect
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> transcript = QDSSession(
    ...     ProtocolParams(key_length=384, check_fraction=0.25),
    ...     rng=np.random.default_rng(11),
    ... ).run(0)
    >>> [
    ...     detect(transcript, eps=e).detected
    ...     for e in (1e-3, 1e-6, 1e-9)
    ... ]
    [False, False, False]
    >>> f"{detect(transcript, eps=1e-3).false_positive_bound:.4e}"
    '4.0167e-04'

    >>> from qsecure.detect.statistics import TranscriptStatistics
    >>> text = transcript.to_json()
    >>> a = detect(transcript, eps=1e-9)
    >>> b = detect(text, eps=1e-9)
    >>> c = detect(TranscriptStatistics.from_json(text), eps=1e-9)
    >>> a.to_dict() == b.to_dict() == c.to_dict()
    True
    """
    stats = _as_statistics(transcript)
    budget = _as_budget(eps)
    noise = _as_probability(channel_error_rate, "channel_error_rate")
    strength = _as_probability(
        tolerated_depolarising, "tolerated_depolarising"
    )
    split = family_budget(
        stats.params, eps=budget, counts_exchanged=stats.counts_exchanged
    )

    rate_family = RateCountThresholds.for_transcript(
        stats, eps=split.rate, channel_error_rate=noise, method=method
    )
    rate_verdict = rate_family.evaluate(stats)
    report = structural_report(stats, eps=split.structural)
    per_link = split.per_link
    screens = {
        link: screen_link(
            stats.links[link],
            epsilon=per_link,
            tolerated_depolarising=strength,
            qber_method=qber_method,
        )
        for link in LINK_ROSTER
        if link in stats.links
    }

    signals = tuple(
        sorted(
            _rate_signals(rate_family, rate_verdict)
            + _structural_signals(report)
            + _channel_signals(screens),
            key=lambda signal: signal.name,
        )
    )

    # Counted once per *threshold*, from each family's own union bound, so two
    # refusal reasons out of one structural check do not appear as two terms.
    bound = min(
        1.0,
        math.fsum(
            (
                rate_verdict.false_positive_bound,
                report.false_positive_bound,
                *(screen.false_positive_bound for screen in screens.values()),
            )
        ),
    )
    if bound > budget * (1.0 + _BOUND_SLACK):
        raise ValueError(  # pragma: no cover - an arithmetic error here
            f"the composite proves {bound!r} against a budget of {budget!r}. "
            f"The union bound and the eps allocation disagree, "
            f"which is a bug in this module and not a property of the run."
        )

    evaluated = _requirement_kinds(rate_verdict, report, screens)
    by_kind: dict[SignalKind, list[Signal]] = {}
    for signal in signals:
        if signal.is_detection:
            by_kind.setdefault(signal.kind, []).append(signal)
    detection_signals = [signal for signal in signals if signal.is_detection]

    withheld = [
        f"structural:{check}" for check in sorted(report.withheld)
    ] + [
        f"channel:{party}/{bit}:{name}"
        for (party, bit), screen in sorted(screens.items())
        for name in sorted(screen.unavailable)
    ]
    if not screens:
        withheld.append(
            "channel: this run published no check rounds, so the whole "
            "channel family is unevaluable -- an unmonitored link is not a "
            "clean one"
        )

    return Detection(
        eps=budget,
        detected=bool(detection_signals),
        signals=signals,
        false_positive_bound=bound,
        bound_is_unconditional=rate_family.bound_is_unconditional,
        channel_error_rate=noise,
        evidence_bound=(
            min(signal.false_positive_bound for signal in detection_signals)
            if detection_signals
            else None
        ),
        budget=split,
        attributions=_attribute(by_kind, evaluated),
        outcomes=report.outcomes,
        not_scored=rate_verdict.not_scored,
        withheld=tuple(withheld),
        grouping_key=rate_verdict.grouping_key,
        requirements_enforced=not rate_verdict.not_scored,
        security_claim=stats.security_claim,
        key_length=stats.key_length,
    )
