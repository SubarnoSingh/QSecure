"""Sampled parameter estimation from check rounds.

A random subset of positions, drawn from the recipients' stream, is spent on
measuring the entanglement resource instead of carrying key. QBER rounds
measure both halves in the same Pauli basis; CHSH rounds measure at Bell-test
settings. The outcomes are published and the positions are sifted out.

.. _check-round-qber-identity:

QBER identity
-------------
For every Bell-diagonal resource, the check-round QBER in basis ``a`` equals the
key mismatch rate in basis ``a``; for a Werner state both are ``p/2``.

.. _check-timing:

Who may know the check set, and when
------------------------------------
The channel adversary must not know the check set while attacking (every seam
is called identically on check and key rounds), and the reserved rounds are
dealt between the links. Alice learns the set only after the pairs are through
the channel, so this is not a device-independent test.

Examples
--------
The default sizing and its cost at ``DEFAULT_PARAMS``:

>>> import math
>>> from qsecure.protocol.checkrounds import (
...     CHECK_CHSH_WEIGHT,
...     CHECK_CONFIDENCE,
...     CLASSICAL_CHSH_BOUND,
...     IDEAL_CHSH,
...     depolarising_chsh,
...     normal_quantile,
...     required_check_rounds,
... )
>>> from qsecure.protocol.params import (
...     CHECKED_PARAMS, DEFAULT_CHECK_FRACTION, DEFAULT_PARAMS,
... )
>>> from qsecure.protocol.verify import enforced_repudiation_bound

>>> f"{IDEAL_CHSH:.4f}", f"{CLASSICAL_CHSH_BOUND:.4f}"
('2.8284', '2.0000')
>>> f"{depolarising_chsh(0.0):.4f}", f"{depolarising_chsh(0.3):.4f}"
('2.8284', '1.9799')
>>> depolarising_chsh(1.0)
0.0

>>> f"{normal_quantile(0.5 + CHECK_CONFIDENCE / 2):.4f}"
'2.5758'
>>> budget = required_check_rounds(DEFAULT_PARAMS)
>>> budget.qber_rounds, budget.chsh_rounds, budget.total
(6688, 6795, 13483)
>>> f"{budget.chsh_rounds / budget.total:.4f}"
'0.5040'
>>> CHECK_CHSH_WEIGHT
0.5
>>> f"{budget.fraction:.4f}"
'0.1170'
>>> budget.fraction <= DEFAULT_CHECK_FRACTION == 0.125
True

>>> z = normal_quantile(0.5 + CHECK_CONFIDENCE / 2)
>>> math.ceil(8 * z ** 2 / (IDEAL_CHSH - CLASSICAL_CHSH_BOUND) ** 2)
78

>>> spent = DEFAULT_PARAMS.with_check_fraction(DEFAULT_CHECK_FRACTION)
>>> spent.check_count, spent.signing_length, spent.expected_matched
(14400, 100800, 33600.0)
>>> f"{enforced_repudiation_bound(spent):.4e}"
'1.8853e-08'
>>> f"{enforced_repudiation_bound(DEFAULT_PARAMS):.4e}"
'1.4139e-09'

>>> CHECKED_PARAMS.key_length, CHECKED_PARAMS.check_count
(131664, 16458)
>>> CHECKED_PARAMS.signing_length, CHECKED_PARAMS.expected_matched
(115206, 38402.0)
>>> f"{enforced_repudiation_bound(CHECKED_PARAMS):.4e}"
'1.4124e-09'
>>> enforced_repudiation_bound(CHECKED_PARAMS) <= enforced_repudiation_bound(
...     DEFAULT_PARAMS)
True

>>> CHECKED_PARAMS.check_count // 2 >= budget.qber_rounds
True
>>> CHECKED_PARAMS.check_count - CHECKED_PARAMS.check_count // 2 >= budget.chsh_rounds
True
"""

from __future__ import annotations

import enum
import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias

import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator

from qsecure.core.measure import projective_measure
from qsecure.core.paulis import PauliBasis
from qsecure.core.rng import resolve_rng
from qsecure.core.states import StateLike, as_density
from qsecure.protocol.keys import PrivateKey
from qsecure.protocol.params import (
    VERIFIERS,
    Party,
    ProtocolParams,
    _as_message_bit,
    _as_party,
)
from qsecure.protocol.records import RecipientRecord

__all__ = [
    "CHECK_CONFIDENCE",
    "CHECK_CHSH_WEIGHT",
    "CHSH_ALICE_ANGLES",
    "CHSH_RECIPIENT_ANGLES",
    "CLASSICAL_CHSH_BOUND",
    "IDEAL_CHSH",
    "IDEAL_PAULI_CORRELATION",
    "CheckLog",
    "CheckRole",
    "CheckRoundBudget",
    "CheckRoundPlan",
    "ChshEstimate",
    "ChshObservation",
    "ChshRound",
    "Interval",
    "QberEstimate",
    "QberObservation",
    "QberRound",
    "depolarising_chsh",
    "draw_check_plan",
    "estimate_chsh",
    "estimate_qber",
    "normal_quantile",
    "observe_chsh_round",
    "observe_qber_round",
    "required_check_rounds",
]


IDEAL_CHSH: Final[float] = 2.0 * math.sqrt(2.0)
"""float: Tsirelson's bound ``2 sqrt(2)``, reached by the shipped settings."""

CLASSICAL_CHSH_BOUND: Final[float] = 2.0
"""float: The CHSH bound of every local hidden-variable model."""

CHSH_ALICE_ANGLES: Final[tuple[float, float]] = (0.0, math.pi / 2.0)
"""Alice's two CHSH settings, as angles from Z towards X."""

CHSH_RECIPIENT_ANGLES: Final[tuple[float, float]] = (
    math.pi / 4.0,
    -math.pi / 4.0,
)
"""The recipient's two CHSH settings, at ``+-pi/4``."""

IDEAL_PAULI_CORRELATION: Final[Mapping[PauliBasis, int]] = {
    PauliBasis.X: 1,
    PauliBasis.Y: -1,
    PauliBasis.Z: 1,
}
"""Sign of ``<PP>`` for ``|Phi+>`` per basis (note ``Y`` is anti-correlated)."""

CHECK_CONFIDENCE: Final[float] = 0.99
"""float: Default two-sided confidence level for every interval here."""

CHECK_CHSH_WEIGHT: Final[float] = 0.5
"""float: Share of check rounds spent on the CHSH arm (derived from R1 and R2)."""

_QBER_METHODS: Final[frozenset[str]] = frozenset({"wilson", "hoeffding"})
_CHSH_METHODS: Final[frozenset[str]] = frozenset({"normal", "hoeffding"})

QberMethod: TypeAlias = Literal["wilson", "hoeffding"]
"""Interval method for :func:`estimate_qber`."""

ChshMethod: TypeAlias = Literal["normal", "hoeffding"]
"""Interval method for :func:`estimate_chsh`."""

_CHSH_ALGEBRAIC_BOUND: Final[float] = 4.0
"""The largest ``|S|`` four numbers in ``[-1, 1]`` can combine to."""

#: Resource qubit carrying Alice's half of the pair (little-endian).
_ALICE_QUBIT: Final[int] = 0

#: Resource qubit carrying the recipient's half.
_RECIPIENT_QUBIT: Final[int] = 1


def normal_quantile(probability: float) -> float:
    """Return the standard normal quantile, by bisecting :func:`math.erfc`.

    Examples
    --------
    >>> from qsecure.protocol.checkrounds import normal_quantile
    >>> f"{abs(normal_quantile(0.5)):.12f}"
    '0.000000000000'
    >>> f"{normal_quantile(0.975):.6f}"
    '1.959964'
    >>> f"{normal_quantile(0.995):.6f}"
    '2.575829'
    """
    value = _as_unit_interval(probability, "probability", strict=True)
    low, high = -40.0, 40.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if 0.5 * math.erfc(-middle / math.sqrt(2.0)) < value:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _two_sided_z(confidence: float) -> float:
    """Return the ``z`` multiplier for a two-sided interval at ``confidence``."""
    level = _as_unit_interval(confidence, "confidence", strict=True)
    return normal_quantile(0.5 + level / 2.0)


def _as_unit_interval(value: Any, name: str, *, strict: bool) -> float:
    """Coerce and range-check a probability; ``strict`` excludes ``0``."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, got the boolean {value!r}")
    if not isinstance(value, numbers.Real):
        raise TypeError(
            f"{name} must be a real number, got {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    lower_ok = result > 0.0 if strict else result >= 0.0
    if not (lower_ok and result < 1.0):
        span = "(0, 1)" if strict else "[0, 1)"
        raise ValueError(f"{name} must lie in {span}, got {result!r}")
    return result


def _as_count(value: Any, name: str, *, minimum: int = 0) -> int:
    """Coerce and range-check a non-negative integer count."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got the boolean {value!r}")
    if not isinstance(value, numbers.Integral):
        raise ValueError(
            f"{name} must be an integer, got {type(value).__name__}"
        )
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {result}")
    return result


def _as_eigenvalue_outcome(value: Any, name: str) -> int:
    """Coerce a measured eigenvalue to ``+1`` or ``-1``, refusing booleans."""
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be the Pauli eigenvalue +1 or -1, got the boolean "
            f"{value!r}; convert with eigenvalue = 1 - 2 * bit."
        )
    if not isinstance(value, numbers.Integral):
        raise ValueError(
            f"{name} must be the integer +1 or -1, got {type(value).__name__}"
        )
    result = int(value)
    if result not in (1, -1):
        raise ValueError(
            f"{name} must be +1 or -1, got {result!r}. These are measurement "
            f"eigenvalues, not bits."
        )
    return result


def _as_setting(value: Any, name: str) -> int:
    """Coerce a CHSH setting index to ``0`` or ``1``."""
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be the integer 0 or 1, got the boolean {value!r}"
        )
    if not isinstance(value, numbers.Integral):
        raise ValueError(
            f"{name} must be the integer 0 or 1, got {type(value).__name__}"
        )
    result = int(value)
    if result not in (0, 1):
        raise ValueError(
            f"{name} must index one of the two CHSH settings (0 or 1), got "
            f"{result!r}. Two settings per wing is what makes the inequality a "
            f"CHSH inequality."
        )
    return result


def depolarising_chsh(depolarising_parameter: float) -> float:
    """Return the CHSH statistic ``(1 - p) * 2 sqrt(2)`` of a Werner resource.

    Examples
    --------
    >>> from qsecure.protocol.checkrounds import depolarising_chsh
    >>> f"{depolarising_chsh(0.0):.4f}"
    '2.8284'
    >>> f"{depolarising_chsh(0.3):.4f}"
    '1.9799'
    >>> f"{depolarising_chsh(1 - 1 / 2 ** 0.5):.4f}"
    '2.0000'
    >>> depolarising_chsh(1.0)
    0.0
    """
    if isinstance(depolarising_parameter, bool):
        raise TypeError(
            f"depolarising_parameter must be a real number in [0, 1], got the "
            f"boolean {depolarising_parameter!r}"
        )
    if not isinstance(depolarising_parameter, numbers.Real):
        raise TypeError(
            f"depolarising_parameter must be a real number in [0, 1], got "
            f"{type(depolarising_parameter).__name__}"
        )
    value = float(depolarising_parameter)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"depolarising_parameter must be a finite number in [0, 1], got "
            f"{value!r}. It is the weight of the maximally mixed component of "
            f"the Werner resource, so it is a probability."
        )
    return (1.0 - value) * IDEAL_CHSH


@dataclass(frozen=True)
class CheckRoundBudget:
    """Check-round counts required per link by a parameter set's thresholds.

    Examples
    --------
    >>> from qsecure.protocol.checkrounds import required_check_rounds
    >>> from qsecure.protocol.params import DEFAULT_PARAMS
    >>> budget = required_check_rounds(DEFAULT_PARAMS)
    >>> budget.qber_rounds, budget.chsh_rounds, budget.total
    (6688, 6795, 13483)
    >>> f"{budget.fraction:.4f}"
    '0.1170'

    Each link measures half the reserved positions, so the fraction to deploy is
    twice as big:

    >>> f"{budget.reserved_fraction:.4f}"
    '0.2341'
    """

    qber_rounds: int
    chsh_rounds: int
    confidence: float
    key_length: int

    @property
    def total(self) -> int:
        """int: ``qber_rounds + chsh_rounds``, the sample one link needs."""
        return self.qber_rounds + self.chsh_rounds

    @property
    def fraction(self) -> float:
        """float: :attr:`total` as a fraction of :attr:`key_length`."""
        return self.total / self.key_length

    @property
    def reserved_fraction(self) -> float:
        """float: The ``check_fraction`` that delivers :attr:`total` per link."""
        return 2.0 * self.total / self.key_length

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the budget."""
        return {
            "qber_rounds": self.qber_rounds,
            "chsh_rounds": self.chsh_rounds,
            "total": self.total,
            "fraction": self.fraction,
            "reserved_fraction": self.reserved_fraction,
            "confidence": self.confidence,
            "key_length": self.key_length,
        }


def required_check_rounds(
    params: ProtocolParams,
    *,
    qber_resolution: float | None = None,
    depolarising_resolution: float | None = None,
    confidence: float = CHECK_CONFIDENCE,
) -> CheckRoundBudget:
    """Size the check sample from the thresholds it has to resolve.

    R1 sizes the QBER arm to a half-width ``qber_resolution`` (default
    ``s_a / 4``); R2 sizes the CHSH arm to resolve ``depolarising_resolution``
    (default ``2 * s_a``). Both use the normal form, with the binomial variance
    evaluated at ``s_a``.

    Raises
    ------
    ValueError
        If a resolution is not positive, ``qber_resolution`` exceeds ``s_a``, or
        ``confidence`` is outside ``(0, 1)``.

    Examples
    --------
    >>> from qsecure.protocol.checkrounds import required_check_rounds
    >>> from qsecure.protocol.params import DEFAULT_PARAMS
    >>> required_check_rounds(DEFAULT_PARAMS).total
    13483
    >>> loose = required_check_rounds(
    ...     DEFAULT_PARAMS, qber_resolution=DEFAULT_PARAMS.s_a / 2)
    >>> loose.qber_rounds
    1672
    >>> tight = required_check_rounds(DEFAULT_PARAMS, confidence=0.999)
    >>> tight.total > required_check_rounds(DEFAULT_PARAMS).total
    True
    """
    checked = _as_params(params)
    level = _as_unit_interval(confidence, "confidence", strict=True)
    z = _two_sided_z(level)

    rate = checked.s_a
    width = (
        rate / 4.0
        if qber_resolution is None
        else _as_positive(qber_resolution, "qber_resolution")
    )
    if rate <= 0.0:
        raise ValueError(
            "cannot size a QBER check sample against s_a = 0: a threshold of "
            "zero admits no noise at all, so no finite sample separates a "
            "channel at the threshold from a perfect one. Use a parameter set "
            "with a positive s_a, or pass qber_resolution explicitly."
        )
    if width > rate:
        raise ValueError(
            f"qber_resolution ({width!r}) exceeds s_a ({rate!r}). An interval "
            f"wider than the threshold it is testing cannot decide whether the "
            f"channel is inside the budget: the interval around an estimate of "
            f"0 would already contain s_a. Try s_a / 4 = {rate / 4.0!r}."
        )
    qber_rounds = math.ceil(z * z * rate * (1.0 - rate) / (width * width))

    strength = (
        2.0 * rate
        if depolarising_resolution is None
        else _as_positive(depolarising_resolution, "depolarising_resolution")
    )
    # Var(S) = sum_ij (1 - E_ij^2) / n_ij = 8 / n for balanced cells at the
    # ideal point, and dS/dp = -IDEAL_CHSH, so a half-width of
    # IDEAL_CHSH * strength in S is a half-width of `strength` in p.
    chsh_half_width = IDEAL_CHSH * strength
    chsh_rounds = math.ceil(8.0 * z * z / (chsh_half_width * chsh_half_width))

    return CheckRoundBudget(
        qber_rounds=qber_rounds,
        chsh_rounds=chsh_rounds,
        confidence=level,
        key_length=checked.key_length,
    )


def _as_positive(value: Any, name: str) -> float:
    """Coerce and check a strictly positive finite float."""
    if isinstance(value, bool):
        raise TypeError(
            f"{name} must be a positive real number, got the boolean {value!r}"
        )
    if not isinstance(value, numbers.Real):
        raise TypeError(
            f"{name} must be a positive real number, got "
            f"{type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(
            f"{name} must be a finite, strictly positive resolution, got "
            f"{result!r}. It is the half-width the sample is sized to achieve, "
            f"and a non-positive one demands an infinite sample."
        )
    return result


def _as_params(params: Any) -> ProtocolParams:
    """Return ``params``, refusing anything that is not a ``ProtocolParams``."""
    if not isinstance(params, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(params).__name__}. "
            f"Start from DEFAULT_PARAMS, CHECKED_PARAMS or DEMO_PARAMS."
        )
    return params


class CheckRole(enum.StrEnum):
    """What a check round is spent on.

    Examples
    --------
    >>> from qsecure.protocol.checkrounds import CheckRole
    >>> CheckRole.QBER, str(CheckRole.CHSH)
    (<CheckRole.QBER: 'qber'>, 'chsh')
    """

    QBER = "qber"
    CHSH = "chsh"


def _as_links(parties: Any) -> tuple[Party, ...] | None:
    """Validate the links a plan's rounds are dealt between; ``None`` means shared."""
    if parties is None:
        return None
    if isinstance(parties, (str, bytes)) or not isinstance(parties, Sequence):
        raise TypeError(
            f"parties must be a sequence of Party members or None, got "
            f"{type(parties).__name__}; the default is VERIFIERS, "
            f"(Party.BOB, Party.CHARLIE), and None draws the shared plan."
        )
    resolved: list[Party] = []
    for party in parties:
        if party is None:
            raise ValueError(
                "parties must name links, and None is not one of them: it is "
                "the spelling of 'no split at all'. Pass parties=None -- the "
                "whole sequence, not an element of it -- for the shared plan."
            )
        # _as_party rather than _as_round_party: an element of this sequence is
        # never the "every link" tag, and the None case is refused above.
        link = _as_party(party)
        if link is Party.ALICE:
            raise ValueError(
                "parties names the recipients whose links measure the check "
                "rounds, and Alice is the common endpoint of both rather than "
                "a link of her own. Pass Party.BOB and Party.CHARLIE."
            )
        resolved.append(link)
    if not resolved:
        raise ValueError(
            "parties must name at least one link to deal the check rounds "
            "between; an empty sequence would leave every reserved position "
            "measured by nobody, so the run would spend a share of its key on "
            "statistics it never publishes. Pass None for the shared plan."
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            f"parties must be distinct, got "
            f"{[party.value for party in resolved]}. A repeated link would "
            f"take a double share of the reserved positions and the other "
            f"link's sample would shrink to match."
        )
    return tuple(resolved)


def _as_round_party(party: Any) -> Party | None:
    """Coerce a planned round's link tag (``None`` = every link), refusing Alice."""
    if party is None:
        return None
    resolved = _as_party(party)
    if resolved is Party.ALICE:
        raise ValueError(
            "a check round is measured on one recipient's link, and Alice is "
            "the common endpoint of both, so she cannot be the link that "
            "measures it. Tag the round with Party.BOB or Party.CHARLIE, or "
            "leave it None for a round every link measures."
        )
    return resolved


@dataclass(frozen=True)
class QberRound:
    """One planned QBER round: a position, the basis both wings use, its link."""

    position: int
    basis: PauliBasis
    party: Party | None = None

    def __post_init__(self) -> None:
        """Coerce the position and check the basis has a known ideal sign."""
        object.__setattr__(
            self, "position", _as_count(self.position, "position")
        )
        if not isinstance(self.basis, PauliBasis):
            object.__setattr__(
                self, "basis", PauliBasis(str(self.basis).strip().upper())
            )
        if self.basis not in IDEAL_PAULI_CORRELATION:
            raise ValueError(
                f"basis {self.basis!r} has no ideal two-wing correlation on "
                f"record, so a round in it could not be scored; expected one "
                f"of {sorted(b.value for b in IDEAL_PAULI_CORRELATION)}."
            )
        object.__setattr__(self, "party", _as_round_party(self.party))

    @property
    def role(self) -> CheckRole:
        """CheckRole: :attr:`CheckRole.QBER`, always."""
        return CheckRole.QBER

    @property
    def ideal_correlation(self) -> int:
        """int: The product ``|Phi+>`` gives in this basis."""
        return IDEAL_PAULI_CORRELATION[self.basis]


@dataclass(frozen=True)
class ChshRound:
    """One planned CHSH round: a position, one setting per wing, its link."""

    position: int
    alice_setting: int
    recipient_setting: int
    party: Party | None = None

    def __post_init__(self) -> None:
        """Coerce the position and check both settings index a shipped angle."""
        object.__setattr__(
            self, "position", _as_count(self.position, "position")
        )
        object.__setattr__(
            self,
            "alice_setting",
            _as_setting(self.alice_setting, "alice_setting"),
        )
        object.__setattr__(
            self,
            "recipient_setting",
            _as_setting(self.recipient_setting, "recipient_setting"),
        )
        object.__setattr__(self, "party", _as_round_party(self.party))

    @property
    def role(self) -> CheckRole:
        """CheckRole: :attr:`CheckRole.CHSH`, always."""
        return CheckRole.CHSH

    @property
    def cell(self) -> int:
        """int: The correlator cell ``2 * alice_setting + recipient_setting``."""
        return 2 * self.alice_setting + self.recipient_setting

    @property
    def angles(self) -> tuple[float, float]:
        """tuple of float: ``(alice_angle, recipient_angle)`` in radians."""
        return (
            CHSH_ALICE_ANGLES[self.alice_setting],
            CHSH_RECIPIENT_ANGLES[self.recipient_setting],
        )


@dataclass(frozen=True)
class CheckRoundPlan:
    """Which positions of a run are check rounds, and what each one measures.

    The reserved set is shared by both recipients so they retain the same key.

    .. _check-round-links:

    Which link measures which reserved position
    -------------------------------------------
    Each round carries a ``party`` tag and a link runs only its own rounds; the
    other link's reserved positions run as key rounds and are sifted out. Learning
    one link's check set therefore does not reveal the other's, at the cost of
    reserving twice the per-link sample. Untagged rounds (``party=None``) are
    measured by every link; tagging is all-or-nothing.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.protocol.checkrounds import draw_check_plan
    >>> from qsecure.protocol.params import Party, ProtocolParams
    >>> params = ProtocolParams(key_length=48, check_fraction=0.25)
    >>> plan = draw_check_plan(params, rng=np.random.default_rng(11))
    >>> plan.check_count, len(plan.signing_positions)
    (12, 36)
    >>> plan.check_count == len(plan.qber_rounds) + len(plan.chsh_rounds)
    True
    >>> set(plan.positions) & set(plan.signing_positions)
    set()

    The reserved set is shared and its halves are not:

    >>> bob = set(plan.positions_for(Party.BOB))
    >>> charlie = set(plan.positions_for(Party.CHARLIE))
    >>> len(bob), len(charlie), bob & charlie
    (6, 6, set())
    >>> bob | charlie == set(plan.positions)
    True
    """

    key_length: int
    qber_rounds: tuple[QberRound, ...]
    chsh_rounds: tuple[ChshRound, ...]

    def __post_init__(self) -> None:
        """Coerce the fields and refuse a plan that cannot be run."""
        object.__setattr__(
            self, "key_length", _as_count(self.key_length, "key_length", minimum=1)
        )
        object.__setattr__(self, "qber_rounds", tuple(self.qber_rounds))
        object.__setattr__(self, "chsh_rounds", tuple(self.chsh_rounds))

        positions = [round_.position for round_ in self.qber_rounds]
        positions.extend(round_.position for round_ in self.chsh_rounds)
        for position in positions:
            if not 0 <= position < self.key_length:
                raise ValueError(
                    f"check-round position {position} is outside the key "
                    f"0 .. {self.key_length - 1}. A plan is drawn for one key "
                    f"length and is not portable to another; draw a fresh one "
                    f"with draw_check_plan(params, rng=...)."
                )
        if len(set(positions)) != len(positions):
            repeated = sorted({p for p in positions if positions.count(p) > 1})
            raise ValueError(
                f"check-round positions must be distinct, got repeats "
                f"{repeated}. One position carries one entanglement resource, "
                f"so it can be spent on the key or on one check round, never "
                f"on two."
            )
        if not positions:
            raise ValueError(
                "a check-round plan must designate at least one position; an "
                "empty plan estimates nothing and would publish a channel "
                "statistic with no sample behind it. Run without a plan if no "
                "estimation is wanted."
            )
        if len(positions) >= self.key_length:
            raise ValueError(
                f"a check-round plan must leave at least one signing position, "
                f"but it designates all {len(positions)} of "
                f"{self.key_length}. Check positions are excluded from the key "
                f"(see ProtocolParams.signing_length), so a plan that takes "
                f"every position leaves nothing to sign."
            )
        tagged = [
            round_.party is not None
            for round_ in (*self.qber_rounds, *self.chsh_rounds)
        ]
        if any(tagged) and not all(tagged):
            raise ValueError(
                f"a check-round plan must tag every round with the link that "
                f"measures it or none of them, got {sum(tagged)} of "
                f"{len(tagged)} tagged. A half-tagged plan would run the "
                f"untagged positions on both links and the tagged ones on "
                f"one, so the two links' published samples would differ in "
                f"size for no reason a reader could recover "
                f"(:ref:`check-round-links`)."
            )

    @property
    def check_count(self) -> int:
        """int: How many positions this plan diverts from the key."""
        return len(self.qber_rounds) + len(self.chsh_rounds)

    @property
    def signing_length(self) -> int:
        """int: How many positions survive into the key, ``L - check_count``."""
        return self.key_length - self.check_count

    @property
    def positions(self) -> tuple[int, ...]:
        """tuple of int: Every check position, ascending."""
        return tuple(
            sorted(
                [round_.position for round_ in self.qber_rounds]
                + [round_.position for round_ in self.chsh_rounds]
            )
        )

    @property
    def signing_positions(self) -> tuple[int, ...]:
        """tuple of int: Every retained position, ascending."""
        checked = set(self.positions)
        return tuple(i for i in range(self.key_length) if i not in checked)

    def is_check(self, position: int) -> bool:
        """Return whether ``position`` is a check round."""
        index = _as_count(position, "position")
        if index >= self.key_length:
            raise ValueError(
                f"position {index} is outside the key 0 .. "
                f"{self.key_length - 1}"
            )
        return index in set(self.positions)

    def round_at(self, position: int) -> QberRound | ChshRound | None:
        """Return the planned round at ``position``, or ``None`` for a key round."""
        index = _as_count(position, "position")
        if index >= self.key_length:
            raise ValueError(
                f"position {index} is outside the key 0 .. "
                f"{self.key_length - 1}"
            )
        return self.rounds_by_position().get(index)

    def rounds_by_position(
        self, party: Party | str | None = None
    ) -> dict[int, QberRound | ChshRound]:
        """Return the plan indexed by position, optionally restricted to one link.

        Examples
        --------
        >>> import numpy as np
        >>> from qsecure.protocol.checkrounds import draw_check_plan
        >>> from qsecure.protocol.params import Party, ProtocolParams
        >>> params = ProtocolParams(key_length=48, check_fraction=0.25)
        >>> plan = draw_check_plan(params, rng=np.random.default_rng(11))
        >>> mine = plan.rounds_by_position(Party.BOB)
        >>> len(mine), len(plan.rounds_by_position())
        (6, 12)
        >>> all(round_.party is Party.BOB for round_ in mine.values())
        True
        """
        wanted = _as_round_party(party)
        rounds: dict[int, QberRound | ChshRound] = {}
        for round_ in (*self.qber_rounds, *self.chsh_rounds):
            if (
                wanted is None
                or round_.party is None
                or round_.party is wanted
            ):
                rounds[round_.position] = round_
        return rounds

    def positions_for(self, party: Party | str) -> tuple[int, ...]:
        """Return the reserved positions ``party``'s link measures, ascending."""
        return tuple(sorted(self.rounds_by_position(party)))

    @property
    def measured_by(self) -> tuple[Party, ...]:
        """tuple of Party: The links this plan splits its rounds between, in order."""
        seen = {
            round_.party
            for round_ in (*self.qber_rounds, *self.chsh_rounds)
            if round_.party is not None
        }
        return tuple(party for party in VERIFIERS if party in seen)

    @property
    def is_split(self) -> bool:
        """bool: Whether the reserved positions are divided between the links."""
        return bool(self.measured_by)

    def sift_key(self, key: PrivateKey) -> PrivateKey:
        """Return ``key`` with the check positions removed (same element objects).

        Examples
        --------
        >>> import numpy as np
        >>> from qsecure.protocol.checkrounds import draw_check_plan
        >>> from qsecure.protocol.keys import generate_private_key
        >>> from qsecure.protocol.params import ProtocolParams
        >>> params = ProtocolParams(key_length=48, check_fraction=0.25)
        >>> plan = draw_check_plan(params, rng=np.random.default_rng(2))
        >>> key = generate_private_key(params, 0, rng=np.random.default_rng(3))
        >>> sifted = plan.sift_key(key)
        >>> len(sifted) == params.signing_length == 36
        True
        >>> sifted.check_against(params.sifted())
        >>> all(
        ...     sifted[j] == key[i]
        ...     for j, i in enumerate(plan.signing_positions)
        ... )
        True
        """
        if not isinstance(key, PrivateKey):
            raise TypeError(
                f"key must be a PrivateKey, got {type(key).__name__}"
            )
        if len(key) != self.key_length:
            raise ValueError(
                f"key has {len(key)} elements but this plan was drawn for a key "
                f"length of {self.key_length}. A plan indexes positions of one "
                f"specific run; sifting a key of another length would silently "
                f"retain the wrong positions."
            )
        return PrivateKey(
            message_bit=key.message_bit,
            elements=tuple(key[i] for i in self.signing_positions),
        )

    def sift_record(self, record: RecipientRecord) -> RecipientRecord:
        """Return ``record`` with the check positions removed."""
        if not isinstance(record, RecipientRecord):
            raise TypeError(
                f"record must be a RecipientRecord, got "
                f"{type(record).__name__}"
            )
        if len(record) != self.key_length:
            raise ValueError(
                f"record has {len(record)} entries but this plan was drawn for "
                f"a key length of {self.key_length}."
            )
        kept = self.signing_positions
        sifted = RecipientRecord.from_measurements(
            record.party,
            record.message_bit,
            [record.bases[i] for i in kept],
            [record.eigenvalues[i] for i in kept],
        )
        return sifted.with_session_id(record.session_id)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the plan.

        Examples
        --------
        >>> import json
        >>> import numpy as np
        >>> from qsecure.protocol.checkrounds import CheckRoundPlan, draw_check_plan
        >>> from qsecure.protocol.params import ProtocolParams
        >>> params = ProtocolParams(key_length=32, check_fraction=0.25)
        >>> plan = draw_check_plan(params, rng=np.random.default_rng(5))
        >>> restored = CheckRoundPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
        >>> restored == plan
        True
        """
        return {
            "key_length": self.key_length,
            "qber_rounds": [
                {
                    "position": round_.position,
                    "basis": round_.basis,
                    "party": round_.party,
                }
                for round_ in self.qber_rounds
            ],
            "chsh_rounds": [
                {
                    "position": round_.position,
                    "alice_setting": round_.alice_setting,
                    "recipient_setting": round_.recipient_setting,
                    "party": round_.party,
                }
                for round_ in self.chsh_rounds
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CheckRoundPlan:
        """Rebuild a plan from :meth:`to_dict` output."""
        known = {"key_length", "qber_rounds", "chsh_rounds"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"unknown CheckRoundPlan field(s) {unknown}; expected "
                f"{sorted(known)}."
            )
        missing = sorted(known - set(data))
        if missing:
            raise ValueError(
                f"CheckRoundPlan.from_dict requires {missing}; a plan with a "
                f"missing arm would silently estimate one statistic and report "
                f"nothing about the other."
            )
        return cls(
            key_length=data["key_length"],
            qber_rounds=tuple(
                QberRound(
                    position=entry["position"],
                    basis=PauliBasis(str(entry["basis"]).upper()),
                    party=entry.get("party"),
                )
                for entry in data["qber_rounds"]
            ),
            chsh_rounds=tuple(
                ChshRound(
                    position=entry["position"],
                    alice_setting=entry["alice_setting"],
                    recipient_setting=entry["recipient_setting"],
                    party=entry.get("party"),
                )
                for entry in data["chsh_rounds"]
            ),
        )


def draw_check_plan(
    params: ProtocolParams,
    *,
    rng: np.random.Generator | None = None,
    chsh_weight: float = CHECK_CHSH_WEIGHT,
    parties: Sequence[Party | str] | None = VERIFIERS,
) -> CheckRoundPlan:
    """Draw the check positions, their roles, their settings and their links.

    Give this the recipients' stream, never Alice's. The CHSH arm takes the floor
    of ``chsh_weight * check_count``; ``parties=None`` draws the shared plan.

    Raises
    ------
    ValueError
        If ``params`` has no check rounds, ``chsh_weight`` is outside ``[0, 1)``,
        or ``parties`` is empty, repeated or names Alice.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.protocol.checkrounds import draw_check_plan
    >>> from qsecure.protocol.params import Party, ProtocolParams
    >>> params = ProtocolParams(key_length=120, check_fraction=0.125)
    >>> plan = draw_check_plan(params, rng=np.random.default_rng(7))
    >>> plan.check_count, len(plan.chsh_rounds), len(plan.qber_rounds)
    (15, 7, 8)
    >>> plan.signing_length == params.signing_length == 105
    True
    >>> other = draw_check_plan(params, rng=np.random.default_rng(7))
    >>> other == plan
    True

    Both arms are dealt between the links:

    >>> plan.measured_by
    (<Party.BOB: 'Bob'>, <Party.CHARLIE: 'Charlie'>)
    >>> [len(plan.positions_for(party)) for party in plan.measured_by]
    [8, 7]
    >>> sum(1 for r in plan.chsh_rounds if r.party is Party.BOB)
    4

    The shared plan is still available:

    >>> shared = draw_check_plan(
    ...     params, rng=np.random.default_rng(7), parties=None
    ... )
    >>> shared.is_split, len(shared.positions_for(Party.BOB))
    (False, 15)
    >>> shared.positions == plan.positions
    True
    """
    checked = _as_params(params)
    weight = _as_unit_interval(chsh_weight, "chsh_weight", strict=False)
    links = _as_links(parties)
    if checked.check_count <= 0:
        raise ValueError(
            f"params.check_fraction is {checked.check_fraction!r}, which "
            f"designates no check rounds at L = {checked.key_length}, so there "
            f"is no plan to draw. A plan with no rounds would publish channel "
            f"statistics with an empty sample behind them. Set a check "
            f"fraction -- params.with_check_fraction(DEFAULT_CHECK_FRACTION) "
            f"-- or run without a plan if no estimation is wanted."
        )
    generator = resolve_rng(rng)

    chosen = generator.permutation(checked.key_length)[: checked.check_count]
    # The order the permutation put them in, kept before the sort: it is the
    # only randomness the link assignment needs and it costs nothing.
    drawn_order = {int(index): rank for rank, index in enumerate(chosen)}
    positions = sorted(int(index) for index in chosen)
    chsh_count = int(math.floor(weight * len(positions)))
    chsh_positions = positions[:chsh_count]
    qber_positions = positions[chsh_count:]

    dealt = 0

    def deal(arm: list[int]) -> dict[int, Party | None]:
        """Assign one arm's positions to the links round-robin, or to none."""
        nonlocal dealt
        if links is None:
            return {position: None for position in arm}
        assignment: dict[int, Party | None] = {}
        for position in sorted(arm, key=lambda index: drawn_order[index]):
            assignment[position] = links[dealt % len(links)]
            dealt += 1
        return assignment

    chsh_links = deal(chsh_positions)
    qber_links = deal(qber_positions)

    alphabet = checked.bases
    qber_rounds = tuple(
        QberRound(
            position=position,
            basis=alphabet[int(generator.integers(len(alphabet)))],
            party=qber_links[position],
        )
        for position in qber_positions
    )
    chsh_rounds = tuple(
        ChshRound(
            position=position,
            alice_setting=int(generator.integers(2)),
            recipient_setting=int(generator.integers(2)),
            party=chsh_links[position],
        )
        for position in chsh_positions
    )
    return CheckRoundPlan(
        key_length=checked.key_length,
        qber_rounds=qber_rounds,
        chsh_rounds=chsh_rounds,
    )


@dataclass(frozen=True)
class QberObservation:
    """What one QBER check round produced.

    Examples
    --------
    >>> from qsecure.core.paulis import PauliBasis
    >>> from qsecure.protocol.checkrounds import QberObservation
    >>> QberObservation(0, PauliBasis.Y, 1, -1).is_error
    False
    >>> QberObservation(0, PauliBasis.Y, 1, 1).is_error
    True
    """

    position: int
    basis: PauliBasis
    alice_eigenvalue: int
    recipient_eigenvalue: int

    def __post_init__(self) -> None:
        """Coerce and validate the two outcomes and the basis."""
        object.__setattr__(
            self, "position", _as_count(self.position, "position")
        )
        if not isinstance(self.basis, PauliBasis):
            object.__setattr__(
                self, "basis", PauliBasis(str(self.basis).strip().upper())
            )
        if self.basis not in IDEAL_PAULI_CORRELATION:
            raise ValueError(
                f"basis {self.basis!r} has no ideal two-wing correlation on "
                f"record; expected one of "
                f"{sorted(b.value for b in IDEAL_PAULI_CORRELATION)}."
            )
        object.__setattr__(
            self,
            "alice_eigenvalue",
            _as_eigenvalue_outcome(self.alice_eigenvalue, "alice_eigenvalue"),
        )
        object.__setattr__(
            self,
            "recipient_eigenvalue",
            _as_eigenvalue_outcome(
                self.recipient_eigenvalue, "recipient_eigenvalue"
            ),
        )

    @property
    def correlation(self) -> int:
        """int: The product of the two outcomes, ``+1`` or ``-1``."""
        return self.alice_eigenvalue * self.recipient_eigenvalue

    @property
    def is_error(self) -> bool:
        """bool: Whether the product disagrees with the ideal sign for :attr:`basis`."""
        return self.correlation != IDEAL_PAULI_CORRELATION[self.basis]


@dataclass(frozen=True)
class ChshObservation:
    """What one CHSH check round produced."""

    position: int
    alice_setting: int
    recipient_setting: int
    alice_eigenvalue: int
    recipient_eigenvalue: int

    def __post_init__(self) -> None:
        """Coerce and validate the settings and the two outcomes."""
        object.__setattr__(
            self, "position", _as_count(self.position, "position")
        )
        object.__setattr__(
            self,
            "alice_setting",
            _as_setting(self.alice_setting, "alice_setting"),
        )
        object.__setattr__(
            self,
            "recipient_setting",
            _as_setting(self.recipient_setting, "recipient_setting"),
        )
        object.__setattr__(
            self,
            "alice_eigenvalue",
            _as_eigenvalue_outcome(self.alice_eigenvalue, "alice_eigenvalue"),
        )
        object.__setattr__(
            self,
            "recipient_eigenvalue",
            _as_eigenvalue_outcome(
                self.recipient_eigenvalue, "recipient_eigenvalue"
            ),
        )

    @property
    def correlation(self) -> int:
        """int: The product of the two outcomes, ``+1`` or ``-1``."""
        return self.alice_eigenvalue * self.recipient_eigenvalue

    @property
    def cell(self) -> int:
        """int: ``2 * alice_setting + recipient_setting``, the correlator cell."""
        return 2 * self.alice_setting + self.recipient_setting


@dataclass(frozen=True)
class CheckLog:
    """Every check-round observation one recipient made on one message bit.

    Examples
    --------
    >>> from qsecure.protocol.checkrounds import CheckLog
    >>> from qsecure.protocol.params import Party
    >>> log = CheckLog(Party.BOB, 0, (), ())
    >>> log.round_count
    0
    """

    party: Party
    message_bit: int
    qber: tuple[QberObservation, ...]
    chsh: tuple[ChshObservation, ...]

    def __post_init__(self) -> None:
        """Coerce the tags and refuse Alice."""
        resolved = _as_party(self.party)
        if resolved is Party.ALICE:
            raise ValueError(
                "a CheckLog belongs to a recipient, not to Alice: a check round "
                "measures one link, and she is the common endpoint of both. "
                "Tag it Party.BOB or Party.CHARLIE."
            )
        object.__setattr__(self, "party", resolved)
        object.__setattr__(
            self, "message_bit", _as_message_bit(self.message_bit)
        )
        object.__setattr__(self, "qber", tuple(self.qber))
        object.__setattr__(self, "chsh", tuple(self.chsh))

    @property
    def round_count(self) -> int:
        """int: How many check rounds this log records, both arms together."""
        return len(self.qber) + len(self.chsh)

    @property
    def positions(self) -> tuple[int, ...]:
        """tuple of int: Every observed check position, ascending."""
        return tuple(
            sorted(
                [entry.position for entry in self.qber]
                + [entry.position for entry in self.chsh]
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the log, raw observations included.

        Examples
        --------
        >>> import json
        >>> from qsecure.core.paulis import PauliBasis
        >>> from qsecure.protocol.checkrounds import CheckLog, QberObservation
        >>> from qsecure.protocol.params import Party
        >>> log = CheckLog(
        ...     Party.CHARLIE, 1, (QberObservation(4, PauliBasis.Y, 1, -1),), ()
        ... )
        >>> CheckLog.from_dict(json.loads(json.dumps(log.to_dict()))) == log
        True
        """
        return {
            "party": self.party,
            "message_bit": self.message_bit,
            "qber": [
                {
                    "position": entry.position,
                    "basis": entry.basis,
                    "alice_eigenvalue": entry.alice_eigenvalue,
                    "recipient_eigenvalue": entry.recipient_eigenvalue,
                }
                for entry in self.qber
            ],
            "chsh": [
                {
                    "position": entry.position,
                    "alice_setting": entry.alice_setting,
                    "recipient_setting": entry.recipient_setting,
                    "alice_eigenvalue": entry.alice_eigenvalue,
                    "recipient_eigenvalue": entry.recipient_eigenvalue,
                }
                for entry in self.chsh
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CheckLog:
        """Rebuild a log from :meth:`to_dict` output."""
        known = {"party", "message_bit", "qber", "chsh"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"unknown CheckLog field(s) {unknown}; expected {sorted(known)}."
            )
        missing = sorted(known - set(data))
        if missing:
            raise ValueError(
                f"CheckLog.from_dict requires {missing}; a log with a missing "
                f"arm would restore as though that arm had produced no rounds, "
                f"which is the one thing an empty sample must never be "
                f"confused with."
            )
        return cls(
            party=data["party"],
            message_bit=data["message_bit"],
            qber=tuple(
                QberObservation(
                    position=entry["position"],
                    basis=PauliBasis(str(entry["basis"]).upper()),
                    alice_eigenvalue=entry["alice_eigenvalue"],
                    recipient_eigenvalue=entry["recipient_eigenvalue"],
                )
                for entry in data["qber"]
            ),
            chsh=tuple(
                ChshObservation(
                    position=entry["position"],
                    alice_setting=entry["alice_setting"],
                    recipient_setting=entry["recipient_setting"],
                    alice_eigenvalue=entry["alice_eigenvalue"],
                    recipient_eigenvalue=entry["recipient_eigenvalue"],
                )
                for entry in data["chsh"]
            ),
        )


def _rotation(angle: float) -> Operator:
    """Return ``R_y(-angle)``, which carries the x-z axis at ``angle`` onto Z."""
    half = -angle / 2.0
    cosine, sine = math.cos(half), math.sin(half)
    return Operator(
        np.array([[cosine, -sine], [sine, cosine]], dtype=complex)
    )


def _as_resource(resource: StateLike) -> DensityMatrix:
    """Coerce and check a two-qubit resource, refusing ``None``."""
    if resource is None:
        raise ValueError(
            "a check round needs an explicit two-qubit entanglement resource; "
            "None is teleport()'s way of asking for the ideal |Phi+> pair and "
            "is not accepted here, because a mis-wired attack would then be "
            "indistinguishable from a clean channel. Pass ideal_resource() if "
            "a clean pair is what you meant."
        )
    state = as_density(resource)
    if state.num_qubits != 2:
        raise ValueError(
            f"a check round measures the two halves of one entanglement "
            f"resource, so it needs a two-qubit state, got "
            f"{state.num_qubits}. Qubit 0 is Alice's half and qubit 1 the "
            f"recipient's, matching the register teleport() builds."
        )
    return state


def _measure_both_wings(
    resource: StateLike,
    alice_basis: PauliBasis,
    recipient_basis: PauliBasis,
    *,
    alice_angle: float = 0.0,
    recipient_angle: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[int, int]:
    """Measure both halves of a pair sequentially; return the two eigenvalues.

    Qubit 0 is Alice's half, qubit 1 the recipient's. Consumes exactly two
    variates.
    """
    state: DensityMatrix = _as_resource(resource)
    if alice_angle:
        state = state.evolve(_rotation(alice_angle), qargs=[_ALICE_QUBIT])
    if recipient_angle:
        state = state.evolve(
            _rotation(recipient_angle), qargs=[_RECIPIENT_QUBIT]
        )
    generator = resolve_rng(rng)
    alice = projective_measure(state, _ALICE_QUBIT, alice_basis, rng=generator)
    recipient = projective_measure(
        alice.post_state, _RECIPIENT_QUBIT, recipient_basis, rng=generator
    )
    return alice.eigenvalue, recipient.eigenvalue


def observe_qber_round(
    resource: StateLike,
    round_: QberRound,
    *,
    rng: np.random.Generator | None = None,
) -> QberObservation:
    """Spend one pair on a matched-basis correlation measurement.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.core.paulis import PauliBasis
    >>> from qsecure.core.states import BellState, bell_state
    >>> from qsecure.protocol.checkrounds import QberRound, observe_qber_round
    >>> pair = bell_state(BellState.PHI_PLUS)
    >>> rng = np.random.default_rng(19)
    >>> [
    ...     observe_qber_round(pair, QberRound(0, basis), rng=rng).is_error
    ...     for basis in (PauliBasis.X, PauliBasis.Y, PauliBasis.Z)
    ... ]
    [False, False, False]
    """
    if not isinstance(round_, QberRound):
        raise TypeError(
            f"round_ must be a QberRound, got {type(round_).__name__}. Draw "
            f"one with draw_check_plan(params, rng=...)."
        )
    alice, recipient = _measure_both_wings(
        resource, round_.basis, round_.basis, rng=rng
    )
    return QberObservation(
        position=round_.position,
        basis=round_.basis,
        alice_eigenvalue=alice,
        recipient_eigenvalue=recipient,
    )


def observe_chsh_round(
    resource: StateLike,
    round_: ChshRound,
    *,
    rng: np.random.Generator | None = None,
) -> ChshObservation:
    """Spend one pair on one cell of the CHSH inequality.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.core.states import BellState, bell_state
    >>> from qsecure.protocol.checkrounds import ChshRound, observe_chsh_round
    >>> observation = observe_chsh_round(
    ...     bell_state(BellState.PHI_PLUS),
    ...     ChshRound(3, 0, 0),
    ...     rng=np.random.default_rng(23),
    ... )
    >>> observation.cell, observation.correlation in (1, -1)
    (0, True)
    """
    if not isinstance(round_, ChshRound):
        raise TypeError(
            f"round_ must be a ChshRound, got {type(round_).__name__}. Draw "
            f"one with draw_check_plan(params, rng=...)."
        )
    alice_angle, recipient_angle = round_.angles
    alice, recipient = _measure_both_wings(
        resource,
        PauliBasis.Z,
        PauliBasis.Z,
        alice_angle=alice_angle,
        recipient_angle=recipient_angle,
        rng=rng,
    )
    return ChshObservation(
        position=round_.position,
        alice_setting=round_.alice_setting,
        recipient_setting=round_.recipient_setting,
        alice_eigenvalue=alice,
        recipient_eigenvalue=recipient,
    )


@dataclass(frozen=True)
class Interval:
    """A two-sided confidence interval, with the method that produced it.

    Examples
    --------
    >>> from qsecure.protocol.checkrounds import Interval
    >>> interval = Interval(0.01, 0.03, 0.99, "wilson")
    >>> interval.covers(0.02), interval.covers(0.05)
    (True, False)
    >>> f"{interval.half_width:.4f}"
    '0.0100'
    """

    low: float
    high: float
    confidence: float
    method: str

    def __post_init__(self) -> None:
        """Coerce the endpoints and check they are ordered and finite."""
        object.__setattr__(self, "low", float(self.low))
        object.__setattr__(self, "high", float(self.high))
        object.__setattr__(
            self,
            "confidence",
            _as_unit_interval(self.confidence, "confidence", strict=True),
        )
        object.__setattr__(self, "method", str(self.method))
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            raise ValueError(
                f"interval endpoints must be finite, got "
                f"[{self.low!r}, {self.high!r}]"
            )
        if self.low > self.high:
            raise ValueError(
                f"interval endpoints must satisfy low <= high, got "
                f"[{self.low!r}, {self.high!r}]"
            )

    @property
    def width(self) -> float:
        """float: ``high - low``."""
        return self.high - self.low

    @property
    def half_width(self) -> float:
        """float: Half the width."""
        return 0.5 * self.width

    def covers(self, value: float) -> bool:
        """Return whether ``value`` lies inside the closed interval."""
        return self.low <= float(value) <= self.high

    def excludes(self, value: float) -> bool:
        """Return whether ``value`` lies strictly outside the interval."""
        return not self.covers(value)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the interval."""
        return {
            "low": self.low,
            "high": self.high,
            "confidence": self.confidence,
            "method": self.method,
        }


@dataclass(frozen=True)
class QberEstimate:
    """The QBER arm's answer: a rate, an interval, and the sample behind them.

    Examples
    --------
    >>> from qsecure.protocol.checkrounds import Interval, QberEstimate
    >>> estimate = QberEstimate(30, 2000, Interval(0.010, 0.022, 0.99, "wilson"))
    >>> estimate.estimate
    0.015
    >>> estimate.within_budget(1 / 64)
    False
    >>> estimate.within_budget(1 / 16)
    True
    """

    errors: int
    rounds: int
    interval: Interval

    def __post_init__(self) -> None:
        """Coerce the counts and check the sample is self-consistent."""
        object.__setattr__(self, "errors", _as_count(self.errors, "errors"))
        object.__setattr__(
            self, "rounds", _as_count(self.rounds, "rounds", minimum=1)
        )
        if self.errors > self.rounds:
            raise ValueError(
                f"errors ({self.errors}) cannot exceed rounds ({self.rounds}); "
                f"a rate above 1 is not a rate."
            )

    @property
    def estimate(self) -> float:
        """float: The point estimate ``errors / rounds``."""
        return self.errors / self.rounds

    @property
    def half_width(self) -> float:
        """float: The interval's half-width."""
        return self.interval.half_width

    def within_budget(self, threshold: float) -> bool:
        """Return whether the interval's upper limit is at or below ``threshold``."""
        return self.interval.high <= float(threshold)

    def implied_depolarising(self) -> Interval:
        """Map the rate interval to the depolarising strength ``p = 2 q``, clipped."""
        return Interval(
            low=min(1.0, max(0.0, 2.0 * self.interval.low)),
            high=min(1.0, max(0.0, 2.0 * self.interval.high)),
            confidence=self.interval.confidence,
            method=self.interval.method,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the estimate."""
        return {
            "errors": self.errors,
            "rounds": self.rounds,
            "estimate": self.estimate,
            "interval": self.interval.to_dict(),
        }


@dataclass(frozen=True)
class ChshEstimate:
    """The CHSH arm's answer: four correlators, their counts, an interval on S.

    Examples
    --------
    >>> from qsecure.protocol.checkrounds import ChshEstimate, Interval
    >>> estimate = ChshEstimate(
    ...     (0.70, 0.71, 0.70, -0.71),
    ...     (500, 500, 500, 500),
    ...     Interval(2.74, 2.90, 0.99, "normal"),
    ... )
    >>> f"{estimate.statistic:.2f}"
    '2.82'
    >>> estimate.violates_classical_bound, estimate.consistent_with_ideal
    (True, True)
    """

    correlators: tuple[float, float, float, float]
    counts: tuple[int, int, int, int]
    interval: Interval

    def __post_init__(self) -> None:
        """Coerce the four cells and check they are correlators."""
        correlators = tuple(float(value) for value in self.correlators)
        counts = tuple(_as_count(value, "counts entry") for value in self.counts)
        if len(correlators) != 4 or len(counts) != 4:
            raise ValueError(
                f"a CHSH estimate has exactly four cells -- two settings per "
                f"wing -- got {len(correlators)} correlators and "
                f"{len(counts)} counts."
            )
        for value in correlators:
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(
                    f"each correlator is a mean of +-1 products and must lie "
                    f"in [-1, 1], got {value!r}"
                )
        object.__setattr__(self, "correlators", correlators)
        object.__setattr__(self, "counts", counts)

    @property
    def statistic(self) -> float:
        """float: ``S = E(a0,b0) + E(a0,b1) + E(a1,b0) - E(a1,b1)``."""
        first, second, third, fourth = self.correlators
        return first + second + third - fourth

    @property
    def rounds(self) -> int:
        """int: The whole CHSH sample, summed over the four cells."""
        return sum(self.counts)

    @property
    def violates_classical_bound(self) -> bool:
        """bool: Whether the interval's lower limit exceeds ``2``."""
        return self.interval.low > CLASSICAL_CHSH_BOUND

    @property
    def consistent_with_ideal(self) -> bool:
        """bool: Whether the interval covers :data:`IDEAL_CHSH`."""
        return self.interval.covers(IDEAL_CHSH)

    def implied_depolarising(self) -> Interval:
        """Map the S interval to ``p = 1 - S / (2 sqrt(2))`` (Werner model)."""
        low = 1.0 - self.interval.high / IDEAL_CHSH
        high = 1.0 - self.interval.low / IDEAL_CHSH
        return Interval(
            low=min(1.0, max(0.0, low)),
            high=min(1.0, max(0.0, high)),
            confidence=self.interval.confidence,
            method=self.interval.method,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the estimate."""
        return {
            "correlators": list(self.correlators),
            "counts": list(self.counts),
            "statistic": self.statistic,
            "interval": self.interval.to_dict(),
        }


def estimate_qber(
    observations: Iterable[QberObservation],
    *,
    confidence: float = CHECK_CONFIDENCE,
    method: QberMethod = "wilson",
) -> QberEstimate:
    """Estimate the channel's error rate from the QBER arm, with an interval.

    Parameters
    ----------
    method : {'wilson', 'hoeffding'}, optional
        Wilson score interval (default), or the distribution-free Hoeffding bound.

    Raises
    ------
    ValueError
        If the sample is empty, or ``confidence`` or ``method`` is invalid.

    Examples
    --------
    >>> from qsecure.core.paulis import PauliBasis
    >>> from qsecure.protocol.checkrounds import QberObservation, estimate_qber
    >>> clean = [
    ...     QberObservation(i, PauliBasis.Z, 1, 1) for i in range(400)
    ... ]
    >>> estimate = estimate_qber(clean)
    >>> estimate.errors, estimate.estimate
    (0, 0.0)
    >>> estimate.interval.low
    0.0
    >>> f"{estimate.interval.high:.4f}"
    '0.0163'
    >>> wide = estimate_qber(clean, method="hoeffding")
    >>> wide.half_width > estimate.half_width
    True
    """
    sample = list(observations)
    for entry in sample:
        if not isinstance(entry, QberObservation):
            raise TypeError(
                f"every element must be a QberObservation, got "
                f"{type(entry).__name__}. The QBER arm and the CHSH arm are "
                f"different measurements and cannot be pooled."
            )
    rounds = len(sample)
    if rounds == 0:
        raise ValueError(
            "cannot estimate a QBER from an empty sample: a rate needs a "
            "denominator. Draw a plan with a positive check fraction, or do "
            "not publish a channel statistic for this run."
        )
    level = _as_unit_interval(confidence, "confidence", strict=True)
    chosen = _as_qber_method(method)
    errors = sum(1 for entry in sample if entry.is_error)

    if chosen == "wilson":
        low, high = _wilson_interval(errors, rounds, level)
    else:
        low, high = _hoeffding_rate_interval(errors, rounds, level)
    return QberEstimate(
        errors=errors,
        rounds=rounds,
        interval=Interval(
            low=low, high=high, confidence=level, method=chosen
        ),
    )


def _as_qber_method(method: Any) -> QberMethod:
    """Validate and lower-case the QBER interval method."""
    if not isinstance(method, str):
        raise TypeError(
            f"method must be a string, got {type(method).__name__}"
        )
    chosen = method.strip().lower()
    if chosen not in _QBER_METHODS:
        raise ValueError(
            f"unknown QBER interval method {method!r}; expected 'wilson' (the "
            f"calibrated score interval, for reporting) or 'hoeffding' (the "
            f"distribution-free bound, for a security claim)."
        )
    return chosen  # type: ignore[return-value]


def _wilson_interval(
    errors: int, rounds: int, confidence: float
) -> tuple[float, float]:
    """Return the Wilson score interval for a proportion, clipped to ``[0, 1]``.

    The endpoints are clamped by case so ``0`` and ``1`` come out exact rather
    than as floating-point dust.
    """
    z = _two_sided_z(confidence)
    z_squared = z * z
    denominator = rounds + z_squared
    centre = (errors + z_squared / 2.0) / denominator
    spread = (
        z
        / denominator
        * math.sqrt(
            errors * (rounds - errors) / rounds + z_squared / 4.0
        )
    )
    low = 0.0 if errors == 0 else max(0.0, centre - spread)
    high = 1.0 if errors == rounds else min(1.0, centre + spread)
    return low, high


def _hoeffding_rate_interval(
    errors: int, rounds: int, confidence: float
) -> tuple[float, float]:
    """Return the distribution-free Hoeffding interval for a proportion."""
    alpha = 1.0 - confidence
    half = math.sqrt(math.log(2.0 / alpha) / (2.0 * rounds))
    estimate = errors / rounds
    return max(0.0, estimate - half), min(1.0, estimate + half)


def estimate_chsh(
    observations: Iterable[ChshObservation],
    *,
    confidence: float = CHECK_CONFIDENCE,
    method: ChshMethod = "normal",
) -> ChshEstimate:
    """Estimate the CHSH statistic from the Bell-test arm, with an interval.

    Parameters
    ----------
    method : {'normal', 'hoeffding'}, optional
        Plug-in normal variance (default), or a union-bounded Hoeffding width.

    Raises
    ------
    ValueError
        If any of the four cells is empty, or ``confidence`` or ``method`` is
        invalid.

    Notes
    -----
    The interval is clipped to the algebraic range ``[-4, 4]``, not Tsirelson's
    bound, so that a super-quantum estimate stays visible.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.core.states import BellState, bell_state
    >>> from qsecure.protocol.checkrounds import (
    ...     ChshRound, estimate_chsh, observe_chsh_round
    ... )
    >>> rng = np.random.default_rng(2026)
    >>> pair = bell_state(BellState.PHI_PLUS)
    >>> sample = [
    ...     observe_chsh_round(
    ...         pair, ChshRound(i, i // 2 % 2, i % 2), rng=rng
    ...     )
    ...     for i in range(2000)
    ... ]
    >>> estimate = estimate_chsh(sample)
    >>> estimate.violates_classical_bound, estimate.consistent_with_ideal
    (True, True)
    >>> estimate.counts
    (500, 500, 500, 500)
    """
    sample = list(observations)
    for entry in sample:
        if not isinstance(entry, ChshObservation):
            raise TypeError(
                f"every element must be a ChshObservation, got "
                f"{type(entry).__name__}. The QBER arm and the CHSH arm are "
                f"different measurements and cannot be pooled."
            )
    level = _as_unit_interval(confidence, "confidence", strict=True)
    chosen = _as_chsh_method(method)

    totals = [0, 0, 0, 0]
    counts = [0, 0, 0, 0]
    for entry in sample:
        totals[entry.cell] += entry.correlation
        counts[entry.cell] += 1
    empty = [cell for cell, count in enumerate(counts) if count == 0]
    if empty:
        raise ValueError(
            f"CHSH cell(s) {empty} have no rounds, so their correlators are "
            f"undefined; the statistic needs all four of E(a0,b0), E(a0,b1), "
            f"E(a1,b0) and E(a1,b1). Treating an empty cell as 0 would report "
            f"a two-setting experiment as a Bell test. The sample of "
            f"{len(sample)} round(s) is either too small or was drawn with a "
            f"degenerate setting distribution."
        )
    correlators = tuple(
        total / count for total, count in zip(totals, counts, strict=True)
    )

    statistic = correlators[0] + correlators[1] + correlators[2] - correlators[3]
    if chosen == "normal":
        z = _two_sided_z(level)
        variance = sum(
            (1.0 - value * value) / count
            for value, count in zip(correlators, counts, strict=True)
        )
        half = z * math.sqrt(variance)
    else:
        alpha = 1.0 - level
        half = sum(
            math.sqrt(2.0 * math.log(8.0 / alpha) / count) for count in counts
        )
    return ChshEstimate(
        correlators=correlators,  # type: ignore[arg-type]
        counts=tuple(counts),  # type: ignore[arg-type]
        interval=Interval(
            low=max(-_CHSH_ALGEBRAIC_BOUND, statistic - half),
            high=min(_CHSH_ALGEBRAIC_BOUND, statistic + half),
            confidence=level,
            method=chosen,
        ),
    )


def _as_chsh_method(method: Any) -> ChshMethod:
    """Validate and lower-case the CHSH interval method."""
    if not isinstance(method, str):
        raise TypeError(
            f"method must be a string, got {type(method).__name__}"
        )
    chosen = method.strip().lower()
    if chosen not in _CHSH_METHODS:
        raise ValueError(
            f"unknown CHSH interval method {method!r}; expected 'normal' (the "
            f"plug-in variance form the sample sizes were derived from) or "
            f"'hoeffding' (the distribution-free bound, union-bounded over the "
            f"four cells)."
        )
    return chosen  # type: ignore[return-value]
