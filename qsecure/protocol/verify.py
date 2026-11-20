"""Phase C: scoring a declared key against one recipient's classical log.

Only matched positions (recipient basis equals declared basis) are scored:
``r_R = e_R / |M_R|``, accepted iff ``r_R <= threshold(R)`` (``s_a`` for Bob,
``s_v`` for Charlie). Replay, matched-count floor and provenance checks run
first; failing one is a no-verdict, never a rejection.

.. _pooled-floor:

Pooled floor
------------
Repudiation bounds are exponential in ``M = m_B + m_C``, so per-verifier floors
alone let a signer who reads both logs aim ``M`` at ``2 m_min`` and leave Bob
accepting a signature Charlie cannot score about half the time. The pooled
floor ``M_min`` and :attr:`AbortReason.COUNTERPART_BELOW_FLOOR` close that.

.. _split-coin-provenance:

Measured split-coin counts (``78/200`` and ``87/200`` at ``L = 360``, ``85/200``
at ``L = 600``) predate the two-stream change and are not reproducible from this
tree; re-measurement gives ``86/200``, matching the closed form.

.. _one-declaration:

One declaration
---------------
``m_B + m_C`` only means something when both counts were taken against the same
declaration, so a counterpart count naming another declaration, or none, is
refused.

.. _replay:

Replay
------
A verifier who passes his own :class:`ConsumedRecords` gets one verdict per
distribution round; a record from another round, or a round already decided, is
refused.

Re-aimed at ``M = M_min``, the coins leave Charlie under his own floor with
probability at most:

>>> from math import exp, sqrt
>>> from qsecure.protocol.params import DEFAULT_PARAMS
>>> m_min = minimum_matched_count(DEFAULT_PARAMS)
>>> M_min = minimum_pooled_matched_count(DEFAULT_PARAMS)
>>> f"{exp(-2 * (M_min / 2 - m_min) ** 2 / M_min):.2e}"   # the Chernoff form
'3.86e-04'
>>> f"{HONEST_ABORT_BUDGET ** (3 - 2 * sqrt(2)):.2e}"     # its L-independent limit
'4.95e-04'
"""

from __future__ import annotations

import enum
import hashlib
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from qsecure.protocol.params import (
    VERIFIERS,
    Party,
    ProtocolParams,
    _as_message_bit,
    _as_party,
)
from qsecure.protocol.records import RecipientRecord
from qsecure.protocol.signature import Signature

__all__ = [
    "HONEST_ABORT_BUDGET",
    "AbortReason",
    "ConsumedRecords",
    "MatchedSetTooSmall",
    "VerificationAbort",
    "VerificationResult",
    "enforced_repudiation_bound",
    "guaranteed_pooled_matched_count",
    "matched_positions",
    "minimum_matched_count",
    "minimum_pooled_matched_count",
    "mismatch_positions",
    "verify",
    "verify_all",
    "verify_or_abort",
]


HONEST_ABORT_BUDGET: Final[float] = 2.0**-64
"""float: The per-verifier probability an honest run is allowed to abort."""

_LOG_ABORT_BUDGET: Final[float] = -math.log(HONEST_ABORT_BUDGET)
"""float: ``ln(1/eps) = 64 ln 2``, precomputed for :func:`minimum_matched_count`."""


class AbortReason(enum.StrEnum):
    """Why verification reached no verdict."""

    SESSION_MISMATCH = "session-identifier-mismatch"
    RECORD_ALREADY_VERIFIED = "records-already-verified"
    EMPTY_MATCHED_SET = "empty-matched-set"
    BELOW_FLOOR = "matched-count-below-floor"
    COUNT_OF_UNRECORDED_PROVENANCE = "counterpart-count-of-unrecorded-provenance"
    COUNTS_FROM_TWO_DECLARATIONS = "counts-from-two-declarations"
    POOLED_BELOW_FLOOR = "pooled-matched-count-below-floor"
    COUNTERPART_BELOW_FLOOR = "counterpart-matched-count-below-floor"
    UNAUTHORISED_VERIFIER = "unauthorised-verifier"


#: Reasons that describe this verifier's own matched set.
_OWN_COUNT_REASONS: Final[frozenset[AbortReason]] = frozenset(
    {AbortReason.EMPTY_MATCHED_SET, AbortReason.BELOW_FLOOR}
)

#: Reasons that refuse before any counting rule; they make no claim about counts.
_ROUND_REASONS: Final[frozenset[AbortReason]] = frozenset(
    {AbortReason.SESSION_MISMATCH, AbortReason.RECORD_ALREADY_VERIFIED}
)

#: The one reason that refuses on who is asking rather than on any evidence.
_IDENTITY_REASONS: Final[frozenset[AbortReason]] = frozenset(
    {AbortReason.UNAUTHORISED_VERIFIER}
)

#: Reasons about both verifiers together; written out so a new member fails a test.
_PAIR_REASONS: Final[frozenset[AbortReason]] = frozenset(
    {
        AbortReason.COUNT_OF_UNRECORDED_PROVENANCE,
        AbortReason.COUNTS_FROM_TWO_DECLARATIONS,
        AbortReason.POOLED_BELOW_FLOOR,
        AbortReason.COUNTERPART_BELOW_FLOOR,
    }
)

#: Pair reasons that refuse to apply the pooled rule; no floor to quote.
_UNPOOLABLE_REASONS: Final[frozenset[AbortReason]] = frozenset(
    {
        AbortReason.COUNT_OF_UNRECORDED_PROVENANCE,
        AbortReason.COUNTS_FROM_TWO_DECLARATIONS,
    }
)

#: Bytes of blake2b output kept; it guards against wiring mistakes, not search.
_DECLARATION_DIGEST_BYTES: Final[int] = 16


def _declaration_digest(signature: Signature) -> str:
    """Return a stable fingerprint of one declaration.

    >>> from qsecure.protocol.keys import KeyElement, PrivateKey
    >>> from qsecure.protocol.signature import Signature
    >>> from qsecure.protocol.verify import _declaration_digest
    >>> key = PrivateKey(0, (KeyElement("X", 1), KeyElement("Z", -1)))
    >>> flipped = PrivateKey(0, (KeyElement("X", 1), KeyElement("Z", 1)))
    >>> first = _declaration_digest(Signature(0, key))
    >>> len(first), first == _declaration_digest(Signature(0, key))
    (32, True)
    >>> first == _declaration_digest(Signature(0, flipped))
    False
    """
    if not isinstance(signature, Signature):
        raise TypeError(
            f"signature must be a Signature, got {type(signature).__name__}; "
            f"a declaration digest fingerprints a declaration."
        )
    # Injective: fixed length, one character per basis, one byte per eigenvalue.
    header = f"{signature.message_bit}|{len(signature)}|"
    payload = (header + "".join(signature.bases)).encode("utf-8") + bytes(
        signature.bits
    )
    return hashlib.blake2b(
        payload, digest_size=_DECLARATION_DIGEST_BYTES
    ).hexdigest()


class _BoundMatchedCount(int):
    """A matched count that remembers which declaration produced it.

    >>> from qsecure.protocol.verify import _BoundMatchedCount
    >>> bound = _BoundMatchedCount(212, "0f1e")
    >>> bound, bound + 1, bound == 212
    (212, 213, True)
    >>> bound.declaration_digest
    '0f1e'
    """

    declaration_digest: str | None

    def __new__(
        cls, value: int, declaration_digest: str | None = None
    ) -> _BoundMatchedCount:
        """Build the count and attach its declaration."""
        if declaration_digest is not None and not isinstance(
            declaration_digest, str
        ):
            raise TypeError(
                f"declaration_digest must be a string or None, got "
                f"{type(declaration_digest).__name__}"
            )
        count = super().__new__(cls, value)
        count.declaration_digest = declaration_digest
        return count


def _counterpart_binding(value: Any) -> str | None:
    """Return the declaration a reported count was computed against, if it
    says.
    """
    digest = getattr(value, "declaration_digest", None)
    return digest if isinstance(digest, str) and digest else None


def _claims_provenance(value: Any) -> bool:
    """Return whether a reported count arrived through the count exchange."""
    return hasattr(value, "declaration_digest")


class ConsumedRecords:
    """One verifier's ledger of distribution rounds he has already decided.

    >>> from qsecure.protocol.records import RecipientRecord
    >>> from qsecure.protocol.verify import ConsumedRecords
    >>> ledger = ConsumedRecords("Bob")
    >>> record = RecipientRecord.from_measurements(
    ...     "Bob", 0, ["X"], [1], session_id="9f3c"
    ... )
    >>> ledger.is_spent(record)
    False
    >>> ledger.spend(record)
    >>> ledger.is_spent(record), len(ledger)
    (True, 1)

    A log that names no round is not tracked, and nothing about it is invented:

    >>> unbound = RecipientRecord.from_measurements("Bob", 0, ["X"], [1])
    >>> ledger.spend(unbound)
    >>> ledger.is_spent(unbound), len(ledger)
    (False, 1)

    And one verifier's ledger will not hold the other's evidence:

    >>> ledger.spend(RecipientRecord.from_measurements(
    ...     "Charlie", 0, ["X"], [1], session_id="9f3c"
    ... ))
    Traceback (most recent call last):
        ...
    ValueError: this ledger belongs to Bob and was handed Charlie's record, ...
    """

    __slots__ = ("_party", "_spent")

    def __init__(self, party: Party | str) -> None:
        resolved = _as_party(party)
        if resolved is Party.ALICE:
            raise ValueError(
                "Alice keeps no ledger of consumed records: she is the signer, "
                "holds no measurement record and reaches no verdict, so there "
                "is no round for her to spend. A ConsumedRecords belongs to "
                "Party.BOB or Party.CHARLIE."
            )
        self._party: Final[Party] = resolved
        self._spent: set[tuple[str, int]] = set()

    @property
    def party(self) -> Party:
        """Party: The verifier this ledger belongs to."""
        return self._party

    def __len__(self) -> int:
        """int: How many rounds this verifier has decided."""
        return len(self._spent)

    def __repr__(self) -> str:
        """Return a debugging representation naming the party and the count."""
        return (
            f"ConsumedRecords({self._party.value!r}, "
            f"{len(self._spent)} round(s) spent)"
        )

    def _key(self, record: RecipientRecord) -> tuple[str, int] | None:
        """Return the ledger key for ``record``, or ``None`` if untracked."""
        if not isinstance(record, RecipientRecord):
            raise TypeError(
                f"record must be a RecipientRecord, got "
                f"{type(record).__name__}; a ledger spends rounds of evidence, "
                f"not declarations."
            )
        if record.party is not self._party:
            raise ValueError(
                f"this ledger belongs to {self._party.value} and was handed "
                f"{record.party.value}'s record, which it must not see. A "
                f"consumed-records ledger is per verifier: the two verifiers "
                f"are adversaries to each other in half of this package's "
                f"attacks, so one verifier's history must not reach the "
                f"other's decision. Give {record.party.value} his own "
                f"ConsumedRecords."
            )
        if record.session_id is None:
            return None
        return (record.session_id, record.message_bit)

    def is_spent(self, record: RecipientRecord) -> bool:
        """Return whether this verifier has already decided ``record``'s round."""
        key = self._key(record)
        return key is not None and key in self._spent

    def spend(self, record: RecipientRecord) -> None:
        """Record that this verifier has reached a verdict on ``record``'s
        round.
        """
        key = self._key(record)
        if key is not None:
            self._spent.add(key)

    def spent_rounds(self) -> frozenset[tuple[str, int]]:
        """Return the ``(session_id, message_bit)`` pairs already decided."""
        return frozenset(self._spent)


def minimum_matched_count(params: ProtocolParams) -> int:
    """Return ``m_min``, the smallest matched set a verifier will score.

    >>> from qsecure.protocol.params import (
    ...     DEFAULT_PARAMS, DEMO_PARAMS, ProtocolParams
    ... )
    >>> from qsecure.protocol.verify import minimum_matched_count
    >>> minimum_matched_count(DEFAULT_PARAMS)      # mu = 38400
    36555
    >>> minimum_matched_count(ProtocolParams(key_length=600))   # mu = 200
    67
    >>> minimum_matched_count(DEMO_PARAMS)  # L = 192: no power to spend
    1
    """
    return _chernoff_floor(_as_protocol_params(params).expected_matched)


def minimum_pooled_matched_count(params: ProtocolParams) -> int:
    """Return ``M_min``, the smallest *pooled* matched set a run will score.

    >>> from qsecure.protocol.params import (
    ...     DEFAULT_PARAMS, DEMO_PARAMS, ProtocolParams
    ... )
    >>> from qsecure.protocol.verify import (
    ...     minimum_matched_count, minimum_pooled_matched_count
    ... )
    >>> minimum_pooled_matched_count(DEFAULT_PARAMS)     # mu_M = 76800
    74190
    >>> 2 * minimum_matched_count(DEFAULT_PARAMS)        # what the attack aims at
    73110
    >>> minimum_pooled_matched_count(ProtocolParams(key_length=600))
    212
    >>> minimum_pooled_matched_count(DEMO_PARAMS)   # L = 192, mu_M = 128
    22
    """
    checked = _as_protocol_params(params)
    return _chernoff_floor(2.0 * checked.expected_matched)


def _as_protocol_params(params: Any) -> ProtocolParams:
    """Return ``params`` unchanged, refusing anything that is not a set of
    them.
    """
    if not isinstance(params, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(params).__name__}; "
            f"the floor is a function of the key length and the alphabet size, "
            f"so it can only be read off a parameter set."
        )
    return params


def _chernoff_floor(mean_matched: float) -> int:
    """Return the largest floor whose honest lower tail stays inside the
    budget.
    """
    # d < 1 exactly when 2 ln(1/eps) < mu; outside it, fall back to the
    # "a rate needs a denominator" rule.
    slack = 2.0 * _LOG_ABORT_BUDGET
    if slack >= mean_matched:
        return 1
    deviation = math.sqrt(slack / mean_matched)
    return max(1, math.ceil((1.0 - deviation) * mean_matched))


def enforced_repudiation_bound(params: ProtocolParams) -> float:
    """The unconditional repudiation bound *this module's* floor actually buys.

    >>> from qsecure.protocol.params import DEFAULT_PARAMS, DEMO_PARAMS
    >>> from qsecure.protocol.verify import enforced_repudiation_bound
    >>> f"{enforced_repudiation_bound(DEFAULT_PARAMS):.2e}"
    '1.41e-09'
    >>> round(enforced_repudiation_bound(DEMO_PARAMS), 4)  # no claim at L=192
    0.994
    """
    # Local import: verification's rule does not depend on its analysis.
    from qsecure.protocol.analysis import repudiation_bound_with_abort

    return repudiation_bound_with_abort(
        params, minimum_matched_records=guaranteed_pooled_matched_count(params)
    )


def guaranteed_pooled_matched_count(params: ProtocolParams) -> int:
    """Return the floor on ``M = m_B + m_C`` that the shipped rules guarantee.

    >>> from qsecure.protocol.params import DEFAULT_PARAMS, ProtocolParams
    >>> from qsecure.protocol.verify import guaranteed_pooled_matched_count
    >>> guaranteed_pooled_matched_count(DEFAULT_PARAMS)
    74190
    >>> guaranteed_pooled_matched_count(ProtocolParams(key_length=24))
    2
    """
    checked = _as_protocol_params(params)
    return max(
        2 * minimum_matched_count(checked),
        minimum_pooled_matched_count(checked),
    )


@dataclass(frozen=True)
class VerificationAbort:
    """A recorded **no-verdict**: verification refused to score, and why.

    >>> from qsecure.protocol.verify import AbortReason, VerificationAbort
    >>> abort = VerificationAbort(
    ...     party="Bob",
    ...     reason=AbortReason.EMPTY_MATCHED_SET,
    ...     matched_count=0,
    ...     minimum_matched=67,
    ...     expected_matched=200.0,
    ...     key_length=600,
    ...     message_bit=0,
    ... )
    >>> abort.shortfall, abort.accepted_is_undefined
    (67, True)
    """

    party: Party
    reason: AbortReason
    matched_count: int
    minimum_matched: int
    expected_matched: float
    key_length: int
    message_bit: int
    counterpart_matched: int | None = None
    minimum_pooled: int | None = None

    def __post_init__(self) -> None:
        """Coerce the fields and enforce that this really is a no-verdict."""
        resolved = _as_party(self.party)
        if resolved is Party.ALICE:
            raise ValueError(
                "Alice reaches no verdict to abort: she is the signer, holds "
                "no measurement record and has no acceptance threshold. A "
                "VerificationAbort belongs to Party.BOB or Party.CHARLIE."
            )
        object.__setattr__(self, "party", resolved)
        object.__setattr__(
            self, "message_bit", _as_message_bit(self.message_bit)
        )
        object.__setattr__(self, "reason", _as_abort_reason(self.reason))
        object.__setattr__(
            self, "key_length", _as_count(self.key_length, "key_length")
        )
        object.__setattr__(
            self, "matched_count", _as_count(self.matched_count, "matched_count")
        )
        object.__setattr__(
            self,
            "minimum_matched",
            _as_count(self.minimum_matched, "minimum_matched"),
        )
        object.__setattr__(
            self,
            "counterpart_matched",
            _as_optional_count(
                self.counterpart_matched, "counterpart_matched"
            ),
        )
        object.__setattr__(
            self,
            "minimum_pooled",
            _as_optional_count(self.minimum_pooled, "minimum_pooled"),
        )
        if isinstance(self.expected_matched, bool) or not isinstance(
            self.expected_matched, numbers.Real
        ):
            raise TypeError(
                f"expected_matched must be a real number, got "
                f"{type(self.expected_matched).__name__}; it is L/|B|."
            )
        object.__setattr__(
            self, "expected_matched", float(self.expected_matched)
        )

        if self.key_length < 1:
            raise ValueError(
                f"key_length must be at least 1, got {self.key_length}; a run "
                f"with no key positions has nothing to verify."
            )
        if self.minimum_matched < 1:
            raise ValueError(
                f"minimum_matched must be at least 1, got "
                f"{self.minimum_matched}. Even where the Chernoff tail is "
                f"vacuous the floor is 1, because a rate needs a denominator; "
                f"see minimum_matched_count."
            )
        for name, value in (
            ("matched_count", self.matched_count),
            ("minimum_matched", self.minimum_matched),
            ("counterpart_matched", self.counterpart_matched),
        ):
            if value is not None and value > self.key_length:
                raise ValueError(
                    f"{name} ({value}) cannot exceed key_length "
                    f"({self.key_length}); the matched set is a subset of the "
                    f"key positions."
                )
        if (
            self.minimum_pooled is not None
            and self.minimum_pooled > 2 * self.key_length
        ):
            raise ValueError(
                f"minimum_pooled ({self.minimum_pooled}) cannot exceed "
                f"2 * key_length ({2 * self.key_length}); the pooled count is "
                f"over the two verifiers' matched sets together, so its "
                f"largest possible value is 2 L."
            )
        if not math.isfinite(self.expected_matched) or not (
            0.0 < self.expected_matched <= self.key_length
        ):
            raise ValueError(
                f"expected_matched must be finite and lie in "
                f"(0, {self.key_length}], got {self.expected_matched!r}. It is "
                f"L/|B|, the mean of Binomial(L, 1/|B|)."
            )
        self._check_reason_against_counts()

    def _check_reason_against_counts(self) -> None:
        """Enforce that the label and the numbers are one observation."""
        if self.reason in _ROUND_REASONS or self.reason in _IDENTITY_REASONS:
            # Nothing to cross-check: these refuse before any counting rule.
            return
        own_short = self.matched_count < self.minimum_matched
        if self.reason in _OWN_COUNT_REASONS:
            if not own_short:
                raise ValueError(
                    f"matched_count ({self.matched_count}) is at or above "
                    f"minimum_matched ({self.minimum_matched}), so this "
                    f"verifier met his own floor. Record a VerificationResult, "
                    f"or -- if the pair's evidence is what fell short -- one of "
                    f"the pooled reasons "
                    f"({AbortReason.POOLED_BELOW_FLOOR.value!r}, "
                    f"{AbortReason.COUNTERPART_BELOW_FLOOR.value!r}). A "
                    f"no-verdict outcome that an evaluation table could not "
                    f"distinguish from a decision would defeat the point of "
                    f"having two types."
                )
            expected_reason = (
                AbortReason.EMPTY_MATCHED_SET
                if self.matched_count == 0
                else AbortReason.BELOW_FLOOR
            )
            if self.reason is not expected_reason:
                raise ValueError(
                    f"reason={self.reason.value!r} contradicts matched_count="
                    f"{self.matched_count}: "
                    f"{AbortReason.EMPTY_MATCHED_SET.value!r} means exactly "
                    f"|M_R| == 0 and {AbortReason.BELOW_FLOOR.value!r} exactly "
                    f"0 < |M_R| < m_min. The label and the numbers have to be "
                    f"the same observation."
                )
            return

        # A pair reason: both exchanged numbers are mandatory, and this verifier
        # must have cleared his own floor to get here.
        if self.counterpart_matched is None or self.minimum_pooled is None:
            raise ValueError(
                f"reason={self.reason.value!r} is a statement about the two "
                f"verifiers together, so it needs both counterpart_matched "
                f"(got {self.counterpart_matched!r}) and minimum_pooled (got "
                f"{self.minimum_pooled!r}). Those numbers arrive over the "
                f"recipients' count exchange -- see qsecure.protocol.tally -- "
                f"and an abort that quoted neither would be unreadable and "
                f"uncheckable."
            )
        if own_short:
            raise ValueError(
                f"reason={self.reason.value!r} says the pair's evidence fell "
                f"short, but this verifier's own matched_count "
                f"({self.matched_count}) is already below his floor "
                f"({self.minimum_matched}). verify() tests the local floor "
                f"first, so that run aborts as "
                f"{AbortReason.EMPTY_MATCHED_SET.value!r} or "
                f"{AbortReason.BELOW_FLOOR.value!r}; labelling it pooled would "
                f"blame the pair for a local failure."
            )
        if self.reason in _UNPOOLABLE_REASONS:
            # Not one declaration, so no relation between the counts is claimed.
            return
        pooled = self.matched_count + self.counterpart_matched
        if self.reason is AbortReason.POOLED_BELOW_FLOOR:
            if pooled >= self.minimum_pooled:
                raise ValueError(
                    f"reason={self.reason.value!r} but m_B + m_C = "
                    f"{self.matched_count} + {self.counterpart_matched} = "
                    f"{pooled}, which meets the pooled floor "
                    f"({self.minimum_pooled}). The label and the numbers have "
                    f"to be the same observation."
                )
            return
        # COUNTERPART_BELOW_FLOOR: pooled total fine, the other verifier not.
        if pooled < self.minimum_pooled:
            raise ValueError(
                f"reason={self.reason.value!r} but m_B + m_C = {pooled} is "
                f"below the pooled floor ({self.minimum_pooled}), which "
                f"verify() tests first. That run aborts as "
                f"{AbortReason.POOLED_BELOW_FLOOR.value!r}."
            )
        if self.counterpart_matched >= self.minimum_matched:
            raise ValueError(
                f"reason={self.reason.value!r} but the counterpart reported "
                f"{self.counterpart_matched} matched records, which meets the "
                f"per-verifier floor ({self.minimum_matched}). Both verifiers "
                f"cleared every floor, so both reached a verdict."
            )

    # -- derived views ------------------------------------------------------ #

    @property
    def pooled_count(self) -> int | None:
        """int or None: ``M = m_B + m_C``, or ``None`` with no exchange."""
        if self.counterpart_matched is None:
            return None
        return self.matched_count + self.counterpart_matched

    @property
    def shortfall(self) -> int:
        """int: How far under *the floor that was missed* this run fell."""
        if (
            self.reason in _UNPOOLABLE_REASONS
            or self.reason in _ROUND_REASONS
            or self.reason in _IDENTITY_REASONS
        ):
            return 0
        if self.reason is AbortReason.POOLED_BELOW_FLOOR:
            assert self.minimum_pooled is not None  # enforced in __post_init__
            pooled = self.pooled_count
            assert pooled is not None
            return self.minimum_pooled - pooled
        if self.reason is AbortReason.COUNTERPART_BELOW_FLOOR:
            assert self.counterpart_matched is not None
            return self.minimum_matched - self.counterpart_matched
        return self.minimum_matched - self.matched_count

    @property
    def is_pooled(self) -> bool:
        """bool: ``True`` iff the refusal is a statement about the pair."""
        return self.reason in _PAIR_REASONS

    @property
    def accepted_is_undefined(self) -> bool:
        """bool: Always ``True``, and it exists to be read."""
        return True

    def summary(self) -> str:
        """Return a one-line human-readable account of the refusal.

        >>> from qsecure.protocol.verify import AbortReason, VerificationAbort
        >>> VerificationAbort(
        ...     "Charlie", AbortReason.BELOW_FLOOR, 13, 67, 200.0, 600, 1
        ... ).summary()[:19]
        'Charlie NO VERDICT '
        >>> VerificationAbort(
        ...     "Bob", AbortReason.POOLED_BELOW_FLOOR, 80, 67, 200.0, 600, 0,
        ...     counterpart_matched=60, minimum_pooled=212,
        ... ).summary()
        'Bob NO VERDICT bit 0: 80/600 positions matched, floor 67, honest mean \
200.0; pooled M = 80 + 60 = 140, pooled floor 212 \
(pooled-matched-count-below-floor). Not a rejection.'
        """
        if self.reason is AbortReason.UNAUTHORISED_VERIFIER:
            return (
                f"{self.party.value} NO VERDICT bit {self.message_bit}: not in "
                f"the authorised recipient set for this round, so no count was "
                f"taken ({self.reason.value}). Not a rejection."
            )
        pooled = ""
        if self.counterpart_matched is not None:
            pooled = (
                f"; pooled M = {self.matched_count} + "
                f"{self.counterpart_matched} = {self.pooled_count}, pooled "
                f"floor {self.minimum_pooled}"
            )
        return (
            f"{self.party.value} NO VERDICT bit {self.message_bit}: "
            f"{self.matched_count}/{self.key_length} positions matched, floor "
            f"{self.minimum_matched}, honest mean {self.expected_matched:.1f}"
            f"{pooled} ({self.reason.value}). Not a rejection."
        )

    # -- serialisation ------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the refusal.

        >>> import json
        >>> from qsecure.protocol.verify import AbortReason, VerificationAbort
        >>> blob = VerificationAbort(
        ...     "Bob", AbortReason.EMPTY_MATCHED_SET, 0, 1, 3.0, 9, 0
        ... ).to_dict()
        >>> json.loads(json.dumps(blob))["reason"]
        'empty-matched-set'
        """
        return {
            "party": self.party,
            "reason": self.reason,
            "matched_count": self.matched_count,
            "minimum_matched": self.minimum_matched,
            "expected_matched": self.expected_matched,
            "key_length": self.key_length,
            "message_bit": self.message_bit,
            "counterpart_matched": self.counterpart_matched,
            "minimum_pooled": self.minimum_pooled,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VerificationAbort:
        """Rebuild a refusal from :meth:`to_dict` output."""
        return cls(
            party=data["party"],
            reason=data["reason"],
            matched_count=data["matched_count"],
            minimum_matched=data["minimum_matched"],
            expected_matched=data["expected_matched"],
            key_length=data["key_length"],
            message_bit=data["message_bit"],
            counterpart_matched=data.get("counterpart_matched"),
            minimum_pooled=data.get("minimum_pooled"),
        )


class MatchedSetTooSmall(ValueError):
    """Verification reached no verdict: the matched set was too small to score."""

    def __init__(self, abort: VerificationAbort) -> None:
        if not isinstance(abort, VerificationAbort):
            raise TypeError(
                f"abort must be a VerificationAbort, got "
                f"{type(abort).__name__}; the exception's message is derived "
                f"from it."
            )
        self.abort = abort
        super().__init__(_abort_message(abort))


def _as_abort_reason(reason: AbortReason | str) -> AbortReason:
    """Coerce a label to an :class:`AbortReason`."""
    if isinstance(reason, AbortReason):
        return reason
    if not isinstance(reason, str):
        raise TypeError(
            f"reason must be an AbortReason or its string value, got "
            f"{type(reason).__name__}"
        )
    try:
        return AbortReason(reason)
    except ValueError:
        known = ", ".join(member.value for member in AbortReason)
        raise ValueError(
            f"reason must be one of {known}, got {reason!r}"
        ) from None


def _abort_message(abort: VerificationAbort) -> str:
    """Build the human-facing message for a :class:`MatchedSetTooSmall`."""
    party = abort.party.value
    # Split so the one refusal not about a starved matched set keeps the framing.
    not_a_signature_failure = (
        "This is not a signature failure and is not reported as one"
    )
    recorded_not_lost = (
        "QDSSession records it as a no-verdict outcome (VerificationAbort) "
        "rather than losing the run."
    )
    common = (
        f"{not_a_signature_failure}: accepting would accept any declaration "
        f"whatsoever on zero evidence, and rejecting would be recorded as a "
        f"forgery detection when no evidence of forgery exists. "
        f"{recorded_not_lost}"
    )
    if abort.reason is AbortReason.SESSION_MISMATCH:
        return (
            f"the declaration does not name the distribution round "
            f"{party}'s log was made in, so there is nothing here to score. "
            f"The key states are one-time: a declaration and a record belong to "
            f"exactly one round, and a rate computed across two of them is not "
            f"a weak measurement of anything -- it is the chance noise of "
            f"scoring against an unrelated key, near (1 - 1/|B|)/2, which is "
            f"why this was refused instead of being reported as the rejection "
            f"it would usually have looked like. "
            f"{not_a_signature_failure}: the verifier has learned nothing about "
            f"the signature, only that he was handed the wrong pair. "
            f"{recorded_not_lost} The usual causes are a captured signature "
            f"re-presented after its round closed and a harness that paired one "
            f"run's declaration with another run's logs -- see "
            f"qsecure.protocol.verify on the replay defence -- so what to "
            f"investigate is which round each half came from."
        )
    if abort.reason is AbortReason.RECORD_ALREADY_VERIFIED:
        return (
            f"{party} has already reached a verdict on this distribution round "
            f"and will not reach a second one. One round yields one verdict per "
            f"verifier: the evidence is spent, and re-deciding it would let a "
            f"replayed declaration collect as many acceptances as it was "
            f"presented times. "
            f"{not_a_signature_failure}: nothing new was learned about the "
            f"signature, and the verdict already reached still stands -- read "
            f"it there rather than asking again. {recorded_not_lost} "
            f"If this is a harness verifying twice on purpose, give that "
            f"verification its own ConsumedRecords, or none; if it is not, the "
            f"second presentation is the replay this ledger exists for."
        )
    if abort.reason is AbortReason.POOLED_BELOW_FLOOR:
        return (
            f"the two verifiers together hold {abort.pooled_count} matched "
            f"records ({party} {abort.matched_count}, the other verifier "
            f"{abort.counterpart_matched}), below the pooled matched-count "
            f"floor of {abort.minimum_pooled}, so neither scores. {common} "
            f"{party} cleared his own floor of {abort.minimum_matched}: this "
            f"is a statement about the pair, and it is the check the "
            f"recipients' count exchange exists to make. Under honest "
            f"operation M = m_B + m_C is Binomial(2L, 1/|B|) with mean "
            f"{2 * abort.expected_matched:.1f}, and a Chernoff lower tail puts "
            f"the chance of an honest run falling below the pooled floor at "
            f"{HONEST_ABORT_BUDGET:.3g}. A declaration whose *total* matched "
            f"count is aimed low is what this refuses -- see "
            f"qsecure.protocol.verify on the split-coin route -- so investigate "
            f"the declaration, not the verifiers."
        )
    if abort.reason is AbortReason.COUNTERPART_BELOW_FLOOR:
        return (
            f"the other verifier reported only {abort.counterpart_matched} "
            f"matched records, below the per-verifier floor of "
            f"{abort.minimum_matched}, so {party} reaches no verdict either -- "
            f"even though his own matched set holds {abort.matched_count} of "
            f"the {abort.key_length} key positions and would have been scored. "
            f"{common} The floor's consequence is joint on purpose: a "
            f"signature the other verifier cannot score is not one this "
            f"verifier can be told he should have transferred, and accepting "
            f"here is exactly the asymmetric outcome a signer who splits the "
            f"evidence base is aiming for. The pair's total was "
            f"{abort.pooled_count} against a pooled floor of "
            f"{abort.minimum_pooled}, so the shortfall is in the *split*, not "
            f"in the total."
        )
    if abort.reason is AbortReason.COUNT_OF_UNRECORDED_PROVENANCE:
        return (
            f"the count {party} was given names no declaration, so it cannot "
            f"be shown to have been computed against the one he is scoring: "
            f"the two numbers ({party} {abort.matched_count}, the other "
            f"verifier {abort.counterpart_matched}) may or may not be a pooled "
            f"count, and the pooled floor of {abort.minimum_pooled} is not "
            f"applied to a sum that might belong to no run. "
            f"{not_a_signature_failure}: what failed is the bookkeeping, the "
            f"verifier has learned nothing about the signature either way, and "
            f"a rejection here would be recorded as a forgery detection when "
            f"no evidence of forgery exists. {recorded_not_lost} "
            f"An unprovenanced count is refused rather than taken at its word "
            f"because Phase C' is the recipients' own step and a recipient is "
            f"the adversary this check exists for: a check that can be "
            f"switched off by declining to answer is not one. The usual cause "
            f"is a count exchange that does not fill in the declaration it "
            f"counted -- see qsecure.protocol.tally on the binding being "
            f"mandatory -- so what to investigate is the exchange, not the "
            f"verifiers."
        )
    if abort.reason is AbortReason.COUNTS_FROM_TWO_DECLARATIONS:
        return (
            f"the count {party} was given was computed against a different "
            f"declaration from the one he is scoring, so the two numbers "
            f"({party} {abort.matched_count}, the other verifier "
            f"{abort.counterpart_matched}) are not a pooled count and the "
            f"pooled floor of {abort.minimum_pooled} has nothing to be applied "
            f"to. Their sum, {abort.pooled_count}, is no run's M. "
            f"{not_a_signature_failure}: what failed is the bookkeeping, the "
            f"verifier has learned nothing about the signature either way, and "
            f"a rejection here would be recorded as a forgery detection when "
            f"no evidence of forgery exists. {recorded_not_lost} "
            f"m_B + m_C is conserved across one fixed pair of records against "
            f"one fixed declaration and means nothing across two, so the run "
            f"is refused rather than scored on a mixed total -- see "
            f"qsecure.protocol.verify on counting against one declaration. The "
            f"usual cause is a forwarding hop that altered the declaration "
            f"after the recipients had already exchanged counts: what to "
            f"investigate is the hop, not the verifiers."
        )
    if abort.reason is AbortReason.UNAUTHORISED_VERIFIER:
        return (
            f"{party} is not in the authorised recipient set for this round, "
            f"so verification refused before taking any count: none of the "
            f"numbers on this refusal is a measurement. "
            f"{not_a_signature_failure}: nothing was scored, and a rejection "
            f"here would be recorded as a forgery detection when no evidence "
            f"of forgery exists. The outcome goes back to whoever named the "
            f"authorised set -- verify_or_abort returns it as a "
            f"VerificationAbort, verify raises it as a MatchedSetTooSmall -- "
            f"and not to QDSSession, which names no such set and so cannot "
            f"produce this reason at all. "
            f"The check tests the identity the record declares, so it refuses "
            f"a party who does not claim to be an authorised recipient and is "
            f"powerless against one presenting a leaked record that still "
            f"names its owner -- see qsecure.protocol.verify on what an "
            f"authorised set can and cannot decide."
        )
    if abort.reason is AbortReason.EMPTY_MATCHED_SET:
        return (
            f"{party}'s matched set is empty: his measurement basis differed "
            f"from the declared one at all {abort.key_length} positions, so the "
            f"mismatch rate is 0/0 and there is no verdict to reach. {common} "
            f"Honest operation puts |M| near {abort.expected_matched:.1f}, and "
            f"the matched-count floor for this parameter set is "
            f"{abort.minimum_matched}. In practice this means the key is far "
            f"too short, the record was paired with the wrong key, or the "
            f"declaration was built to avoid both recipients' logged bases -- "
            f"see qsecure.protocol.verify on who can reach it."
        )
    return (
        f"{party}'s matched set holds only {abort.matched_count} of the "
        f"{abort.key_length} key positions, below the matched-count floor of "
        f"{abort.minimum_matched}, so verification aborts instead of scoring. "
        f"{common} Under honest operation |M| is Binomial(L, 1/|B|) with mean "
        f"{abort.expected_matched:.1f}, and a Chernoff lower tail puts the "
        f"chance of an honest verifier falling below the floor at "
        f"{HONEST_ABORT_BUDGET:.3g}. Every bound the scheme quotes is "
        f"exp(-Theta(|M|)), so a rate computed from {abort.matched_count} "
        f"positions would carry a confidence it does not have -- which is "
        f"precisely what a signer who starves the matched set is buying. "
        f"Investigate the declaration and the distribution; raising L raises "
        f"the floor with it, and lowering the floor is not a fix."
    )


@dataclass(frozen=True)
class VerificationResult:
    """One verifier's decision, together with the numbers it was reached from.

    >>> from qsecure.protocol.verify import VerificationResult
    >>> result = VerificationResult(
    ...     party="Bob",
    ...     accepted=True,
    ...     matched_count=32,
    ...     mismatches=1,
    ...     rate=1 / 32,
    ...     threshold=1 / 32,
    ...     key_length=96,
    ...     message_bit=0,
    ... )
    >>> result.unmatched_count, result.margin
    (64, 0.0)
    """

    party: Party
    accepted: bool
    matched_count: int
    mismatches: int
    rate: float
    threshold: float
    key_length: int
    message_bit: int

    def __post_init__(self) -> None:
        """Coerce the fields and enforce the internal consistency of the verdict."""
        resolved = _as_party(self.party)
        if resolved is Party.ALICE:
            raise ValueError(
                "Alice reaches no verdict: she is the signer, holds no "
                "measurement record and has no acceptance threshold. A "
                "VerificationResult belongs to Party.BOB or Party.CHARLIE."
            )
        object.__setattr__(self, "party", resolved)
        object.__setattr__(
            self, "message_bit", _as_message_bit(self.message_bit)
        )

        if not isinstance(self.accepted, bool):
            raise TypeError(
                f"accepted must be a bool, got {type(self.accepted).__name__}; "
                f"it is a verdict, not a count"
            )
        object.__setattr__(
            self, "key_length", _as_count(self.key_length, "key_length")
        )
        object.__setattr__(
            self, "matched_count", _as_count(self.matched_count, "matched_count")
        )
        object.__setattr__(
            self, "mismatches", _as_count(self.mismatches, "mismatches")
        )
        object.__setattr__(self, "rate", _as_rate(self.rate, "rate"))
        object.__setattr__(
            self, "threshold", _as_rate(self.threshold, "threshold")
        )

        if self.matched_count < 1:
            raise ValueError(
                "matched_count must be at least 1: with an empty matched set "
                "the mismatch rate is 0/0 and no decision exists. verify() "
                "raises rather than manufacturing one; see its documentation "
                "for why neither accepting nor rejecting is honest there."
            )
        if self.matched_count > self.key_length:
            raise ValueError(
                f"matched_count ({self.matched_count}) cannot exceed key_length "
                f"({self.key_length}); the matched set is a subset of the key "
                f"positions."
            )
        if self.mismatches > self.matched_count:
            raise ValueError(
                f"mismatches ({self.mismatches}) cannot exceed matched_count "
                f"({self.matched_count}); disagreements are counted only within "
                f"the matched set, never over the whole key. Counting the "
                f"unmatched positions too is the classic bug in this protocol "
                f"family -- see qsecure.protocol.verify."
            )

        expected_rate = self.mismatches / self.matched_count
        if not math.isclose(self.rate, expected_rate, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"rate must be mismatches / matched_count = "
                f"{self.mismatches} / {self.matched_count} = {expected_rate!r}, "
                f"got {self.rate!r}. The reported rate and the reported counts "
                f"have to be the same measurement."
            )
        if self.accepted != (self.rate <= self.threshold):
            raise ValueError(
                f"accepted={self.accepted!r} contradicts rate={self.rate!r} and "
                f"threshold={self.threshold!r}: the rule is accept iff "
                f"rate <= threshold. A result whose verdict and numbers "
                f"disagree would show one thing on a dashboard and mean another."
            )

    # -- derived views ------------------------------------------------------ #

    @property
    def unmatched_count(self) -> int:
        """int: Positions discarded because the bases differed, ``L - |M_R|``."""
        return self.key_length - self.matched_count

    @property
    def margin(self) -> float:
        """float: ``threshold - rate``."""
        return self.threshold - self.rate

    @property
    def agreements(self) -> int:
        """int: Matched positions whose outcome agreed with the declaration."""
        return self.matched_count - self.mismatches

    def summary(self) -> str:
        """Return a one-line human-readable account of the decision.

        >>> from qsecure.protocol.verify import VerificationResult
        >>> VerificationResult(
        ...     "Charlie", False, 30, 12, 0.4, 1 / 6, 90, 1
        ... ).summary()[:16]
        'Charlie REJECTED'
        """
        verdict = "ACCEPTED" if self.accepted else "REJECTED"
        relation = "<=" if self.accepted else ">"
        return (
            f"{self.party.value} {verdict} bit {self.message_bit}: "
            f"{self.mismatches}/{self.matched_count} mismatches on matched "
            f"positions ({self.unmatched_count} discarded), rate "
            f"{self.rate:.5f} {relation} {self.threshold:.5f}"
        )

    # -- serialisation ------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the decision.

        >>> import json
        >>> from qsecure.protocol.verify import VerificationResult
        >>> result = VerificationResult("Bob", True, 4, 0, 0.0, 0.5, 12, 0)
        >>> json.loads(json.dumps(result.to_dict()))["party"]
        'Bob'
        """
        return {
            "party": self.party,
            "accepted": self.accepted,
            "matched_count": self.matched_count,
            "mismatches": self.mismatches,
            "rate": self.rate,
            "threshold": self.threshold,
            "key_length": self.key_length,
            "message_bit": self.message_bit,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VerificationResult:
        """Rebuild a decision from :meth:`to_dict` output."""
        return cls(
            party=data["party"],
            accepted=data["accepted"],
            matched_count=data["matched_count"],
            mismatches=data["mismatches"],
            rate=data["rate"],
            threshold=data["threshold"],
            key_length=data["key_length"],
            message_bit=data["message_bit"],
        )


def _as_count(value: Any, name: str) -> int:
    """Validate a non-negative integer count."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(
            f"{name} must be a non-negative integer, got "
            f"{type(value).__name__}"
        )
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result


def _as_optional_count(value: Any, name: str) -> int | None:
    """Validate a count that is allowed to be absent."""
    if value is None:
        return None
    return _as_count(value, name)


def _as_rate(value: Any, name: str) -> float:
    """Validate a mismatch rate or threshold in ``[0, 1]``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{name} must be a real number in [0, 1], got "
            f"{type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    if not 0.0 <= result <= 1.0:
        raise ValueError(
            f"{name} must lie in [0, 1], got {result!r}. It is a fraction of "
            f"the matched positions."
        )
    return result


def _check_pairing(signature: Signature, record: RecipientRecord) -> None:
    """Check that a signature and a record describe the same distribution run."""
    if not isinstance(signature, Signature):
        raise TypeError(
            f"signature must be a Signature, got {type(signature).__name__}. "
            f"Build one with sign(message_bit, keys)."
        )
    if not isinstance(record, RecipientRecord):
        raise TypeError(
            f"record must be a RecipientRecord, got {type(record).__name__}. "
            f"It is produced by distribute_to_recipient(...)."
        )
    if signature.message_bit != record.message_bit:
        raise ValueError(
            f"the signature declares message bit {signature.message_bit} but "
            f"{record.party.value}'s record was made during the distribution "
            f"for bit {record.message_bit}. The two distribution runs use "
            f"independent keys, so scoring one against the other measures "
            f"nothing but chance: the rate would sit near (1 - 1/|B|)/2 and "
            f"look like a noisy channel."
        )
    if len(signature) != len(record):
        raise ValueError(
            f"the signature declares {len(signature)} key positions but "
            f"{record.party.value}'s record holds {len(record)} entries. "
            f"Verification is positional -- entry i is the measurement of key "
            f"qubit i -- so a length mismatch means the record and the "
            f"signature came from different runs."
        )


def matched_positions(
    signature: Signature, record: RecipientRecord
) -> tuple[int, ...]:
    """Return the positions the verifier is allowed to score, ``M_R``.

    >>> from qsecure.protocol.keys import KeyElement, PrivateKey
    >>> from qsecure.protocol.records import RecipientRecord
    >>> from qsecure.protocol.signature import sign
    >>> from qsecure.protocol.verify import matched_positions
    >>> key = PrivateKey(0, (KeyElement("X", 1), KeyElement("Z", -1)))
    >>> record = RecipientRecord.from_measurements("Bob", 0, ["X", "Y"], [1, 1])
    >>> matched_positions(sign(0, key), record)
    (0,)
    """
    _check_pairing(signature, record)
    declared = signature.bases
    chosen = record.bases
    return tuple(
        index for index in range(len(record)) if chosen[index] == declared[index]
    )


def mismatch_positions(
    signature: Signature, record: RecipientRecord
) -> tuple[int, ...]:
    """Return the matched positions whose outcome contradicts the declaration.

    >>> from qsecure.protocol.keys import KeyElement, PrivateKey
    >>> from qsecure.protocol.records import RecipientRecord
    >>> from qsecure.protocol.signature import sign
    >>> from qsecure.protocol.verify import mismatch_positions
    >>> key = PrivateKey(0, (KeyElement("X", 1), KeyElement("Z", -1)))
    >>> record = RecipientRecord.from_measurements("Bob", 0, ["X", "Z"], [-1, -1])
    >>> mismatch_positions(sign(0, key), record)
    (0,)
    """
    declared = signature.eigenvalues
    measured = record.eigenvalues
    return tuple(
        index
        for index in matched_positions(signature, record)
        if measured[index] != declared[index]
    )


def _round_refusal(
    signature: Signature,
    record: RecipientRecord,
    ledger: ConsumedRecords | None,
    *,
    matched_count: int,
    params: ProtocolParams,
    floor: int,
) -> VerificationAbort | None:
    """Apply the two replay checks and return the first refusal."""
    common = {
        "party": record.party,
        "matched_count": matched_count,
        "minimum_matched": floor,
        "expected_matched": params.expected_matched,
        "key_length": params.key_length,
        "message_bit": record.message_bit,
    }
    stamped = record.session_id
    if stamped is not None and signature.session_id != stamped:
        return VerificationAbort(reason=AbortReason.SESSION_MISMATCH, **common)
    if ledger is not None and ledger.is_spent(record):
        return VerificationAbort(
            reason=AbortReason.RECORD_ALREADY_VERIFIED, **common
        )
    return None


def _evidence_refusal(
    *,
    party: Party,
    matched_count: int,
    counterpart_matched: int | None,
    params: ProtocolParams,
    floor: int,
    message_bit: int,
    counterpart_declaration: str | None = None,
    scored_declaration: str | None = None,
    counterpart_binding_missing: bool = False,
) -> VerificationAbort | None:
    """Apply the matched-count rule and return the first refusal."""
    common = {
        "party": party,
        "minimum_matched": floor,
        "expected_matched": params.expected_matched,
        "key_length": params.key_length,
        "message_bit": message_bit,
        "counterpart_matched": counterpart_matched,
        "minimum_pooled": (
            None
            if counterpart_matched is None
            else minimum_pooled_matched_count(params)
        ),
    }
    if matched_count < floor:
        return VerificationAbort(
            reason=(
                AbortReason.EMPTY_MATCHED_SET
                if matched_count == 0
                else AbortReason.BELOW_FLOOR
            ),
            matched_count=matched_count,
            **common,
        )
    if counterpart_matched is None:
        return None
    if counterpart_binding_missing:
        return VerificationAbort(
            reason=AbortReason.COUNT_OF_UNRECORDED_PROVENANCE,
            matched_count=matched_count,
            **common,
        )
    if (
        counterpart_declaration is not None
        and scored_declaration is not None
        and counterpart_declaration != scored_declaration
    ):
        return VerificationAbort(
            reason=AbortReason.COUNTS_FROM_TWO_DECLARATIONS,
            matched_count=matched_count,
            **common,
        )
    pooled_floor = minimum_pooled_matched_count(params)
    if matched_count + counterpart_matched < pooled_floor:
        return VerificationAbort(
            reason=AbortReason.POOLED_BELOW_FLOOR,
            matched_count=matched_count,
            **common,
        )
    if counterpart_matched < floor:
        return VerificationAbort(
            reason=AbortReason.COUNTERPART_BELOW_FLOOR,
            matched_count=matched_count,
            **common,
        )
    return None


def verify(
    signature: Signature,
    record: RecipientRecord,
    params: ProtocolParams,
    *,
    counterpart_matched: int | None = None,
    ledger: ConsumedRecords | None = None,
    authorised: frozenset[Party] | None = None,
) -> VerificationResult:
    """Score a signature against one recipient's record and reach a verdict.

    >>> import numpy as np
    >>> from qsecure.protocol.distribute import distribute_to_recipient
    >>> from qsecure.protocol.keys import generate_private_key
    >>> from qsecure.protocol.params import Party, ProtocolParams
    >>> from qsecure.protocol.signature import sign
    >>> from qsecure.protocol.verify import verify
    >>> params = ProtocolParams(key_length=24)
    >>> key = generate_private_key(params, 0, rng=np.random.default_rng(1))
    >>> record = distribute_to_recipient(
    ...     key, params, party=Party.BOB, rng=np.random.default_rng(2)
    ... )
    >>> result = verify(sign(0, key, params), record, params)
    >>> result.accepted, result.rate, result.mismatches
    (True, 0.0, 0)

    >>> from qsecure.protocol.signature import Signature
    >>> from qsecure.protocol.verify import verify_or_abort
    >>> bound = record.with_session_id(
    ...     Signature(0, key, session_opening="round-a").session_id
    ... )
    >>> verify(sign(0, key, params, session_opening="round-a"), bound, params).rate
    0.0
    >>> verify_or_abort(
    ...     sign(0, key, params, session_opening="round-b"), bound, params
    ... ).reason.value
    'session-identifier-mismatch'

    One round, one verdict, once the verifier keeps a ledger:

    >>> from qsecure.protocol.verify import ConsumedRecords
    >>> ledger = ConsumedRecords(Party.BOB)
    >>> declaration = sign(0, key, params, session_opening="round-a")
    >>> verify(declaration, bound, params, ledger=ledger).accepted
    True
    >>> verify_or_abort(
    ...     declaration, bound, params, ledger=ledger
    ... ).reason.value
    'records-already-verified'
    """
    if not isinstance(params, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(params).__name__}"
        )
    resolved_authorised: frozenset[Party] | None = None
    if authorised is not None:
        if not isinstance(authorised, frozenset):
            raise TypeError(
                f"authorised must be a frozenset of Party or None, got "
                f"{type(authorised).__name__}. Frozen because it is the "
                f"round's recipient set, fixed before any declaration arrives; "
                f"None runs no authorisation check, which is the default."
            )
        if not authorised:
            raise ValueError(
                "authorised must name at least one recipient. An empty set "
                "authorises nobody, so every verifier is refused with "
                "unauthorised-verifier -- a structural reason, which "
                "qsecure.detect.statistics reports at a false-positive "
                "probability of exactly zero -- and a run that authorised "
                "nobody would be published as a run that caught somebody. To "
                "run no authorisation check, pass authorised=None, which is "
                "the default."
            )
        # Coerce the members too: a mis-wired set must raise, not refuse every verifier.
        resolved_authorised = frozenset(
            _as_party(member) for member in authorised
        )
    if ledger is not None and not isinstance(ledger, ConsumedRecords):
        raise TypeError(
            f"ledger must be a ConsumedRecords or None, got "
            f"{type(ledger).__name__}. It is the verifier's own record of "
            f"rounds he has already decided; build one per verifier with "
            f"ConsumedRecords(record.party)."
        )
    _check_pairing(signature, record)
    signature.check_against(params)
    record.check_against(params)
    # After every check that raises, but before reading evidence, so an
    # unauthorised party learns no count.
    if (
        resolved_authorised is not None
        and record.party not in resolved_authorised
    ):
        raise MatchedSetTooSmall(
            VerificationAbort(
                party=record.party,
                reason=AbortReason.UNAUTHORISED_VERIFIER,
                # No count was taken; the other numbers are the parameter set's own.
                matched_count=0,
                minimum_matched=minimum_matched_count(params),
                expected_matched=params.expected_matched,
                key_length=params.key_length,
                message_bit=record.message_bit,
            )
        )
    # Read provenance before coercing: _as_count returns a plain int.
    counterpart_declaration = _counterpart_binding(counterpart_matched)
    binding_missing = (
        counterpart_matched is not None
        and _claims_provenance(counterpart_matched)
        and counterpart_declaration is None
    )
    reported = _as_optional_count(counterpart_matched, "counterpart_matched")
    if reported is not None and reported > params.key_length:
        raise ValueError(
            f"counterpart_matched ({reported}) exceeds the key length "
            f"({params.key_length}); the other verifier's matched set is a "
            f"subset of the same key positions, so a larger count means the "
            f"two verifiers scored different runs."
        )

    matched = matched_positions(signature, record)
    floor = minimum_matched_count(params)
    # Replay checks first: no floor applies to a count of the wrong round.
    refusal = _round_refusal(
        signature,
        record,
        ledger,
        matched_count=len(matched),
        params=params,
        floor=floor,
    )
    if refusal is not None:
        raise MatchedSetTooSmall(refusal)
    refusal = _evidence_refusal(
        party=record.party,
        matched_count=len(matched),
        counterpart_matched=reported,
        params=params,
        floor=floor,
        message_bit=record.message_bit,
        counterpart_declaration=counterpart_declaration,
        # Fingerprint only when there is a binding to compare against.
        scored_declaration=(
            None
            if counterpart_declaration is None
            else _declaration_digest(signature)
        ),
        counterpart_binding_missing=binding_missing,
    )
    if refusal is not None:
        # A no-verdict outcome, not a rejection.
        raise MatchedSetTooSmall(refusal)

    mismatches = len(mismatch_positions(signature, record))
    rate = mismatches / len(matched)
    threshold = params.threshold_for(record.party)

    # Spent only on a verdict, so a refusal cannot burn an unscored round.
    if ledger is not None:
        ledger.spend(record)

    return VerificationResult(
        party=record.party,
        accepted=rate <= threshold,
        matched_count=len(matched),
        mismatches=mismatches,
        rate=rate,
        threshold=threshold,
        key_length=params.key_length,
        message_bit=record.message_bit,
    )


def verify_or_abort(
    signature: Signature,
    record: RecipientRecord,
    params: ProtocolParams,
    *,
    counterpart_matched: int | None = None,
    ledger: ConsumedRecords | None = None,
    authorised: frozenset[Party] | None = None,
) -> VerificationResult | VerificationAbort:
    """Score a signature, returning the refusal instead of raising it.

    >>> from qsecure.protocol.keys import KeyElement, PrivateKey
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.records import RecipientRecord
    >>> from qsecure.protocol.signature import sign
    >>> from qsecure.protocol.verify import verify_or_abort
    >>> params = ProtocolParams(key_length=8, bases=("X", "Z"))
    >>> key = PrivateKey(0, tuple(KeyElement("X", 1) for _ in range(8)))
    >>> record = RecipientRecord.from_measurements("Bob", 0, ["Z"] * 8, [1] * 8)
    >>> outcome = verify_or_abort(sign(0, key, params), record, params)
    >>> outcome.reason.value, outcome.matched_count, outcome.minimum_matched
    ('empty-matched-set', 0, 1)
    """
    try:
        return verify(
            signature,
            record,
            params,
            counterpart_matched=counterpart_matched,
            ledger=ledger,
            authorised=authorised,
        )
    except MatchedSetTooSmall as too_small:
        return too_small.abort


def verify_all(
    signature: Signature,
    records: Mapping[Party | str, RecipientRecord],
    params: ProtocolParams,
    *,
    require_symmetrised: bool = True,
    exchange_counts: bool = True,
    ledgers: Mapping[Party | str, ConsumedRecords] | None = None,
    authorised: frozenset[Party] | None = None,
) -> dict[Party, VerificationResult]:
    """Verify one signature against every recipient's record.

    >>> import numpy as np
    >>> from qsecure.protocol.distribute import distribute_public_key
    >>> from qsecure.protocol.keys import generate_private_key
    >>> from qsecure.protocol.params import Party, ProtocolParams
    >>> from qsecure.protocol.signature import sign
    >>> from qsecure.protocol.symmetrise import symmetrise_records
    >>> from qsecure.protocol.verify import verify_all
    >>> params = ProtocolParams(key_length=24)
    >>> key = generate_private_key(params, 0, rng=np.random.default_rng(3))
    >>> raw = distribute_public_key(key, params, rng=np.random.default_rng(4))
    >>> records = symmetrise_records(raw, rng=np.random.default_rng(5))
    >>> results = verify_all(sign(0, key, params), records, params)
    >>> [results[party].accepted for party in (Party.BOB, Party.CHARLIE)]
    [True, True]
    >>> results[Party.BOB].threshold < results[Party.CHARLIE].threshold
    True
    >>> verify_all(sign(0, key, params), raw, params)
    Traceback (most recent call last):
        ...
    ValueError: Bob's record has not been symmetrised, ...
    """
    if not isinstance(records, Mapping):
        raise TypeError(
            f"records must be a mapping of Party to RecipientRecord, got "
            f"{type(records).__name__}; distribute_public_key returns exactly "
            f"that."
        )
    if not records:
        raise ValueError(
            "records must hold at least one recipient's log. Verifying against "
            "nobody reaches no verdict, and the protocol's claims are about "
            "two verifiers: transferability and non-repudiation are both "
            "statements about Bob and Charlie together."
        )
    if not isinstance(exchange_counts, bool):
        raise TypeError(
            f"exchange_counts must be a bool, got "
            f"{type(exchange_counts).__name__}; it selects whether the "
            f"recipients run Phase C', not a count."
        )
    if ledgers is not None and not isinstance(ledgers, Mapping):
        raise TypeError(
            f"ledgers must be a mapping of Party to ConsumedRecords or None, "
            f"got {type(ledgers).__name__}. One ledger per verifier: a single "
            f"shared object would put one verifier's history inside the "
            f"other's decision."
        )
    by_party: dict[Party, ConsumedRecords] = (
        {}
        if ledgers is None
        else {_as_party(party): held for party, held in ledgers.items()}
    )

    # Shape first, so a mis-wired pair fails on the wiring.
    checked: dict[Party, RecipientRecord] = {}
    for party, record in records.items():
        resolved = _as_party(party)
        if not isinstance(record, RecipientRecord):
            raise TypeError(
                f"records[{resolved.value!r}] must be a RecipientRecord, got "
                f"{type(record).__name__}"
            )
        if record.party is not resolved:
            raise ValueError(
                f"records is keyed by {resolved.value!r} but the record stored "
                f"there belongs to {record.party.value!r}. The key selects the "
                f"acceptance threshold, so a swap would score one verifier's "
                f"evidence against the other's cut -- which is precisely the "
                f"confusion the s_a < s_v gap exists to prevent."
            )
        if require_symmetrised and not record.symmetrised:
            raise ValueError(
                f"{resolved.value}'s record has not been symmetrised, so this "
                f"pair supports no non-repudiation claim. Alice prepares the "
                f"two recipients' copies independently and consumes separate "
                f"entanglement for each, so against raw logs she can send Bob "
                f"the eigenstate she declares and Charlie its orthogonal "
                f"partner: Bob accepts at rate 0 and Charlie rejects, with "
                f"probability 1, at every key length. Pass the pair through "
                f"symmetrise_records(records, rng=...) first -- "
                f"QDSSession.distribute() already does. If you are deliberately "
                f"running the insecure variant to measure that attack, say so "
                f"with verify_all(..., require_symmetrised=False)."
            )
        checked[resolved] = record

    reported = _exchanged_counts(
        signature, checked, params, enabled=exchange_counts
    )
    return {
        party: verify(
            signature,
            record,
            params,
            counterpart_matched=reported.get(party),
            ledger=by_party.get(party),
            authorised=authorised,
        )
        for party, record in checked.items()
    }


def _exchanged_counts(
    signature: Signature,
    records: Mapping[Party, RecipientRecord],
    params: ProtocolParams,
    *,
    enabled: bool,
) -> dict[Party, int]:
    """Run Phase C' for :func:`verify_all` and return each party's
    *counterpart* count.
    """
    if not enabled or set(records) != set(VERIFIERS):
        # Nobody to exchange with.
        return {}
    # Deferred: ``tally`` imports the floors from this module.
    from qsecure.protocol.tally import (
        exchange_matched_counts,
        matched_count_message,
    )

    messages = {
        party: matched_count_message(signature, record, params)
        for party, record in records.items()
    }
    pooled = exchange_matched_counts(messages, params)
    if pooled is None:
        return {}
    # counterpart_of, never by hand: inverting it would hand a verifier his own count.
    return {party: pooled.counterpart_of(party) for party in VERIFIERS}
