"""Protocol parameters, parties and the two acceptance thresholds.

Bob accepts iff ``r_B <= s_a``; Charlie iff ``r_C <= s_v``. Validation enforces
``0 <= s_a < s_v < forger_floor`` (transferability, non-repudiation room,
unforgeability). ``forger_floor = (|B|-1) / (2|B|(|B|+1))`` (``1/12`` for XYZ)
is optimal over all POVMs. Key figures:

>>> import math
>>> from qsecure.protocol.analysis import (
...     averaged_repudiation_bound,
...     binary_kl_divergence,
...     depolarising_error_rate,
...     matched_statistics,
...     recipient_forgery_bound,
...     repudiation_bound,
... )
>>> from qsecure.protocol.params import (
...     CHECKED_PARAMS, DEFAULT_PARAMS, DEMO_PARAMS,
... )
>>> from qsecure.protocol.verify import (
...     enforced_repudiation_bound,
...     minimum_matched_count,
...     minimum_pooled_matched_count,
... )

Defaults and gap:

>>> DEFAULT_PARAMS.key_length, DEFAULT_PARAMS.s_a, DEFAULT_PARAMS.s_v
(115200, 0.015625, 0.0625)
>>> DEFAULT_PARAMS.gap == 1 / 16 - 1 / 64 == 3 / 64
True
>>> DEFAULT_PARAMS.forger_floor == 1 / 12
True
>>> DEFAULT_PARAMS.s_v == 0.75 * DEFAULT_PARAMS.forger_floor
True

The ``s_v`` margin and the ``s_a`` noise budget:

>>> f"{binary_kl_divergence(DEFAULT_PARAMS.s_v, DEFAULT_PARAMS.forger_floor):.6f}"
'0.003088'
>>> f"{recipient_forgery_bound(DEFAULT_PARAMS, method='kl'):.1e}"
'1.1e-103'
>>> depolarising_error_rate(0.03125) == DEFAULT_PARAMS.s_a
True

Expected matched counts:

>>> statistics = matched_statistics(DEFAULT_PARAMS)
>>> statistics.expected, 2 * statistics.expected
(38400.0, 76800.0)

Conditional repudiation figures (assume the signer cannot see logged bases):

>>> f"{repudiation_bound(DEFAULT_PARAMS, matched_records=76800):.4e}"
'6.9040e-10'
>>> averaged = averaged_repudiation_bound(
...     DEFAULT_PARAMS, signer_sees_recipient_bases=False)
>>> f"{averaged:.4e}"
'6.9173e-10'

The unconditional figure needs both floors and the joint abort; the pooled floor
binds. The last line is the weaker per-verifier-only number:

>>> m_min = minimum_matched_count(DEFAULT_PARAMS)
>>> M_min = minimum_pooled_matched_count(DEFAULT_PARAMS)
>>> m_min, M_min, M_min - 2 * m_min
(36555, 74190, 1080)
>>> max(2 * m_min, M_min) == M_min
True
>>> f"{enforced_repudiation_bound(DEFAULT_PARAMS):.4e}"
'1.4139e-09'
>>> f"{math.exp(-M_min * DEFAULT_PARAMS.gap ** 2 / 8):.4e}"
'1.4139e-09'
>>> f"{math.exp(-2 * m_min * DEFAULT_PARAMS.gap ** 2 / 8):.4e}"
'1.9022e-09'

:data:`DEMO_PARAMS` carries no security claim:

>>> f"{repudiation_bound(DEMO_PARAMS, matched_records=128):.2f}"
'0.97'

:data:`CHECKED_PARAMS` spends 1/8 of rounds on estimation; bounds use the
shortened signing length, and the longer ``L`` keeps them no weaker:

>>> CHECKED_PARAMS.check_count, CHECKED_PARAMS.signing_length
(16458, 115206)
>>> CHECKED_PARAMS.expected_matched
38402.0
>>> minimum_matched_count(CHECKED_PARAMS), minimum_pooled_matched_count(
...     CHECKED_PARAMS)
(36557, 74194)
>>> f"{enforced_repudiation_bound(CHECKED_PARAMS):.4e}"
'1.4124e-09'
>>> enforced_repudiation_bound(CHECKED_PARAMS) <= enforced_repudiation_bound(
...     DEFAULT_PARAMS)
True

Verification runs under :meth:`ProtocolParams.sifted`, which agrees in every
derived quantity:

>>> sifted = CHECKED_PARAMS.sifted()
>>> sifted.key_length, sifted.check_fraction
(115206, 0.0)
>>> (sifted.expected_matched, minimum_matched_count(sifted),
...  minimum_pooled_matched_count(sifted)) == (
...     CHECKED_PARAMS.expected_matched,
...     minimum_matched_count(CHECKED_PARAMS),
...     minimum_pooled_matched_count(CHECKED_PARAMS))
True

The shipped fraction meets its own requirement:

>>> from qsecure.protocol.checkrounds import required_check_rounds
>>> required_check_rounds(CHECKED_PARAMS).total
13483
>>> CHECKED_PARAMS.check_count >= required_check_rounds(CHECKED_PARAMS).total
True
>>> DEFAULT_PARAMS.check_count, DEFAULT_PARAMS.has_check_rounds
(0, False)
"""

from __future__ import annotations

import enum
import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

from qsecure.core.paulis import PauliBasis

__all__ = [
    "Party",
    "VERIFIERS",
    "ProtocolParams",
    "COMPLIANT_FORGER_RATE_THREE_BASIS",
    "UNSYMMETRISED_FORGER_RATE_THREE_BASIS",
    "DEFAULT_S_A",
    "DEFAULT_S_V",
    "DEFAULT_BASES",
    "DEFAULT_CHECK_FRACTION",
    "DEFAULT_PARAMS",
    "DEMO_PARAMS",
    "CHECKED_PARAMS",
    "DEMO_CHECKED_PARAMS",
]


class Party(enum.StrEnum):
    """One of the three roles; a ``StrEnum``, so members serialise as their labels.

    Examples
    --------
    >>> import json
    >>> from qsecure.protocol.params import Party
    >>> json.dumps({Party.BOB: 0, Party.CHARLIE: 1})
    '{"Bob": 0, "Charlie": 1}'
    >>> Party.CHARLIE.is_verifier
    True
    """

    ALICE = "Alice"
    BOB = "Bob"
    CHARLIE = "Charlie"

    @property
    def is_verifier(self) -> bool:
        """bool: ``True`` for the two recipients, ``False`` for Alice."""
        return self is not Party.ALICE


VERIFIERS: Final[tuple[Party, Party]] = (Party.BOB, Party.CHARLIE)
"""The two verifying parties, Bob then Charlie."""


COMPLIANT_FORGER_RATE_THREE_BASIS: Final[float] = 1.0 / 12.0
"""Optimal recipient-forger mismatch rate for ``{X,Y,Z}`` after symmetrisation (``1/12``)."""

UNSYMMETRISED_FORGER_RATE_THREE_BASIS: Final[float] = 1.0 / 3.0
"""The same rate without symmetrisation (``1/3``); also the unmatched-position bug rate."""

DEFAULT_BASES: Final[tuple[PauliBasis, ...]] = (
    PauliBasis.X,
    PauliBasis.Y,
    PauliBasis.Z,
)
"""The three-basis (six-state) alphabet."""

DEFAULT_S_A: Final[float] = 1.0 / 64.0
"""Bob's threshold, ``1/64``: tolerates Werner noise up to ``p = 2 * s_a``."""

DEFAULT_S_V: Final[float] = 1.0 / 16.0
"""Charlie's threshold, ``1/16``: three quarters of the ``1/12`` forger floor."""

DEFAULT_CHECK_FRACTION: Final[float] = 0.125
"""Share of positions spent on check rounds, ``1/8``."""

_DEFAULT_KEY_LENGTH: Final[int] = 115200
_DEMO_KEY_LENGTH: Final[int] = 192

_CHECKED_KEY_LENGTH: Final[int] = 131664
"""Smallest multiple of 24 above ``115200 * 8/7``."""


def _as_basis(basis: PauliBasis | str, *, name: str = "basis") -> PauliBasis:
    """Coerce a :class:`PauliBasis` or case-insensitive label to :class:`PauliBasis`."""
    if isinstance(basis, PauliBasis):
        return basis
    if isinstance(basis, str):
        try:
            return PauliBasis(basis.strip().upper())
        except ValueError:
            raise ValueError(
                f"{name} must name a measurement basis: expected 'X', 'Y' or "
                f"'Z' (case-insensitive) or a PauliBasis member, got "
                f"{basis!r}"
            ) from None
    raise TypeError(
        f"{name} must be a PauliBasis (or the string 'X', 'Y', 'Z'), got "
        f"{type(basis).__name__}"
    )


def _as_eigenvalue(eigenvalue: int, *, name: str = "eigenvalue") -> int:
    """Validate a Pauli eigenvalue ``+1``/``-1`` (booleans rejected)."""
    if isinstance(eigenvalue, bool):
        raise ValueError(
            f"{name} must be the Pauli eigenvalue +1 or -1, got the boolean "
            f"{eigenvalue!r}. Booleans look like measurement bits and would be "
            f"read with the wrong sign; convert with "
            f"eigenvalue = 1 - 2 * bit (bit 0 -> +1, bit 1 -> -1)."
        )
    if not isinstance(eigenvalue, numbers.Integral):
        raise ValueError(
            f"{name} must be the integer +1 or -1, got {type(eigenvalue).__name__}"
        )
    value = int(eigenvalue)
    if value not in (1, -1):
        raise ValueError(
            f"{name} must be +1 or -1, got {value!r}. These are Pauli "
            f"eigenvalues, not measurement bits; convert with "
            f"eigenvalue = 1 - 2 * bit (bit 0 -> +1, bit 1 -> -1)."
        )
    return value


def _as_message_bit(message_bit: int, *, name: str = "message_bit") -> int:
    """Validate a message bit ``0``/``1`` (booleans rejected)."""
    if isinstance(message_bit, bool):
        raise ValueError(
            f"{name} must be the integer 0 or 1, got the boolean "
            f"{message_bit!r}. Pass int({message_bit!r}) if you really mean "
            f"the key for message bit {int(message_bit)}."
        )
    if not isinstance(message_bit, numbers.Integral):
        raise ValueError(
            f"{name} must be the integer 0 or 1, got {type(message_bit).__name__}"
        )
    value = int(message_bit)
    if value not in (0, 1):
        raise ValueError(
            f"{name} must be 0 or 1, got {value!r}. Distribution runs once per "
            f"future message bit, so exactly two key families exist."
        )
    return value


def _as_threshold(value: Any, name: str) -> float:
    """Coerce and check a threshold: a finite, non-negative real (booleans rejected)."""
    if isinstance(value, bool):
        raise TypeError(
            f"{name} must be a real number in [0, 0.5), got the boolean "
            f"{value!r}"
        )
    if not isinstance(value, numbers.Real):
        raise TypeError(
            f"{name} must be a real number in [0, 0.5), got "
            f"{type(value).__name__}"
        )
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(
            f"{name} must be a finite mismatch rate in [0, 0.5), got {result!r}"
        )
    if result < 0.0:
        raise ValueError(
            f"{name} must be non-negative, got {result!r}. It is a mismatch "
            f"*rate*, the fraction of matched positions on which the "
            f"recipient's outcome disagrees with the declared eigenvalue, so it "
            f"lives in [0, 1) by construction. A negative threshold would make "
            f"the party reject every signature including honest noiseless ones, "
            f"which have rate exactly 0."
        )
    return result


def _as_check_fraction(value: Any) -> float:
    """Coerce and check the check-round fraction, in ``[0, 1)``."""
    if isinstance(value, bool):
        raise TypeError(
            f"check_fraction must be a real number in [0, 1), got the boolean "
            f"{value!r}"
        )
    if not isinstance(value, numbers.Real):
        raise TypeError(
            f"check_fraction must be a real number in [0, 1), got "
            f"{type(value).__name__}"
        )
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(
            f"check_fraction must be a finite fraction in [0, 1), got "
            f"{result!r}"
        )
    if result < 0.0:
        raise ValueError(
            f"check_fraction must be non-negative, got {result!r}. It is the "
            f"share of the run's positions diverted from the key to sampled "
            f"parameter estimation; 0 disables estimation entirely."
        )
    if result >= 1.0:
        raise ValueError(
            f"check_fraction must be strictly less than 1, got {result!r}. "
            f"Check positions are excluded from the key, so a fraction of 1 "
            f"leaves signing_length = 0 and there is nothing left to sign. "
            f"Try DEFAULT_CHECK_FRACTION = {DEFAULT_CHECK_FRACTION!r}."
        )
    return result


@dataclass(frozen=True)
class ProtocolParams:
    """The frozen parameter set one QDS run executes under.

    ``key_length`` is ``L`` rounds; with ``check_fraction > 0``, ``check_count``
    rounds are spent on estimation and every derived security quantity uses
    :attr:`signing_length`. ``allow_forgeable=True`` permits ``s_v >= forger_floor``
    for deliberate sweeps.

    Raises
    ------
    TypeError
        If a field has the wrong type.
    ValueError
        If ``L`` is not positive, ``bases`` is invalid, a threshold is out of range,
        ``s_a >= s_v``, ``s_v >= 0.5``, ``s_v >= forger_floor`` without
        ``allow_forgeable``, or ``check_fraction`` is invalid or yields no rounds.

    Examples
    --------
    >>> from qsecure.protocol.params import ProtocolParams, Party
    >>> params = ProtocolParams(key_length=115200)
    >>> params.expected_matched
    38400.0
    >>> params.threshold_for(Party.BOB) < params.threshold_for(Party.CHARLIE)
    True
    >>> ProtocolParams(key_length=64, s_a=0.4, s_v=0.2)
    Traceback (most recent call last):
        ...
    ValueError: s_a must be strictly less than s_v, ...
    >>> ProtocolParams(key_length=600, s_a=0.02, s_v=0.2)
    Traceback (most recent call last):
        ...
    ValueError: s_v must be strictly below the forger floor, ...
    >>> ProtocolParams(key_length=600, s_a=0.02, s_v=0.2, allow_forgeable=True).s_v
    0.2

    Check rounds shorten the key and everything follows it:

    >>> checked = ProtocolParams(key_length=115200, check_fraction=0.125)
    >>> checked.check_count, checked.signing_length
    (14400, 100800)
    >>> checked.expected_matched
    33600.0
    >>> checked.sifted().key_length, checked.sifted().check_fraction
    (100800, 0.0)
    """

    key_length: int
    s_a: float = DEFAULT_S_A
    s_v: float = DEFAULT_S_V
    bases: tuple[PauliBasis, ...] = DEFAULT_BASES
    allow_forgeable: bool = False
    check_fraction: float = 0.0

    def __post_init__(self) -> None:
        """Coerce the fields and reject every parameter set that is not secure."""
        object.__setattr__(self, "bases", _validate_bases(self.bases))
        object.__setattr__(
            self, "key_length", _validate_key_length(self.key_length, self.bases)
        )
        object.__setattr__(self, "s_a", _as_threshold(self.s_a, "s_a"))
        object.__setattr__(self, "s_v", _as_threshold(self.s_v, "s_v"))
        if not isinstance(self.allow_forgeable, bool):
            raise TypeError(
                f"allow_forgeable must be a bool, got "
                f"{type(self.allow_forgeable).__name__}. It is an explicit "
                f"opt-in to an insecure parameter set, not a tuning knob."
            )
        object.__setattr__(
            self, "check_fraction", _as_check_fraction(self.check_fraction)
        )
        if self.check_fraction > 0.0 and self.check_count < 1:
            raise ValueError(
                f"check_fraction={self.check_fraction!r} designates "
                f"floor({self.check_fraction!r} * {self.key_length}) = 0 check "
                f"rounds, so nothing would be estimated while the parameter "
                f"set claims estimation is happening -- the worst of both. "
                f"Either raise the fraction to at least "
                f"{1.0 / self.key_length!r}, lengthen the key, or set "
                f"check_fraction=0.0 to say plainly that this run publishes no "
                f"channel statistics."
            )
        if self.check_count >= self.key_length:
            raise ValueError(
                f"check_fraction={self.check_fraction!r} designates all "
                f"{self.check_count} of {self.key_length} positions as check "
                f"rounds, leaving signing_length=0. Check positions are "
                f"excluded from the key, so there would be nothing to sign."
            )

        if self.s_a >= self.s_v:
            raise ValueError(
                f"s_a must be strictly less than s_v, got s_a={self.s_a!r} and "
                f"s_v={self.s_v!r}. The whole scheme lives in the gap between "
                f"them. s_v > s_a is TRANSFERABILITY: Bob's acceptance region "
                f"is contained in Charlie's, so a signature inside Bob's tight "
                f"cut sits inside Charlie's looser one and can be forwarded. "
                f"s_a > 0 is the room that makes NON-REPUDIATION work: after "
                f"the recipients' symmetrisation exchange, Alice cannot craft a "
                f"signature that squeaks past Bob and fails Charlie, because "
                f"the exchange splits her preparations between them and pushing "
                f"Charlie's rate above s_v pushes Bob's above s_a with it. With "
                f"s_a >= s_v both "
                f"guarantees invert -- Charlie becomes strictly harder to "
                f"satisfy than Bob, so Bob can accept signatures he cannot "
                f"forward. Try s_a={DEFAULT_S_A!r}, s_v={DEFAULT_S_V!r}."
            )
        if self.s_v >= 0.5:
            raise ValueError(
                f"s_v must be strictly less than 0.5, got {self.s_v!r}. On "
                f"matched positions an adversary who knows nothing about the "
                f"private key produces mismatches at rate exactly 1/2, so a "
                f"verification threshold at or above 1/2 accepts pure noise and "
                f"UNFORGEABILITY is gone outright. 0.5 is only the crude bound: "
                f"the rate an optimal recipient forger actually achieves "
                f"against this alphabet is {self.forger_floor!r} "
                f"(= (|B| - 1) / (2 |B| (|B| + 1)) with "
                f"|B| = {len(self.bases)}), and s_v must sit below that, not "
                f"merely below 0.5. Try s_v={DEFAULT_S_V!r}."
            )
        if self.s_v >= self.forger_floor and not self.allow_forgeable:
            raise ValueError(
                f"s_v must be strictly below the forger floor, got "
                f"s_v={self.s_v!r} with forger_floor={self.forger_floor!r} "
                f"(= (|B| - 1) / (2 |B| (|B| + 1)) for |B| = "
                f"{len(self.bases)}). That floor is the mismatch rate an "
                f"optimal recipient forger achieves against the second "
                f"verifier -- Bob declaring, per position, what he knows of "
                f"Charlie's record -- so a threshold at or above it accepts "
                f"that forgery outright and UNFORGEABILITY is gone. It is the "
                f"threshold that binds, not the crude 1/2. Try "
                f"s_v={DEFAULT_S_V!r}, or pass allow_forgeable=True if this is "
                f"a deliberate sweep past the floor; the flag is part "
                f"of the parameter set's identity and serialisation, so the "
                f"resulting transcripts record that the run was known to be "
                f"forgeable."
            )


    @property
    def match_probability(self) -> float:
        """float: ``1/|B|``."""
        return 1.0 / len(self.bases)

    @property
    def check_count(self) -> int:
        """int: ``floor(check_fraction * key_length)``."""
        return int(math.floor(self.check_fraction * self.key_length))

    @property
    def signing_length(self) -> int:
        """int: ``L - check_count``, the key length every security quantity uses."""
        return self.key_length - self.check_count

    @property
    def has_check_rounds(self) -> bool:
        """bool: Whether this parameter set runs sampled parameter estimation."""
        return self.check_count > 0

    @property
    def expected_matched(self) -> float:
        """float: ``signing_length / |B|``; both matched-count floors read only this."""
        return self.signing_length * self.match_probability

    @property
    def gap(self) -> float:
        """float: ``s_v - s_a``; the repudiation exponent scales with its square."""
        return self.s_v - self.s_a

    @property
    def unmatched_noise_rate(self) -> float:
        """float: ``(1 - 1/|B|) / 2``, the rate added by wrongly scoring unmatched positions."""
        return (1.0 - self.match_probability) / 2.0

    @property
    def forger_scored_fraction(self) -> float:
        """float: ``(|B| + 1) / (2|B|)``, the fraction a recipient forger gets scored."""
        return (1.0 + self.match_probability) / 2.0

    @property
    def forger_floor(self) -> float:
        """float: ``(|B| - 1) / (2 |B| (|B| + 1))``, the optimal recipient-forger rate."""
        size = float(len(self.bases))
        return (size - 1.0) / (2.0 * size * (size + 1.0))

    @property
    def is_forgeable(self) -> bool:
        """bool: ``True`` when ``s_v >= forger_floor`` (only with ``allow_forgeable``)."""
        return self.s_v >= self.forger_floor

    def threshold_for(self, party: Party | str) -> float:
        """Return ``s_a`` for Bob or ``s_v`` for Charlie; Alice has none (``ValueError``).

        Examples
        --------
        >>> from qsecure.protocol.params import DEMO_PARAMS, Party
        >>> DEMO_PARAMS.threshold_for("Bob") == DEMO_PARAMS.s_a
        True
        """
        resolved = _as_party(party)
        if resolved is Party.ALICE:
            raise ValueError(
                "Alice has no acceptance threshold: she is the signer, not a "
                "verifier, and keeps no measurement record to compute a "
                "mismatch rate from. Ask for Party.BOB (threshold s_a) or "
                "Party.CHARLIE (threshold s_v)."
            )
        return self.s_a if resolved is Party.BOB else self.s_v

    def with_changes(self, **changes: Any) -> ProtocolParams:
        """Return a re-validated copy with some fields replaced.

        Examples
        --------
        >>> from qsecure.protocol.params import DEMO_PARAMS
        >>> DEMO_PARAMS.with_changes(key_length=48).key_length
        48
        """
        return replace(self, **changes)

    def with_check_fraction(self, check_fraction: float) -> ProtocolParams:
        """Return the same parameter set with a different check fraction.

        Examples
        --------
        >>> from qsecure.protocol.params import DEFAULT_CHECK_FRACTION, DEMO_PARAMS
        >>> checked = DEMO_PARAMS.with_check_fraction(DEFAULT_CHECK_FRACTION)
        >>> checked.check_count, checked.signing_length
        (24, 168)
        >>> checked.with_check_fraction(0.0) == DEMO_PARAMS
        True
        """
        return replace(self, check_fraction=check_fraction)

    def sifted(self) -> ProtocolParams:
        """Return the set the retained key is verified under (``key_length = signing_length``).

        Pass this, not the checked set, to verification and to anything in
        :mod:`qsecure.protocol.analysis`, which counts trials against ``key_length``.

        Examples
        --------
        >>> from qsecure.protocol.params import DEFAULT_PARAMS, ProtocolParams
        >>> checked = ProtocolParams(key_length=115200, check_fraction=0.125)
        >>> checked.sifted().key_length
        100800
        >>> checked.sifted().expected_matched == checked.expected_matched
        True
        >>> checked.sifted().sifted() == checked.sifted()
        True
        >>> DEFAULT_PARAMS.sifted() == DEFAULT_PARAMS
        True
        """
        if not self.has_check_rounds:
            return replace(self, check_fraction=0.0)
        return replace(
            self, key_length=self.signing_length, check_fraction=0.0
        )


    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the parameter set.

        Examples
        --------
        >>> import json
        >>> from qsecure.protocol.params import ProtocolParams
        >>> json.loads(json.dumps(ProtocolParams(key_length=9).to_dict()))["bases"]
        ['X', 'Y', 'Z']
        >>> ProtocolParams(key_length=9).to_dict()["check_fraction"]
        0.0
        """
        return {
            "key_length": self.key_length,
            "s_a": self.s_a,
            "s_v": self.s_v,
            "bases": list(self.bases),
            "allow_forgeable": self.allow_forgeable,
            "check_fraction": self.check_fraction,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProtocolParams:
        """Rebuild from a mapping; ``key_length`` is required, unknown keys are rejected.

        Examples
        --------
        >>> from qsecure.protocol.params import ProtocolParams
        >>> ProtocolParams.from_dict({"key_length": 12, "bases": ["z", "x"]}).bases
        (<PauliBasis.Z: 'Z'>, <PauliBasis.X: 'X'>)
        """
        known = {
            "key_length",
            "s_a",
            "s_v",
            "bases",
            "allow_forgeable",
            "check_fraction",
        }
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"unknown ProtocolParams field(s) {unknown}; expected any of "
                f"{sorted(known)}. Unknown keys are rejected rather than "
                f"ignored so a mistyped config entry cannot leave a threshold "
                f"silently at its default."
            )
        if "key_length" not in data:
            raise ValueError(
                "ProtocolParams.from_dict requires 'key_length'; there is no "
                "sensible default key length, because it is what every security "
                "bound is exponential in."
            )
        return cls(
            key_length=data["key_length"],
            s_a=data.get("s_a", DEFAULT_S_A),
            s_v=data.get("s_v", DEFAULT_S_V),
            bases=data.get("bases", DEFAULT_BASES),
            allow_forgeable=data.get("allow_forgeable", False),
            check_fraction=data.get("check_fraction", 0.0),
        )


def _as_party(party: Party | str) -> Party:
    """Coerce a :class:`Party` or case-insensitive label to :class:`Party`."""
    if isinstance(party, Party):
        return party
    if isinstance(party, str):
        cleaned = party.strip().capitalize()
        try:
            return Party(cleaned)
        except ValueError:
            raise ValueError(
                f"unknown party {party!r}; expected 'Alice', 'Bob' or "
                f"'Charlie' (case-insensitive) or a Party member"
            ) from None
    raise TypeError(
        f"party must be a Party (or the string 'Alice', 'Bob', 'Charlie'), got "
        f"{type(party).__name__}"
    )


def _validate_bases(bases: Sequence[PauliBasis | str]) -> tuple[PauliBasis, ...]:
    """Coerce and check the alphabet: a non-string sequence of at least two distinct bases."""
    if isinstance(bases, str):
        raise TypeError(
            f"bases must be a sequence of PauliBasis members, not the string "
            f"{bases!r}. A string would iterate character by character; write "
            f"(PauliBasis.X, PauliBasis.Y, PauliBasis.Z) or ['X', 'Y', 'Z']."
        )
    if not isinstance(bases, Sequence):
        raise TypeError(
            f"bases must be a sequence of PauliBasis members, got "
            f"{type(bases).__name__}"
        )
    resolved = tuple(_as_basis(basis, name="bases entry") for basis in bases)
    if not resolved:
        raise ValueError(
            "bases must not be empty: Alice and the recipients both draw their "
            "measurement bases uniformly from this alphabet, so an empty one "
            "leaves nothing to draw."
        )
    if len(set(resolved)) != len(resolved):
        duplicates = sorted({b for b in resolved if resolved.count(b) > 1})
        raise ValueError(
            f"bases must not repeat a basis, got duplicates "
            f"{[b.value for b in duplicates]}. A repeated entry silently "
            f"doubles that basis's draw probability, which changes the match "
            f"probability and therefore every security bound, without changing "
            f"anything visible in a run."
        )
    if len(resolved) < 2:
        raise ValueError(
            f"bases must hold at least two bases, got "
            f"{[b.value for b in resolved]}. With a single basis every "
            f"recipient's basis matches Alice's, the public key states are "
            f"perfectly distinguishable by the measurement the protocol itself "
            f"prescribes, and the compliant-forger mismatch rate drops to 0: "
            f"UNFORGEABILITY is not merely weakened but absent."
        )
    return resolved


def _validate_key_length(key_length: int, bases: tuple[PauliBasis, ...]) -> int:
    """Coerce and check ``L``: a positive, non-bool integer."""
    if isinstance(key_length, bool):
        raise ValueError(
            f"key_length must be a positive integer, got the boolean "
            f"{key_length!r}"
        )
    if not isinstance(key_length, numbers.Integral):
        raise ValueError(
            f"key_length must be a positive integer, got "
            f"{type(key_length).__name__}. It counts key elements, so a "
            f"fractional value is meaningless; use int(...) at the config "
            f"boundary."
        )
    value = int(key_length)
    if value <= 0:
        raise ValueError(
            f"key_length must be positive, got {value}. It is the number of "
            f"key elements drawn per message bit, and only the matched "
            f"positions -- about key_length / {len(bases)} of them -- carry any "
            f"information. Every failure probability in the scheme is "
            f"exp(-Theta(matched positions)), so a non-positive length leaves "
            f"nothing to verify and no bound at all."
        )
    return value


DEFAULT_PARAMS: Final[ProtocolParams] = ProtocolParams(
    key_length=_DEFAULT_KEY_LENGTH,
    s_a=DEFAULT_S_A,
    s_v=DEFAULT_S_V,
    bases=DEFAULT_BASES,
)
"""Security-grade set: ``L = 115200``, ``s_a = 1/64``, ``s_v = 1/16``, bases ``XYZ``.

Unconditional enforced repudiation bound ``1.4139e-09``; recipient-forgery bound
``1e-103``. Checked in the module doctests.
"""

DEMO_PARAMS: Final[ProtocolParams] = ProtocolParams(
    key_length=_DEMO_KEY_LENGTH,
    s_a=DEFAULT_S_A,
    s_v=DEFAULT_S_V,
    bases=DEFAULT_BASES,
)
"""Short set (``L = 192``) with the default thresholds; no security claim."""

CHECKED_PARAMS: Final[ProtocolParams] = ProtocolParams(
    key_length=_CHECKED_KEY_LENGTH,
    s_a=DEFAULT_S_A,
    s_v=DEFAULT_S_V,
    bases=DEFAULT_BASES,
    check_fraction=DEFAULT_CHECK_FRACTION,
)
"""Security-grade set with check rounds: ``L = 131664``, ``check_fraction = 1/8``.

``signing_length = 115206``, so the enforced bound (``1.4124e-09``) is no weaker
than :data:`DEFAULT_PARAMS`'.
"""

DEMO_CHECKED_PARAMS: Final[ProtocolParams] = ProtocolParams(
    key_length=_DEMO_KEY_LENGTH,
    s_a=DEFAULT_S_A,
    s_v=DEFAULT_S_V,
    bases=DEFAULT_BASES,
    check_fraction=DEFAULT_CHECK_FRACTION,
)
""":data:`DEMO_PARAMS` with check rounds (24 of 192); no security or statistical claim."""
