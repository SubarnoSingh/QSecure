"""Phase C': Bob and Charlie exchange matched counts so both can check ``M = m_B + m_C``.

A per-verifier floor alone lets a log-reading signer aim ``M`` at ``2 m_min``
and let the coins split it (Bob accepts, Charlie has no verdict, probability
-> 1/2). Each message carries a count, never positions, plus a digest of the
declaration counted.

.. _binding-is-mandatory:

The declaration digest is mandatory on both types, neither may be subclassed,
and :func:`~qsecure.protocol.verify.verify` refuses a count without one.
:func:`no_count_exchange` skips the step, for measuring the attack.

Examples
--------
>>> import numpy as np
>>> from qsecure.protocol.distribute import distribute_public_key
>>> from qsecure.protocol.keys import generate_private_key
>>> from qsecure.protocol.params import Party, ProtocolParams
>>> from qsecure.protocol.signature import sign
>>> from qsecure.protocol.symmetrise import symmetrise_records
>>> from qsecure.protocol.tally import (
...     exchange_matched_counts, matched_count_message
... )
>>> params = ProtocolParams(key_length=600)
>>> key = generate_private_key(params, 0, rng=np.random.default_rng(1))
>>> raw = distribute_public_key(key, params, rng=np.random.default_rng(2))
>>> records = symmetrise_records(raw, rng=np.random.default_rng(3))
>>> declaration = sign(0, key, params)
>>> messages = {
...     party: matched_count_message(declaration, record, params)
...     for party, record in records.items()
... }
>>> pooled = exchange_matched_counts(messages, params)
>>> pooled.pooled == pooled.bob_count + pooled.charlie_count
True
>>> pooled.meets_pooled_floor, pooled.meets_every_floor
(True, True)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from qsecure.protocol.params import (
    VERIFIERS,
    Party,
    ProtocolParams,
    _as_message_bit,
    _as_party,
)
from qsecure.protocol.records import RecipientRecord
from qsecure.protocol.signature import Signature
from qsecure.protocol.verify import (
    _BoundMatchedCount,
    _as_count,
    _declaration_digest,
    matched_positions,
    minimum_matched_count,
    minimum_pooled_matched_count,
)

__all__ = [
    "CountExchange",
    "MatchedCountMessage",
    "PooledMatchedCounts",
    "exchange_matched_counts",
    "matched_count_message",
    "no_count_exchange",
    "counterpart_count",
]


@dataclass(frozen=True)
class MatchedCountMessage:
    """One recipient's Phase C' message: ``|M_R|`` and the declaration it was counted against.

    Raises
    ------
    TypeError
        If a count is not an integer or the digest is not a string.
    ValueError
        If ``party`` is Alice, the bit is invalid, ``key_length < 1``, the count
        exceeds ``key_length``, or the digest is ``None`` or empty.

    Examples
    --------
    >>> from qsecure.protocol.tally import MatchedCountMessage
    >>> message = MatchedCountMessage("Bob", 0, 204, 600, "0f1e")
    >>> message.party.value, message.matched_count
    ('Bob', 204)
    """

    party: Party
    message_bit: int
    matched_count: int
    key_length: int
    declaration_digest: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Refuse subclasses; see :func:`_final`."""
        _final(cls)

    def __post_init__(self) -> None:
        """Coerce the fields and refuse a message no recipient could have sent."""
        resolved = _as_party(self.party)
        if resolved is Party.ALICE:
            raise ValueError(
                "Alice sends no matched-count message: she is the signer, "
                "holds no measurement record and is not a party to the "
                "recipients' exchange. A MatchedCountMessage belongs to "
                "Party.BOB or Party.CHARLIE."
            )
        object.__setattr__(self, "party", resolved)
        object.__setattr__(
            self, "message_bit", _as_message_bit(self.message_bit)
        )
        object.__setattr__(
            self, "matched_count", _as_count(self.matched_count, "matched_count")
        )
        object.__setattr__(
            self, "key_length", _as_count(self.key_length, "key_length")
        )
        if self.key_length < 1:
            raise ValueError(
                f"key_length must be at least 1, got {self.key_length}; a run "
                f"with no key positions has nothing to count."
            )
        if self.matched_count > self.key_length:
            raise ValueError(
                f"matched_count ({self.matched_count}) cannot exceed "
                f"key_length ({self.key_length}); the matched set is a subset "
                f"of the key positions."
            )
        object.__setattr__(
            self,
            "declaration_digest",
            _as_declaration_digest(self.declaration_digest),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the message.

        Examples
        --------
        >>> import json
        >>> from qsecure.protocol.tally import MatchedCountMessage
        >>> json.loads(
        ...     json.dumps(MatchedCountMessage("Bob", 1, 3, 9, "0f1e").to_dict())
        ... )
        {'party': 'Bob', 'message_bit': 1, 'matched_count': 3, 'key_length': 9, \
'declaration_digest': '0f1e'}
        """
        return {
            "party": self.party,
            "message_bit": self.message_bit,
            "matched_count": self.matched_count,
            "key_length": self.key_length,
            "declaration_digest": self.declaration_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MatchedCountMessage:
        """Rebuild a message from :meth:`to_dict` output; a missing digest raises ``KeyError``."""
        return cls(
            party=data["party"],
            message_bit=data["message_bit"],
            matched_count=data["matched_count"],
            key_length=data["key_length"],
            declaration_digest=data["declaration_digest"],
        )


@dataclass(frozen=True)
class PooledMatchedCounts:
    """Both recipients' view after the exchange: counts, floors and declaration digest.

    Floors are stored, so a transcript records the rule in force at the time.

    Raises
    ------
    TypeError
        If a count is not an integer or the digest is not a string.
    ValueError
        If the fields are inconsistent or the digest is ``None`` or empty.

    Examples
    --------
    >>> from qsecure.protocol.tally import PooledMatchedCounts
    >>> pooled = PooledMatchedCounts(
    ...     bob_count=204, charlie_count=196, minimum_matched=67,
    ...     minimum_pooled=212, key_length=600, message_bit=0,
    ...     declaration_digest="0f1e",
    ... )
    >>> pooled.pooled, pooled.meets_every_floor
    (400, True)
    >>> pooled.counterpart_of("Bob")
    196
    """

    bob_count: int
    charlie_count: int
    minimum_matched: int
    minimum_pooled: int
    key_length: int
    message_bit: int
    declaration_digest: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Refuse subclasses; see :func:`_final`."""
        _final(cls)

    def __post_init__(self) -> None:
        """Coerce the fields and check the two messages describe one run."""
        for name in ("bob_count", "charlie_count", "key_length"):
            object.__setattr__(self, name, _as_count(getattr(self, name), name))
        for name in ("minimum_matched", "minimum_pooled"):
            object.__setattr__(self, name, _as_count(getattr(self, name), name))
        object.__setattr__(
            self, "message_bit", _as_message_bit(self.message_bit)
        )
        if self.key_length < 1:
            raise ValueError(
                f"key_length must be at least 1, got {self.key_length}; a run "
                f"with no key positions has nothing to count."
            )
        for name in ("minimum_matched", "minimum_pooled"):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be at least 1, got {getattr(self, name)}. "
                    f"Even where the Chernoff tail is vacuous a floor is 1, "
                    f"because a rate needs a denominator; see "
                    f"qsecure.protocol.verify.minimum_matched_count."
                )
        for name in ("bob_count", "charlie_count"):
            if getattr(self, name) > self.key_length:
                raise ValueError(
                    f"{name} ({getattr(self, name)}) cannot exceed key_length "
                    f"({self.key_length}); each matched set is a subset of the "
                    f"same key positions."
                )
        if self.minimum_pooled > 2 * self.key_length:
            raise ValueError(
                f"minimum_pooled ({self.minimum_pooled}) cannot exceed "
                f"2 * key_length ({2 * self.key_length}); the pooled count is "
                f"over both verifiers' matched sets, so 2 L is its ceiling."
            )
        object.__setattr__(
            self,
            "declaration_digest",
            _as_declaration_digest(self.declaration_digest),
        )


    @property
    def pooled(self) -> int:
        """int: ``M = m_B + m_C``."""
        return self.bob_count + self.charlie_count

    @property
    def meets_pooled_floor(self) -> bool:
        """bool: ``True`` iff ``M >= M_min``."""
        return self.pooled >= self.minimum_pooled

    @property
    def parties_below_floor(self) -> tuple[Party, ...]:
        """tuple of Party: verifiers below ``m_min``; any entry takes both verifiers down."""
        return tuple(
            party
            for party in VERIFIERS
            if self.count_for(party) < self.minimum_matched
        )

    @property
    def meets_every_floor(self) -> bool:
        """bool: ``True`` iff every floor is met and both verifiers may score."""
        return self.meets_pooled_floor and not self.parties_below_floor

    def count_for(self, party: Party | str) -> int:
        """Return the count ``party`` (Bob or Charlie) reported."""
        resolved = _as_verifier(party)
        return (
            self.bob_count if resolved is Party.BOB else self.charlie_count
        )

    def counterpart_of(self, party: Party | str) -> int:
        """Return the other verifier's count, carrying :attr:`declaration_digest`."""
        resolved = _as_verifier(party)
        count = (
            self.charlie_count if resolved is Party.BOB else self.bob_count
        )
        return _BoundMatchedCount(count, self.declaration_digest)

    def summary(self) -> str:
        """Return a one-line account of the exchange.

        Examples
        --------
        >>> from qsecure.protocol.tally import PooledMatchedCounts
        >>> PooledMatchedCounts(204, 196, 67, 212, 600, 0, "0f1e").summary()
        'POOLED EVIDENCE bit 0: m_B = 204, m_C = 196, M = 400 (floors: \
m_min = 67, M_min = 212). Every floor met.'
        """
        if self.meets_every_floor:
            verdict = "Every floor met."
        elif not self.meets_pooled_floor:
            verdict = (
                f"Pooled floor NOT met: M = {self.pooled} < "
                f"{self.minimum_pooled}. Neither verifier scores."
            )
        else:
            short = ", ".join(
                party.value for party in self.parties_below_floor
            )
            verdict = (
                f"Below the per-verifier floor: {short}. Neither verifier "
                f"scores."
            )
        return (
            f"POOLED EVIDENCE bit {self.message_bit}: m_B = {self.bob_count}, "
            f"m_C = {self.charlie_count}, M = {self.pooled} (floors: "
            f"m_min = {self.minimum_matched}, M_min = {self.minimum_pooled}). "
            f"{verdict}"
        )


    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view; derived values are not stored.

        Examples
        --------
        >>> import json
        >>> from qsecure.protocol.tally import PooledMatchedCounts
        >>> blob = PooledMatchedCounts(204, 196, 67, 212, 600, 0, "0f1e").to_dict()
        >>> json.loads(json.dumps(blob))["bob_count"]
        204
        """
        return {
            "bob_count": self.bob_count,
            "charlie_count": self.charlie_count,
            "minimum_matched": self.minimum_matched,
            "minimum_pooled": self.minimum_pooled,
            "key_length": self.key_length,
            "message_bit": self.message_bit,
            "declaration_digest": self.declaration_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PooledMatchedCounts:
        """Rebuild from :meth:`to_dict` output; a missing digest raises ``KeyError``."""
        return cls(
            bob_count=data["bob_count"],
            charlie_count=data["charlie_count"],
            minimum_matched=data["minimum_matched"],
            minimum_pooled=data["minimum_pooled"],
            key_length=data["key_length"],
            message_bit=data["message_bit"],
            declaration_digest=data["declaration_digest"],
        )


class CountExchange(Protocol):
    """Callable running Phase C'; returns pooled counts, or ``None`` for "did not compare"."""

    def __call__(
        self,
        messages: Mapping[Party | str, MatchedCountMessage],
        params: ProtocolParams,
    ) -> PooledMatchedCounts | None:
        """Return the pooled view both recipients end up holding."""
        ...


def matched_count_message(
    signature: Signature,
    record: RecipientRecord,
    params: ProtocolParams,
) -> MatchedCountMessage:
    """Compute one recipient's Phase C' message from his own record and the declaration.

    Raises
    ------
    TypeError
        If any argument is of the wrong type.
    ValueError
        If the signature and record describe different runs or don't fit ``params``.

    Examples
    --------
    >>> from qsecure.protocol.keys import KeyElement, PrivateKey
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.records import RecipientRecord
    >>> from qsecure.protocol.signature import sign
    >>> from qsecure.protocol.tally import matched_count_message
    >>> params = ProtocolParams(key_length=2)
    >>> key = PrivateKey(0, (KeyElement("X", 1), KeyElement("Z", -1)))
    >>> record = RecipientRecord.from_measurements("Bob", 0, ["X", "Y"], [1, 1])
    >>> message = matched_count_message(sign(0, key, params), record, params)
    >>> message.party.value, message.matched_count, message.key_length
    ('Bob', 1, 2)
    >>> other = RecipientRecord.from_measurements("Charlie", 0, ["X", "Z"], [1, -1])
    >>> message.declaration_digest == matched_count_message(
    ...     sign(0, key, params), other, params
    ... ).declaration_digest
    True
    """
    if not isinstance(params, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(params).__name__}"
        )
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
    signature.check_against(params)
    record.check_against(params)
    return MatchedCountMessage(
        party=record.party,
        message_bit=record.message_bit,
        matched_count=len(matched_positions(signature, record)),
        key_length=params.key_length,
        declaration_digest=_declaration_digest(signature),
    )


def _checked_messages(
    messages: Mapping[Party | str, MatchedCountMessage],
    params: ProtocolParams,
) -> tuple[MatchedCountMessage, MatchedCountMessage]:
    """Validate the mapping and return ``(Bob's message, Charlie's message)``.

    Raises
    ------
    TypeError
        If ``params``, ``messages`` or a value has the wrong type.
    ValueError
        If a verifier is missing or misfiled, there are extra entries, the messages
        disagree with each other or ``params``, or they name different declarations.
    """
    if not isinstance(params, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(params).__name__}; "
            f"the two floors are read off it."
        )
    if not isinstance(messages, Mapping):
        raise TypeError(
            f"messages must be a mapping of Party to MatchedCountMessage, got "
            f"{type(messages).__name__}; build one with "
            f"matched_count_message(...) per recipient."
        )
    resolved: dict[Party, MatchedCountMessage] = {}
    for party, message in messages.items():
        key = _as_party(party)
        if not isinstance(message, MatchedCountMessage):
            raise TypeError(
                f"messages[{key.value!r}] must be a MatchedCountMessage, got "
                f"{type(message).__name__}"
            )
        if message.party is not key:
            raise ValueError(
                f"messages is keyed by {key.value!r} but the message stored "
                f"there was sent by {message.party.value!r}. The key decides "
                f"whose count is whose, so a mislabelled entry would hand each "
                f"verifier his own number back as the counterpart's."
            )
        resolved[key] = message

    missing = [party.value for party in VERIFIERS if party not in resolved]
    if missing:
        raise ValueError(
            f"the count exchange needs both verifiers' messages, missing "
            f"{missing}. The step *is* a comparison between Bob and Charlie: "
            f"with one recipient there is no pooled count, and with one "
            f"recipient the protocol has no transferability and no "
            f"non-repudiation to protect in the first place."
        )
    extra = sorted(set(resolved) - set(VERIFIERS))
    if extra:
        raise ValueError(
            f"the count exchange takes exactly the two verifiers' messages, "
            f"got extra entries for {[party.value for party in extra]}. The "
            f"comparison is defined pairwise."
        )

    bob, charlie = resolved[Party.BOB], resolved[Party.CHARLIE]
    if bob.message_bit != charlie.message_bit:
        raise ValueError(
            f"the two messages are from different runs: Bob's counts a "
            f"declaration for message bit {bob.message_bit} and Charlie's for "
            f"bit {charlie.message_bit}. Only counts against the *same* "
            f"declaration may be pooled -- m_B + m_C is conserved for one "
            f"declaration and means nothing across two."
        )
    for message in (bob, charlie):
        if message.key_length != params.key_length:
            raise ValueError(
                f"{message.party.value}'s message reports key_length "
                f"{message.key_length} but these parameters say "
                f"{params.key_length}. The floors are derived from the key "
                f"length, so a mismatch means the message and the rule come "
                f"from different runs."
            )
    # Same run (bit, length) is not the same declaration.
    if bob.declaration_digest != charlie.declaration_digest:
        raise ValueError(
            f"the two messages counted different declarations: Bob's names "
            f"{bob.declaration_digest} and Charlie's {charlie.declaration_digest}. "
            f"m_B + m_C is conserved across one fixed pair of records against "
            f"one fixed declaration and means nothing across two, so these two "
            f"counts cannot be pooled -- their sum is no run's M. See "
            f"qsecure.protocol.verify on counting against one declaration."
        )
    return bob, charlie


def exchange_matched_counts(
    messages: Mapping[Party | str, MatchedCountMessage],
    params: ProtocolParams,
) -> PooledMatchedCounts:
    """Combine the two recipients' messages into the pooled view (symmetric in the two).

    Raises
    ------
    TypeError
        If ``params`` or ``messages`` is of the wrong type.
    ValueError
        As :func:`_checked_messages`.

    Examples
    --------
    >>> from qsecure.protocol.params import Party, ProtocolParams
    >>> from qsecure.protocol.tally import (
    ...     MatchedCountMessage, exchange_matched_counts
    ... )
    >>> params = ProtocolParams(key_length=600)
    >>> messages = {
    ...     Party.BOB: MatchedCountMessage("Bob", 0, 68, 600, "0f1e"),
    ...     Party.CHARLIE: MatchedCountMessage("Charlie", 0, 66, 600, "0f1e"),
    ... }
    >>> pooled = exchange_matched_counts(messages, params)
    >>> pooled.pooled, pooled.minimum_pooled, pooled.meets_pooled_floor
    (134, 212, False)
    >>> pooled.parties_below_floor        # Charlie is under m_min = 67 too
    (<Party.CHARLIE: 'Charlie'>,)
    """
    bob, charlie = _checked_messages(messages, params)
    return PooledMatchedCounts(
        bob_count=bob.matched_count,
        charlie_count=charlie.matched_count,
        minimum_matched=minimum_matched_count(params),
        minimum_pooled=minimum_pooled_matched_count(params),
        key_length=params.key_length,
        message_bit=bob.message_bit,
        declaration_digest=bob.declaration_digest,
    )


def counterpart_count(message: MatchedCountMessage) -> int:
    """Return a counterpart's message as an ``int`` carrying the digest its sender counted.

    Lets a verifier refuse to pool counts from two declarations instead of crashing.

    Examples
    --------
    >>> from qsecure.protocol.tally import MatchedCountMessage, counterpart_count
    >>> count = counterpart_count(
    ...     MatchedCountMessage("Charlie", 0, 196, 600, "0f1e")
    ... )
    >>> count, count + 4, count == 196
    (196, 200, True)

    It names the sender's declaration:

    >>> count.declaration_digest
    '0f1e'
    """
    if not isinstance(message, MatchedCountMessage):
        raise TypeError(
            f"message must be a MatchedCountMessage, got "
            f"{type(message).__name__}. It is what the counterpart sent over "
            f"Phase C'; build one with matched_count_message(...)."
        )
    return _BoundMatchedCount(
        message.matched_count, message.declaration_digest
    )


def no_count_exchange(
    messages: Mapping[Party | str, MatchedCountMessage],
    params: ProtocolParams,
) -> None:
    """Skip Phase C' and return ``None``. **Leaves the split-coin repudiation route open.**

    Messages are validated exactly as :func:`exchange_matched_counts` does.

    Examples
    --------
    >>> from qsecure.protocol.params import Party, ProtocolParams
    >>> from qsecure.protocol.tally import (
    ...     MatchedCountMessage, no_count_exchange
    ... )
    >>> params = ProtocolParams(key_length=600)
    >>> messages = {
    ...     Party.BOB: MatchedCountMessage("Bob", 0, 68, 600, "0f1e"),
    ...     Party.CHARLIE: MatchedCountMessage("Charlie", 0, 66, 600, "0f1e"),
    ... }
    >>> no_count_exchange(messages, params) is None
    True
    """
    _checked_messages(messages, params)
    return None


def _as_declaration_digest(digest: Any) -> str:
    """Validate a mandatory, non-empty declaration digest."""
    if digest is None:
        raise ValueError(
            "declaration_digest must name the declaration this count was "
            "computed against; it is not optional. A count whose provenance "
            "is unrecorded cannot be shown to be about the declaration a "
            "verifier is scoring, and m_B + m_C is conserved for one fixed "
            "declaration and is a quantity of nothing across two -- so such a "
            "count is refused rather than pooled on trust. Phase C' is the "
            "recipients' own step and a recipient is the adversary here, so "
            "'the exchange is honest' is not available as a defence. Pass "
            "qsecure.protocol.verify._declaration_digest(signature), which "
            "matched_count_message(...) already does."
        )
    if not isinstance(digest, str):
        raise TypeError(
            f"declaration_digest must be a string, got "
            f"{type(digest).__name__}; it is the fingerprint of the "
            f"declaration the count was computed against."
        )
    if not digest:
        raise ValueError(
            "declaration_digest must be a non-empty string. An empty string "
            "is a second spelling of 'not recorded' that compares *unequal* "
            "to every real digest, so it would turn a count that named "
            "nothing into one claiming a declaration nothing hashes to."
        )
    return digest


def _final(cls: type) -> None:
    """Always raise ``TypeError``; used by ``__init_subclass__``.

    A subclass could override :meth:`PooledMatchedCounts.counterpart_of` and drop
    the provenance.
    """
    raise TypeError(
        f"{cls.__name__}: this type may not be subclassed. It is a frozen "
        f"scalar message whose whole job is to carry a count together with "
        f"the declaration it was counted against; a subclass could override "
        f"that pairing and hand a verifier a count of unrecorded provenance, "
        f"which is the omission the mandatory declaration_digest exists to "
        f"refuse. Build one, or convert to and from dict."
    )


def _as_verifier(party: Party | str) -> Party:
    """Coerce a label to Bob or Charlie, refusing Alice."""
    resolved = _as_party(party)
    if resolved is Party.ALICE:
        raise ValueError(
            "Alice holds no matched count: she is the signer and is not a "
            "party to the recipients' exchange. Ask for Party.BOB or "
            "Party.CHARLIE."
        )
    return resolved
