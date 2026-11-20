"""Phase B: Alice signs bit ``b`` by declaring the whole private key ``k_b``.

Publishing the key is safe because the recipients measured before it was
declared. The channel must be authenticated, not secret. No uncovered free-text
label is carried; :attr:`Signature.context` is allowed because the session id covers it.

.. _session-binding:

Session binding: ``session_id = blake2b(tag | bit | L | opening | context)``,
announced in Phase A, stamped on records, recomputed at verification. The
declared key is deliberately excluded, so forgeries are rejected on rate rather
than refused as the wrong round.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from qsecure.core.paulis import PauliBasis
from qsecure.core.rng import resolve_rng
from qsecure.protocol.keys import KeyElement, PrivateKey
from qsecure.protocol.params import ProtocolParams, _as_message_bit

__all__ = [
    "Signature",
    "fresh_context",
    "fresh_opening",
    "session_identifier",
    "sign",
]


_OPENING_BYTES: Final[int] = 16
"""int: Width of a session opening (128 bits)."""

_SESSION_ID_BYTES: Final[int] = 16
"""int: Bytes of blake2b output kept in a session identifier."""

_SESSION_TAG: Final[bytes] = b"qsecure.protocol.signature/session-id/v1"
"""bytes: Domain separator for :func:`session_identifier`."""


def session_identifier(
    message_bit: int,
    key_length: int,
    *,
    opening: str,
    context: str | None = None,
) -> str:
    """Return the 32-hex-character identifier of one distribution round.

    The declared key is not an input. ``context=None`` and ``context=""`` differ.
    The encoding is length-prefixed and injective.

    Raises
    ------
    TypeError
        If ``opening`` is not a string or ``context`` is neither string nor ``None``.
    ValueError
        If ``message_bit`` is not 0/1, ``key_length`` is not positive, or ``opening``
        is empty.

    Examples
    --------
    >>> from qsecure.protocol.signature import session_identifier
    >>> first = session_identifier(0, 192, opening="a3f1")
    >>> len(first), first == session_identifier(0, 192, opening="a3f1")
    (32, True)
    >>> first == session_identifier(1, 192, opening="a3f1")
    False
    >>> first == session_identifier(0, 192, opening="a3f2")
    False

    The context is covered:

    >>> labelled = session_identifier(0, 192, opening="a3f1", context="pay 10")
    >>> labelled in (first, session_identifier(
    ...     0, 192, opening="a3f1", context="pay 10000"
    ... ))
    False

    ``None`` and ``""`` differ:

    >>> session_identifier(0, 192, opening="a3f1", context="") == first
    False
    """
    bit = _as_message_bit(message_bit)
    length = _as_key_length(key_length)
    if not isinstance(opening, str):
        raise TypeError(
            f"opening must be a string, got {type(opening).__name__}; "
            f"fresh_opening(rng=...) draws one."
        )
    if not opening:
        raise ValueError(
            "opening must be non-empty. It is the secret this round's "
            "identifier commits to, and an empty one commits to nothing: two "
            "rounds would share an identifier and a replayed declaration would "
            "pair with the wrong records."
        )
    if context is not None and not isinstance(context, str):
        raise TypeError(
            f"context must be a string or None, got {type(context).__name__}"
        )
    opening_bytes = opening.encode("utf-8")
    # A leading marker byte separates "no context" from the empty context, and a
    # length prefix on each variable-length field keeps the encoding injective.
    if context is None:
        context_bytes = b"\x00"
    else:
        context_bytes = b"\x01" + context.encode("utf-8")
    payload = b"|".join(
        (
            _SESSION_TAG,
            str(bit).encode("ascii"),
            str(length).encode("ascii"),
            str(len(opening_bytes)).encode("ascii"),
            opening_bytes,
            str(len(context_bytes)).encode("ascii"),
            context_bytes,
        )
    )
    return hashlib.blake2b(payload, digest_size=_SESSION_ID_BYTES).hexdigest()


def _as_key_length(value: Any) -> int:
    """Coerce and range-check a key length (a non-bool integer >= 1)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"key_length must be a positive integer, got {value!r} of type "
            f"{type(value).__name__}"
        )
    length = int(value)
    if length < 1:
        raise ValueError(
            f"key_length must be at least 1, got {length}; a run with no key "
            f"positions distributes nothing and has no round to identify."
        )
    return length


def fresh_opening(*, rng: np.random.Generator | None = None) -> str:
    """Draw a secret 128-bit opening (32 hex chars) for one distribution round.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.protocol.signature import fresh_opening
    >>> opening = fresh_opening(rng=np.random.default_rng(0))
    >>> len(opening), opening == fresh_opening(rng=np.random.default_rng(0))
    (32, True)
    >>> opening == fresh_opening(rng=np.random.default_rng(1))
    False
    """
    return resolve_rng(rng).bytes(_OPENING_BYTES).hex()


def fresh_context(instruction: str, nonce: str) -> str:
    """Return ``"<len>:<instruction>#<nonce>"`` for :attr:`Signature.context`.

    The nonce is supplied by the application; a distinct nonce makes a distinct
    round, so one signed instruction cannot be executed twice.

    Raises
    ------
    TypeError
        If either argument is not a string.
    ValueError
        If ``nonce`` is empty.

    Examples
    --------
    >>> from qsecure.protocol.signature import fresh_context
    >>> fresh_context("pay 10", "2026-08-31T09:00:00Z")
    '6:pay 10#2026-08-31T09:00:00Z'
    >>> fresh_context("pay 10", "a") == fresh_context("pay 10", "b")
    False

    Length prefixing prevents collisions:

    >>> fresh_context("pay", "10#x") == fresh_context("pay#10", "x")
    False
    """
    if not isinstance(instruction, str):
        raise TypeError(
            f"instruction must be a string, got {type(instruction).__name__}"
        )
    if not isinstance(nonce, str):
        raise TypeError(f"nonce must be a string, got {type(nonce).__name__}")
    if not nonce:
        raise ValueError(
            "nonce must be non-empty: it is the whole freshness content of the "
            "context, and an empty one makes two authorisations of the same "
            "instruction indistinguishable."
        )
    return f"{len(instruction)}:{instruction}#{nonce}"


@dataclass(frozen=True)
class Signature:
    """Phase B declaration: a message bit, the key claimed for it, and optional session fields.

    ``session_opening=None`` is an unbound declaration. ``declared_key.message_bit``
    must equal ``message_bit``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.protocol.keys import generate_key_pair
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.signature import sign
    >>> params = ProtocolParams(key_length=9)
    >>> keys = generate_key_pair(params, rng=np.random.default_rng(20260141))
    >>> signature = sign(1, keys, params)
    >>> signature.message_bit, len(signature)
    (1, 9)
    >>> signature.declared_key is keys[1]
    True

    Unbound vs bound:

    >>> signature.session_id is None
    True
    >>> bound = sign(1, keys, params, session_opening="9c2e", context="pay 10")
    >>> len(bound.session_id)
    32
    >>> bound.session_id == sign(
    ...     1, keys, params, session_opening="9c2e", context="pay 11"
    ... ).session_id
    False
    """

    message_bit: int
    declared_key: PrivateKey
    session_opening: str | None = None
    context: str | None = None

    def __post_init__(self) -> None:
        """Validate the bit, the key's own bit, and the session fields."""
        object.__setattr__(
            self, "message_bit", _as_message_bit(self.message_bit)
        )
        if self.session_opening is not None and not isinstance(
            self.session_opening, str
        ):
            raise TypeError(
                f"session_opening must be a string or None, got "
                f"{type(self.session_opening).__name__}; fresh_opening(rng=...) "
                f"draws one and QDSSession.distribute() keeps it."
            )
        if self.session_opening == "":
            raise ValueError(
                "session_opening must be non-empty or None. An empty opening "
                "would commit to nothing, so every round would share one "
                "identifier and a replayed declaration would pair with the "
                "wrong records; declare no opening instead, which says "
                "honestly that this declaration names no round."
            )
        if self.context is not None and not isinstance(self.context, str):
            raise TypeError(
                f"context must be a string or None, got "
                f"{type(self.context).__name__}; it is the application's "
                f"instruction-and-nonce string, not a structured object."
            )
        if not isinstance(self.declared_key, PrivateKey):
            raise TypeError(
                f"declared_key must be a PrivateKey, got "
                f"{type(self.declared_key).__name__}. The signature *is* the "
                f"private key for the signed bit; build one with "
                f"generate_private_key(params, message_bit, rng=...)."
            )
        if self.declared_key.message_bit != self.message_bit:
            raise ValueError(
                f"cannot sign message bit {self.message_bit} with the key drawn "
                f"for bit {self.declared_key.message_bit}. A run distributes "
                f"quantum states for k_0 and k_1 separately, so a recipient's "
                f"record for one bit has no relationship to the other key: "
                f"declaring the wrong one gives a mismatch rate near "
                f"(1 - 1/|B|)/2 at both verifiers and looks like a channel "
                f"problem rather than a mix-up. Use sign(message_bit, keys) and "
                f"let it pick."
            )


    def __len__(self) -> int:
        """int: The declared key length ``L``."""
        return len(self.declared_key)

    def __iter__(self) -> Iterator[KeyElement]:
        """Iterate over the declared key elements, in key order."""
        return iter(self.declared_key)

    @property
    def length(self) -> int:
        """int: The declared key length ``L``."""
        return len(self.declared_key)

    @property
    def bases(self) -> tuple[PauliBasis, ...]:
        """tuple of PauliBasis: The declared bases, in key order."""
        return self.declared_key.bases

    @property
    def eigenvalues(self) -> tuple[int, ...]:
        """tuple of int: The declared eigenvalues, in key order."""
        return self.declared_key.eigenvalues

    @property
    def bits(self) -> tuple[int, ...]:
        """tuple of int: The declared eigenvalues as ``0``/``1`` bits."""
        return self.declared_key.bits

    @property
    def session_id(self) -> str | None:
        """str or None: :func:`session_identifier` of the opening and context; not settable.

        Examples
        --------
        >>> from qsecure.protocol.keys import KeyElement, PrivateKey
        >>> from qsecure.protocol.signature import Signature
        >>> key = PrivateKey(0, (KeyElement("X", 1),))
        >>> Signature(0, key).session_id is None
        True
        >>> Signature(0, key, session_opening="7b").session_id == Signature(
        ...     0, key, session_opening="7b"
        ... ).session_id
        True

        Independent of the declared key:

        >>> forged = PrivateKey(0, (KeyElement("X", -1),))
        >>> Signature(0, key, session_opening="7b").session_id == Signature(
        ...     0, forged, session_opening="7b"
        ... ).session_id
        True
        """
        if self.session_opening is None:
            return None
        return session_identifier(
            self.message_bit,
            len(self.declared_key),
            opening=self.session_opening,
            context=self.context,
        )

    def check_against(self, params: ProtocolParams) -> None:
        """Raise ``ValueError`` if the declared key does not belong to ``params``.

        Examples
        --------
        >>> import numpy as np
        >>> from qsecure.protocol.keys import generate_private_key
        >>> from qsecure.protocol.params import ProtocolParams
        >>> from qsecure.protocol.signature import sign
        >>> params = ProtocolParams(key_length=5)
        >>> key = generate_private_key(params, 0, rng=np.random.default_rng(1))
        >>> sign(0, key).check_against(params) is None
        True
        """
        self.declared_key.check_against(params)


    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view; ``session_id`` is derived, so not stored.

        Examples
        --------
        >>> import json
        >>> import numpy as np
        >>> from qsecure.protocol.keys import generate_private_key
        >>> from qsecure.protocol.params import ProtocolParams
        >>> from qsecure.protocol.signature import Signature, sign
        >>> key = generate_private_key(
        ...     ProtocolParams(key_length=3), 1, rng=np.random.default_rng(2)
        ... )
        >>> blob = json.dumps(sign(1, key).to_dict())
        >>> Signature.from_dict(json.loads(blob)) == sign(1, key)
        True
        >>> sorted(sign(1, key).to_dict())
        ['context', 'declared_key', 'message_bit', 'session_opening']

        A bound declaration round-trips to the same round:

        >>> bound = sign(1, key, session_opening="4d", context="pay 10")
        >>> restored = Signature.from_dict(json.loads(json.dumps(bound.to_dict())))
        >>> restored == bound and restored.session_id == bound.session_id
        True
        """
        return {
            "message_bit": self.message_bit,
            "declared_key": self.declared_key.to_dict(),
            "session_opening": self.session_opening,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Signature:
        """Rebuild a signature from :meth:`to_dict` output (session fields default to ``None``)."""
        return cls(
            message_bit=data["message_bit"],
            declared_key=PrivateKey.from_dict(data["declared_key"]),
            session_opening=data.get("session_opening"),
            context=data.get("context"),
        )


def sign(
    message_bit: int,
    keys: PrivateKey | Sequence[PrivateKey],
    params: ProtocolParams | None = None,
    *,
    session_opening: str | None = None,
    context: str | None = None,
) -> Signature:
    """Sign one bit by declaring its key.

    ``keys`` is either the single key for the bit or the ``(k_0, k_1)`` pair, from
    which ``keys[message_bit]`` is selected. If ``params`` is given the key is
    checked against it.

    Raises
    ------
    TypeError
        If ``keys`` or ``params`` has the wrong type, or a session field is not a
        string or ``None``.
    ValueError
        If the bit is invalid, the pair is not ``(k_0, k_1)``, the key is tagged for
        the other bit, or it does not belong to ``params``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.protocol.keys import generate_key_pair
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.signature import sign
    >>> params = ProtocolParams(key_length=12)
    >>> keys = generate_key_pair(params, rng=np.random.default_rng(7))
    >>> sign(0, keys, params).declared_key is keys[0]
    True
    >>> sign(1, keys, params).declared_key is keys[1]
    True

    Naming the round does not change the key:

    >>> bound = sign(0, keys, params, session_opening="e01d")
    >>> bound.declared_key is keys[0], len(bound.session_id)
    (True, 32)
    """
    bit = _as_message_bit(message_bit)

    if isinstance(keys, PrivateKey):
        selected = keys
    elif isinstance(keys, (str, bytes)) or not isinstance(keys, Sequence):
        raise TypeError(
            f"keys must be a PrivateKey or the (k_0, k_1) pair from "
            f"generate_key_pair, got {type(keys).__name__}"
        )
    else:
        pair = tuple(keys)
        for position, candidate in enumerate(pair):
            if not isinstance(candidate, PrivateKey):
                raise TypeError(
                    f"keys[{position}] must be a PrivateKey, got "
                    f"{type(candidate).__name__}"
                )
        if len(pair) != 2:
            raise ValueError(
                f"keys must hold exactly the two keys a run commits to, "
                f"(k_0, k_1), got {len(pair)}. Alice draws both before she knows "
                f"which bit she will sign, and distributes quantum states for "
                f"both; a run with one key can only ever sign one predetermined "
                f"bit, which is not a signature scheme."
            )
        tags = tuple(candidate.message_bit for candidate in pair)
        if tags != (0, 1):
            raise ValueError(
                f"keys must be ordered (key_for_bit_0, key_for_bit_1); the pair "
                f"given is tagged {tags}. Index it with the message bit -- "
                f"keys[b] is the key for bit b -- which is what "
                f"generate_key_pair returns."
            )
        selected = pair[bit]

    if selected.message_bit != bit:
        raise ValueError(
            f"cannot sign message bit {bit} with the key drawn for bit "
            f"{selected.message_bit}. Pass the (k_0, k_1) pair instead of a "
            f"single key and let sign() select, or draw the key with "
            f"generate_private_key(params, {bit}, rng=...)."
        )

    if params is not None:
        if not isinstance(params, ProtocolParams):
            raise TypeError(
                f"params must be a ProtocolParams or None, got "
                f"{type(params).__name__}"
            )
        selected.check_against(params)

    return Signature(
        message_bit=bit,
        declared_key=selected,
        session_opening=session_opening,
        context=context,
    )
