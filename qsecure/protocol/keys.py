"""Alice's private keys and the quantum public keys derived from them.

A key is ``L`` uniform ``(basis, eigenvalue)`` pairs; the public key is the
product of the matching eigenstates, returned factor by factor and never
materialised. Preparing two copies from a known classical description is not
cloning. A recipient's log has the shape of a key, which is the forgery strategy;
it buys the second verifier's ``1/12`` floor, which is why ``s_v`` sits below it:

>>> from qsecure.protocol.params import DEFAULT_PARAMS
>>> DEFAULT_PARAMS.forger_floor == 1 / 12
True
>>> DEFAULT_PARAMS.s_v < DEFAULT_PARAMS.forger_floor
True
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, overload

import numpy as np
from qiskit.quantum_info import Statevector

from qsecure.core.paulis import PauliBasis, eigenstate, eigenstate_label
from qsecure.core.rng import resolve_rng
from qsecure.protocol.params import (
    ProtocolParams,
    _as_basis,
    _as_eigenvalue,
    _as_message_bit,
)
from qsecure.protocol.records import RecipientRecord

__all__ = [
    "KeyElement",
    "PrivateKey",
    "generate_private_key",
    "generate_key_pair",
    "key_from_record",
    "public_key_states",
]


@dataclass(frozen=True)
class KeyElement:
    """One key position: a basis and a Pauli eigenvalue (``+1``/``-1``, never a bit).

    Examples
    --------
    >>> from qsecure.core.paulis import PauliBasis
    >>> from qsecure.protocol.keys import KeyElement
    >>> element = KeyElement(PauliBasis.Y, -1)
    >>> element.label, element.bit
    ('-i', 1)
    """

    basis: PauliBasis
    eigenvalue: int

    def __post_init__(self) -> None:
        """Coerce ``basis`` to a member and validate ``eigenvalue``."""
        object.__setattr__(self, "basis", _as_basis(self.basis))
        object.__setattr__(self, "eigenvalue", _as_eigenvalue(self.eigenvalue))

    @property
    def bit(self) -> int:
        """int: The classical bit, ``(1 - eigenvalue) // 2``."""
        return (1 - self.eigenvalue) // 2

    @property
    def label(self) -> str:
        """str: The ket label of the state, e.g. ``'+'``, ``'-i'``, ``'0'``."""
        return eigenstate_label(self.basis, self.eigenvalue)

    def state(self) -> Statevector:
        """Return a freshly constructed eigenstate for this element.

        Examples
        --------
        >>> from qsecure.protocol.keys import KeyElement
        >>> KeyElement("X", -1).state().data.round(6)
        array([ 0.707107+0.j, -0.707107+0.j])
        """
        return eigenstate(self.basis, self.eigenvalue)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable ``{"basis", "eigenvalue"}`` view."""
        return {"basis": self.basis, "eigenvalue": self.eigenvalue}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KeyElement:
        """Rebuild an element from :meth:`to_dict` output."""
        return cls(basis=data["basis"], eigenvalue=data["eigenvalue"])


@dataclass(frozen=True)
class PrivateKey:
    """Alice's private key for one message bit: a non-empty immutable tuple of elements.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.protocol.keys import generate_private_key
    >>> from qsecure.protocol.params import ProtocolParams
    >>> key = generate_private_key(
    ...     ProtocolParams(key_length=4), 0, rng=np.random.default_rng(20260141)
    ... )
    >>> len(key), key.message_bit
    (4, 0)
    >>> isinstance(key.elements, tuple)
    True
    """

    message_bit: int
    elements: tuple[KeyElement, ...]

    def __post_init__(self) -> None:
        """Validate the tag and copy the elements into an immutable tuple."""
        object.__setattr__(
            self, "message_bit", _as_message_bit(self.message_bit)
        )
        elements = self.elements
        if isinstance(elements, (str, bytes)) or not isinstance(
            elements, Sequence
        ):
            raise TypeError(
                f"elements must be a sequence of KeyElement, got "
                f"{type(elements).__name__}"
            )
        frozen = tuple(elements)
        for position, element in enumerate(frozen):
            if not isinstance(element, KeyElement):
                raise TypeError(
                    f"elements[{position}] must be a KeyElement, got "
                    f"{type(element).__name__}. Build one with "
                    f"KeyElement(basis, eigenvalue); a bare (basis, eigenvalue) "
                    f"tuple is not accepted, because it has no validation and "
                    f"would let a measurement bit through as an eigenvalue."
                )
        if not frozen:
            raise ValueError(
                "a private key must have at least one element: with no "
                "elements there are no matched positions, the mismatch rate is "
                "0/0, and no verifier can reach a decision."
            )
        object.__setattr__(self, "elements", frozen)


    def __len__(self) -> int:
        """int: The key length ``L``."""
        return len(self.elements)

    @overload
    def __getitem__(self, index: int) -> KeyElement: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[KeyElement, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> KeyElement | tuple[KeyElement, ...]:
        """Return element ``index``, or a tuple for a slice."""
        return self.elements[index]

    def __iter__(self) -> Iterator[KeyElement]:
        """Iterate over the elements in key order."""
        return iter(self.elements)


    @property
    def length(self) -> int:
        """int: The key length ``L``."""
        return len(self.elements)

    @property
    def bases(self) -> tuple[PauliBasis, ...]:
        """tuple of PauliBasis: The declared bases, in key order."""
        return tuple(element.basis for element in self.elements)

    @property
    def eigenvalues(self) -> tuple[int, ...]:
        """tuple of int: The declared eigenvalues, in key order."""
        return tuple(element.eigenvalue for element in self.elements)

    @property
    def bits(self) -> tuple[int, ...]:
        """tuple of int: The declared eigenvalues as classical bits."""
        return tuple(element.bit for element in self.elements)

    @property
    def labels(self) -> tuple[str, ...]:
        """tuple of str: Ket labels of the public-key factors."""
        return tuple(element.label for element in self.elements)

    def check_against(self, params: ProtocolParams) -> None:
        """Raise ``ValueError`` if the key length or bases do not match ``params``.

        Examples
        --------
        >>> import numpy as np
        >>> from qsecure.protocol.keys import generate_private_key
        >>> from qsecure.protocol.params import ProtocolParams
        >>> params = ProtocolParams(key_length=8)
        >>> key = generate_private_key(params, 1, rng=np.random.default_rng(1))
        >>> key.check_against(params) is None
        True
        """
        if len(self.elements) != params.key_length:
            raise ValueError(
                f"private key has {len(self.elements)} elements but "
                f"params.key_length is {params.key_length}. The verifier's "
                f"record has one entry per distributed qubit, so a length "
                f"mismatch means the key and the record came from different "
                f"runs."
            )
        allowed = set(params.bases)
        stray = sorted({b.value for b in self.bases if b not in allowed})
        if stray:
            raise ValueError(
                f"private key uses basis/bases {stray}, which are not in "
                f"params.bases {[b.value for b in params.bases]}. The match "
                f"probability, and therefore every security bound, is computed "
                f"from the declared alphabet."
            )


    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the key."""
        return {
            "message_bit": self.message_bit,
            "elements": [element.to_dict() for element in self.elements],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PrivateKey:
        """Rebuild a key from :meth:`to_dict` output."""
        return cls(
            message_bit=data["message_bit"],
            elements=tuple(
                KeyElement.from_dict(item) for item in data["elements"]
            ),
        )


def generate_private_key(
    params: ProtocolParams,
    message_bit: int,
    *,
    rng: np.random.Generator | None = None,
) -> PrivateKey:
    """Draw one private key uniformly (two variates per element: basis, then eigenvalue).

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.protocol.keys import generate_private_key
    >>> from qsecure.protocol.params import ProtocolParams
    >>> params = ProtocolParams(key_length=6)
    >>> first = generate_private_key(params, 0, rng=np.random.default_rng(7))
    >>> second = generate_private_key(params, 0, rng=np.random.default_rng(7))
    >>> first == second
    True
    """
    if not isinstance(params, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(params).__name__}"
        )
    bit = _as_message_bit(message_bit)
    generator = resolve_rng(rng)
    alphabet = params.bases
    elements = []
    for _ in range(params.key_length):
        basis = alphabet[int(generator.integers(len(alphabet)))]
        eigenvalue = 1 - 2 * int(generator.integers(2))
        elements.append(KeyElement(basis, eigenvalue))
    return PrivateKey(message_bit=bit, elements=tuple(elements))


def generate_key_pair(
    params: ProtocolParams, *, rng: np.random.Generator | None = None
) -> tuple[PrivateKey, PrivateKey]:
    """Draw ``(key_for_bit_0, key_for_bit_1)`` from one threaded generator.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.protocol.keys import generate_key_pair
    >>> from qsecure.protocol.params import ProtocolParams
    >>> keys = generate_key_pair(
    ...     ProtocolParams(key_length=32), rng=np.random.default_rng(20260141)
    ... )
    >>> [key.message_bit for key in keys]
    [0, 1]
    >>> keys[0].elements == keys[1].elements
    False
    """
    generator = resolve_rng(rng)
    return (
        generate_private_key(params, 0, rng=generator),
        generate_private_key(params, 1, rng=generator),
    )


def public_key_states(key: PrivateKey) -> tuple[Statevector, ...]:
    """Return the public key as ``L`` freshly prepared single-qubit factors.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.core.paulis import PauliBasis, pauli_matrix
    >>> from qsecure.protocol.keys import KeyElement, PrivateKey, public_key_states
    >>> key = PrivateKey(0, (KeyElement(PauliBasis.Y, -1), KeyElement("Z", 1)))
    >>> states = public_key_states(key)
    >>> vector = states[0].data
    >>> bool(np.allclose(pauli_matrix("Y") @ vector, -1 * vector))
    True
    """
    if not isinstance(key, PrivateKey):
        raise TypeError(
            f"key must be a PrivateKey, got {type(key).__name__}. Draw one with "
            f"generate_private_key(params, message_bit, rng=...)."
        )
    return tuple(element.state() for element in key.elements)


def key_from_record(
    record: RecipientRecord, *, message_bit: int | None = None
) -> PrivateKey:
    """Read a recipient's measurement log as a private key (the recipient-forgery declaration).

    Element ``i`` is ``(chosen basis, measured eigenvalue)``. ``message_bit``
    defaults to the record's own. A forging recipient should pass his pre-exchange
    ``raw_record``.

    Examples
    --------
    >>> from qsecure.protocol.keys import key_from_record
    >>> from qsecure.protocol.records import RecipientRecord
    >>> record = RecipientRecord.from_measurements(
    ...     "Bob", 1, ["X", "Z", "Y"], [1, -1, -1]
    ... )
    >>> declared = key_from_record(record)
    >>> declared.message_bit, len(declared)
    (1, 3)
    >>> declared.bases == record.bases
    True
    >>> declared.eigenvalues == record.eigenvalues
    True

    Eigenvalues, not bits:

    >>> declared.eigenvalues, record.bits
    ((1, -1, -1), (0, 1, 1))
    """
    if not isinstance(record, RecipientRecord):
        raise TypeError(
            f"record must be a RecipientRecord, got {type(record).__name__}. "
            f"This function reads one recipient's measurement log as a key; "
            f"pass view.raw_record for a forging recipient, or "
            f"session.records[bit][party] if you are the harness."
        )
    bit = (
        record.message_bit
        if message_bit is None
        else _as_message_bit(message_bit)
    )
    return PrivateKey(
        message_bit=bit,
        elements=tuple(
            KeyElement(entry.basis, entry.eigenvalue) for entry in record
        ),
    )
