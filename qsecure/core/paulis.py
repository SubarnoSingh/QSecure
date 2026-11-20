"""Pauli algebra and the three single-qubit measurement bases.

Defines the Pauli matrices, the six basis eigenstates and the rotations that map
each basis onto Z. ``eigenvalue`` arguments are :math:`\\pm 1`, never bits
(``+1 -> 0``, ``-1 -> 1``). Rotations: :math:`U_X = H`, :math:`U_Y = H S^{\\dagger}`
(circuit order ``sdg`` then ``h``), :math:`U_Z = I`.
"""

from __future__ import annotations

import enum
from typing import Final

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qsecure.core.rng import resolve_rng

__all__ = [
    "PauliBasis",
    "PAULI_I",
    "PAULI_X",
    "PAULI_Y",
    "PAULI_Z",
    "pauli_matrix",
    "eigenstate",
    "eigenstate_label",
    "BASIS_CHANGE",
    "basis_change_circuit",
    "random_basis",
    "verify_eigenstate",
]

ComplexMatrix = NDArray[np.complex128]


class PauliBasis(enum.StrEnum):
    """A single-qubit measurement basis; a ``StrEnum``, so members are their labels.

        >>> import json
        >>> json.dumps({PauliBasis.Z: 12, PauliBasis.X: 4})
        '{"Z": 12, "X": 4}'

    Members sort ``X < Y < Z``, the order :func:`random_basis` draws from.

        >>> [b.value for b in sorted(PauliBasis)]
        ['X', 'Y', 'Z']
    """

    X = "X"
    Y = "Y"
    Z = "Z"


def _make_pauli(rows: list[list[complex]]) -> ComplexMatrix:
    """Build a read-only 2x2 complex128 matrix for a module constant."""
    matrix = np.array(rows, dtype=np.complex128)
    matrix.setflags(write=False)
    return matrix


PAULI_I: Final[ComplexMatrix] = _make_pauli([[1, 0], [0, 1]])
"""Identity. Read-only; use :func:`pauli_matrix` for a writeable copy."""

PAULI_X: Final[ComplexMatrix] = _make_pauli([[0, 1], [1, 0]])

PAULI_Y: Final[ComplexMatrix] = _make_pauli([[0, -1j], [1j, 0]])

PAULI_Z: Final[ComplexMatrix] = _make_pauli([[1, 0], [0, -1]])

_PAULI_BY_LABEL: Final[dict[str, ComplexMatrix]] = {
    "I": PAULI_I,
    "X": PAULI_X,
    "Y": PAULI_Y,
    "Z": PAULI_Z,
}

_INV_SQRT2: Final[float] = float(1.0 / np.sqrt(2.0))

_BASIS_ORDER: Final[tuple[PauliBasis, ...]] = (
    PauliBasis.X,
    PauliBasis.Y,
    PauliBasis.Z,
)

_EIGENSTATE_AMPLITUDES: Final[dict[tuple[PauliBasis, int], tuple[complex, complex]]] = {
    (PauliBasis.Z, 1): (1.0 + 0.0j, 0.0 + 0.0j),
    (PauliBasis.Z, -1): (0.0 + 0.0j, 1.0 + 0.0j),
    (PauliBasis.X, 1): (_INV_SQRT2 + 0.0j, _INV_SQRT2 + 0.0j),
    (PauliBasis.X, -1): (_INV_SQRT2 + 0.0j, -_INV_SQRT2 + 0.0j),
    (PauliBasis.Y, 1): (_INV_SQRT2 + 0.0j, 1j * _INV_SQRT2),
    (PauliBasis.Y, -1): (_INV_SQRT2 + 0.0j, -1j * _INV_SQRT2),
}

_EIGENSTATE_LABELS: Final[dict[tuple[PauliBasis, int], str]] = {
    (PauliBasis.Z, 1): "0",
    (PauliBasis.Z, -1): "1",
    (PauliBasis.X, 1): "+",
    (PauliBasis.X, -1): "-",
    (PauliBasis.Y, 1): "+i",
    (PauliBasis.Y, -1): "-i",
}


def _as_basis(basis: PauliBasis | str) -> PauliBasis:
    """Coerce a :class:`PauliBasis` or case-insensitive string to :class:`PauliBasis`."""
    if isinstance(basis, PauliBasis):
        return basis
    if isinstance(basis, str):
        try:
            return PauliBasis(basis.strip().upper())
        except ValueError:
            raise ValueError(
                f"unknown measurement basis {basis!r}; expected one of "
                f"'X', 'Y', 'Z' (case-insensitive) or a PauliBasis member"
            ) from None
    raise TypeError(
        f"basis must be a PauliBasis (or the string 'X', 'Y', 'Z'), got "
        f"{type(basis).__name__}"
    )


def _as_eigenvalue(eigenvalue: int) -> int:
    """Validate a Pauli eigenvalue (``+1``/``-1``; bools and bits rejected)."""
    if isinstance(eigenvalue, bool):
        raise ValueError(
            f"eigenvalue must be the Pauli eigenvalue +1 or -1, got the boolean "
            f"{eigenvalue!r}. Booleans look like measurement bits; convert with "
            f"eigenvalue = 1 - 2 * bit (bit 0 -> +1, bit 1 -> -1)."
        )
    if eigenvalue == 1:
        return 1
    if eigenvalue == -1:
        return -1
    raise ValueError(
        f"eigenvalue must be +1 or -1, got {eigenvalue!r}. These are Pauli "
        f"eigenvalues, not measurement bits; convert with "
        f"eigenvalue = 1 - 2 * bit (bit 0 -> +1, bit 1 -> -1)."
    )


def pauli_matrix(label: str) -> ComplexMatrix:
    """Return a fresh writeable 2x2 matrix for ``"I"``, ``"X"``, ``"Y"`` or ``"Z"``.

    Multi-qubit strings are rejected; use :class:`qiskit.quantum_info.Pauli`.

    Examples
    --------
    >>> from qsecure.core.paulis import pauli_matrix
    >>> pauli_matrix("z")
    array([[ 1.+0.j,  0.+0.j],
           [ 0.+0.j, -1.+0.j]])
    """
    if not isinstance(label, str):
        raise TypeError(
            f"label must be a string from 'IXYZ', got {type(label).__name__}"
        )
    key = label.strip().upper()
    if key not in _PAULI_BY_LABEL:
        raise ValueError(
            f"unknown Pauli label {label!r}; expected a single character from "
            f"'IXYZ' (case-insensitive). Multi-qubit Pauli strings are not "
            f"handled here: use qiskit.quantum_info.Pauli instead."
        )
    return _PAULI_BY_LABEL[key].copy()


def eigenstate(basis: PauliBasis, eigenvalue: int) -> Statevector:
    """Return the eigenvector of ``basis`` for ``eigenvalue`` (``<0|psi>`` real, positive).

    Examples
    --------
    >>> from qsecure.core.paulis import PauliBasis, eigenstate
    >>> eigenstate(PauliBasis.Z, -1).data
    array([0.+0.j, 1.+0.j])
    """
    key = (_as_basis(basis), _as_eigenvalue(eigenvalue))
    amplitudes = _EIGENSTATE_AMPLITUDES[key]
    return Statevector(np.array(amplitudes, dtype=np.complex128))


def eigenstate_label(basis: PauliBasis, eigenvalue: int) -> str:
    """Return the ket label: ``"0"``, ``"1"``, ``"+"``, ``"-"``, ``"+i"`` or ``"-i"``.

    Examples
    --------
    >>> from qsecure.core.paulis import PauliBasis, eigenstate_label
    >>> eigenstate_label(PauliBasis.Y, -1)
    '-i'
    """
    return _EIGENSTATE_LABELS[(_as_basis(basis), _as_eigenvalue(eigenvalue))]


def _build_basis_change(basis: PauliBasis) -> QuantumCircuit:
    """Construct the rotation ``U`` with ``U|b_{+1}> = |0>``, ``U|b_{-1}> = |1>``."""
    circuit = QuantumCircuit(1, name=f"basis_change_{basis.value}")
    if basis is PauliBasis.X:
        circuit.h(0)
    elif basis is PauliBasis.Y:
        # S instead of S-dagger would swap the two outcomes.
        circuit.sdg(0)
        circuit.h(0)
    return circuit


BASIS_CHANGE: Final[dict[PauliBasis, QuantumCircuit]] = {
    basis: _build_basis_change(basis) for basis in _BASIS_ORDER
}
"""Shared basis-change templates; do not mutate, use :func:`basis_change_circuit`."""


def basis_change_circuit(basis: PauliBasis) -> QuantumCircuit:
    """Return a fresh circuit rotating ``basis`` onto the computational basis.

    Examples
    --------
    >>> from qsecure.core.paulis import PauliBasis, basis_change_circuit
    >>> [inst.operation.name for inst in basis_change_circuit(PauliBasis.Y).data]
    ['sdg', 'h']
    """
    return BASIS_CHANGE[_as_basis(basis)].copy()


def random_basis(*, rng: np.random.Generator | None = None) -> PauliBasis:
    """Draw a basis uniformly from ``{X, Y, Z}``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.core.paulis import random_basis
    >>> random_basis(rng=np.random.default_rng(7)) == random_basis(
    ...     rng=np.random.default_rng(7)
    ... )
    True
    """
    generator = resolve_rng(rng)
    return _BASIS_ORDER[int(generator.integers(len(_BASIS_ORDER)))]


def verify_eigenstate(
    basis: PauliBasis, eigenvalue: int, tol: float = 1e-9
) -> bool:
    """Check numerically that :func:`eigenstate` satisfies :math:`P\\psi = \\lambda\\psi`.

    Examples
    --------
    >>> from qsecure.core.paulis import PauliBasis, verify_eigenstate
    >>> all(
    ...     verify_eigenstate(basis, value)
    ...     for basis in PauliBasis
    ...     for value in (1, -1)
    ... )
    True
    """
    resolved_basis = _as_basis(basis)
    value = _as_eigenvalue(eigenvalue)
    if not tol > 0:
        raise ValueError(f"tol must be a positive tolerance, got {tol!r}")
    vector = eigenstate(resolved_basis, value).data
    observable = _PAULI_BY_LABEL[resolved_basis.value]
    return bool(
        np.allclose(observable @ vector, value * vector, atol=float(tol), rtol=0.0)
    )
