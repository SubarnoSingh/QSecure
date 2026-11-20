"""Quantum state construction and state metrics (fidelity, purity, concurrence).

:func:`as_density` is the shared normaliser for state input and rejects
unphysical matrices. :func:`fidelity` returns the squared Uhlmann fidelity
:math:`F = (\\mathrm{Tr}\\sqrt{\\sqrt{\\rho}\\sigma\\sqrt{\\rho}})^2`, matching
:func:`qiskit.quantum_info.state_fidelity`.
"""

from __future__ import annotations

import enum
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity

__all__ = [
    "BellState",
    "BELL_ORDER",
    "StateLike",
    "bell_state",
    "bell_circuit",
    "as_density",
    "fidelity",
    "purity",
    "concurrence",
    "is_normalised",
]

StateLike: TypeAlias = (
    Statevector
    | DensityMatrix
    | NDArray[np.complexfloating]
    | NDArray[np.floating]
)

_SQRT1_2: Final[float] = float(np.sqrt(0.5))

# Looser than is_normalised's 1e-9: inputs have been through several matrix products.
_VALIDATION_TOL: Final[float] = 1e-8

# Gates fidelity's exact pure-state closed form, so it must be tight.
_PURE_STATE_TOL: Final[float] = 1e-12


_BELL_SORT_ORDER: Final[tuple[str, ...]] = (
    "PHI_PLUS",
    "PSI_PLUS",
    "PHI_MINUS",
    "PSI_MINUS",
)


class BellState(enum.StrEnum):
    """The four Bell states; a ``StrEnum``, so members serialise as their labels.

        >>> import json
        >>> json.dumps({BellState.PHI_PLUS: 2, BellState.PSI_MINUS: 1})
        '{"Phi+": 2, "Psi-": 1}'

    Members sort by :data:`BELL_ORDER`, not lexicographically:

        >>> [b.value for b in sorted(BellState)]
        ['Phi+', 'Psi+', 'Phi-', 'Psi-']
    """

    PHI_PLUS = "Phi+"
    PHI_MINUS = "Phi-"
    PSI_PLUS = "Psi+"
    PSI_MINUS = "Psi-"

    @property
    def order_index(self) -> int:
        """Position in :data:`BELL_ORDER` (``2 * m1 + m0`` of the correction bits)."""
        return _BELL_SORT_ORDER.index(self.name)

    def __lt__(self, other: object) -> bool:
        """Order by :data:`BELL_ORDER`, not by the string value."""
        if isinstance(other, BellState):
            return self.order_index < other.order_index
        return NotImplemented

    def __le__(self, other: object) -> bool:
        """Order by :data:`BELL_ORDER`, not by the string value."""
        if isinstance(other, BellState):
            return self.order_index <= other.order_index
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        """Order by :data:`BELL_ORDER`, not by the string value."""
        if isinstance(other, BellState):
            return self.order_index > other.order_index
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        """Order by :data:`BELL_ORDER`, not by the string value."""
        if isinstance(other, BellState):
            return self.order_index >= other.order_index
        return NotImplemented


# Index is 2 * m1 + m0; the receiver's correction is X**m0 @ Z**m1.
BELL_ORDER: Final[tuple[BellState, ...]] = (
    BellState.PHI_PLUS,
    BellState.PSI_PLUS,
    BellState.PHI_MINUS,
    BellState.PSI_MINUS,
)

# Little-endian index order: [|00>, |01>, |10>, |11>].
_BELL_AMPLITUDES: Final[dict[BellState, tuple[float, float, float, float]]] = {
    BellState.PHI_PLUS: (_SQRT1_2, 0.0, 0.0, _SQRT1_2),
    BellState.PHI_MINUS: (_SQRT1_2, 0.0, 0.0, -_SQRT1_2),
    BellState.PSI_PLUS: (0.0, _SQRT1_2, _SQRT1_2, 0.0),
    BellState.PSI_MINUS: (0.0, _SQRT1_2, -_SQRT1_2, 0.0),
}

# Pauli fixups on qubit 0 after H(0), CX(0, 1), in circuit order; Z before X
# for Psi- avoids a -1 global phase.
_BELL_FIXUP: Final[dict[BellState, tuple[str, ...]]] = {
    BellState.PHI_PLUS: (),
    BellState.PHI_MINUS: ("z",),
    BellState.PSI_PLUS: ("x",),
    BellState.PSI_MINUS: ("z", "x"),
}


def _as_array(state: StateLike) -> NDArray[np.complex128]:
    """Return the raw ``complex128`` array backing ``state`` without validating it."""
    if isinstance(state, (Statevector, DensityMatrix)):
        return np.asarray(state.data, dtype=np.complex128)
    try:
        arr = np.asarray(state, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "state must be a Statevector, a DensityMatrix or a numeric array; "
            f"got {type(state).__name__!r} which could not be converted: {exc}"
        ) from exc
    return arr


def _check_dimension(arr: NDArray[np.complex128]) -> int:
    """Validate a 1-D or square 2-D array and return its power-of-two dimension."""
    if arr.ndim == 1:
        dim = arr.shape[0]
    elif arr.ndim == 2:
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(
                "a density matrix must be square; got shape "
                f"{arr.shape}. Pass a 1-D array for a statevector."
            )
        dim = arr.shape[0]
    else:
        raise ValueError(
            "a state array must be 1-D (statevector) or 2-D (density matrix); "
            f"got {arr.ndim} dimensions with shape {arr.shape}."
        )
    if dim < 2 or (dim & (dim - 1)) != 0:
        raise ValueError(
            f"state dimension must be a power of two and at least 2; got {dim}. "
            "This project only models qubit systems."
        )
    return dim


def _coerce_state(
    state: StateLike, *, tol: float = _VALIDATION_TOL
) -> Statevector | DensityMatrix:
    """Validate state input and return a ``Statevector`` (1-D) or ``DensityMatrix`` (2-D).

    The result owns a copy of the data. Density matrices must have unit trace and be
    Hermitian and positive semidefinite: the metrics clip into ``[0, 1]``, so an
    unphysical input would otherwise be biased towards "pure and undisturbed".

    Raises
    ------
    ValueError
        If the state is non-finite, wrongly shaped, unnormalised or unphysical.
    """
    arr = _as_array(state)
    if not np.isfinite(arr).all():
        raise ValueError("state contains NaN or infinite entries.")

    _check_dimension(arr)

    if arr.ndim == 1:
        norm_sq = float(np.vdot(arr, arr).real)
        if abs(norm_sq - 1.0) > tol:
            raise ValueError(
                f"statevector is not normalised: <psi|psi> = {norm_sq!r}, expected 1 "
                f"within {tol}. Divide by numpy.linalg.norm(amplitudes) first."
            )
        return Statevector(arr.copy())

    trace = complex(np.trace(arr))
    if abs(trace - 1.0) > tol:
        raise ValueError(
            f"density matrix is not normalised: Tr(rho) = {trace!r}, expected 1 "
            f"within {tol}. Divide by numpy.trace(rho) first."
        )
    if not np.allclose(arr, arr.conj().T, atol=tol, rtol=0.0):
        raise ValueError(
            "density matrix is not Hermitian: rho != rho.conj().T within "
            f"{tol}. A physical density operator must be self-adjoint."
        )
    # eigvalsh reads one triangle only, so symmetrise first.
    smallest = float(np.linalg.eigvalsh(0.5 * (arr + arr.conj().T))[0])
    if smallest < -tol:
        raise ValueError(
            "density matrix is not positive semidefinite; smallest eigenvalue "
            f"{smallest!r}, expected >= {-tol}. A physical density operator has "
            "no negative eigenvalues -- check the Kraus operators or mixing "
            "probability that produced it (a depolarising strength outside "
            "[0, 1] is the usual cause)."
        )
    return DensityMatrix(arr.copy())


def _psd_eigenvalues(vals: NDArray[np.floating]) -> NDArray[np.float64]:
    """Snap eigenvalues below ``d * eps * max(lambda, 1)`` to zero before square-rooting.

    ``sqrt`` amplifies rounding debris near zero (``1e-17 -> 3e-9``), which would
    make a product state report a nonzero concurrence.
    """
    real_vals = np.asarray(vals, dtype=np.float64)
    if real_vals.size == 0:
        return real_vals
    scale = max(float(real_vals.max()), 1.0)
    cutoff = scale * real_vals.size * float(np.finfo(np.float64).eps)
    return np.where(real_vals > cutoff, real_vals, 0.0)


def _sqrtm_psd(mat: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Principal square root of a Hermitian PSD matrix via eigendecomposition."""
    vals, vecs = np.linalg.eigh(mat)
    return (vecs * np.sqrt(_psd_eigenvalues(vals))) @ vecs.conj().T


def bell_state(which: BellState = BellState.PHI_PLUS) -> Statevector:
    """Return a Bell state as an exact statevector (amplitudes written out literally).

    Examples
    --------
    >>> bell_state(BellState.PSI_PLUS).data.round(3)
    array([0.   +0.j, 0.707+0.j, 0.707+0.j, 0.   +0.j])
    """
    if not isinstance(which, BellState):
        raise ValueError(
            f"which must be a BellState member, got {which!r}. "
            "Use e.g. BellState.PHI_PLUS."
        )
    return Statevector(np.asarray(_BELL_AMPLITUDES[which], dtype=np.complex128))


def bell_circuit(which: BellState = BellState.PHI_PLUS) -> QuantumCircuit:
    """Return a circuit preparing ``which`` from ``|00>``, matching :func:`bell_state` exactly."""
    if not isinstance(which, BellState):
        raise ValueError(
            f"which must be a BellState member, got {which!r}. "
            "Use e.g. BellState.PHI_PLUS."
        )
    circuit = QuantumCircuit(2, name=f"bell_{which.value}")
    circuit.h(0)
    circuit.cx(0, 1)
    for gate in _BELL_FIXUP[which]:
        getattr(circuit, gate)(0)
    return circuit


def as_density(state: StateLike, *, tol: float = _VALIDATION_TOL) -> DensityMatrix:
    """Convert any accepted state representation to a validated ``DensityMatrix``.

    Raises
    ------
    ValueError
        If the input is malformed, unnormalised, non-Hermitian or not PSD.

    Examples
    --------
    >>> round(float(as_density(bell_state()).data[0, 0].real), 6)
    0.5
    >>> as_density(bell_state()).dims()
    (2, 2)
    """
    coerced = _coerce_state(state, tol=tol)
    if isinstance(coerced, DensityMatrix):
        return coerced
    return DensityMatrix(coerced)


def fidelity(a: StateLike, b: StateLike) -> float:
    """Squared Uhlmann fidelity between two states, clipped into ``[0, 1]``.

    When either state is pure the exact closed form :math:`\\mathrm{Tr}(\\rho\\sigma)`
    is used, so the value does not depend on whether a pure state was passed as a
    ket or a matrix.

    Raises
    ------
    ValueError
        If either state is invalid or the dimensions differ.
    """
    state_a = _coerce_state(a)
    state_b = _coerce_state(b)
    if state_a.dim != state_b.dim:
        raise ValueError(
            "cannot compare states of different dimension: "
            f"{state_a.dim} vs {state_b.dim}. Trace out the extra subsystems "
            "with qiskit.quantum_info.partial_trace first."
        )
    if isinstance(state_a, Statevector) or isinstance(state_b, Statevector):
        value = float(np.real(state_fidelity(state_a, state_b, validate=False)))
    else:
        rho = np.asarray(state_a.data, dtype=np.complex128)
        sigma = np.asarray(state_b.data, dtype=np.complex128)
        if _is_pure_matrix(rho) or _is_pure_matrix(sigma):
            value = float(np.real(np.trace(rho @ sigma)))
        else:
            value = float(
                np.real(state_fidelity(state_a, state_b, validate=False))
            )
    return float(np.clip(value, 0.0, 1.0))


def _is_pure_matrix(rho: NDArray[np.complex128]) -> bool:
    """Return ``True`` when :math:`|\\mathrm{Tr}(\\rho^2) - 1| \\le` :data:`_PURE_STATE_TOL`."""
    return abs(float(np.real(np.trace(rho @ rho))) - 1.0) <= _PURE_STATE_TOL


def purity(state: StateLike) -> float:
    """Return the purity :math:`\\mathrm{Tr}(\\rho^{2})`, clipped into ``[0, 1]``."""
    rho = np.asarray(as_density(state).data, dtype=np.complex128)
    value = float(np.real(np.trace(rho @ rho)))
    return float(np.clip(value, 0.0, 1.0))


def concurrence(state: StateLike) -> float:
    """Wootters concurrence of a two-qubit state, clipped into ``[0, 1]``.

    Eigenvalues come from the Hermitian :math:`\\sqrt{\\rho}\\tilde{\\rho}\\sqrt{\\rho}`,
    which shares its spectrum with :math:`\\rho\\tilde{\\rho}`.

    Raises
    ------
    ValueError
        If the state is invalid or not two-qubit.
    """
    rho = np.asarray(as_density(state).data, dtype=np.complex128)
    if rho.shape != (4, 4):
        raise ValueError(
            "concurrence is only defined for two-qubit states (4x4 density "
            f"matrix); got shape {rho.shape}. Trace the system down to two "
            "qubits with qiskit.quantum_info.partial_trace first."
        )

    sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    yy = np.kron(sigma_y, sigma_y)
    rho_tilde = yy @ rho.conj() @ yy

    sqrt_rho = _sqrtm_psd(rho)
    hermitian_r = sqrt_rho @ rho_tilde @ sqrt_rho
    # Re-symmetrise to kill the ~1e-17 anti-Hermitian residue of the products.
    hermitian_r = 0.5 * (hermitian_r + hermitian_r.conj().T)

    eigenvalues = _psd_eigenvalues(np.linalg.eigvalsh(hermitian_r).real)
    roots = np.sqrt(np.sort(eigenvalues)[::-1])  # descending
    value = float(roots[0] - roots[1] - roots[2] - roots[3])
    return float(np.clip(value, 0.0, 1.0))


def is_normalised(state: StateLike, tol: float = 1e-9) -> bool:
    """Return whether a state has unit norm (1-D) or unit trace (2-D) within ``tol``.

    Never raises for an unnormalised state; still raises for a malformed array or a
    negative ``tol``.
    """
    if tol < 0.0:
        raise ValueError(f"tol must be non-negative, got {tol!r}.")
    arr = _as_array(state)
    _check_dimension(arr)
    if not np.isfinite(arr).all():
        return False
    if arr.ndim == 1:
        return bool(abs(float(np.vdot(arr, arr).real) - 1.0) <= tol)
    return bool(abs(complex(np.trace(arr)) - 1.0) <= tol)
