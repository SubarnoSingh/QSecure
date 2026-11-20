"""Quantum state teleportation.

:func:`teleport_in_register` moves one qubit of a register, keeping its
correlations; :func:`teleport` is the one-qubit case (payload on qubit 0,
resource on qubits 1-2, receiver qubit 2); :func:`teleport_channel` gives the
exact induced Pauli channel. The receiver applies :math:`X^{m_0} Z^{m_1}` with
:math:`2 m_1 + m_0` = ``BELL_ORDER.index(outcome)``. A Werner resource with
noise ``p`` gives fidelity :math:`1 - p/2`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import DensityMatrix, Kraus, Statevector, partial_trace

from qsecure.core.measure import bell_measure
from qsecure.core.paulis import PAULI_X, PAULI_Z
from qsecure.core.rng import resolve_rng
from qsecure.core.states import (
    BELL_ORDER,
    BellState,
    StateLike,
    as_density,
    bell_state,
)
from qsecure.core.states import fidelity as state_fidelity

from qsecure.core.states import _coerce_state, _psd_eigenvalues

from qsecure.core.measure import _embed_single

__all__ = [
    "TeleportationResult",
    "RegisterTeleportationResult",
    "pauli_correction",
    "correction_bits",
    "teleport",
    "teleport_in_register",
    "teleport_channel",
    "teleport_circuit",
]

ComplexMatrix = NDArray[np.complex128]

_PAYLOAD_QUBIT: Final[int] = 0

_SENDER_QUBIT: Final[int] = 1

_RECEIVER_QUBIT: Final[int] = 2


def _validate_outcome(bell_outcome: BellState) -> BellState:
    """Reject anything that is not a :class:`BellState` (strings are not coerced)."""
    if not isinstance(bell_outcome, BellState):
        raise ValueError(
            f"bell_outcome must be a BellState member, got {bell_outcome!r}. "
            "Use e.g. BellState.PHI_PLUS; strings are not coerced because a "
            "typo would silently select the wrong Pauli correction."
        )
    return bell_outcome


def correction_bits(bell_outcome: BellState) -> tuple[int, int]:
    """Return the transmitted bits ``(m0, m1)``: the ``X`` and ``Z`` exponents.

    Examples
    --------
    >>> from qsecure.core.states import BellState
    >>> correction_bits(BellState.PHI_PLUS)
    (0, 0)
    >>> correction_bits(BellState.PSI_MINUS)
    (1, 1)
    """
    index = BELL_ORDER.index(_validate_outcome(bell_outcome))
    return index & 1, (index >> 1) & 1


def _build_correction(bell_outcome: BellState) -> ComplexMatrix:
    """Construct the read-only :math:`X^{m_0} Z^{m_1}` (``Z`` acts first)."""
    m0, m1 = correction_bits(bell_outcome)
    x_part = np.linalg.matrix_power(np.asarray(PAULI_X, dtype=np.complex128), m0)
    z_part = np.linalg.matrix_power(np.asarray(PAULI_Z, dtype=np.complex128), m1)
    matrix: ComplexMatrix = np.asarray(x_part @ z_part, dtype=np.complex128)
    matrix.setflags(write=False)
    return matrix


_CORRECTIONS: Final[dict[BellState, ComplexMatrix]] = {
    which: _build_correction(which) for which in BELL_ORDER
}


def pauli_correction(bell_outcome: BellState) -> ComplexMatrix:
    """Return a fresh copy of the 2x2 correction :math:`X^{m_0} Z^{m_1}` for a Bell outcome.

    Examples
    --------
    >>> from qsecure.core.states import BellState
    >>> pauli_correction(BellState.PHI_PLUS)
    array([[1.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j]])
    >>> pauli_correction(BellState.PSI_MINUS)
    array([[ 0.+0.j, -1.+0.j],
           [ 1.+0.j,  0.+0.j]])
    """
    return _CORRECTIONS[_validate_outcome(bell_outcome)].copy()


@dataclass(frozen=True)
class TeleportationResult:
    """Record of one :func:`teleport` run.

    ``payload`` is kept as submitted (pure or mixed); ``received`` is the corrected
    receiver state; ``fidelity`` is measured against ``payload``.
    """

    payload: StateLike
    bell_outcome: BellState
    classical_bits: tuple[int, int]
    received: DensityMatrix
    fidelity: float

    def __post_init__(self) -> None:
        """Validate the record's internal consistency."""
        expected_bits = correction_bits(self.bell_outcome)
        if tuple(self.classical_bits) != expected_bits:
            raise ValueError(
                f"classical_bits {self.classical_bits!r} disagree with "
                f"bell_outcome {self.bell_outcome!r}, which requires "
                f"{expected_bits!r}. The bits are 2*m1 + m0 == "
                "BELL_ORDER.index(outcome)."
            )
        if _coerce_state(self.payload).num_qubits != 1:
            raise ValueError(
                f"payload must be a single-qubit state, got "
                f"{_coerce_state(self.payload).num_qubits} qubits."
            )
        if self.received.num_qubits != 1:
            raise ValueError(
                f"received must be a single-qubit state, got "
                f"{self.received.num_qubits} qubits."
            )
        if not 0.0 <= float(self.fidelity) <= 1.0:
            raise ValueError(
                f"fidelity must lie in [0, 1], got {self.fidelity!r}."
            )


@dataclass(frozen=True)
class RegisterTeleportationResult:
    """Record of one :func:`teleport_in_register` run.

    ``register`` has the two consumed qubits traced out and the correction applied;
    ``payload_index`` is where the teleported qubit now lives.
    """

    register: DensityMatrix
    bell_outcome: BellState
    classical_bits: tuple[int, int]
    payload_index: int

    def __post_init__(self) -> None:
        """Validate the record's internal consistency."""
        expected_bits = correction_bits(self.bell_outcome)
        if tuple(self.classical_bits) != expected_bits:
            raise ValueError(
                f"classical_bits {self.classical_bits!r} disagree with "
                f"bell_outcome {self.bell_outcome!r}, which requires "
                f"{expected_bits!r}. The bits are 2*m1 + m0 == "
                "BELL_ORDER.index(outcome)."
            )
        num_qubits = self.register.num_qubits
        if not 0 <= int(self.payload_index) < num_qubits:
            raise ValueError(
                f"payload_index {self.payload_index!r} is out of range for a "
                f"{num_qubits}-qubit register; it must satisfy "
                f"0 <= payload_index < {num_qubits}."
            )


def _payload_state(payload: StateLike) -> Statevector | DensityMatrix:
    """Validate a single-qubit payload, pure (``Statevector``) or mixed (``DensityMatrix``)."""
    coerced = _coerce_state(payload)
    if coerced.num_qubits != 1:
        raise ValueError(
            f"payload must be a single-qubit state, got "
            f"{coerced.num_qubits} qubits (dimension {coerced.dim}). "
            "Teleportation as modelled here moves one qubit."
        )
    if isinstance(coerced, Statevector):
        return Statevector(np.asarray(coerced.data, dtype=np.complex128))
    return coerced


def _resource_density(resource: StateLike | None) -> DensityMatrix:
    """Validate the two-qubit resource; ``None`` means :math:`|\\Phi^{+}\\rangle`."""
    if resource is None:
        return as_density(bell_state(BellState.PHI_PLUS))
    density = as_density(resource)
    if density.num_qubits != 2:
        raise ValueError(
            f"resource must be a two-qubit state (the shared entangled pair), "
            f"got {density.num_qubits} qubits (dimension {density.dim})."
        )
    return density


def _check_register_index(index: int, num_qubits: int, *, name: str) -> int:
    """Validate one qubit index against a register size (bools rejected)."""
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError(
            f"{name} must be an integer qubit index, got {index!r} of type "
            f"{type(index).__name__!r}."
        )
    value = int(index)
    if not 0 <= value < num_qubits:
        raise ValueError(
            f"{name} must satisfy 0 <= {name} < {num_qubits} for this "
            f"{num_qubits}-qubit register, got {value}."
        )
    return value


def teleport_in_register(
    register: StateLike,
    *,
    payload_qubit: int,
    resource_qubits: tuple[int, int],
    rng: np.random.Generator | None = None,
) -> RegisterTeleportationResult:
    """Teleport one qubit of a register, keeping its correlations with the other qubits.

    ``resource_qubits`` is ``(sender_half, receiver_half)``; order matters. The
    result has two fewer qubits; survivors keep their relative order.

    Raises
    ------
    ValueError
        If the register has fewer than three qubits, the indices are invalid or not
        distinct, or the collapsed branch cannot be renormalised.

    Examples
    --------
    Teleport half of a Bell pair; the entanglement survives:

    >>> import numpy as np
    >>> from qiskit.quantum_info import DensityMatrix
    >>> from qsecure.core.states import bell_state, concurrence
    >>> pair = DensityMatrix(bell_state())
    >>> register = DensityMatrix(bell_state()).tensor(pair)
    >>> moved = teleport_in_register(
    ...     register,
    ...     payload_qubit=0,
    ...     resource_qubits=(2, 3),
    ...     rng=np.random.default_rng(20260141),
    ... )
    >>> moved.payload_index
    1
    >>> round(concurrence(moved.register), 12)
    1.0
    """
    generator = resolve_rng(rng)
    density = as_density(register)
    num_qubits = density.num_qubits
    if num_qubits < 3:
        raise ValueError(
            f"teleport_in_register needs at least three qubits (payload, "
            f"sender's half, receiver's half), got a {num_qubits}-qubit "
            "register. Use teleport() for a standalone single-qubit payload; "
            "it builds the three-qubit register for you."
        )
    if not isinstance(resource_qubits, tuple) or len(resource_qubits) != 2:
        raise ValueError(
            "resource_qubits must be a tuple of two distinct qubit indices "
            f"(sender's half, receiver's half), got {resource_qubits!r}."
        )
    payload = _check_register_index(payload_qubit, num_qubits, name="payload_qubit")
    sender = _check_register_index(
        resource_qubits[0], num_qubits, name="resource_qubits[0]"
    )
    receiver = _check_register_index(
        resource_qubits[1], num_qubits, name="resource_qubits[1]"
    )
    if len({payload, sender, receiver}) != 3:
        raise ValueError(
            f"payload_qubit ({payload}) and resource_qubits "
            f"({sender}, {receiver}) must be three distinct qubits: the "
            "payload and the sender's half are both consumed by the Bell "
            "measurement and the receiver's half must survive it."
        )

    outcome, collapsed = bell_measure(density, (payload, sender), rng=generator)

    reduced = partial_trace(collapsed, [payload, sender])
    remaining = num_qubits - 2
    new_index = receiver - sum(1 for consumed in (payload, sender) if consumed < receiver)

    unitary = _embed_single(_CORRECTIONS[outcome], new_index, remaining)
    corrected = (
        unitary
        @ np.asarray(reduced.data, dtype=np.complex128)
        @ unitary.conj().T
    )
    # Remove round-off anti-Hermitian residue before as_density sees it.
    corrected = 0.5 * (corrected + corrected.conj().T)
    trace = float(np.trace(corrected).real)
    if not np.isfinite(trace) or trace <= 0.0:
        raise ValueError(
            f"the collapsed register has non-positive trace {trace!r} and "
            "cannot be renormalised; the input register was accepted as "
            "physical, so this indicates a bug in the Bell projectors."
        )
    corrected = corrected / trace

    return RegisterTeleportationResult(
        register=DensityMatrix(corrected, dims=(2,) * remaining),
        bell_outcome=outcome,
        classical_bits=correction_bits(outcome),
        payload_index=new_index,
    )


def teleport(
    payload: StateLike,
    *,
    resource: StateLike | None = None,
    rng: np.random.Generator | None = None,
) -> TeleportationResult:
    """Teleport a single-qubit payload (pure or mixed) across a shared pair.

    A wrapper over :func:`teleport_in_register`; ``resource`` defaults to
    :math:`|\\Phi^{+}\\rangle`.

    Raises
    ------
    ValueError
        If the payload or resource is invalid, or the Bell branch cannot be
        renormalised.

    Examples
    --------
    >>> import numpy as np
    >>> from qiskit.quantum_info import Statevector
    >>> payload = Statevector(np.array([0.6, 0.8j]))
    >>> result = teleport(payload, rng=np.random.default_rng(20260141))
    >>> round(result.fidelity, 12)
    1.0
    >>> result.classical_bits == correction_bits(result.bell_outcome)
    True
    """
    generator = resolve_rng(rng)
    payload_state = _payload_state(payload)
    resource_state = _resource_density(resource)

    # tensor() puts its argument on the low qubits, so the payload is qubit 0.
    register = resource_state.tensor(as_density(payload_state))

    moved = teleport_in_register(
        register,
        payload_qubit=_PAYLOAD_QUBIT,
        resource_qubits=(_SENDER_QUBIT, _RECEIVER_QUBIT),
        rng=generator,
    )
    received = moved.register

    return TeleportationResult(
        payload=payload_state,
        bell_outcome=moved.bell_outcome,
        classical_bits=moved.classical_bits,
        received=received,
        # Clipped helper: qiskit's raw state_fidelity can return 1.000000000000001.
        fidelity=state_fidelity(payload_state, received),
    )


def _branch_map(bell_vector: ComplexMatrix, resource_vector: ComplexMatrix) -> ComplexMatrix:
    """Return the payload map :math:`M[r, p] = \\sum_s \\overline{B_{2s+p}}\\,\\chi_{2r+s}`."""
    bra = np.conj(bell_vector).reshape(2, 2)  # [s, p]
    ket = resource_vector.reshape(2, 2)  # [r, s]
    return np.asarray(ket @ bra, dtype=np.complex128)  # [r, p]


def teleport_channel(resource: StateLike | None = None) -> Kraus:
    """Return the exact single-qubit channel induced by teleporting over ``resource``.

    Kraus operators are :math:`U_B M_{B,k}` over Bell outcomes and resource
    eigen-branches. The result is always the Pauli channel
    :math:`\\sum_B q_B U_B \\rho U_B^{\\dagger}` with
    :math:`q_B = \\langle B|\\rho_{\\text{res}}|B\\rangle`; compare channels via ``Choi``.

    Examples
    --------
    >>> import numpy as np
    >>> from qiskit.quantum_info import Choi, Operator
    >>> channel = teleport_channel()
    >>> bool(np.allclose(Choi(channel).data, Choi(Operator(np.eye(2))).data))
    True
    """
    density = np.asarray(
        _resource_density(resource).data, dtype=np.complex128
    )
    # eigh reads one triangle only, so symmetrise first.
    weights, vectors = np.linalg.eigh(0.5 * (density + density.conj().T))
    weights = _psd_eigenvalues(np.asarray(weights, dtype=np.float64))

    bell_vectors = {
        which: np.asarray(bell_state(which).data, dtype=np.complex128)
        for which in BELL_ORDER
    }

    operators: list[ComplexMatrix] = []
    for index, weight in enumerate(weights):
        if weight <= 0.0:
            continue
        branch = np.sqrt(weight) * np.asarray(
            vectors[:, index], dtype=np.complex128
        )
        for which in BELL_ORDER:
            operators.append(
                _CORRECTIONS[which] @ _branch_map(bell_vectors[which], branch)
            )
    return Kraus(operators)


def _teleport_core() -> QuantumCircuit:
    """Return the measurement-free part: ``h(1); cx(1, 2)`` then ``cx(0, 1); h(0)``."""
    circuit = QuantumCircuit(3, name="teleport_core")
    circuit.h(_SENDER_QUBIT)
    circuit.cx(_SENDER_QUBIT, _RECEIVER_QUBIT)
    circuit.barrier()
    circuit.cx(_PAYLOAD_QUBIT, _SENDER_QUBIT)
    circuit.h(_PAYLOAD_QUBIT)
    return circuit


def teleport_circuit() -> QuantumCircuit:
    """Return the textbook three-qubit teleportation circuit (for display).

    ``m[0]`` is qubit 1's result (``X`` exponent) and ``m[1]`` qubit 0's (``Z``
    exponent); the correction applies ``Z`` then ``X``.
    """
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(2, "m")
    circuit = QuantumCircuit(qreg, creg, name="teleport")
    circuit.compose(_teleport_core(), qubits=qreg, inplace=True)

    circuit.measure(_SENDER_QUBIT, creg[0])  # m0: the X exponent
    circuit.measure(_PAYLOAD_QUBIT, creg[1])  # m1: the Z exponent
    circuit.barrier()

    with circuit.if_test((creg[1], 1)):
        circuit.z(_RECEIVER_QUBIT)
    with circuit.if_test((creg[0], 1)):
        circuit.x(_RECEIVER_QUBIT)
    return circuit
