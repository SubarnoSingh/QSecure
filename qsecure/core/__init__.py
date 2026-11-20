"""Quantum core: Pauli algebra, states, measurement, teleportation and randomness.

Conventions: post-measurement states are ``DensityMatrix``; Qiskit little-endian
qubit order (qubit 0 is the rightmost label character); all randomness goes
through :func:`resolve_rng`. Note ``import qsecure.core.teleport as t`` binds the
:func:`teleport` function, not the submodule.
"""

from qsecure.core.measure import (
    MeasurementOutcome,
    bell_measure,
    born_probabilities,
    expectation,
    joint_probability,
    measure_qubits,
    projective_measure,
    sample_counts,
)
from qsecure.core.paulis import (
    BASIS_CHANGE,
    PAULI_I,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    PauliBasis,
    basis_change_circuit,
    eigenstate,
    eigenstate_label,
    pauli_matrix,
    random_basis,
    verify_eigenstate,
)
from qsecure.core.rng import resolve_rng, seed_to_generator
from qsecure.core.states import (
    BELL_ORDER,
    BellState,
    StateLike,
    as_density,
    bell_circuit,
    bell_state,
    concurrence,
    fidelity,
    is_normalised,
    purity,
)
from qsecure.core.teleport import (
    RegisterTeleportationResult,
    TeleportationResult,
    correction_bits,
    pauli_correction,
    teleport,
    teleport_channel,
    teleport_circuit,
    teleport_in_register,
)

__all__ = [
    "resolve_rng",
    "seed_to_generator",
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
    "MeasurementOutcome",
    "born_probabilities",
    "projective_measure",
    "measure_qubits",
    "joint_probability",
    "sample_counts",
    "expectation",
    "bell_measure",
    "TeleportationResult",
    "RegisterTeleportationResult",
    "pauli_correction",
    "correction_bits",
    "teleport",
    "teleport_in_register",
    "teleport_channel",
    "teleport_circuit",
]
