"""Projective measurement: Born probabilities, collapse, sampling, expectations.

States are accepted in any form and normalised via :func:`as_density`; returned
states are ``DensityMatrix``. Results carry both the eigenvalue and the bit
(``+1 -> 0``, ``-1 -> 1``). Collapsing functions never select a branch below
:data:`_MIN_BRANCH_PROB`; :func:`sample_counts` has no such floor.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import DensityMatrix, Operator, Pauli

from qsecure.core.paulis import (
    PAULI_I,
    PauliBasis,
    _as_basis,
    basis_change_circuit,
    pauli_matrix,
)
from qsecure.core.rng import resolve_rng
from qsecure.core.states import (
    BELL_ORDER,
    BellState,
    StateLike,
    _VALIDATION_TOL,
    as_density,
    bell_state,
)

__all__ = [
    "MeasurementOutcome",
    "born_probabilities",
    "projective_measure",
    "measure_qubits",
    "joint_probability",
    "sample_counts",
    "expectation",
    "bell_measure",
]

ComplexMatrix = NDArray[np.complex128]

# Must accept every state as_density accepts.
_PROB_TOL: Final[float] = _VALIDATION_TOL

# eps / _VALIDATION_TOL (~2.2e-8): below this, P rho P / p is dominated by
# round-off and is not a valid density operator.
_MIN_BRANCH_PROB: Final[float] = float(np.finfo(np.float64).eps) / _VALIDATION_TOL

# Negative-eigenvalue mass beyond this is a real defect, not round-off.
_PSD_REPAIR_LIMIT: Final[float] = 100.0 * _VALIDATION_TOL

_BIT_FROM_EIGENVALUE: Final[dict[int, int]] = {1: 0, -1: 1}

_PAULI_LABEL_CHARS: Final[frozenset[str]] = frozenset("IXYZ")


def _embed_single(operator: ComplexMatrix, qubit: int, num_qubits: int) -> ComplexMatrix:
    """Embed a one-qubit operator at ``qubit`` (little-endian: kron from the top qubit down).

    ``Z`` on qubit 0 of two qubits gives ``diag(1, -1, 1, -1)``.
    """
    result: ComplexMatrix = np.asarray(
        operator if num_qubits - 1 == qubit else PAULI_I, dtype=np.complex128
    )
    for index in range(num_qubits - 2, -1, -1):
        factor = operator if index == qubit else PAULI_I
        result = np.kron(result, np.asarray(factor, dtype=np.complex128))
    return np.asarray(result, dtype=np.complex128)


def _check_qubit(qubit: int, num_qubits: int, *, name: str = "qubit") -> int:
    """Validate a qubit index; negative indices are rejected rather than wrapped."""
    if isinstance(qubit, bool) or not isinstance(qubit, (int, np.integer)):
        raise ValueError(
            f"{name} must be an integer index, got {qubit!r} "
            f"({type(qubit).__name__})."
        )
    index = int(qubit)
    if not 0 <= index < num_qubits:
        raise ValueError(
            f"{name} must satisfy 0 <= {name} < {num_qubits} for this "
            f"{num_qubits}-qubit state, got {index}. Qubit 0 is the rightmost "
            f"bit of a little-endian label; negative indices are not wrapped."
        )
    return index


def _projectors(
    basis: PauliBasis, qubit: int, num_qubits: int
) -> tuple[ComplexMatrix, ComplexMatrix]:
    """Return the embedded projectors ``(P_plus, P_minus) = (1 +/- sigma)/2``."""
    sigma = _embed_single(pauli_matrix(basis.value), qubit, num_qubits)
    identity = np.eye(2**num_qubits, dtype=np.complex128)
    return 0.5 * (identity + sigma), 0.5 * (identity - sigma)


def _collapse(
    rho: ComplexMatrix, projector: ComplexMatrix, probability: float
) -> ComplexMatrix:
    """Apply ``rho -> P rho P / Tr(P rho P)``, projected back onto the PSD cone.

    Normalises by the realised trace, not ``probability``; ``probability`` only
    guards against branches at or below :data:`_MIN_BRANCH_PROB`.

    Raises
    ------
    ValueError
        If the branch is too improbable, the trace is non-positive, or the
        negative-eigenvalue mass exceeds :data:`_PSD_REPAIR_LIMIT`.
    """
    if probability <= _MIN_BRANCH_PROB:
        raise ValueError(
            f"cannot renormalise a measurement branch of probability "
            f"{probability!r}: it is at or below {_MIN_BRANCH_PROB} "
            "(= float64 eps / the project's state-validation tolerance), below "
            "which the quotient P rho P / p is dominated by round-off and is "
            "not a valid density operator."
        )
    collapsed = projector @ rho @ projector
    # eigh reads one triangle only, so symmetrise first.
    collapsed = 0.5 * (collapsed + collapsed.conj().T)

    eigenvalues, vectors = np.linalg.eigh(collapsed)
    trace = float(eigenvalues.sum())
    if trace <= 0.0:
        raise ValueError(
            f"collapsed branch has non-positive trace {trace!r}; the projector "
            "or the input density matrix is malformed."
        )
    negative_mass = -float(eigenvalues[eigenvalues < 0.0].sum())
    if negative_mass > _PSD_REPAIR_LIMIT * trace:
        raise ValueError(
            f"collapsed branch is not positive semidefinite: {negative_mass!r} "
            f"of negative eigenvalue mass against a trace of {trace!r}, which "
            f"exceeds the round-off budget {_PSD_REPAIR_LIMIT}. This is a "
            "malformed projector or a non-physical input state, not numerical "
            "noise."
        )
    weights = np.clip(eigenvalues, 0.0, None)
    weights /= weights.sum()
    repaired: ComplexMatrix = np.asarray(
        (vectors * weights) @ vectors.conj().T, dtype=np.complex128
    )
    return 0.5 * (repaired + repaired.conj().T)


def _density_array(state: StateLike) -> tuple[ComplexMatrix, DensityMatrix]:
    """Return ``(array, DensityMatrix)`` for validated state input."""
    rho = as_density(state)
    return np.asarray(rho.data, dtype=np.complex128), rho


def _as_dm(array: ComplexMatrix, template: DensityMatrix) -> DensityMatrix:
    """Wrap ``array`` as a ``DensityMatrix`` with ``template``'s dims."""
    return DensityMatrix(array, dims=template.dims())


def _validate_bases(
    bases: Mapping[int, PauliBasis], num_qubits: int
) -> dict[int, PauliBasis]:
    """Validate a ``{qubit: basis}`` mapping and return it sorted by qubit."""
    if not isinstance(bases, Mapping):
        raise TypeError(
            "bases must be a mapping from qubit index to PauliBasis, e.g. "
            f"{{0: PauliBasis.Z}}; got {type(bases).__name__}."
        )
    resolved: dict[int, PauliBasis] = {}
    for qubit, basis in bases.items():
        index = _check_qubit(qubit, num_qubits, name="basis key")
        resolved[index] = _as_basis(basis)
    return {index: resolved[index] for index in sorted(resolved)}


def _bell_projectors(
    num_qubits: int, qubits: tuple[int, int]
) -> dict[BellState, ComplexMatrix]:
    """Build the four Bell projectors on ``qubits``, padded with identity on the rest.

    ``qubits[0]`` is the Bell state's qubit 0; swapping the pair gives identical
    projectors.
    """
    dim = 2**num_qubits
    spectators = [q for q in range(num_qubits) if q not in qubits]
    projectors: dict[BellState, ComplexMatrix] = {}
    for which in BELL_ORDER:
        amplitudes = np.asarray(bell_state(which).data, dtype=np.complex128)
        projector = np.zeros((dim, dim), dtype=np.complex128)
        for assignment in itertools.product((0, 1), repeat=len(spectators)):
            base = 0
            for bit, spectator in zip(assignment, spectators):
                base |= bit << spectator
            vector = np.zeros(dim, dtype=np.complex128)
            for pair_index, amplitude in enumerate(amplitudes):
                if amplitude == 0:
                    continue
                full_index = (
                    base
                    | ((pair_index & 1) << qubits[0])
                    | (((pair_index >> 1) & 1) << qubits[1])
                )
                vector[full_index] = amplitude
            projector += np.outer(vector, vector.conj())
        projectors[which] = projector
    return projectors


def _select_branch(
    probabilities: Iterable[float], rng: np.random.Generator | None
) -> int:
    """Draw one branch index by inverse-CDF sampling (one uniform variate).

    Branches at or below :data:`_MIN_BRANCH_PROB` are never returned.
    """
    values = [float(p) for p in probabilities]
    draw = float(resolve_rng(rng).random())
    cumulative = 0.0
    fallback = -1
    for index, probability in enumerate(values):
        if probability > _MIN_BRANCH_PROB:
            fallback = index
            cumulative += probability
            if draw < cumulative:
                return index
    if fallback < 0:
        raise ValueError(
            "no measurement branch has non-negligible probability; the state "
            f"is not trace-one (probabilities were {values!r})."
        )
    # Cumulative sum fell a few ULPs short of the draw.
    return fallback


@dataclass(frozen=True)
class MeasurementOutcome:
    """Record of one single-qubit projective measurement.

    ``probability`` is the Born weight given the state passed to the producing call,
    so in :func:`measure_qubits` it is conditional on earlier outcomes; combine with
    :func:`joint_probability`, never by averaging. ``post_state`` is the collapsed
    state of the whole register.
    """

    eigenvalue: int
    bit: int
    basis: PauliBasis
    qubit: int
    probability: float
    post_state: DensityMatrix

    def __post_init__(self) -> None:
        """Validate the eigenvalue/bit correspondence and the probability."""
        if self.eigenvalue not in (1, -1):
            raise ValueError(
                f"eigenvalue must be +1 or -1, got {self.eigenvalue!r}. These "
                "are Pauli eigenvalues, not measurement bits."
            )
        expected_bit = _BIT_FROM_EIGENVALUE[int(self.eigenvalue)]
        if self.bit != expected_bit:
            raise ValueError(
                f"bit {self.bit!r} is inconsistent with eigenvalue "
                f"{self.eigenvalue!r}; the project-wide mapping is "
                "+1 -> bit 0 and -1 -> bit 1 (bit = (1 - eigenvalue) // 2)."
            )
        if not 0.0 <= float(self.probability) <= 1.0:
            raise ValueError(
                f"probability must lie in [0, 1], got {self.probability!r}."
            )


def born_probabilities(
    state: StateLike, qubit: int, basis: PauliBasis
) -> tuple[float, float]:
    """Return ``(p_plus, p_minus) = Tr(rho P_+/-)`` for measuring ``qubit`` in ``basis``.

    ``p_minus`` is ``1 - p_plus`` exactly.

    Raises
    ------
    ValueError
        If the state, qubit or basis is invalid, or completeness fails within
        :data:`_PROB_TOL`.

    Examples
    --------
    :math:`|01\\rangle` has qubit 0 in :math:`|1\\rangle`:

    >>> import numpy as np
    >>> from qiskit.quantum_info import Statevector
    >>> from qsecure.core.paulis import PauliBasis
    >>> ket01 = Statevector(np.array([0, 1, 0, 0], dtype=complex))
    >>> born_probabilities(ket01, 0, PauliBasis.Z)
    (0.0, 1.0)
    >>> born_probabilities(ket01, 1, PauliBasis.Z)
    (1.0, 0.0)
    """
    rho, dm = _density_array(state)
    index = _check_qubit(qubit, dm.num_qubits)
    resolved_basis = _as_basis(basis)

    proj_plus, proj_minus = _projectors(resolved_basis, index, dm.num_qubits)
    p_plus = float(np.real(np.trace(rho @ proj_plus)))
    p_minus = float(np.real(np.trace(rho @ proj_minus)))

    total = p_plus + p_minus
    if abs(total - 1.0) > _PROB_TOL:
        raise ValueError(
            f"Born probabilities do not sum to 1 (got {total!r} for qubit "
            f"{index} in basis {resolved_basis.value}, tolerance {_PROB_TOL}); "
            "the projectors are malformed, since any state as_density accepts "
            "is trace-one to within this tolerance."
        )
    # as_density admits a small trace drift; don't leak it into the result.
    clipped = float(np.clip(p_plus / total, 0.0, 1.0))
    return clipped, 1.0 - clipped


def projective_measure(
    state: StateLike,
    qubit: int,
    basis: PauliBasis,
    *,
    rng: np.random.Generator | None = None,
) -> MeasurementOutcome:
    """Measure one qubit and collapse the whole register.

    Exact Born sampling except that branches at or below :data:`_MIN_BRANCH_PROB`
    are never selected; use :func:`sample_counts` for rare-event rates.

    Raises
    ------
    ValueError
        If the state, qubit or basis is invalid, or the branch cannot be renormalised.
    """
    rho, dm = _density_array(state)
    index = _check_qubit(qubit, dm.num_qubits)
    resolved_basis = _as_basis(basis)

    p_plus, p_minus = born_probabilities(rho, index, resolved_basis)
    branch = _select_branch((p_plus, p_minus), rng)

    proj_plus, proj_minus = _projectors(resolved_basis, index, dm.num_qubits)
    if branch == 0:
        eigenvalue, probability, projector = 1, p_plus, proj_plus
    else:
        eigenvalue, probability, projector = -1, p_minus, proj_minus

    collapsed = _collapse(rho, projector, probability)
    return MeasurementOutcome(
        eigenvalue=eigenvalue,
        bit=_BIT_FROM_EIGENVALUE[eigenvalue],
        basis=resolved_basis,
        qubit=index,
        probability=probability,
        post_state=_as_dm(collapsed, dm),
    )


def measure_qubits(
    state: StateLike,
    bases: Mapping[int, PauliBasis],
    *,
    rng: np.random.Generator | None = None,
) -> tuple[list[MeasurementOutcome], DensityMatrix]:
    """Measure several qubits in ascending order, threading the collapse through each.

    Returns
    -------
    outcomes : list of MeasurementOutcome
        One per measured qubit; each ``probability`` is conditional on the earlier
        outcomes (use :func:`joint_probability`).
    final_state : qiskit.quantum_info.DensityMatrix
        The state after all measurements (``as_density(state)`` if ``bases`` is empty).
    """
    _, dm = _density_array(state)
    resolved = _validate_bases(bases, dm.num_qubits)

    generator = resolve_rng(rng)
    current: DensityMatrix = dm
    outcomes: list[MeasurementOutcome] = []
    for qubit, basis in resolved.items():
        outcome = projective_measure(current, qubit, basis, rng=generator)
        outcomes.append(outcome)
        current = outcome.post_state
    return outcomes, current


def joint_probability(outcomes: Sequence[MeasurementOutcome]) -> float:
    """Return the chain-rule product of a measurement record's conditional probabilities.

    An empty record gives ``1.0``.

    Raises
    ------
    TypeError
        If ``outcomes`` is not a sequence of :class:`MeasurementOutcome` (e.g. the
        whole ``(outcomes, final_state)`` pair).

    Examples
    --------
    On :math:`|\\Phi^{+}\\rangle` the joint probability is ``0.5``, not the mean ``0.75``.

    >>> import numpy as np
    >>> from qsecure.core.states import BellState, bell_state
    >>> from qsecure.core.paulis import PauliBasis
    >>> records, _ = measure_qubits(
    ...     bell_state(BellState.PHI_PLUS),
    ...     {0: PauliBasis.Z, 1: PauliBasis.Z},
    ...     rng=np.random.default_rng(20260141),
    ... )
    >>> [round(record.probability, 3) for record in records]
    [0.5, 1.0]
    >>> round(joint_probability(records), 3)
    0.5
    """
    if isinstance(outcomes, MeasurementOutcome) or not isinstance(
        outcomes, Sequence
    ):
        raise TypeError(
            f"outcomes must be a sequence of MeasurementOutcome, got "
            f"{type(outcomes).__name__}. measure_qubits returns the pair "
            f"(outcomes, final_state); pass the first element, e.g. "
            f"joint_probability(measure_qubits(state, bases)[0])."
        )
    product = 1.0
    for position, outcome in enumerate(outcomes):
        if not isinstance(outcome, MeasurementOutcome):
            raise TypeError(
                f"outcomes[{position}] must be a MeasurementOutcome, got "
                f"{type(outcome).__name__}. This function multiplies the "
                f"conditional probabilities of one measurement record; pass the "
                f"list measure_qubits returned, not raw floats."
            )
        product *= float(outcome.probability)
    return float(np.clip(product, 0.0, 1.0))


def sample_counts(
    state: StateLike,
    bases: Mapping[int, PauliBasis],
    shots: int,
    *,
    rng: np.random.Generator | None = None,
) -> dict[str, int]:
    """Sample ``shots`` repetitions of a multi-qubit measurement in one multinomial draw.

    The joint distribution is computed once (rotate onto Z, read the diagonal,
    marginalise spectators); there is no branch-probability floor.

    Returns
    -------
    dict
        Bitstring -> count, one character per measured qubit, highest index first;
        zero counts omitted. Empty ``bases`` gives ``{"": shots}``.

    Raises
    ------
    ValueError
        If the state or ``bases`` is invalid, or ``shots`` is not a non-negative int.
    """
    if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)):
        raise ValueError(
            f"shots must be a non-negative integer, got {shots!r} "
            f"({type(shots).__name__})."
        )
    shot_count = int(shots)
    if shot_count < 0:
        raise ValueError(f"shots must be non-negative, got {shot_count}.")

    rho, dm = _density_array(state)
    num_qubits = dm.num_qubits
    resolved = _validate_bases(bases, num_qubits)
    measured = list(resolved)

    rotation = np.eye(2**num_qubits, dtype=np.complex128)
    for qubit, basis in resolved.items():
        gate = np.asarray(
            Operator(basis_change_circuit(basis)).data, dtype=np.complex128
        )
        rotation = _embed_single(gate, qubit, num_qubits) @ rotation
    rotated = rotation @ rho @ rotation.conj().T

    diagonal = np.clip(np.real(np.diag(rotated)), 0.0, None)
    num_outcomes = 2 ** len(measured)
    probabilities = np.zeros(num_outcomes, dtype=np.float64)
    for full_index in range(2**num_qubits):
        outcome_index = 0
        for position, qubit in enumerate(measured):
            outcome_index |= ((full_index >> qubit) & 1) << position
        probabilities[outcome_index] += diagonal[full_index]

    total = float(probabilities.sum())
    if abs(total - 1.0) > _PROB_TOL:
        raise ValueError(
            f"measurement distribution does not sum to 1 (got {total!r}, "
            f"tolerance {_PROB_TOL}); the basis-change rotation or the "
            "spectator marginalisation is malformed, since any state "
            "as_density accepts is trace-one to within this tolerance."
        )
    probabilities /= total

    draws = resolve_rng(rng).multinomial(shot_count, probabilities)
    keys = [
        "".join(
            str((outcome_index >> position) & 1)
            for position in reversed(range(len(measured)))
        )
        for outcome_index in range(num_outcomes)
    ]
    return {
        key: int(count) for key, count in zip(keys, draws) if int(count) > 0
    }


def expectation(state: StateLike, pauli_label: str) -> float:
    """Return :math:`\\mathrm{Tr}(\\rho P)`; the label's rightmost character acts on qubit 0.

    Raises
    ------
    ValueError
        If the label has the wrong length or characters outside ``"IXYZ"``.
    TypeError
        If ``pauli_label`` is not a string.
    """
    if not isinstance(pauli_label, str):
        raise TypeError(
            f"pauli_label must be a string over 'IXYZ', got "
            f"{type(pauli_label).__name__}."
        )
    rho, dm = _density_array(state)
    label = pauli_label.strip().upper()
    if not label or not set(label) <= _PAULI_LABEL_CHARS:
        raise ValueError(
            f"pauli_label {pauli_label!r} must be a non-empty string over "
            "'IXYZ' (case-insensitive). Signed or phase-prefixed labels such "
            "as '-iXY' are not accepted: the observable must be Hermitian."
        )
    if len(label) != dm.num_qubits:
        raise ValueError(
            f"pauli_label {pauli_label!r} has length {len(label)} but the "
            f"state has {dm.num_qubits} qubits. Pad with 'I', remembering that "
            "the rightmost character acts on qubit 0."
        )
    observable = np.asarray(Pauli(label).to_matrix(), dtype=np.complex128)
    return float(np.real(np.trace(rho @ observable)))


def bell_measure(
    state: StateLike,
    qubits: tuple[int, int] = (0, 1),
    *,
    rng: np.random.Generator | None = None,
) -> tuple[BellState, DensityMatrix]:
    """Measure a qubit pair in the Bell basis, leaving other qubits in their conditional state.

    Branches at or below :data:`_MIN_BRANCH_PROB` are never selected. Swapping the
    pair order changes nothing.

    Raises
    ------
    ValueError
        If the state has fewer than two qubits, ``qubits`` is not a distinct valid
        pair, or the branch cannot be renormalised.
    """
    rho, dm = _density_array(state)
    if dm.num_qubits < 2:
        raise ValueError(
            f"bell_measure needs at least two qubits, got a "
            f"{dm.num_qubits}-qubit state."
        )
    if not isinstance(qubits, tuple) or len(qubits) != 2:
        raise ValueError(
            f"qubits must be a tuple of two distinct qubit indices, got "
            f"{qubits!r}."
        )
    pair = (
        _check_qubit(qubits[0], dm.num_qubits, name="qubits[0]"),
        _check_qubit(qubits[1], dm.num_qubits, name="qubits[1]"),
    )
    if pair[0] == pair[1]:
        raise ValueError(
            f"qubits must be two distinct indices, got {pair[0]} twice. A Bell "
            "measurement acts on a pair."
        )

    projectors = _bell_projectors(dm.num_qubits, pair)
    probabilities = [
        float(np.real(np.trace(rho @ projectors[which]))) for which in BELL_ORDER
    ]
    total = sum(probabilities)
    if abs(total - 1.0) > _PROB_TOL:
        raise ValueError(
            f"Bell-basis probabilities do not sum to 1 (got {total!r}, "
            f"tolerance {_PROB_TOL}); the Bell projectors are malformed, since "
            "any state as_density accepts is trace-one to within this "
            "tolerance."
        )
    probabilities = [float(np.clip(p, 0.0, 1.0)) / total for p in probabilities]

    branch = _select_branch(probabilities, rng)
    outcome = BELL_ORDER[branch]
    collapsed = _collapse(rho, projectors[outcome], probabilities[branch])
    return outcome, _as_dm(collapsed, dm)
