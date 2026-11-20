"""Channel manipulation: an eavesdropper between Alice and one recipient.

Attacks only the Phase A distribution, through the
:data:`~qsecure.protocol.distribute.ResourceFactory` and
:data:`~qsecure.protocol.distribute.PayloadMap` seams.

Examples
--------
>>> from qsecure.attacks.channel import (
...     chsh_from_tensor, collapse_tensor, depolarising_tensor
... )
>>> f"{chsh_from_tensor(depolarising_tensor(0.0)):.4f}"
'2.8284'
>>> f"{chsh_from_tensor(depolarising_tensor(0.3)):.4f}"
'1.9799'
>>> f"{chsh_from_tensor(collapse_tensor()):.4f}"
'0.9428'
>>> f"{chsh_from_tensor(collapse_tensor(('Z',))):.4f}"
'1.4142'
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, TypeAlias

import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator, Statevector

from qsecure.core.paulis import PauliBasis, pauli_matrix
from qsecure.core.rng import resolve_rng
from qsecure.core.states import BellState, StateLike, as_density, bell_state
from qsecure.protocol.checkrounds import (
    ChshEstimate,
    ChshRound,
    QberEstimate,
    QberRound,
    estimate_chsh,
    estimate_qber,
    observe_chsh_round,
    observe_qber_round,
)
from qsecure.protocol.distribute import ResourceContext
from qsecure.protocol.params import Party, ProtocolParams, _as_party
from qsecure.protocol.symmetrise import no_symmetrisation

__all__ = [
    "PAULI_AXES",
    "IDEAL_TENSOR",
    "AttackDecision",
    "ChannelAttack",
    "DepolarisingChannel",
    "InterceptResend",
    "KeptShareSwap",
    "CorrelationTensor",
    "attribution_survives_symmetrisation",
    "chsh_from_tensor",
    "collapse_tensor",
    "depolarising_tensor",
    "measure_chsh",
    "measure_qber",
    "payload_line_is_unwatched",
    "qber_from_tensor",
]


PAULI_AXES: Final[tuple[PauliBasis, PauliBasis, PauliBasis]] = (
    PauliBasis.X,
    PauliBasis.Y,
    PauliBasis.Z,
)
"""The three axes an eavesdropper draws from unless told otherwise."""

CorrelationTensor: TypeAlias = tuple[float, float, float]
"""``(T_xx, T_yy, T_zz)``, the diagonal of a Bell-diagonal correlation tensor.
"""

IDEAL_TENSOR: Final[CorrelationTensor] = (1.0, -1.0, 1.0)
""":math:`|\\Phi^{+}\\rangle`'s tensor: correlated in :math:`X` and :math:`Z`,
**anti**-correlated in :math:`Y`.
"""

_ALICE_QUBIT: Final[int] = 0

_TRAVELLING_QUBIT: Final[int] = 1

_AXIS_INDEX: Final[Mapping[PauliBasis, int]] = {
    PauliBasis.X: 0,
    PauliBasis.Y: 1,
    PauliBasis.Z: 2,
}

_TWIRL_LABELS: Final[tuple[str, str, str, str]] = ("I", "X", "Y", "Z")


def _as_tensor(tensor: Any, name: str = "tensor") -> CorrelationTensor:
    """Coerce and range-check a correlation tensor."""
    if isinstance(tensor, (str, bytes)) or not isinstance(tensor, Sequence):
        raise TypeError(
            f"{name} must be a sequence (T_xx, T_yy, T_zz) of three "
            f"correlators, got {type(tensor).__name__}. Build one with "
            f"depolarising_tensor(p) or collapse_tensor(axes)."
        )
    if len(tensor) != 3:
        raise ValueError(
            f"{name} must hold exactly three correlators (T_xx, T_yy, T_zz), "
            f"got {len(tensor)}. A tensor with a missing axis would silently "
            f"read as zero correlation on that axis, which is a maximally "
            f"noisy channel and not a missing measurement."
        )
    values: list[float] = []
    for axis, entry in zip("xyz", tensor, strict=True):
        if isinstance(entry, bool) or not isinstance(entry, numbers.Real):
            raise TypeError(
                f"{name}[T_{axis}{axis}] must be a real number in [-1, 1], got "
                f"{type(entry).__name__}"
            )
        value = float(entry)
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError(
                f"{name}[T_{axis}{axis}] must be a finite correlator in "
                f"[-1, 1], got {value!r}. It is the expectation of a product of "
                f"two eigenvalues, so it cannot leave that range."
            )
        values.append(value)
    return values[0], values[1], values[2]


def chsh_from_tensor(tensor: Sequence[float]) -> float:
    """Return ``S = sqrt(2) * (T_zz + T_xx)`` for a Bell-diagonal resource.

    Parameters
    ----------
    tensor : sequence of float
        ``(T_xx, T_yy, T_zz)``.

    Returns
    -------
    float
        The predicted CHSH statistic.

    Examples
    --------
    >>> from qsecure.attacks.channel import IDEAL_TENSOR, chsh_from_tensor
    >>> f"{chsh_from_tensor(IDEAL_TENSOR):.4f}"
    '2.8284'

    >>> chsh_from_tensor((1.0, 0.0, 1.0)) == chsh_from_tensor(IDEAL_TENSOR)
    True
    """
    t_xx, _t_yy, t_zz = _as_tensor(tensor)
    return math.sqrt(2.0) * (t_zz + t_xx)


def qber_from_tensor(
    tensor: Sequence[float],
    bases: Iterable[PauliBasis | str] = PAULI_AXES,
) -> float:
    """Return the matched-basis error rate a Bell-diagonal resource produces.

    Parameters
    ----------
    tensor : sequence of float
        ``(T_xx, T_yy, T_zz)``.
    bases : iterable of PauliBasis or str, optional
        The alphabet the round's basis is drawn from, uniformly.

    Returns
    -------
    float
        The predicted error rate.

    Examples
    --------
    >>> from qsecure.attacks.channel import (
    ...     IDEAL_TENSOR, collapse_tensor, depolarising_tensor,
    ...     qber_from_tensor,
    ... )
    >>> qber_from_tensor(IDEAL_TENSOR)
    0.0
    >>> f"{qber_from_tensor(depolarising_tensor(0.14)):.5f}"
    '0.07000'
    >>> f"{qber_from_tensor(collapse_tensor()):.5f}"
    '0.33333'

    >>> [
    ...     f"{qber_from_tensor(collapse_tensor(('Z',)), (basis,)):.4f}"
    ...     for basis in "XYZ"
    ... ]
    ['0.5000', '0.5000', '0.0000']
    """
    values = _as_tensor(tensor)
    alphabet = [PauliBasis(str(basis).upper()) for basis in bases]
    if not alphabet:
        raise ValueError(
            "qber_from_tensor needs at least one basis to average over: an "
            "empty alphabet has no error rate, it has no rounds."
        )
    total = 0.0
    for basis in alphabet:
        index = _AXIS_INDEX[basis]
        total += (1.0 - IDEAL_TENSOR[index] * values[index]) / 2.0
    return total / len(alphabet)


def depolarising_tensor(strength: float) -> CorrelationTensor:
    """Return ``(1 - p) * IDEAL_TENSOR``, the Werner family's tensor.

    Parameters
    ----------
    strength : float
        ``p`` in ``[0, 1]``.

    Examples
    --------
    >>> from qsecure.attacks.channel import chsh_from_tensor, depolarising_tensor
    >>> depolarising_tensor(0.0)
    (1.0, -1.0, 1.0)
    >>> depolarising_tensor(1.0)
    (0.0, -0.0, 0.0)

    >>> from qsecure.protocol.checkrounds import depolarising_chsh
    >>> all(
    ...     abs(chsh_from_tensor(depolarising_tensor(p / 20))
    ...         - depolarising_chsh(p / 20)) < 1e-12
    ...     for p in range(21)
    ... )
    True
    """
    value = _as_strength(strength, "strength")
    return (
        (1.0 - value) * IDEAL_TENSOR[0],
        (1.0 - value) * IDEAL_TENSOR[1],
        (1.0 - value) * IDEAL_TENSOR[2],
    )


def collapse_tensor(
    axes: Iterable[PauliBasis | str] = PAULI_AXES,
) -> CorrelationTensor:
    """Return the tensor left by collapsing :math:`|\\Phi^{+}\\rangle` onto
    ``axes``.

    Parameters
    ----------
    axes : iterable of PauliBasis or str, optional
        The axes Eve draws from, uniformly.

    Examples
    --------
    >>> from qsecure.attacks.channel import (
    ...     chsh_from_tensor, collapse_tensor, qber_from_tensor
    ... )
    >>> collapse_tensor(("Z",))
    (0.0, 0.0, 1.0)
    >>> tuple(round(value, 6) for value in collapse_tensor())
    (0.333333, -0.333333, 0.333333)

    >>> f"{chsh_from_tensor(collapse_tensor(('Z',))):.4f}"
    '1.4142'
    >>> f"{chsh_from_tensor(collapse_tensor()):.4f}"
    '0.9428'
    >>> qber_from_tensor(collapse_tensor(("Z",))) == qber_from_tensor(
    ...     collapse_tensor()
    ... )
    True
    """
    alphabet = _as_axes(axes)
    totals = [0.0, 0.0, 0.0]
    for basis in alphabet:
        index = _AXIS_INDEX[basis]
        totals[index] += IDEAL_TENSOR[index]
    count = float(len(alphabet))
    return (totals[0] / count, totals[1] / count, totals[2] / count)


def _as_strength(value: Any, name: str) -> float:
    """Coerce and range-check a probability-valued attack strength."""
    if isinstance(value, bool):
        raise TypeError(
            f"{name} must be a real number in [0, 1], got the boolean "
            f"{value!r}"
        )
    if not isinstance(value, numbers.Real):
        raise TypeError(
            f"{name} must be a real number in [0, 1], got "
            f"{type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(
            f"{name} must be a finite probability in [0, 1], got {result!r}. "
            f"It is the fraction of hops the attack acts on."
        )
    return result


def _as_axes(axes: Any) -> tuple[PauliBasis, ...]:
    """Coerce an axis alphabet, refusing an empty one."""
    if isinstance(axes, (str, bytes)):
        raise TypeError(
            f"axes must be an iterable of PauliBasis, not the string "
            f"{axes!r}; a bare string would iterate character by character and "
            f"'XY' would silently mean two axes. Pass ('X', 'Y')."
        )
    try:
        candidates = tuple(axes)
    except TypeError:
        raise TypeError(
            f"axes must be an iterable of PauliBasis or their labels, got "
            f"{type(axes).__name__}"
        ) from None
    if not candidates:
        raise ValueError(
            "axes must name at least one measurement axis: an eavesdropper "
            "with no basis to guess in is not an attack, and an empty "
            "alphabet would divide by zero in collapse_tensor."
        )
    resolved: list[PauliBasis] = []
    for candidate in candidates:
        try:
            resolved.append(PauliBasis(str(candidate).upper()))
        except ValueError:
            raise ValueError(
                f"axes must name Pauli bases X, Y or Z, got {candidate!r}. "
                f"The default PAULI_AXES is the protocol's own alphabet."
            ) from None
    return tuple(resolved)


def _as_target(target: Any) -> Party | None:
    """Coerce the party-targeting argument."""
    if target is None:
        return None
    resolved = _as_party(target)
    if resolved is Party.ALICE:
        raise ValueError(
            "target names the recipient whose link is attacked, and Alice is "
            "at the far end of both links, so targeting her designates no "
            "link. Pass Party.BOB, Party.CHARLIE, or None to attack both."
        )
    return resolved


@dataclass(frozen=True)
class AttackDecision:
    """One hop, and what Eve chose to do to it.

    Attributes
    ----------
    party : Party
        The recipient whose hop this was.
    message_bit : int
        Which distribution.
    position : int
        The key index, in the unsifted ``0 .. L-1`` numbering.
    engaged : bool
        Whether the attack acted.
    choice : str
        What was done, as a short label: ``"pass"`` for an untouched hop, a
        Pauli label for a twirl, an axis label for a measurement or a
        dephasing.

    Examples
    --------
    >>> from qsecure.attacks.channel import AttackDecision
    >>> from qsecure.protocol.params import Party
    >>> AttackDecision(Party.BOB, 0, 7, True, "Z").as_tuple()
    ('Bob', 0, 7, True, 'Z')
    """

    party: Party
    message_bit: int
    position: int
    engaged: bool
    choice: str

    def as_tuple(self) -> tuple[str, int, int, bool, str]:
        """Return a plain comparable tuple.

        Returns
        -------
        tuple
            ``(party, message_bit, position, engaged, choice)``, with the party
            as its label so the value survives
            :func:`qsecure.attacks.isolation.canonical` and :func:`json.dumps`
            unchanged.
        """
        return (
            self.party.value,
            self.message_bit,
            self.position,
            self.engaged,
            self.choice,
        )


class ChannelAttack:
    """Base class: an eavesdropper who owns a generator and targets a link.

    Parameters
    ----------
    rng : numpy.random.Generator
        **required**.
    target : Party, str or None, optional
        Restrict the attack to one recipient's link, read off
        :attr:`~qsecure.protocol.distribute.ResourceContext.party`.
    """

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        target: Party | str | None = None,
    ) -> None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                f"rng must be a numpy.random.Generator and is required, got "
                f"{type(rng).__name__}. An adversary owns its randomness: "
                f"it is constructed with a generator of its own, never with the "
                f"session's seed and never with None, because an adversary that "
                f"cannot be replayed cannot be shown to be isolated."
            )
        self._rng = rng
        self._target = _as_target(target)
        self._log: list[AttackDecision] = []

    @property
    def target(self) -> Party | None:
        """Party or None: The link this attack is restricted to."""
        return self._target

    @property
    def log(self) -> tuple[AttackDecision, ...]:
        """tuple of AttackDecision: Every hop offered, in call order."""
        return tuple(self._log)

    @property
    def engaged_count(self) -> int:
        """int: How many hops the attack actually acted on."""
        return sum(1 for entry in self._log if entry.engaged)

    def decisions(self) -> tuple[tuple[str, int, int, bool, str], ...]:
        """Return the log as plain tuples, for an isolation probe.

        Returns
        -------
        tuple of tuple
            One :meth:`AttackDecision.as_tuple` per hop.
        """
        return tuple(entry.as_tuple() for entry in self._log)

    def __repr__(self) -> str:
        """Return a debugging representation naming the target and the log
        size.
        """
        where = "both links" if self._target is None else self._target.value
        return (
            f"<{type(self).__name__} on {where}, {self.engaged_count} of "
            f"{len(self._log)} hop(s) engaged>"
        )

    def resource(self, context: ResourceContext) -> StateLike:
        """Mount the attack on the entanglement resource.

        Parameters
        ----------
        context : ResourceContext
            The hop being served.

        Returns
        -------
        StateLike
            The two-qubit pair actually delivered.
        """
        pair = bell_state(BellState.PHI_PLUS)
        if not self._targets(context):
            self._record(context, engaged=False, choice="pass")
            return pair
        return self._act_on_resource(pair, context)

    def payload(self, state: StateLike, context: ResourceContext) -> StateLike:
        """Mount the attack on the payload line.

        Parameters
        ----------
        state : StateLike
            The eigenstate Alice prepared for this position.
        context : ResourceContext
            The hop being served.

        Returns
        -------
        StateLike
            The one-qubit state actually teleported.
        """
        if not self._targets(context):
            self._record(context, engaged=False, choice="pass")
            return state
        return self._act_on_payload(state, context)

    def _act_on_resource(
        self, pair: Statevector, context: ResourceContext
    ) -> StateLike:
        """Transform one targeted hop's entanglement resource."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _act_on_resource to be "
            f"mounted on the resource_factory seam."
        )

    def _act_on_payload(
        self, state: StateLike, context: ResourceContext
    ) -> StateLike:
        """Transform one targeted hop's payload."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _act_on_payload to be "
            f"mounted on the payload_map seam."
        )

    def _targets(self, context: ResourceContext) -> bool:
        """Return whether this hop is on the attacked link."""
        return self._target is None or context.party is self._target

    def _record(
        self, context: ResourceContext, *, engaged: bool, choice: str
    ) -> None:
        """Append one decision to the log."""
        self._log.append(
            AttackDecision(
                party=context.party,
                message_bit=context.message_bit,
                position=context.position,
                engaged=engaged,
                choice=choice,
            )
        )

    def _draw_axis(self, axes: tuple[PauliBasis, ...]) -> PauliBasis:
        """Draw one measurement axis from Eve's own stream."""
        return axes[int(self._rng.integers(len(axes)))]


class DepolarisingChannel(ChannelAttack):
    """Injected depolarising noise, realised as a Pauli twirl on the wire.

    Parameters
    ----------
    strength : float
        ``p`` in ``[0, 1]``.
    rng : numpy.random.Generator
        Required.
    target : Party, str or None, optional
        One link, or both.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.channel import DepolarisingChannel
    >>> from qsecure.protocol.distribute import ResourceContext
    >>> from qsecure.protocol.params import Party
    >>> attack = DepolarisingChannel(0.5, rng=np.random.default_rng(7))
    >>> hops = [
    ...     ResourceContext(party=Party.BOB, message_bit=0, position=i)
    ...     for i in range(6)
    ... ]
    >>> [attack.resource(hop) is not None for hop in hops]
    [True, True, True, True, True, True]
    >>> [entry.choice for entry in attack.log]
    ['pass', 'pass', 'pass', 'I', 'pass', 'X']

    >>> aimed = DepolarisingChannel(
    ...     1.0, rng=np.random.default_rng(7), target=Party.BOB
    ... )
    >>> charlie = ResourceContext(
    ...     party=Party.CHARLIE, message_bit=0, position=0
    ... )
    >>> _ = aimed.resource(charlie)
    >>> aimed.log[0].engaged, aimed.engaged_count
    (False, 0)
    """

    def __init__(
        self,
        strength: float,
        *,
        rng: np.random.Generator,
        target: Party | str | None = None,
    ) -> None:
        super().__init__(rng=rng, target=target)
        self._strength = _as_strength(strength, "strength")

    @property
    def strength(self) -> float:
        """float: ``p``, the fraction of hops that take a Pauli."""
        return self._strength

    @property
    def tensor(self) -> CorrelationTensor:
        """CorrelationTensor: The correlation tensor of the resource ensemble.
        """
        return depolarising_tensor(self._strength)

    def _draw_twirl(self) -> str:
        """Draw this hop's Pauli label from Eve's own stream."""
        if float(self._rng.random()) >= self._strength:
            return "pass"
        return _TWIRL_LABELS[int(self._rng.integers(len(_TWIRL_LABELS)))]

    def _act_on_resource(
        self, pair: Statevector, context: ResourceContext
    ) -> StateLike:
        """Apply the drawn Pauli to the travelling half only."""
        label = self._draw_twirl()
        self._record(context, engaged=label != "pass", choice=label)
        if label in ("pass", "I"):
            return pair
        return as_density(pair).evolve(
            Operator(pauli_matrix(label)), qargs=[_TRAVELLING_QUBIT]
        )

    def _act_on_payload(
        self, state: StateLike, context: ResourceContext
    ) -> StateLike:
        """Apply the drawn Pauli to the one-qubit payload."""
        label = self._draw_twirl()
        self._record(context, engaged=label != "pass", choice=label)
        if label in ("pass", "I"):
            return state
        return as_density(state).evolve(Operator(pauli_matrix(label)))


class InterceptResend(ChannelAttack):
    """Eve measures the travelling half and re-sends what she found.

    Parameters
    ----------
    rng : numpy.random.Generator
        Required.
    target : Party, str or None, optional
        One link, or both.
    axes : iterable of PauliBasis or str, optional
        The alphabet she guesses in.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.channel import InterceptResend
    >>> from qsecure.core.states import concurrence, purity
    >>> from qsecure.protocol.distribute import ResourceContext
    >>> from qsecure.protocol.params import Party
    >>> attack = InterceptResend(rng=np.random.default_rng(11))
    >>> hop = ResourceContext(party=Party.BOB, message_bit=0, position=0)
    >>> delivered = attack.resource(hop)
    >>> attack.log[0].choice
    'X'

    >>> round(purity(delivered), 9), round(concurrence(delivered), 9)
    (1.0, 0.0)
    """

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        target: Party | str | None = None,
        axes: Iterable[PauliBasis | str] = PAULI_AXES,
    ) -> None:
        super().__init__(rng=rng, target=target)
        self._axes = _as_axes(axes)

    @property
    def axes(self) -> tuple[PauliBasis, ...]:
        """tuple of PauliBasis: The alphabet Eve guesses in."""
        return self._axes

    @property
    def tensor(self) -> CorrelationTensor:
        """CorrelationTensor: The correlation tensor of the resource ensemble.
        """
        return collapse_tensor(self._axes)

    def _act_on_resource(
        self, pair: Statevector, context: ResourceContext
    ) -> StateLike:
        """Measure qubit 1 in a drawn axis and hand on the collapsed register.
        """
        from qsecure.core.measure import projective_measure

        axis = self._draw_axis(self._axes)
        self._record(context, engaged=True, choice=axis.value)
        outcome = projective_measure(
            as_density(pair), _TRAVELLING_QUBIT, axis, rng=self._rng
        )
        return outcome.post_state

    def _act_on_payload(
        self, state: StateLike, context: ResourceContext
    ) -> StateLike:
        """Measure the payload in a drawn axis and re-send the eigenstate
        found.
        """
        from qsecure.core.measure import projective_measure

        axis = self._draw_axis(self._axes)
        self._record(context, engaged=True, choice=axis.value)
        outcome = projective_measure(as_density(state), 0, axis, rng=self._rng)
        return outcome.post_state


class KeptShareSwap(ChannelAttack):
    """Eve copies the travelling half onto an ancilla and keeps it.

    Parameters
    ----------
    rng : numpy.random.Generator
        Required.
    target : Party, str or None, optional
        One link, or both.
    axes : iterable of PauliBasis or str, optional
        The axes she copies in.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.channel import KeptShareSwap
    >>> from qsecure.core.states import concurrence, purity
    >>> from qsecure.protocol.distribute import ResourceContext
    >>> from qsecure.protocol.params import Party
    >>> attack = KeptShareSwap(rng=np.random.default_rng(5), axes=("Z",))
    >>> hop = ResourceContext(party=Party.CHARLIE, message_bit=1, position=2)
    >>> delivered = attack.resource(hop)
    >>> [round(value, 9) for value in (purity(delivered), concurrence(delivered))]
    [0.5, 0.0]

    >>> delivered.data.real.round(6).tolist()
    [[0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5]]
    """

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        target: Party | str | None = None,
        axes: Iterable[PauliBasis | str] = PAULI_AXES,
    ) -> None:
        super().__init__(rng=rng, target=target)
        self._axes = _as_axes(axes)

    @property
    def axes(self) -> tuple[PauliBasis, ...]:
        """tuple of PauliBasis: The axes Eve copies in."""
        return self._axes

    @property
    def tensor(self) -> CorrelationTensor:
        """CorrelationTensor: The correlation tensor of the resource ensemble.
        """
        return collapse_tensor(self._axes)

    @staticmethod
    def _dephase(
        state: DensityMatrix, axis: PauliBasis, qubit: int | None
    ) -> DensityMatrix:
        """Return ``(rho + sigma_a rho sigma_a) / 2`` on one qubit."""
        operator = Operator(pauli_matrix(axis.value))
        flipped = (
            state.evolve(operator)
            if qubit is None
            else state.evolve(operator, qargs=[qubit])
        )
        return DensityMatrix(
            (np.asarray(state.data) + np.asarray(flipped.data)) / 2.0
        )

    def _act_on_resource(
        self, pair: Statevector, context: ResourceContext
    ) -> StateLike:
        """Dephase the travelling half into Eve's retained ancilla."""
        axis = self._draw_axis(self._axes)
        self._record(context, engaged=True, choice=axis.value)
        return self._dephase(as_density(pair), axis, _TRAVELLING_QUBIT)

    def _act_on_payload(
        self, state: StateLike, context: ResourceContext
    ) -> StateLike:
        """Dephase the payload into Eve's retained ancilla."""
        axis = self._draw_axis(self._axes)
        self._record(context, engaged=True, choice=axis.value)
        return self._dephase(as_density(state), axis, None)


def _sample_contexts(
    rounds: int, party: Party, message_bit: int
) -> list[ResourceContext]:
    """Build ``rounds`` consecutive hop contexts for one link."""
    if not isinstance(rounds, numbers.Integral) or isinstance(rounds, bool):
        raise TypeError(
            f"rounds must be a positive integer, got {type(rounds).__name__}"
        )
    count = int(rounds)
    if count < 1:
        raise ValueError(
            f"rounds must be at least 1, got {count}: an estimate needs a "
            f"denominator, the same rule estimate_qber and estimate_chsh apply."
        )
    return [
        ResourceContext(party=party, message_bit=message_bit, position=index)
        for index in range(count)
    ]


def measure_qber(
    factory: Any,
    *,
    rounds: int,
    rng: np.random.Generator | None = None,
    party: Party | str = Party.BOB,
    message_bit: int = 0,
    bases: Iterable[PauliBasis | str] = PAULI_AXES,
    **kwargs: Any,
) -> QberEstimate:
    """Run the QBER arm against a resource factory and return the estimate.

    Parameters
    ----------
    factory : callable
        A :data:`~qsecure.protocol.distribute.ResourceFactory` taking a
        :class:`~qsecure.protocol.distribute.ResourceContext` -- normally
        ``attack.resource``.
    rounds : int
        Sample size.
    rng : numpy.random.Generator or None, optional
        The recipients' measurement randomness, distinct from the attack's
       .
    party : Party or str, optional
        Which link the contexts name, so a targeted attack can be measured on
        the link it targets and on the one it does not.
    message_bit : int, optional
        Which distribution the contexts name.
    bases : iterable of PauliBasis or str, optional
        The alphabet the round's basis is drawn from, uniformly.
    **kwargs
        Forwarded to :func:`~qsecure.protocol.checkrounds.estimate_qber`, e.g.
        ``method="hoeffding"`` or ``confidence=0.999``.

    Returns
    -------
    QberEstimate
        Count, sample size and interval.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.channel import (
    ...     InterceptResend, measure_qber, qber_from_tensor
    ... )
    >>> attack = InterceptResend(rng=np.random.default_rng(101))
    >>> estimate = measure_qber(
    ...     attack.resource, rounds=3000, rng=np.random.default_rng(202)
    ... )
    >>> f"{estimate.estimate:.4f}"
    '0.3233'
    >>> estimate.interval.covers(qber_from_tensor(attack.tensor))
    True

    >>> from qsecure.protocol.distribute import ideal_resource
    >>> clean = measure_qber(
    ...     lambda context: ideal_resource(),
    ...     rounds=500,
    ...     rng=np.random.default_rng(303),
    ... )
    >>> clean.errors
    0
    """
    if not callable(factory):
        raise TypeError(
            f"factory must be a callable factory(context) returning a "
            f"two-qubit resource, got {type(factory).__name__}. Pass the "
            f"attack's bound method, e.g. attack.resource."
        )
    generator = resolve_rng(rng)
    alphabet = _as_axes(bases)
    contexts = _sample_contexts(rounds, _as_party(party), message_bit)
    observations = []
    for context in contexts:
        basis = alphabet[int(generator.integers(len(alphabet)))]
        resource = factory(context)
        observations.append(
            observe_qber_round(
                resource, QberRound(context.position, basis), rng=generator
            )
        )
    return estimate_qber(observations, **kwargs)


def measure_chsh(
    factory: Any,
    *,
    rounds: int,
    rng: np.random.Generator | None = None,
    party: Party | str = Party.BOB,
    message_bit: int = 0,
    **kwargs: Any,
) -> ChshEstimate:
    """Run the CHSH arm against a resource factory and return the estimate.

    Parameters
    ----------
    factory : callable
        As :func:`measure_qber`.
    rounds : int
        Sample size; at least four, one per cell.
    rng : numpy.random.Generator or None, optional
        The recipients' measurement randomness, distinct from the attack's
       .
    party : Party or str, optional
        Which link the contexts name.
    message_bit : int, optional
        Which distribution.
    **kwargs
        Forwarded to :func:`~qsecure.protocol.checkrounds.estimate_chsh`.

    Returns
    -------
    ChshEstimate
        The four correlators, their counts and the interval on ``S``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.channel import chsh_from_tensor, measure_chsh
    >>> from qsecure.protocol.distribute import ideal_resource
    >>> ideal = measure_chsh(
    ...     lambda context: ideal_resource(),
    ...     rounds=4000,
    ...     rng=np.random.default_rng(404),
    ... )
    >>> f"{ideal.statistic:.4f}"
    '2.7600'
    >>> ideal.violates_classical_bound, ideal.consistent_with_ideal
    (True, True)

    >>> from qsecure.attacks.channel import InterceptResend
    >>> attack = InterceptResend(rng=np.random.default_rng(505))
    >>> broken = measure_chsh(
    ...     attack.resource, rounds=4000, rng=np.random.default_rng(606)
    ... )
    >>> f"{broken.statistic:.4f}"
    '0.8400'
    >>> broken.violates_classical_bound
    False
    >>> broken.interval.covers(chsh_from_tensor(attack.tensor))
    True
    """
    if not callable(factory):
        raise TypeError(
            f"factory must be a callable factory(context) returning a "
            f"two-qubit resource, got {type(factory).__name__}. Pass the "
            f"attack's bound method, e.g. attack.resource."
        )
    contexts = _sample_contexts(rounds, _as_party(party), message_bit)
    if len(contexts) < 4:
        raise ValueError(
            f"a CHSH estimate needs at least one round in each of the four "
            f"cells, so rounds must be at least 4, got {len(contexts)}."
        )
    generator = resolve_rng(rng)
    observations = []
    for index, context in enumerate(contexts):
        alice_setting = (index // 2) % 2
        recipient_setting = index % 2
        resource = factory(context)
        observations.append(
            observe_chsh_round(
                resource,
                ChshRound(context.position, alice_setting, recipient_setting),
                rng=generator,
            )
        )
    return estimate_chsh(observations, **kwargs)


@dataclass(frozen=True)
class AttributionOutcome:
    """What a party-targeted attack looked like from each of the two vantages.

    Attributes
    ----------
    target : Party
        The link the attack was aimed at.
    check_qber : dict of Party to float
        Each recipient's own QBER, from his own
        :class:`~qsecure.protocol.checkrounds.CheckLog`s, pooled over both
        message bits because both tested the same link.
    check_qber_interval : dict of Party to tuple of float
        The Wilson interval around each, at
        :data:`~qsecure.protocol.checkrounds.CHECK_CONFIDENCE`.
    check_chsh : dict of Party to float
        The same logs' CHSH statistics.
    check_chsh_interval : dict of Party to tuple of float
        Their intervals.
    record_rate : dict of Party to float
        Each recipient's matched-position mismatch rate against Alice's
        declared key, **after** Phase A'.
    unsymmetrised_rate : dict of Party to float
        The same rate from a companion run with
        :func:`~qsecure.protocol.symmetrise.no_symmetrisation`, which is what
        the smearing is smearing.

    Examples
    --------
    >>> from qsecure.attacks.channel import AttributionOutcome
    >>> from qsecure.protocol.params import Party
    >>> outcome = AttributionOutcome(
    ...     target=Party.BOB,
    ...     check_qber={Party.BOB: 0.07, Party.CHARLIE: 0.0},
    ...     check_qber_interval={
    ...         Party.BOB: (0.04, 0.10), Party.CHARLIE: (0.0, 0.02)
    ...     },
    ...     check_chsh={Party.BOB: 2.43, Party.CHARLIE: 2.83},
    ...     check_chsh_interval={
    ...         Party.BOB: (2.2, 2.7), Party.CHARLIE: (2.6, 3.0)
    ...     },
    ...     record_rate={Party.BOB: 0.034, Party.CHARLIE: 0.039},
    ...     unsymmetrised_rate={Party.BOB: 0.069, Party.CHARLIE: 0.0},
    ... )
    >>> outcome.attributable(margin=0.02)
    True
    >>> outcome.records_attributable(margin=0.02)
    False

    >>> outcome.intervals_disjoint()
    True
    """

    target: Party
    check_qber: Mapping[Party, float]
    check_qber_interval: Mapping[Party, tuple[float, float]]
    check_chsh: Mapping[Party, float]
    check_chsh_interval: Mapping[Party, tuple[float, float]]
    record_rate: Mapping[Party, float]
    unsymmetrised_rate: Mapping[Party, float]

    def _separates(
        self, values: Mapping[Party, float], margin: float
    ) -> bool:
        """Return whether ``values`` puts the target clear of the other link.
        """
        others = [
            value for party, value in values.items() if party is not self.target
        ]
        if not others:
            return False
        return values[self.target] - max(others) >= margin

    def attributable(self, *, margin: float = 0.0) -> bool:
        """bool: Whether the **check logs** name the attacked link.

        Parameters
        ----------
        margin : float, optional
            How far the targeted link's QBER must exceed the other's before the
            separation is called real.
        """
        return self._separates(self.check_qber, margin)

    def records_attributable(self, *, margin: float = 0.0) -> bool:
        """bool: Whether the **post-exchange records** name the attacked link.

        Parameters
        ----------
        margin : float, optional
            As :meth:`attributable`.
        """
        return self._separates(self.record_rate, margin)

    def intervals_disjoint(self) -> bool:
        """bool: Whether the two links' QBER intervals fail to overlap."""
        target_low = self.check_qber_interval[self.target][0]
        others = [
            high
            for party, (_low, high) in self.check_qber_interval.items()
            if party is not self.target
        ]
        return bool(others) and target_low > max(others)


def _record_rates(session: Any, message_bit: int) -> dict[Party, float]:
    """Return each recipient's matched-position mismatch rate."""
    from qsecure.protocol.verify import matched_positions, mismatch_positions

    signature = session.signature
    rates: dict[Party, float] = {}
    for party, record in session.records[message_bit].items():
        matched = len(matched_positions(signature, record))
        errors = len(mismatch_positions(signature, record))
        rates[party] = errors / matched if matched else float("nan")
    return rates


def attribution_survives_symmetrisation(
    strength: float = 0.14,
    *,
    key_length: int = 960,
    check_fraction: float = 0.5,
    target: Party | str = Party.BOB,
    attack_seed: int = 3141,
    session_seed: int = 2718,
    message_bit: int = 0,
) -> AttributionOutcome:
    """Mount one party-targeted attack and read it from both vantages.

    Parameters
    ----------
    strength : float, optional
        The depolarising strength on the targeted link.
    key_length : int, optional
        ``L``.
    check_fraction : float, optional
        The share of positions spent on check rounds.
    target : Party or str, optional
        The attacked link.
    attack_seed, session_seed : int, optional
        **deliberately different**.
    message_bit : int, optional
        Which bit is signed, and so which records are scored.

    Examples
    --------
    >>> from qsecure.attacks.channel import attribution_survives_symmetrisation
    >>> outcome = attribution_survives_symmetrisation()

    >>> f"{outcome.check_qber[outcome.target]:.4f}"
    '0.0583'
    >>> f"{min(outcome.check_qber.values()):.4f}"
    '0.0000'
    >>> outcome.attributable(margin=0.02), outcome.intervals_disjoint()
    (True, True)

    >>> outcome.records_attributable(margin=0.02)
    False
    >>> sorted(f"{value:.4f}" for value in outcome.record_rate.values())
    ['0.0197', '0.0316']
    >>> sorted(f"{value:.4f}" for value in outcome.unsymmetrised_rate.values())
    ['0.0000', '0.0537']
    """
    from qsecure.protocol.session import QDSSession

    if attack_seed == session_seed:
        raise ValueError(
            f"attack_seed and session_seed must differ, got {attack_seed!r} "
            f"for both. An adversary built from the session's seed can rebuild "
            f"the session's streams and predict every symmetrisation coin "
            f", and the smearing this function measures would then be "
            f"reported from a run in which Eve already knew the answer."
        )
    aimed = _as_target(target)
    params = ProtocolParams(
        key_length=key_length, check_fraction=check_fraction
    )

    def run(symmetriser: Any) -> Any:
        """Run one session to the end of Phase B and return it."""
        attack = DepolarisingChannel(
            strength, rng=np.random.default_rng(attack_seed), target=aimed
        )
        session = QDSSession(
            params,
            resource_factory=attack.resource,
            symmetriser=symmetriser,
            rng=np.random.default_rng(session_seed),
        )
        session.distribute()
        session.sign(message_bit)
        return session

    session = run(None)
    plain = run(no_symmetrisation)

    check_qber: dict[Party, float] = {}
    check_qber_interval: dict[Party, tuple[float, float]] = {}
    check_chsh: dict[Party, float] = {}
    check_chsh_interval: dict[Party, tuple[float, float]] = {}
    for party in (Party.BOB, Party.CHARLIE):
        logs = [session.check_logs[(party, bit)] for bit in (0, 1)]
        qber = estimate_qber([entry for log in logs for entry in log.qber])
        chsh = estimate_chsh([entry for log in logs for entry in log.chsh])
        check_qber[party] = qber.estimate
        check_qber_interval[party] = (qber.interval.low, qber.interval.high)
        check_chsh[party] = chsh.statistic
        check_chsh_interval[party] = (chsh.interval.low, chsh.interval.high)
    return AttributionOutcome(
        target=aimed,
        check_qber=check_qber,
        check_qber_interval=check_qber_interval,
        check_chsh=check_chsh,
        check_chsh_interval=check_chsh_interval,
        record_rate=_record_rates(session, message_bit),
        unsymmetrised_rate=_record_rates(plain, message_bit),
    )


def payload_line_is_unwatched(
    *,
    key_length: int = 960,
    check_fraction: float = 0.5,
    attack_seed: int = 271828,
    session_seed: int = 314159,
    message_bit: int = 0,
) -> dict[str, float]:
    """Show that a payload-line attack moves the key and not the statistics.

    Parameters
    ----------
    key_length : int, optional
        ``L``, chosen for statistics.
    attack_seed, session_seed : int, optional
        Required to differ.

    Returns
    -------
    dict of str to float
        ``"resource_qber"`` / ``"resource_chsh"`` / ``"resource_record_rate"``
        and the three ``"payload_"`` counterparts, each pooled over both
        recipients.

    Examples
    --------
    >>> from qsecure.attacks.channel import payload_line_is_unwatched
    >>> seen = payload_line_is_unwatched()

    >>> f"{seen['resource_qber']:.4f}", f"{seen['resource_chsh']:.4f}"
    ('0.3604', '0.5782')

    >>> f"{seen['payload_qber']:.4f}", f"{seen['payload_chsh']:.4f}"
    ('0.0000', '2.6676')
    >>> f"{seen['resource_record_rate']:.4f}"
    '0.3458'
    >>> f"{seen['payload_record_rate']:.4f}"
    '0.3458'
    """
    from qsecure.protocol.session import QDSSession

    if attack_seed == session_seed:
        raise ValueError(
            f"attack_seed and session_seed must differ, got {attack_seed!r} "
            f"for both; an adversary sharing the session's seed is exactly what "
            f"isolation forbids."
        )
    params = ProtocolParams(
        key_length=key_length, check_fraction=check_fraction
    )
    seen: dict[str, float] = {}
    for line in ("resource", "payload"):
        attack = InterceptResend(rng=np.random.default_rng(attack_seed))
        seams = (
            {"resource_factory": attack.resource}
            if line == "resource"
            else {"payload_map": attack.payload}
        )
        session = QDSSession(
            params, rng=np.random.default_rng(session_seed), **seams
        )
        session.distribute()
        session.sign(message_bit)
        qber = []
        chsh = []
        for party in (Party.BOB, Party.CHARLIE):
            for bit in (0, 1):
                log = session.check_logs[(party, bit)]
                qber.extend(log.qber)
                chsh.extend(log.chsh)
        rates = _record_rates(session, message_bit)
        seen[f"{line}_qber"] = estimate_qber(qber).estimate
        seen[f"{line}_chsh"] = estimate_chsh(chsh).statistic
        seen[f"{line}_record_rate"] = sum(rates.values()) / len(rates)
    return seen
