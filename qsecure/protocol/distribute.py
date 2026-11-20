"""Phase A: teleport the quantum public key and measure it on arrival.

For each key position Alice prepares the eigenstate named by her private key,
teleports it over a fresh Bell pair, and the recipient measures it immediately
in a random basis, keeping only ``(i, basis, eigenvalue)``.

.. _payload-seam:

The ``payload_map`` seam
------------------------
An optional callable on the payload line, between preparation and Bell
measurement. It is called and its result adopted on every position, check
rounds included, and it consumes no randomness from the run's generator.

.. _check-round-lockstep:

Check rounds and the variate budget
-----------------------------------
Check rounds must be indistinguishable from key rounds to every seam: each seam
is called and its answer read once per position before the branch, malformed
answers are refused identically on both branches, and exactly three variates
are consumed per position either way.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector

from qsecure.core.measure import projective_measure
from qsecure.core.paulis import PauliBasis
from qsecure.core.rng import resolve_rng
from qsecure.core.states import BellState, StateLike, _coerce_state, bell_state
# Shared with teleport so the payload precondition has one definition.
from qsecure.core.teleport import _payload_state, teleport
from qsecure.protocol.checkrounds import (
    CheckLog,
    CheckRoundPlan,
    ChshObservation,
    ChshRound,
    QberObservation,
    QberRound,
    observe_chsh_round,
    observe_qber_round,
)
from qsecure.protocol.keys import PrivateKey
from qsecure.protocol.params import (
    VERIFIERS,
    Party,
    ProtocolParams,
    _as_party,
)
from qsecure.protocol.records import RecipientRecord

__all__ = [
    "RecipientDistribution",
    "ResourceContext",
    "ResourceFactory",
    "PayloadMap",
    "ideal_resource",
    "identity_payload",
    "distribute_to_recipient",
    "distribute_to_recipient_with_checks",
    "distribute_public_key",
    "distribute_public_key_with_checks",
    "accepts_context",
]


@dataclass(frozen=True)
class ResourceContext:
    """Which hop a ``resource_factory`` or ``payload_map`` is being asked for.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.protocol.distribute import (
    ...     ResourceContext, distribute_public_key, ideal_resource
    ... )
    >>> from qsecure.protocol.keys import generate_private_key
    >>> from qsecure.protocol.params import Party, ProtocolParams
    >>> seen = []
    >>> def watcher(context: ResourceContext):
    ...     seen.append((context.party, context.position))
    ...     return ideal_resource()
    >>> params = ProtocolParams(key_length=4)
    >>> key = generate_private_key(params, 1, rng=np.random.default_rng(0))
    >>> _ = distribute_public_key(
    ...     key, params, resource_factory=watcher, rng=np.random.default_rng(1)
    ... )
    >>> seen[0], seen[-1]
    ((<Party.BOB: 'Bob'>, 0), (<Party.CHARLIE: 'Charlie'>, 3))
    """

    party: Party
    message_bit: int
    position: int


ResourceFactory: TypeAlias = (
    Callable[[], StateLike] | Callable[[ResourceContext], StateLike]
)
"""Callable ``factory()`` or ``factory(context)`` returning a 2-qubit pair."""


PayloadMap: TypeAlias = Callable[[StateLike, ResourceContext], StateLike]
"""Callable ``payload_map(state, context)`` returning the state to send."""


def ideal_resource() -> Statevector:
    """Return a fresh, clean ``|Phi+>`` pair (the default resource).

    Examples
    --------
    >>> from qsecure.protocol.distribute import ideal_resource
    >>> ideal_resource().data.round(6)
    array([0.707107+0.j, 0.      +0.j, 0.      +0.j, 0.707107+0.j])
    """
    return bell_state(BellState.PHI_PLUS)


def identity_payload(payload: StateLike, context: ResourceContext) -> StateLike:
    """Return ``payload`` unchanged: the honest payload line.

    Examples
    --------
    >>> from qsecure.protocol.distribute import ResourceContext, identity_payload
    >>> from qsecure.protocol.keys import KeyElement
    >>> from qsecure.protocol.params import Party
    >>> state = KeyElement("X", 1).state()
    >>> context = ResourceContext(party=Party.BOB, message_bit=0, position=3)
    >>> identity_payload(state, context) is state
    True
    """
    del context  # An honest preparation is the same wherever it is sent.
    return payload


def _adopt_state(state: StateLike) -> StateLike:
    """Copy a seam's state into a protocol-owned object, reading it once.

    Keeps the representation (Statevector or DensityMatrix) and dimensions.
    Anything not array-like is returned untouched for the downstream validator.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.protocol.distribute import _adopt_state, ideal_resource
    >>> pair = ideal_resource()
    >>> adopted = _adopt_state(pair)
    >>> type(adopted).__name__
    'Statevector'
    >>> bool(np.allclose(adopted.data, pair.data))
    True
    >>> adopted is pair
    False

    A counting subclass is read exactly once:

    >>> from qiskit.quantum_info import Statevector
    >>> class Counting(Statevector):
    ...     reads = 0
    ...     @property
    ...     def data(self):
    ...         Counting.reads += 1
    ...         return self._data
    >>> adopted = _adopt_state(Counting(ideal_resource().data))
    >>> Counting.reads
    1
    >>> type(adopted).__name__
    'Statevector'
    """
    if isinstance(state, DensityMatrix):
        return DensityMatrix(
            np.array(state.data, copy=True), dims=state.dims()
        )
    if isinstance(state, Statevector):
        return Statevector(np.array(state.data, copy=True), dims=state.dims())
    try:
        return np.array(state, dtype=np.complex128)
    except (TypeError, ValueError):
        # Not array-like: let the downstream validator report it.
        return state


def _map_payload(
    payload_map: PayloadMap | None,
    payload: StateLike,
    context: ResourceContext,
) -> StateLike:
    """Put one payload through ``payload_map`` and validate the result.

    Validation happens here, above the check/key branch, so a malformed payload
    is refused identically on both branches.

    Raises
    ------
    TypeError
        If ``payload_map`` is not callable.
    ValueError
        If it returned ``None`` or anything teleport would refuse as a payload.

    Examples
    --------
    >>> import numpy as np
    >>> from qiskit.quantum_info import DensityMatrix, Statevector
    >>> from qsecure.protocol.distribute import _map_payload, ResourceContext
    >>> from qsecure.protocol.params import Party
    >>> context = ResourceContext(
    ...     party=Party.BOB, message_bit=0, position=7)
    >>> malformed = [
    ...     Statevector([1.0, 0.0, 0.0, 0.0]),                  # two qubits
    ...     Statevector(np.array([2.0, 0.0])),                  # unnormalised
    ...     DensityMatrix(np.array([[0.5, 1.0], [0.0, 0.5]])),  # non-Hermitian
    ...     DensityMatrix(np.diag([2.0, -1.0])),                # negative eig
    ...     np.array([np.nan, 0.0], dtype=complex),             # non-finite
    ...     np.array([1.0, 0.0, 0.0]),                          # dimension 3
    ...     "not a state at all",
    ... ]
    >>> def refused(bad):
    ...     try:
    ...         _map_payload(lambda state, ctx: bad,
    ...                      Statevector([1.0, 0.0]), context)
    ...     except ValueError as exc:
    ...         return str(exc).startswith(
    ...             "payload_map returned a state teleport cannot send, "
    ...             "for Bob at key position 7 of message bit 0")
    ...     return False
    >>> [refused(bad) for bad in malformed]
    [True, True, True, True, True, True, True]

    ``payload_map=None`` returns the payload directly:

    >>> sent = _map_payload(None, Statevector([1.0, 0.0]), context)
    >>> sent.data.round(6)
    array([1.+0.j, 0.+0.j])
    """
    if payload_map is None:
        return payload
    if not callable(payload_map):
        raise TypeError(
            f"payload_map must be a callable payload_map(state, context) "
            f"returning a one-qubit state, got "
            f"{type(payload_map).__name__}. Leave it None for the honest "
            f"payload line, or pass identity_payload to say so explicitly."
        )
    mapped = payload_map(payload, context)
    if mapped is None:
        raise ValueError(
            f"payload_map returned None for {context.party.value} at key "
            f"position {context.position} of message bit "
            f"{context.message_bit}; it must return the one-qubit state to "
            f"teleport. Return the payload it was given -- or "
            f"identity_payload(state, context) -- to send the eigenstate "
            f"unaltered."
        )
    adopted = _adopt_state(mapped)
    try:
        # Validate the copy: re-reading the seam's object leaks a read count.
        _payload_state(adopted)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"payload_map returned a state teleport cannot send, for "
            f"{context.party.value} at key position {context.position} of "
            f"message bit {context.message_bit}: {exc}"
        ) from exc
    return adopted


def _resolve_factory(
    resource_factory: ResourceFactory | None,
) -> tuple[ResourceFactory, bool]:
    """Validate the factory; return it and whether it takes a context."""
    if resource_factory is None:
        return ideal_resource, False
    if not callable(resource_factory):
        raise TypeError(
            f"resource_factory must be a callable returning a two-qubit state "
            f"-- either factory() or factory(context) -- got "
            f"{type(resource_factory).__name__}. It is invoked once per key "
            f"position so that the resource may vary along the run; to use one "
            f"fixed pair, wrap it: resource_factory=lambda: my_pair."
        )
    return resource_factory, accepts_context(resource_factory)


def accepts_context(resource_factory: ResourceFactory) -> bool:
    """Return ``True`` when ``resource_factory`` requires one positional arg.

    A callable that can be invoked with no arguments (e.g. ``lambda pair=pair:
    pair``) is called with none.

    Raises
    ------
    TypeError
        If the callable accepts neither zero arguments nor one.
    """
    try:
        signature = inspect.signature(resource_factory)
    except (TypeError, ValueError):
        # No introspectable signature: treat as zero-argument.
        return False
    try:
        signature.bind()
    except TypeError:
        pass
    else:
        return False
    try:
        signature.bind(
            ResourceContext(party=Party.BOB, message_bit=0, position=0)
        )
    except TypeError:
        raise TypeError(
            f"resource_factory must be callable either with no arguments or "
            f"with one positional ResourceContext, but {resource_factory!r} "
            f"accepts neither: its signature is {signature}. The context "
            f"carries the party, message bit and key position of the hop being "
            f"requested; a factory that does not want it should take no "
            f"arguments at all."
        ) from None
    return True


_accepts_context = accepts_context


def _draw_resource(
    resource_factory: ResourceFactory,
    wants_context: bool,
    context: ResourceContext,
) -> StateLike:
    """Call the factory for one position; adopt and validate the result.

    Raises
    ------
    ValueError
        If the factory returned ``None`` (teleport would treat it as the
        ideal pair) or anything that is not a two-qubit state.
    """
    resource = (
        resource_factory(context)  # type: ignore[call-arg]
        if wants_context
        else resource_factory()  # type: ignore[call-arg]
    )
    index = context.position
    if resource is None:
        raise ValueError(
            f"resource_factory returned None for key position {index}; it must "
            f"return a two-qubit entanglement resource. None is not forwarded "
            f"to teleport(), where it would silently select the ideal "
            f"|Phi+> pair -- a mis-wired attack would then be indistinguishable "
            f"from a clean channel. Return ideal_resource() explicitly if a "
            f"clean pair is what you meant."
        )
    adopted = _adopt_state(resource)
    try:
        # Check the copy, not the factory's own object (one read per position).
        if _coerce_state(adopted).num_qubits != 2:
            raise ValueError(
                f"a resource must be a two-qubit state (the shared entangled "
                f"pair), got {_coerce_state(adopted).num_qubits} qubits."
            )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"resource_factory returned a state the protocol cannot use, for "
            f"{context.party.value} at key position {index} of message bit "
            f"{context.message_bit}: {exc}"
        ) from exc
    return adopted


def _resolve_verifier(party: Party | str) -> Party:
    """Coerce ``party`` to a :class:`Party`, refusing Alice."""
    resolved = _as_party(party)
    if resolved is Party.ALICE:
        raise ValueError(
            "cannot distribute a public key to Alice: she is the signer, "
            "prepares the states and keeps the private key, and never measures, "
            "so she has no record to build. Distribute to Party.BOB and "
            "Party.CHARLIE -- both, since transferability is a statement about "
            "a second verifier."
        )
    return resolved


@dataclass(frozen=True)
class RecipientDistribution:
    """One recipient's sifted record, published check log, and plan.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.protocol.checkrounds import draw_check_plan, estimate_qber
    >>> from qsecure.protocol.distribute import (
    ...     distribute_to_recipient_with_checks
    ... )
    >>> from qsecure.protocol.keys import generate_private_key
    >>> from qsecure.protocol.params import Party, ProtocolParams
    >>> params = ProtocolParams(key_length=64, check_fraction=0.25)
    >>> plan = draw_check_plan(params, rng=np.random.default_rng(1))
    >>> key = generate_private_key(params, 0, rng=np.random.default_rng(2))
    >>> outcome = distribute_to_recipient_with_checks(
    ...     key, params, party=Party.BOB, check_plan=plan,
    ...     rng=np.random.default_rng(3),
    ... )
    >>> len(outcome.record) == params.signing_length == 48
    True
    >>> outcome.record.check_against(params.sifted())

    This link measures its own half of the reserved positions:

    >>> params.check_count, outcome.log.round_count
    (16, 8)
    >>> outcome.log.positions == plan.positions_for(Party.BOB)
    True
    >>> estimate_qber(outcome.log.qber).errors        # an ideal channel
    0
    """

    record: RecipientRecord
    log: CheckLog
    plan: CheckRoundPlan | None


def _resolve_plan(
    check_plan: CheckRoundPlan | None, params: ProtocolParams
) -> CheckRoundPlan | None:
    """Validate a check-round plan against ``params``.

    Raises
    ------
    TypeError
        If ``check_plan`` is not a ``CheckRoundPlan`` or ``None``.
    ValueError
        If the plan disagrees with ``params``, or ``params`` reserves check
        rounds and no plan was given.
    """
    if check_plan is None:
        if params.has_check_rounds:
            raise ValueError(
                f"params.check_fraction is {params.check_fraction!r}, which "
                f"reserves {params.check_count} of {params.key_length} "
                f"positions for sampled parameter estimation and shortens the "
                f"effective key to {params.signing_length}, but no check_plan "
                f"was given. Distributing anyway would build a full-length "
                f"record and then score it against matched-count floors "
                f"computed for the shorter key -- every bound weakened with "
                f"nothing to show for it. Draw a plan from the RECIPIENTS' "
                f"stream: draw_check_plan(params, rng=recipient_rng). To run "
                f"without estimation, use params.with_check_fraction(0.0)."
            )
        return None
    if not isinstance(check_plan, CheckRoundPlan):
        raise TypeError(
            f"check_plan must be a CheckRoundPlan or None, got "
            f"{type(check_plan).__name__}. Draw one with "
            f"draw_check_plan(params, rng=...)."
        )
    if check_plan.key_length != params.key_length:
        raise ValueError(
            f"check_plan was drawn for a key length of "
            f"{check_plan.key_length} but params.key_length is "
            f"{params.key_length}. A plan indexes positions of one specific "
            f"run and is not portable to another length."
        )
    if check_plan.check_count != params.check_count:
        raise ValueError(
            f"check_plan designates {check_plan.check_count} check rounds but "
            f"params.check_fraction={params.check_fraction!r} implies "
            f"{params.check_count} at L={params.key_length}. The two must "
            f"agree: params.signing_length is what every matched-count floor "
            f"is derived from, and the plan is what actually decides how many "
            f"positions survive."
        )
    return check_plan


def distribute_to_recipient(
    key: PrivateKey,
    params: ProtocolParams,
    *,
    party: Party | str,
    resource_factory: ResourceFactory | None = None,
    rng: np.random.Generator | None = None,
    check_plan: CheckRoundPlan | None = None,
    payload_map: PayloadMap | None = None,
) -> RecipientRecord:
    """Teleport the public key to one recipient and measure it on arrival.

    Wrapper over :func:`distribute_to_recipient_with_checks` without the log.

    Parameters
    ----------
    party : Party or str
        Bob or Charlie; Alice is refused.
    resource_factory : callable or None, optional
        Entanglement resource per position; defaults to :func:`ideal_resource`.
    rng : numpy.random.Generator or None, optional
        Three variates are consumed per position.
    check_plan : CheckRoundPlan or None, optional
        Positions spent on parameter estimation; the record is returned sifted.
    payload_map : callable or None, optional
        See :ref:`payload-seam`.

    Returns
    -------
    RecipientRecord
        ``len(key)`` entries, or ``check_plan.signing_length`` with a plan.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.protocol.distribute import distribute_to_recipient
    >>> from qsecure.protocol.keys import generate_private_key
    >>> from qsecure.protocol.params import Party, ProtocolParams
    >>> params = ProtocolParams(key_length=6)
    >>> key = generate_private_key(params, 0, rng=np.random.default_rng(1))
    >>> record = distribute_to_recipient(
    ...     key, params, party=Party.BOB, rng=np.random.default_rng(2)
    ... )
    >>> len(record), record.party, record.message_bit
    (6, <Party.BOB: 'Bob'>, 0)
    >>> matched = [
    ...     i for i in range(len(key)) if record.bases[i] == key.bases[i]
    ... ]
    >>> all(record.eigenvalues[i] == key.eigenvalues[i] for i in matched)
    True
    """
    return distribute_to_recipient_with_checks(
        key,
        params,
        party=party,
        resource_factory=resource_factory,
        rng=rng,
        check_plan=check_plan,
        payload_map=payload_map,
    ).record


def distribute_to_recipient_with_checks(
    key: PrivateKey,
    params: ProtocolParams,
    *,
    party: Party | str,
    resource_factory: ResourceFactory | None = None,
    rng: np.random.Generator | None = None,
    check_plan: CheckRoundPlan | None = None,
    payload_map: PayloadMap | None = None,
) -> RecipientDistribution:
    """Run Phase A for one recipient, keeping the check-round diagnostics.

    A key round teleports the key element and measures it; a check round
    measures both halves of the pair and discards the payload. Reserved
    positions dealt to the other link run as key rounds and are dropped.

    Parameters
    ----------
    key : PrivateKey
        Alice's full-length key for one message bit.
    party : Party or str
        Bob or Charlie; Alice is refused.
    check_plan : CheckRoundPlan or None, optional
        Shared by both recipients; ``None`` returns an empty log.

    Other parameters and errors as :func:`distribute_to_recipient`.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.protocol.checkrounds import draw_check_plan, estimate_chsh
    >>> from qsecure.protocol.distribute import (
    ...     distribute_to_recipient_with_checks
    ... )
    >>> from qsecure.protocol.keys import generate_private_key
    >>> from qsecure.protocol.params import Party, ProtocolParams
    >>> params = ProtocolParams(key_length=800, check_fraction=0.5)
    >>> plan = draw_check_plan(params, rng=np.random.default_rng(41))
    >>> key = generate_private_key(params, 1, rng=np.random.default_rng(42))
    >>> outcome = distribute_to_recipient_with_checks(
    ...     key, params, party=Party.CHARLIE, check_plan=plan,
    ...     rng=np.random.default_rng(43),
    ... )
    >>> len(outcome.record), outcome.log.round_count
    (400, 200)
    >>> estimate_chsh(outcome.log.chsh).violates_classical_bound
    True

    The payload seam sees every position, in order:

    >>> small = ProtocolParams(key_length=40, check_fraction=0.25)
    >>> small_plan = draw_check_plan(small, rng=np.random.default_rng(1))
    >>> small_key = generate_private_key(small, 0, rng=np.random.default_rng(2))
    >>> seen = []
    >>> def watch(state, context):
    ...     seen.append(context.position)
    ...     return state
    >>> watched = distribute_to_recipient_with_checks(
    ...     small_key, small, party=Party.BOB, check_plan=small_plan,
    ...     payload_map=watch, rng=np.random.default_rng(3),
    ... )
    >>> seen == list(range(40))
    True
    >>> set(seen) >= set(small_plan.positions)
    True
    """
    if not isinstance(key, PrivateKey):
        raise TypeError(
            f"key must be a PrivateKey, got {type(key).__name__}. Draw one with "
            f"generate_private_key(params, message_bit, rng=...)."
        )
    if not isinstance(params, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(params).__name__}"
        )
    recipient = _resolve_verifier(party)
    key.check_against(params)
    plan = _resolve_plan(check_plan, params)
    factory, wants_context = _resolve_factory(resource_factory)
    generator = resolve_rng(rng)

    alphabet = params.bases
    chosen_bases: list[PauliBasis] = []
    eigenvalues: list[int] = []
    qber_seen: list[QberObservation] = []
    chsh_seen: list[ChshObservation] = []
    # Positions this link checks vs. all reserved positions; reserved positions
    # dealt to the other link run as key rounds and are dropped.
    planned = {} if plan is None else plan.rounds_by_position(recipient)
    reserved: frozenset[int] = (
        frozenset() if plan is None else frozenset(plan.positions)
    )

    for index, element in enumerate(key.elements):
        # Drawn on every position (discarded on check rounds) to keep three
        # variates per position.
        basis = alphabet[int(generator.integers(len(alphabet)))]

        # Drawn and adopted before the branch so it cannot be distinguished.
        context = ResourceContext(
            party=recipient,
            message_bit=key.message_bit,
            position=index,
        )
        resource = _draw_resource(factory, wants_context, context)

        # Called and adopted on every position; discarded on check rounds.
        payload = _map_payload(payload_map, element.state(), context)

        scheduled = planned.get(index)
        if scheduled is None:
            # Key round. The received state is one qubit, so index 0.
            hop = teleport(payload, resource=resource, rng=generator)
            outcome = projective_measure(hop.received, 0, basis, rng=generator)
            if index not in reserved:
                chosen_bases.append(basis)
                eigenvalues.append(outcome.eigenvalue)
            # Reserved for the other link: measured but not recorded.
        elif isinstance(scheduled, QberRound):
            # Check round, QBER arm; the payload is discarded.
            qber_seen.append(
                observe_qber_round(resource, scheduled, rng=generator)
            )
        elif isinstance(scheduled, ChshRound):
            # Check round, CHSH arm.
            chsh_seen.append(
                observe_chsh_round(resource, scheduled, rng=generator)
            )
        else:
            # Unreachable via draw_check_plan.
            raise TypeError(
                f"check_plan schedules an unknown kind of round at position "
                f"{index}: {type(scheduled).__name__}. Expected a QberRound or "
                f"a ChshRound."
            )

    return RecipientDistribution(
        record=RecipientRecord.from_measurements(
            recipient, key.message_bit, chosen_bases, eigenvalues
        ),
        log=CheckLog(
            party=recipient,
            message_bit=key.message_bit,
            qber=tuple(qber_seen),
            chsh=tuple(chsh_seen),
        ),
        plan=plan,
    )


def distribute_public_key(
    key: PrivateKey,
    params: ProtocolParams,
    *,
    parties: Sequence[Party | str] = VERIFIERS,
    resource_factory: ResourceFactory | None = None,
    rng: np.random.Generator | None = None,
    check_plan: CheckRoundPlan | None = None,
    payload_map: PayloadMap | None = None,
) -> dict[Party, RecipientRecord]:
    """Distribute one copy of the public key to each recipient.

    Returns one raw record per party; symmetrise them before verifying.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.protocol.distribute import distribute_public_key
    >>> from qsecure.protocol.keys import generate_private_key
    >>> from qsecure.protocol.params import Party, ProtocolParams
    >>> params = ProtocolParams(key_length=8)
    >>> key = generate_private_key(params, 1, rng=np.random.default_rng(3))
    >>> records = distribute_public_key(key, params, rng=np.random.default_rng(4))
    >>> sorted(records)
    [<Party.BOB: 'Bob'>, <Party.CHARLIE: 'Charlie'>]
    >>> records[Party.BOB].bases == records[Party.CHARLIE].bases
    False
    """
    return {
        party: outcome.record
        for party, outcome in distribute_public_key_with_checks(
            key,
            params,
            parties=parties,
            resource_factory=resource_factory,
            rng=rng,
            check_plan=check_plan,
            payload_map=payload_map,
        ).items()
    }


def distribute_public_key_with_checks(
    key: PrivateKey,
    params: ProtocolParams,
    *,
    parties: Sequence[Party | str] = VERIFIERS,
    resource_factory: ResourceFactory | None = None,
    rng: np.random.Generator | None = None,
    check_plan: CheckRoundPlan | None = None,
    payload_map: PayloadMap | None = None,
) -> dict[Party, RecipientDistribution]:
    """Distribute to each recipient, keeping every check-round diagnostic.

    One generator and one plan are shared by all parties. Keep the two check
    logs separate: they sample two different channels.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.protocol.checkrounds import draw_check_plan, estimate_qber
    >>> from qsecure.protocol.distribute import (
    ...     distribute_public_key_with_checks
    ... )
    >>> from qsecure.protocol.keys import generate_private_key
    >>> from qsecure.protocol.params import Party, ProtocolParams
    >>> params = ProtocolParams(key_length=200, check_fraction=0.5)
    >>> plan = draw_check_plan(params, rng=np.random.default_rng(8))
    >>> key = generate_private_key(params, 0, rng=np.random.default_rng(9))
    >>> outcomes = distribute_public_key_with_checks(
    ...     key, params, check_plan=plan, rng=np.random.default_rng(10)
    ... )

    The two links measure disjoint halves of the reserved set:

    >>> bob = set(outcomes[Party.BOB].log.positions)
    >>> charlie = set(outcomes[Party.CHARLIE].log.positions)
    >>> bob & charlie, bob | charlie == set(plan.positions)
    (set(), True)
    >>> estimate_qber(outcomes[Party.BOB].log.qber).estimate
    0.0
    """
    if isinstance(parties, (str, bytes)) or not isinstance(parties, Sequence):
        raise TypeError(
            f"parties must be a sequence of Party members, got "
            f"{type(parties).__name__}; the default is VERIFIERS, "
            f"(Party.BOB, Party.CHARLIE)."
        )
    resolved = [_resolve_verifier(party) for party in parties]
    if not resolved:
        raise ValueError(
            "parties must name at least one recipient; distributing to nobody "
            "produces no record and no verification is possible."
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            f"parties must be distinct, got "
            f"{[party.value for party in resolved]}. One recipient holds one "
            f"record per message bit; a repeat would silently overwrite it."
        )
    generator = resolve_rng(rng)
    return {
        party: distribute_to_recipient_with_checks(
            key,
            params,
            party=party,
            resource_factory=resource_factory,
            rng=generator,
            check_plan=check_plan,
            payload_map=payload_map,
        )
        for party in resolved
    }
