"""Orchestration: one QDS run, from entanglement to a transferred signature.

:class:`QDSSession` runs the phases in order -- ``distribute()`` (A, plus the
recipients' symmetrisation A'), ``sign(b)`` (B), ``exchange_counts()`` (C'),
``verify(party)`` and ``transfer()`` (C) -- and ``transcript()`` freezes the run
into a JSON-serialisable :class:`SessionTranscript`. ``run(b)`` does all of it.

.. _seams:

Seams
-----
Every place an adversary can stand is a keyword-only constructor argument
holding a callable that defaults to the honest implementation, so an attacked
run and a clean run execute the same code.

.. _two-log-signer:

The ``signer`` seam is shown no recipient log (:data:`NO_RECIPIENT_LOGS`) unless
the session is built with ``signer_sees_recipient_logs=True``; that over-powered
variant gets the raw, pre-exchange logs and is flagged in the transcript.

.. _forger-route:

A forging recipient stands on the ``forwarder`` seam, not the ``signer``: Bob
scores Alice's real declaration, and the hop substitutes what he would have
Charlie believe.

.. _two-streams:

Session streams
---------------
The caller's generator is drawn from once and split into labelled streams:
Alice's (key generation and the ``distributor`` seam), the recipients'
(check-round plans and symmetrisation coins) and the session's own binding
stream. No Alice-side seam is ever handed the recipients' stream.

>>> import numpy as np
>>> from qsecure.protocol.params import ProtocolParams
>>> from qsecure.protocol.session import QDSSession
>>> params = ProtocolParams(key_length=24)
>>> transcript = QDSSession(params, rng=np.random.default_rng(0)).run(1)
>>> transcript.transferable
True
>>> transcript.bob.rate, transcript.charlie.rate
(0.0, 0.0)
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Protocol

import numpy as np
from qiskit.quantum_info import DensityMatrix, partial_trace

from qsecure.core.rng import resolve_rng
from qsecure.core.states import (
    BellState,
    StateLike,
    as_density,
    bell_state,
    concurrence,
    fidelity,
    purity,
)
from qsecure.protocol.analysis import repudiation_bound
from qsecure.protocol.checkrounds import (
    _ALICE_QUBIT,
    _RECIPIENT_QUBIT,
    CheckLog,
    CheckRole,
    CheckRoundPlan,
    ChshRound,
    QberRound,
    draw_check_plan,
)
from qsecure.protocol.distribute import (
    PayloadMap,
    RecipientDistribution,
    ResourceContext,
    ResourceFactory,
    _draw_resource,
    _resolve_factory,
    distribute_public_key_with_checks,
)
from qsecure.protocol.keys import PrivateKey, generate_key_pair
from qsecure.protocol.params import (
    DEMO_PARAMS,
    VERIFIERS,
    Party,
    ProtocolParams,
    _as_message_bit,
    _as_party,
)
from qsecure.protocol.records import RecipientRecord, RecipientView
from qsecure.protocol.signature import (
    Signature,
    fresh_opening,
    session_identifier,
    sign,
)
from qsecure.protocol.symmetrise import Symmetriser, symmetrise_records
from qsecure.protocol.tally import (
    CountExchange,
    MatchedCountMessage,
    PooledMatchedCounts,
    exchange_matched_counts,
    matched_count_message,
)
from qsecure.protocol.verify import (
    AbortReason,
    ConsumedRecords,
    MatchedSetTooSmall,
    VerificationAbort,
    VerificationResult,
    minimum_matched_count,
    minimum_pooled_matched_count,
    verify,
)

__all__ = [
    "COUNTS_AFTER_FORWARDING",
    "COUNTS_BEFORE_FORWARDING",
    "COUNT_EXCHANGE_TIMINGS",
    "MESSAGE_BITS",
    "NO_RECIPIENT_LOGS",
    "ChannelMonitor",
    "ChannelSample",
    "Distributor",
    "Signer",
    "Forwarder",
    "WithheldRecords",
    "honest_signer",
    "honest_forwarder",
    "forwarder_wants_view",
    "SessionTranscript",
    "QDSSession",
]


COUNTS_BEFORE_FORWARDING: Final[str] = "before-forwarding"
"""Phase C' runs before the hop; both counts name the declaration Alice signed."""

COUNTS_AFTER_FORWARDING: Final[str] = "after-forwarding"
"""Phase C' runs after the hop; each recipient counts what he actually
received.
"""

COUNT_EXCHANGE_TIMINGS: Final[tuple[str, str]] = (
    COUNTS_BEFORE_FORWARDING,
    COUNTS_AFTER_FORWARDING,
)
"""The two orderings :class:`QDSSession` accepts, in order of strength."""

MESSAGE_BITS: Final[tuple[int, int]] = (0, 1)
"""The message bits a session distributes for, in order."""


# --------------------------------------------------------------------------- #
# The two session streams


_STREAM_MATERIAL_BYTES: Final[int] = 32
"""Bytes drawn from the caller's generator to derive the session's two streams."""

_ALICE_STREAM_LABEL: Final[bytes] = b"qsecure.protocol.session/alice"
"""Domain separator for the stream Alice and her seams draw from."""

_RECIPIENT_STREAM_LABEL: Final[bytes] = b"qsecure.protocol.session/recipients"
"""Domain separator for the stream Bob and Charlie's coins come from."""

_BINDING_STREAM_LABEL: Final[bytes] = b"qsecure.protocol.session/binding"
"""Domain separator for the stream the session openings are drawn from."""


def _derive_stream(material: bytes, label: bytes) -> np.random.Generator:
    """Derive one labelled generator from the session's seed material.

    >>> import hashlib
    >>> from qsecure.protocol.session import _derive_stream
    >>> material = bytes(32)
    >>> _derive_stream(material, b"one").bytes(4) == _derive_stream(
    ...     material, b"one"
    ... ).bytes(4)
    True
    >>> _derive_stream(material, b"one").bytes(4) == _derive_stream(
    ...     material, b"two"
    ... ).bytes(4)
    False
    >>> seeded = _derive_stream(material, b"one")
    >>> seeded.bit_generator.seed_seq.entropy == int.from_bytes(
    ...     hashlib.sha256(b"one:" + material).digest(), "big"
    ... )
    True
    """
    digest = hashlib.sha256(label + b":" + material).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


# --------------------------------------------------------------------------- #
# What the signer seam is shown


class WithheldRecords(Mapping[int, Mapping[Party, RecipientRecord]]):
    """An empty records mapping that says why it is empty.

    >>> from qsecure.protocol.session import NO_RECIPIENT_LOGS
    >>> len(NO_RECIPIENT_LOGS), list(NO_RECIPIENT_LOGS)
    (0, [])
    >>> NO_RECIPIENT_LOGS.get(0, {})
    {}
    >>> try:
    ...     NO_RECIPIENT_LOGS[0]
    ... except KeyError as missing:
    ...     print(missing.args[0].split(":")[0])
    the signer seam was shown no recipient log
    """

    __slots__ = ("_reason",)

    def __init__(self, reason: str | None = None) -> None:
        self._reason = (
            "the signer seam was shown no recipient log: no adversary in the "
            "threat model holds one at signing time. Construct "
            "QDSSession(..., signer_sees_recipient_logs=True) to run the "
            "over-powered variant deliberately; the transcript flags it. A "
            "forging recipient belongs on the forwarder seam, which is handed "
            "his own RecipientView."
            if reason is None
            else reason
        )

    def __getitem__(self, key: int) -> Mapping[Party, RecipientRecord]:
        """Raise :exc:`KeyError` carrying the reason, for every key."""
        raise KeyError(self._reason)

    def __iter__(self) -> Iterator[int]:
        """Yield nothing: there is no log here to iterate over."""
        return iter(())

    def __len__(self) -> int:
        """int: ``0``, always."""
        return 0

    def __repr__(self) -> str:
        """Return a representation that names the class, not an empty dict."""
        return "WithheldRecords()"


NO_RECIPIENT_LOGS: Final[WithheldRecords] = WithheldRecords()
"""The default ``records`` argument to the :class:`Signer` seam."""


# --------------------------------------------------------------------------- #
# The channel monitor


class ChannelMonitor(Protocol):
    """Callable that summarises one check round's entanglement resource."""

    def __call__(
        self, resource: StateLike, context: ResourceContext
    ) -> Mapping[str, Any]:
        """Return extra diagnostics for this check round."""
        ...


def _as_json_value(value: Any, path: str) -> Any:
    """Coerce one monitor-supplied value to something :func:`json.dumps`
    accepts.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(
                f"channel_monitor returned a non-finite number at {path}: "
                f"{value!r}. json.dumps would write it as NaN or Infinity, "
                f"which is not JSON and which no other reader accepts, so the "
                f"transcript would stop being portable. Report a sentinel "
                f"(None) or a flag instead."
            )
        return number
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (complex, np.complexfloating)):
        raise TypeError(
            f"channel_monitor returned a complex number at {path}: {value!r}. "
            f"JSON has no complex type, and picking a convention here -- a "
            f"two-element list? an object with re and im? -- would make every "
            f"reader guess. Publish the parts explicitly, "
            f"[float(z.real), float(z.imag)], or the modulus, and say in the "
            f"key which it is. A whole density matrix comes out as "
            f"rho.real.tolist() and rho.imag.tolist() under two keys."
        )
    if isinstance(value, np.ndarray):
        return _as_json_value(value.tolist(), path)
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"channel_monitor returned a non-string key at {path}: "
                    f"{key!r} of type {type(key).__name__}. JSON object keys "
                    f"are strings, and a transcript that needed an encoder "
                    f"would not round-trip."
                )
            converted[key] = _as_json_value(item, f"{path}.{key}")
        return converted
    if isinstance(value, (list, tuple)):
        return [
            _as_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"channel_monitor returned {type(value).__name__} at {path}, which has "
        f"no JSON representation. A ChannelSample is carried in the transcript "
        f"and the transcript must survive json.dumps end to end; summarise the "
        f"object into numbers or strings in the monitor itself, which is the "
        f"one place that knows what the summary should be."
    )


def _as_pair(
    resource: StateLike, context: ResourceContext
) -> DensityMatrix:
    """Coerce one hop's resource and refuse anything that is not a pair."""
    density = as_density(resource)
    if density.num_qubits != 2:
        raise ValueError(
            f"a check round measures both halves of an entanglement "
            f"resource, so the resource must be a two-qubit state; got "
            f"{density.num_qubits} qubit(s) for {context.party.value} at "
            f"position {context.position} of message bit "
            f"{context.message_bit}. The resource_factory is what produced "
            f"it."
        )
    return density


def _call_monitor(
    monitor: ChannelMonitor | None,
    density: DensityMatrix,
    context: ResourceContext,
) -> dict[str, Any]:
    """Put one hop's resource past the channel-monitor seam."""
    if monitor is None:
        return {}
    if not callable(monitor):
        raise TypeError(
            f"channel_monitor must be a callable "
            f"monitor(resource, context), got {type(monitor).__name__}."
        )
    # A copy: a monitor handed the live object could edit the pair before it is
    # measured, which would be a channel attack the transcript never shows.
    returned = monitor(
        DensityMatrix(np.array(density.data, copy=True)), context
    )
    if not isinstance(returned, Mapping):
        raise TypeError(
            f"channel_monitor must return a mapping of JSON-safe "
            f"diagnostics, got {type(returned).__name__} for "
            f"{context.party.value} at position {context.position}. "
            f"Return an empty mapping to record nothing beyond the "
            f"built-in summary."
        )
    # Read once, at every hop; the monitor's own mapping is not referenced again.
    return _as_json_value(returned, "extra")


@dataclass(frozen=True)
class ChannelSample:
    """What one check round's entanglement resource was, before it was
    measured.

    >>> from qsecure.core.states import BellState, bell_state
    >>> from qsecure.protocol.checkrounds import QberRound
    >>> from qsecure.protocol.distribute import ResourceContext
    >>> from qsecure.protocol.params import Party
    >>> from qsecure.protocol.session import ChannelSample
    >>> context = ResourceContext(party=Party.BOB, message_bit=1, position=7)
    >>> sample = ChannelSample.of(
    ...     bell_state(BellState.PHI_PLUS), context, QberRound(7, "Z")
    ... )
    >>> sample.role, sample.fidelity, sample.purity
    (<CheckRole.QBER: 'qber'>, 1.0, 1.0)
    >>> round(sample.concurrence, 12)     # 1, to the last few bits of a 4x4
    1.0
    >>> sample.is_ideal
    True
    >>> ChannelSample.from_dict(sample.to_dict()) == sample
    True

    >>> round(sample.alice_purity, 9), round(sample.recipient_purity, 9)
    (0.5, 0.5)
    >>> sample.wings_agree
    True

    >>> import numpy as np
    >>> from qiskit.quantum_info import DensityMatrix, Statevector
    >>> split = DensityMatrix(Statevector([1, 0])).tensor(
    ...     DensityMatrix(np.eye(2) / 2)
    ... )
    >>> broken = ChannelSample.of(split, context, QberRound(7, "Z"))
    >>> round(broken.alice_purity, 9), round(broken.recipient_purity, 9)
    (0.5, 1.0)
    >>> broken.concurrence, broken.wings_agree
    (0.0, False)
    """

    party: Party
    message_bit: int
    position: int
    role: CheckRole
    fidelity: float
    purity: float
    concurrence: float
    alice_purity: float
    recipient_purity: float
    extra: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        """Coerce the tags and check the three summaries are real numbers."""
        resolved = _as_party(self.party)
        if resolved is Party.ALICE:
            raise ValueError(
                "a ChannelSample describes one link, and Alice is the common "
                "endpoint of both: tag it with the recipient whose pair this "
                "was, Party.BOB or Party.CHARLIE."
            )
        object.__setattr__(self, "party", resolved)
        object.__setattr__(
            self, "message_bit", _as_message_bit(self.message_bit)
        )
        if not isinstance(self.position, (int, np.integer)) or isinstance(
            self.position, bool
        ):
            raise TypeError(
                f"position must be an integer key index, got "
                f"{type(self.position).__name__}."
            )
        object.__setattr__(self, "position", int(self.position))
        if self.position < 0:
            raise ValueError(
                f"position must be a key index in 0 .. L-1, got "
                f"{self.position}."
            )
        if not isinstance(self.role, CheckRole):
            object.__setattr__(
                self, "role", CheckRole(str(self.role).strip().lower())
            )
        for name, value, low, high in (
            ("fidelity", self.fidelity, 0.0, 1.0),
            ("purity", self.purity, 0.0, 1.0),
            ("concurrence", self.concurrence, 0.0, 1.0),
            # A single qubit's purity floors at 1/2; the wider bound is checked here.
            ("alice_purity", self.alice_purity, 0.0, 1.0),
            ("recipient_purity", self.recipient_purity, 0.0, 1.0),
        ):
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise TypeError(
                    f"{name} must be a real number, got "
                    f"{type(value).__name__} ({value!r}). A ChannelSample "
                    f"carries the summary of a two-qubit state, not a label "
                    f"for it."
                ) from None
            if not np.isfinite(number) or not low <= number <= high:
                raise ValueError(
                    f"{name} must be a finite number in [{low}, {high}], got "
                    f"{value!r}. It is a property of a physical two-qubit "
                    f"state; a value outside the range means the sample was "
                    f"built from something that is not one."
                )
            object.__setattr__(self, name, number)
        if not isinstance(self.extra, Mapping):
            raise TypeError(
                f"extra must be a mapping of the channel_monitor's extra "
                f"diagnostics, got {type(self.extra).__name__}. A monitor that "
                f"has nothing to add should return an empty mapping."
            )
        object.__setattr__(
            self,
            "extra",
            MappingProxyType(dict(_as_json_value(self.extra, "extra"))),
        )

    # -- construction ------------------------------------------------------- #

    @classmethod
    def of(
        cls,
        resource: StateLike,
        context: ResourceContext,
        scheduled: QberRound | ChshRound,
        *,
        monitor: ChannelMonitor | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ChannelSample:
        """Summarise the resource one check round was handed."""
        if scheduled.position != context.position:
            raise ValueError(
                f"the scheduled check round is at position "
                f"{scheduled.position} but the hop being summarised is at "
                f"{context.position}. The two come from one plan and one loop; "
                f"a mismatch would file this pair's diagnostics against another "
                f"position's outcomes."
            )
        if monitor is not None and extra is not None:
            raise ValueError(
                "pass either monitor= to invoke the seam here or extra= to "
                "record what it already returned, not both: two calls for one "
                "hop would double every count a monitor keeps and leave the "
                "sample carrying the second one's answer."
            )
        density = _as_pair(resource, context)
        if extra is None:
            extra = _call_monitor(monitor, density, context)
        reference = bell_state(BellState.PHI_PLUS)
        return cls(
            party=context.party,
            message_bit=context.message_bit,
            position=context.position,
            role=scheduled.role,
            fidelity=fidelity(density, reference),
            purity=purity(density),
            concurrence=concurrence(density),
            # Wing order comes from checkrounds so the two modules cannot disagree.
            alice_purity=purity(partial_trace(density, [_RECIPIENT_QUBIT])),
            recipient_purity=purity(partial_trace(density, [_ALICE_QUBIT])),
            extra=extra,
        )

    # -- derived views ------------------------------------------------------ #

    @property
    def is_ideal(self) -> bool:
        """bool: Whether this pair was the clean :math:`|\\Phi^{+}\\rangle`."""
        return (
            abs(self.fidelity - 1.0) <= 1e-9
            and abs(self.purity - 1.0) <= 1e-9
            and abs(self.concurrence - 1.0) <= 1e-9
        )

    @property
    def wings_agree(self) -> bool:
        """bool: Whether the two halves are equally disturbed."""
        return abs(self.alice_purity - self.recipient_purity) <= 1e-9

    # -- serialisation ------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the sample."""
        return {
            "party": self.party,
            "message_bit": self.message_bit,
            "position": self.position,
            "role": self.role,
            "fidelity": self.fidelity,
            "purity": self.purity,
            "concurrence": self.concurrence,
            "alice_purity": self.alice_purity,
            "recipient_purity": self.recipient_purity,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChannelSample:
        """Rebuild a sample from :meth:`to_dict` output."""
        return cls(
            party=data["party"],
            message_bit=data["message_bit"],
            position=data["position"],
            role=data["role"],
            fidelity=data["fidelity"],
            purity=data["purity"],
            concurrence=data["concurrence"],
            alice_purity=data["alice_purity"],
            recipient_purity=data["recipient_purity"],
            extra=data.get("extra", {}),
        )


class _ChannelTap:
    """The ``resource_factory`` with a check-round summary taken on the way
    past.
    """

    __slots__ = (
        "_factory",
        "_wants_context",
        "_plan",
        "_planned",
        "_monitor",
        "_sink",
    )

    def __init__(
        self,
        factory: ResourceFactory | None,
        *,
        plan: CheckRoundPlan,
        monitor: ChannelMonitor | None,
        sink: list[ChannelSample],
    ) -> None:
        self._factory, self._wants_context = _resolve_factory(factory)
        self._plan = plan
        self._planned: dict[Party, dict[int, QberRound | ChshRound]] = {}
        self._monitor = monitor
        self._sink = sink

    def _rounds_for(self, party: Party) -> dict[int, QberRound | ChshRound]:
        """Return this link's planned rounds, indexed once per link."""
        rounds = self._planned.get(party)
        if rounds is None:
            rounds = self._plan.rounds_by_position(party)
            self._planned[party] = rounds
        return rounds

    def __call__(self, context: ResourceContext) -> StateLike:
        """Draw one hop's resource, monitor it, and record it if it is watched."""
        # Adopted on the way in, so the factory's object is read once per position.
        resource = _draw_resource(self._factory, self._wants_context, context)
        # Every hop, so the seam cannot read the plan off its call sequence; guarded
        # so an unmonitored run skips the pair check entirely.
        extra: Mapping[str, Any] = {}
        if self._monitor is not None:
            extra = _call_monitor(
                self._monitor, _as_pair(resource, context), context
            )
        scheduled = self._rounds_for(context.party).get(context.position)
        if scheduled is not None:
            self._sink.append(
                ChannelSample.of(resource, context, scheduled, extra=extra)
            )
        return resource


# --------------------------------------------------------------------------- #
# The callable seams


class Distributor(Protocol):
    """Callable that runs Phase A for one message bit."""

    def __call__(
        self,
        key: PrivateKey,
        params: ProtocolParams,
        *,
        parties: Sequence[Party | str] = VERIFIERS,
        resource_factory: ResourceFactory | None = None,
        rng: np.random.Generator | None = None,
        check_plan: CheckRoundPlan | None = None,
        payload_map: PayloadMap | None = None,
    ) -> Mapping[Party, RecipientRecord | RecipientDistribution]:
        """Distribute ``key`` and return one record per party."""
        ...


class Signer(Protocol):
    """Callable that runs Phase B and produces the declaration to be verified."""

    def __call__(
        self,
        message_bit: int,
        keys: tuple[PrivateKey, PrivateKey],
        params: ProtocolParams,
        *,
        records: Mapping[int, Mapping[Party, RecipientRecord]],
    ) -> Signature:
        """Return the signature declared for ``message_bit``."""
        ...


def honest_signer(
    message_bit: int,
    keys: tuple[PrivateKey, PrivateKey],
    params: ProtocolParams,
    *,
    records: Mapping[int, Mapping[Party, RecipientRecord]],
) -> Signature:
    """Declare exactly the key whose states were distributed.

    >>> import numpy as np
    >>> from qsecure.protocol.keys import generate_key_pair
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import honest_signer
    >>> params = ProtocolParams(key_length=9)
    >>> keys = generate_key_pair(params, rng=np.random.default_rng(5))
    >>> honest_signer(1, keys, params, records={}).declared_key is keys[1]
    True
    """
    del records  # An honest Alice holds no recipient's measurement log.
    return sign(message_bit, keys, params)


class Forwarder(Protocol):
    """Callable that carries the declaration from Bob to Charlie."""

    def __call__(
        self,
        signature: Signature,
        params: ProtocolParams,
        *,
        view: RecipientView | None = None,
    ) -> Signature:
        """Return the declaration Charlie will be given."""
        ...


def honest_forwarder(
    signature: Signature,
    params: ProtocolParams,
    *,
    view: RecipientView | None = None,
) -> Signature:
    """Forward the declaration unchanged.

    >>> import numpy as np
    >>> from qsecure.protocol.keys import generate_key_pair
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import honest_forwarder
    >>> from qsecure.protocol.signature import sign
    >>> params = ProtocolParams(key_length=9)
    >>> keys = generate_key_pair(params, rng=np.random.default_rng(5))
    >>> declaration = sign(0, keys, params)
    >>> honest_forwarder(declaration, params) is declaration
    True
    """
    del params, view  # An honest hop needs nothing but the declaration itself.
    return signature


# --------------------------------------------------------------------------- #
# The transcript
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SessionTranscript:
    """Everything one run produced, frozen and JSON-serialisable end to end.

    >>> import json
    >>> import numpy as np
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession, SessionTranscript
    >>> transcript = QDSSession(
    ...     ProtocolParams(key_length=24), rng=np.random.default_rng(11)
    ... ).run(0)
    >>> restored = SessionTranscript.from_json(transcript.to_json())
    >>> restored == transcript
    True
    >>> transcript.symmetrised, transcript.forwarded_signature is None
    (True, True)
    """

    params: ProtocolParams
    message_bit: int
    signature: Signature
    records: tuple[RecipientRecord, ...]
    results: tuple[VerificationResult, ...]
    forwarded_signature: Signature | None = None
    run_id: str | None = None
    aborts: tuple[VerificationAbort, ...] = ()
    pooled: PooledMatchedCounts | None = None
    check_logs: tuple[CheckLog, ...] = ()
    channel: tuple[ChannelSample, ...] = ()
    signer_saw_recipient_logs: bool = False
    count_exchange_timing: str = COUNTS_BEFORE_FORWARDING
    spent_rounds: tuple[tuple[str, str, int], ...] = ()
    replay_refusals: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples and check the run hangs
        together.
        """
        if not isinstance(self.params, ProtocolParams):
            raise TypeError(
                f"params must be a ProtocolParams, got "
                f"{type(self.params).__name__}"
            )
        object.__setattr__(
            self, "message_bit", _as_message_bit(self.message_bit)
        )
        if not isinstance(self.signature, Signature):
            raise TypeError(
                f"signature must be a Signature, got "
                f"{type(self.signature).__name__}"
            )
        if self.signature.message_bit != self.message_bit:
            raise ValueError(
                f"transcript is tagged with message bit {self.message_bit} but "
                f"holds a signature for bit {self.signature.message_bit}. The "
                f"bit selects which distribution the verifiers scored against, "
                f"so a mismatch means the verdicts in this transcript were "
                f"reached on the wrong records."
            )
        if self.count_exchange_timing not in COUNT_EXCHANGE_TIMINGS:
            raise ValueError(
                f"count_exchange_timing must be one of "
                f"{list(COUNT_EXCHANGE_TIMINGS)}, got "
                f"{self.count_exchange_timing!r}. It names which experiment "
                f"this run was, and the two give different answers to the same "
                f"attack, so an unrecognised value is a transcript nobody can "
                f"interpret."
            )
        object.__setattr__(
            self,
            "spent_rounds",
            tuple(
                (str(party), str(session_id), _as_message_bit(bit))
                for party, session_id, bit in self.spent_rounds
            ),
        )
        object.__setattr__(
            self,
            "replay_refusals",
            tuple(
                (str(party), int(count))
                for party, count in self.replay_refusals
            ),
        )
        object.__setattr__(self, "records", tuple(self.records))
        for position, record in enumerate(self.records):
            if not isinstance(record, RecipientRecord):
                raise TypeError(
                    f"records[{position}] must be a RecipientRecord, got "
                    f"{type(record).__name__}"
                )
        if self.forwarded_signature is not None:
            if not isinstance(self.forwarded_signature, Signature):
                raise TypeError(
                    f"forwarded_signature must be a Signature or None, got "
                    f"{type(self.forwarded_signature).__name__}"
                )
            if self.forwarded_signature.message_bit != self.message_bit:
                raise ValueError(
                    f"the forwarded declaration is for message bit "
                    f"{self.forwarded_signature.message_bit} but the transcript "
                    f"is tagged with bit {self.message_bit}. Bob forwards the "
                    f"declaration he was asked to verify; a different bit means "
                    f"Charlie was scored against another run entirely."
                )
        if self.run_id is not None and not isinstance(self.run_id, str):
            raise TypeError(
                f"run_id must be a string or None, got "
                f"{type(self.run_id).__name__}. It is a ledger key, carried "
                f"verbatim and interpreted by nothing in this module."
            )

        if not isinstance(self.signer_saw_recipient_logs, bool):
            raise TypeError(
                f"signer_saw_recipient_logs must be a bool, got "
                f"{type(self.signer_saw_recipient_logs).__name__}. It records "
                f"whether the signer seam was shown both recipients' raw logs, "
                f"which decides whether this run is inside the threat model at "
                f"all."
            )

        # Everything here was scored under the sifted set: check rounds carry no key.
        scored = self.params.sifted()

        object.__setattr__(self, "results", tuple(self.results))
        seen: set[Party] = set()
        for position, result in enumerate(self.results):
            if not isinstance(result, VerificationResult):
                raise TypeError(
                    f"results[{position}] must be a VerificationResult, got "
                    f"{type(result).__name__}"
                )
            if result.party in seen:
                raise ValueError(
                    f"results holds two verdicts for "
                    f"{result.party.value!r}. One verifier reaches one decision "
                    f"per signature; a duplicate would make "
                    f"'did Charlie accept?' depend on which entry is read."
                )
            seen.add(result.party)
            _check_result_against(result, scored, self.message_bit)

        object.__setattr__(self, "aborts", tuple(self.aborts))
        for position, abort in enumerate(self.aborts):
            if not isinstance(abort, VerificationAbort):
                raise TypeError(
                    f"aborts[{position}] must be a VerificationAbort, got "
                    f"{type(abort).__name__}"
                )
            if abort.party in seen:
                raise ValueError(
                    f"{abort.party.value} appears in both results and aborts, "
                    f"or twice in aborts. Phase C leaves each verifier with "
                    f"exactly one outcome -- a verdict or a refusal to score, "
                    f"never both -- and a party in both fields would let "
                    f"'did Charlie accept?' depend on which field is read."
                )
            seen.add(abort.party)
            _check_abort_against(abort, scored, self.message_bit)

        if self.pooled is not None:
            if not isinstance(self.pooled, PooledMatchedCounts):
                raise TypeError(
                    f"pooled must be a PooledMatchedCounts or None, got "
                    f"{type(self.pooled).__name__}; "
                    f"tally.exchange_matched_counts returns exactly that."
                )
            _check_pooled_against(self.pooled, scored, self.message_bit)

        object.__setattr__(self, "check_logs", tuple(self.check_logs))
        published: dict[tuple[Party, int], CheckLog] = {}
        for position, log in enumerate(self.check_logs):
            if not isinstance(log, CheckLog):
                raise TypeError(
                    f"check_logs[{position}] must be a CheckLog, got "
                    f"{type(log).__name__}; "
                    f"distribute_public_key_with_checks returns one per party."
                )
            if (log.party, log.message_bit) in published:
                raise ValueError(
                    f"check_logs holds two logs for {log.party.value} on "
                    f"message bit {log.message_bit}. One recipient measures one "
                    f"link once per distribution, so a duplicate would double "
                    f"the sample behind every interval computed from it."
                )
            published[(log.party, log.message_bit)] = log

        object.__setattr__(self, "channel", tuple(self.channel))
        for position, sample in enumerate(self.channel):
            if not isinstance(sample, ChannelSample):
                raise TypeError(
                    f"channel[{position}] must be a ChannelSample, got "
                    f"{type(sample).__name__}."
                )
            _check_sample_against(sample, published, self.params)

    # -- derived views ------------------------------------------------------ #

    @property
    def results_by_party(self) -> dict[Party, VerificationResult]:
        """dict: The verdicts keyed by party, in the order they were reached."""
        return {result.party: result for result in self.results}

    @property
    def bob(self) -> VerificationResult | None:
        """VerificationResult or None: Bob's verdict, if he reached one."""
        return self.results_by_party.get(Party.BOB)

    @property
    def charlie(self) -> VerificationResult | None:
        """VerificationResult or None: Charlie's verdict, if he reached one."""
        return self.results_by_party.get(Party.CHARLIE)

    @property
    def aborts_by_party(self) -> dict[Party, VerificationAbort]:
        """dict: The refusals to score, keyed by party, in the order recorded."""
        return {abort.party: abort for abort in self.aborts}

    @property
    def aborted(self) -> bool:
        """bool: ``True`` iff some verifier reached no verdict at all."""
        return bool(self.aborts)

    @property
    def is_complete(self) -> bool:
        """bool: ``True`` when both verifiers have reached a verdict."""
        return self.bob is not None and self.charlie is not None

    @property
    def counts_exchanged(self) -> bool:
        """bool: ``True`` iff the recipients ran Phase C'."""
        return self.pooled is not None

    @property
    def channel_monitored(self) -> bool:
        """bool: ``True`` iff this run published per-check-round channel
        samples.
        """
        return bool(self.channel)

    @property
    def check_logs_by_party(self) -> dict[tuple[Party, int], CheckLog]:
        """dict: The published check logs, keyed by ``(party, message_bit)``."""
        return {(log.party, log.message_bit): log for log in self.check_logs}

    def check_log_for(
        self, party: Party | str, message_bit: int
    ) -> CheckLog | None:
        """Return one link's published check-round observations."""
        resolved = _as_party(party)
        if resolved is Party.ALICE:
            raise ValueError(
                "Alice holds no check log: a check round is a statement about "
                "one link and she is at the far end of both. Ask for "
                "Party.BOB or Party.CHARLIE."
            )
        return self.check_logs_by_party.get(
            (resolved, _as_message_bit(message_bit))
        )

    def channel_for(
        self, party: Party | str, message_bit: int
    ) -> tuple[ChannelSample, ...]:
        """Return one link's channel samples, in the order the hops happened."""
        resolved = _as_party(party)
        if resolved is Party.ALICE:
            raise ValueError(
                "Alice holds no channel samples: they summarise the resource "
                "delivered on one recipient's link. Ask for Party.BOB or "
                "Party.CHARLIE."
            )
        bit = _as_message_bit(message_bit)
        return tuple(
            sample
            for sample in self.channel
            if sample.party is resolved and sample.message_bit == bit
        )

    @property
    def symmetrised(self) -> bool:
        """bool: ``True`` iff every log went through Phase A'."""
        return bool(self.records) and all(
            record.symmetrised for record in self.records
        )

    def signature_for(self, party: Party | str) -> Signature:
        """Return the declaration a given verifier actually scored."""
        resolved = _as_party(party)
        if resolved is Party.ALICE:
            raise ValueError(
                "Alice scores no declaration: she is the signer. Ask for "
                "Party.BOB (who received it) or Party.CHARLIE (who was "
                "forwarded it)."
            )
        if resolved is Party.CHARLIE and self.forwarded_signature is not None:
            return self.forwarded_signature
        return self.signature

    @property
    def forwarding_altered_signature(self) -> bool:
        """bool: ``True`` iff Charlie scored a different declaration from Bob."""
        return (
            self.forwarded_signature is not None
            and self.forwarded_signature != self.signature
        )

    @property
    def session_coherent(self) -> bool:
        """bool: ``True`` iff every scored log belongs to the declaration's
        round.
        """
        for record in self.records:
            if (
                record.message_bit != self.message_bit
                or record.session_id is None
            ):
                continue
            scored = self.signature_for(record.party)
            if scored.session_id != record.session_id:
                return False
        return True

    @property
    def transferable(self) -> bool:
        """bool: ``True`` iff Bob accepted **and** Charlie accepted, in one
        round.
        """
        return (
            self.bob is not None
            and self.charlie is not None
            and self.bob.accepted
            and self.charlie.accepted
            and self.session_coherent
        )

    @property
    def repudiated(self) -> bool:
        """bool: ``True`` iff Bob accepted and Charlie rejected *the same key*."""
        return (
            self.bob is not None
            and self.charlie is not None
            and self.bob.accepted
            and not self.charlie.accepted
            and not self.forwarding_altered_signature
            and self.session_coherent
        )

    @property
    def pooled_matched_count(self) -> int | None:
        """int or None: ``M = m_B + m_C`` as this run actually produced it."""
        if self.forwarding_altered_signature or not self.symmetrised:
            return None
        bob, charlie = self.bob, self.charlie
        if bob is None or charlie is None:
            return None
        return bob.matched_count + charlie.matched_count

    @property
    def repudiation_guarantee(self) -> float | None:
        """float or None: the per-run repudiation bound, from the observed
        ``M``.

        >>> import numpy as np
        >>> from qsecure.protocol.params import ProtocolParams
        >>> from qsecure.protocol.session import QDSSession
        >>> transcript = QDSSession(
        ...     ProtocolParams(key_length=600), rng=np.random.default_rng(3)
        ... ).run(0)
        >>> transcript.pooled_matched_count > 2 * 600 // 3 - 60
        True
        >>> 0.0 < transcript.repudiation_guarantee < 1.0
        True
        """
        pooled = self.pooled_matched_count
        if pooled is None or pooled < 1:
            return None
        # Sifted set: counting check rounds as key would make the bound too good.
        return repudiation_bound(self.params.sifted(), matched_records=pooled)

    def records_for(self, message_bit: int) -> dict[Party, RecipientRecord]:
        """Return the logs distributed for one message bit, keyed by party."""
        bit = _as_message_bit(message_bit)
        return {
            record.party: record
            for record in self.records
            if record.message_bit == bit
        }

    def verdict_for(self, party: Party | str) -> VerificationResult:
        """Return one verifier's decision."""
        resolved = _as_party(party)
        result = self.results_by_party.get(resolved)
        if result is None:
            reached = sorted(item.value for item in self.results_by_party)
            abort = self.aborts_by_party.get(resolved)
            because = (
                f" He was asked and refused to score: {abort.summary()}"
                if abort is not None
                else ""
            )
            advice = (
                " Read transcript.aborts for the refusal."
                if abort is not None
                else (
                    " Run the session to completion with "
                    "QDSSession.run(message_bit), or call verify(Party.BOB) "
                    "and then transfer()."
                )
            )
            raise ValueError(
                f"{resolved.value} reached no verdict in this run; verdicts "
                f"present: {reached or 'none'}. This is not a rejection: no "
                f"decision was made.{because}{advice}"
            )
        return result

    def summary(self) -> str:
        """Return a short human-readable account of the whole run.

        >>> import numpy as np
        >>> from qsecure.protocol.params import ProtocolParams
        >>> from qsecure.protocol.session import QDSSession
        >>> text = QDSSession(
        ...     ProtocolParams(key_length=24), rng=np.random.default_rng(4)
        ... ).run(0).summary()
        >>> text.splitlines()[-1]
        'TRANSFERABLE: Bob accepted and Charlie accepted.'
        """
        lines = [
            f"QDS run on message bit {self.message_bit}: L="
            f"{self.params.key_length}, |B|={len(self.params.bases)}, "
            f"s_a={self.params.s_a:.5f}, s_v={self.params.s_v:.5f}"
        ]
        if not self.symmetrised:
            lines.append(
                "UNSYMMETRISED: the recipients did not exchange their copies, "
                "so this run has no non-repudiation guarantee at any L."
            )
        if not self.counts_exchanged:
            lines.append(
                "UNPOOLED: the recipients did not compare matched counts, so "
                "only the per-verifier floor applied and a signer who splits "
                "the evidence base repudiates at about 1/2 at any L."
            )
        if self.signer_saw_recipient_logs:
            lines.append(
                "OVER-POWERED SIGNER: the signer seam was shown both "
                "recipients' raw logs, which no single adversary in the threat "
                "model holds. Nothing here checks what it read, so no number "
                "from this run describes the shipped scheme unless the attack "
                "itself stayed inside the model."
            )
        if self.check_logs or self.channel:
            rounds = sum(log.round_count for log in self.check_logs)
            lines.append(
                f"CHECK ROUNDS: {self.params.check_count} of "
                f"{self.params.key_length} positions were reserved for "
                f"channel estimation and dealt between the links, so the key "
                f"was scored at L={self.params.signing_length}; {rounds} "
                f"observations and {len(self.channel)} channel samples "
                f"published."
            )
        if self.forwarding_altered_signature:
            lines.append(
                "FORWARDING ALTERED THE DECLARATION: Charlie scored a "
                "different key from the one Bob was given."
            )
        if self.pooled is not None:
            lines.append(self.pooled.summary())
        lines.extend(result.summary() for result in self.results)
        lines.extend(abort.summary() for abort in self.aborts)
        guarantee = self.repudiation_guarantee
        if guarantee is not None:
            lines.append(
                f"EVIDENCE: M = m_B + m_C = {self.pooled_matched_count} matched "
                f"records, so P(repudiation | this run) <= {guarantee:.3e}. "
                f"This is the per-run bound and the only one that assumes "
                f"nothing about the signer."
            )
        if self.aborted:
            # Name the reason each verifier actually gave, never an assumed one.
            refused = ", ".join(
                f"{abort.party.value} ({abort.reason.value})" for abort in self.aborts
            )
            lines.append(
                f"NO VERDICT: {refused} could not score this declaration. "
                f"This is not a rejection and must not be counted as one."
            )
        elif not self.is_complete:
            lines.append("INCOMPLETE: not every verifier reached a verdict.")
        elif self.transferable:
            lines.append("TRANSFERABLE: Bob accepted and Charlie accepted.")
        elif self.repudiated:
            lines.append(
                "REPUDIATION: Bob accepted a signature Charlie rejected."
            )
        elif not self.verdict_for(Party.BOB).accepted:
            lines.append("REJECTED: Bob did not accept the signature.")
        elif not self.session_coherent:
            # Bob accepted, so "REJECTED" would be false; say which gate closed.
            lines.append(
                "INCOHERENT: the verdicts in this transcript were reached on "
                "logs from a different distribution round than the declaration "
                "they are filed against, so neither transferability nor "
                "repudiation is a statement about it."
            )
        else:
            # The hop altered the declaration, so this is not Alice's repudiation.
            lines.append(
                "NOT TRANSFERRED: Bob accepted, but Charlie was handed a "
                "different declaration, so this run measures the forwarding "
                "hop and not the signer. It is not a repudiation."
            )
        return "\n".join(lines)

    # -- serialisation ------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the entire run.

        >>> import json
        >>> import numpy as np
        >>> from qsecure.protocol.params import ProtocolParams
        >>> from qsecure.protocol.session import QDSSession
        >>> blob = QDSSession(
        ...     ProtocolParams(key_length=24), rng=np.random.default_rng(6)
        ... ).run(1).to_dict()
        >>> json.loads(json.dumps(blob))["message_bit"]
        1
        """
        return {
            "params": self.params.to_dict(),
            "message_bit": self.message_bit,
            "signature": self.signature.to_dict(),
            "records": [record.to_dict() for record in self.records],
            "results": [result.to_dict() for result in self.results],
            "aborts": [abort.to_dict() for abort in self.aborts],
            "pooled": None if self.pooled is None else self.pooled.to_dict(),
            "forwarded_signature": (
                None
                if self.forwarded_signature is None
                else self.forwarded_signature.to_dict()
            ),
            "run_id": self.run_id,
            "check_logs": [log.to_dict() for log in self.check_logs],
            "channel": [sample.to_dict() for sample in self.channel],
            "signer_saw_recipient_logs": self.signer_saw_recipient_logs,
            "count_exchange_timing": self.count_exchange_timing,
            "spent_rounds": [list(entry) for entry in self.spent_rounds],
            "replay_refusals": [list(entry) for entry in self.replay_refusals],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SessionTranscript:
        """Rebuild a transcript from :meth:`to_dict` output."""
        forwarded = data.get("forwarded_signature")
        return cls(
            params=ProtocolParams.from_dict(data["params"]),
            message_bit=data["message_bit"],
            signature=Signature.from_dict(data["signature"]),
            records=tuple(
                RecipientRecord.from_dict(item) for item in data["records"]
            ),
            results=tuple(
                VerificationResult.from_dict(item) for item in data["results"]
            ),
            forwarded_signature=(
                None if forwarded is None else Signature.from_dict(forwarded)
            ),
            run_id=data.get("run_id"),
            # Optional fields below default so older transcripts still restore.
            aborts=tuple(
                VerificationAbort.from_dict(item)
                for item in data.get("aborts", ())
            ),
            pooled=(
                None
                if data.get("pooled") is None
                else PooledMatchedCounts.from_dict(data["pooled"])
            ),
            check_logs=tuple(
                CheckLog.from_dict(item) for item in data.get("check_logs", ())
            ),
            channel=tuple(
                ChannelSample.from_dict(item)
                for item in data.get("channel", ())
            ),
            signer_saw_recipient_logs=bool(
                data.get("signer_saw_recipient_logs", False)
            ),
            count_exchange_timing=str(
                data.get("count_exchange_timing", COUNTS_BEFORE_FORWARDING)
            ),
            spent_rounds=tuple(
                (str(party), str(session_id), int(bit))
                for party, session_id, bit in data.get("spent_rounds", ())
            ),
            replay_refusals=tuple(
                (str(party), int(count))
                for party, count in data.get("replay_refusals", ())
            ),
        )

    def to_json(self, **kwargs: Any) -> str:
        """Serialise the run to a JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str) -> SessionTranscript:
        """Rebuild a transcript from :meth:`to_json` output."""
        return cls.from_dict(json.loads(text))


# --------------------------------------------------------------------------- #
# The session
# --------------------------------------------------------------------------- #


class QDSSession:
    """One run of the protocol: distribute, sign, verify, transfer.

    >>> import numpy as np
    >>> from qsecure.protocol.params import Party, ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> session = QDSSession(
    ...     ProtocolParams(key_length=24), rng=np.random.default_rng(2)
    ... )
    >>> _ = session.distribute()
    >>> _ = session.sign(0)
    >>> session.verify(Party.BOB).accepted
    True
    >>> session.transfer().accepted
    True
    >>> session.transcript().transferable
    True
    """

    def __init__(
        self,
        params: ProtocolParams = DEMO_PARAMS,
        *,
        resource_factory: ResourceFactory | None = None,
        payload_map: PayloadMap | None = None,
        channel_monitor: ChannelMonitor | None = None,
        distributor: Distributor | None = None,
        symmetriser: Symmetriser | None = None,
        signer: Signer | None = None,
        signer_sees_recipient_logs: bool = False,
        count_exchange: CountExchange | None = None,
        count_exchange_timing: str = COUNTS_BEFORE_FORWARDING,
        forwarder: Forwarder | None = None,
        run_id: str | None = None,
        context: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not isinstance(params, ProtocolParams):
            raise TypeError(
                f"params must be a ProtocolParams, got {type(params).__name__}. "
                f"Use DEMO_PARAMS for a quick run or DEFAULT_PARAMS for the "
                f"security-grade set."
            )
        _check_seam(
            resource_factory,
            "resource_factory",
            "a callable returning a two-qubit entanglement resource, taking "
            "either no arguments or one ResourceContext; leave it None for the "
            "ideal |Phi+> pair",
        )
        _check_seam(
            payload_map,
            "payload_map",
            "a callable payload_map(state, context) returning the one-qubit "
            "state to teleport; leave it None to send the eigenstate Alice "
            "prepared",
        )
        _check_seam(
            channel_monitor,
            "channel_monitor",
            "a callable monitor(resource, context) returning a JSON-safe "
            "mapping of extra check-round diagnostics",
        )
        _check_seam(
            distributor,
            "distributor",
            "a callable with the signature of distribute_public_key_with_checks",
        )
        _check_seam(
            symmetriser,
            "symmetriser",
            "a callable with the signature of symmetrise_records; pass "
            "no_symmetrisation to run the insecure variant deliberately",
        )
        _check_seam(
            signer, "signer", "a callable with the signature of honest_signer"
        )
        _check_seam(
            count_exchange,
            "count_exchange",
            "a callable with the signature of exchange_matched_counts; pass "
            "no_count_exchange to run the pre-pooled variant deliberately",
        )
        _check_seam(
            forwarder,
            "forwarder",
            "a callable with the signature of honest_forwarder",
        )
        if count_exchange_timing not in COUNT_EXCHANGE_TIMINGS:
            raise ValueError(
                f"count_exchange_timing must be one of "
                f"{list(COUNT_EXCHANGE_TIMINGS)}, got "
                f"{count_exchange_timing!r}. It decides *which declaration* "
                f"each recipient counts against, which is a different "
                f"experiment and not a tuning knob: "
                f"{COUNTS_BEFORE_FORWARDING!r} counts the declaration Alice "
                f"signed at both verifiers, {COUNTS_AFTER_FORWARDING!r} counts "
                f"whatever each one received."
            )
        if not isinstance(signer_sees_recipient_logs, bool):
            raise TypeError(
                f"signer_sees_recipient_logs must be a bool, got "
                f"{type(signer_sees_recipient_logs).__name__}. It is the "
                f"opt-in that hands the signer seam more than any adversary in "
                f"the threat model holds, so it is a deliberate yes or no and "
                f"not a value to be inferred."
            )
        if run_id is not None and not isinstance(run_id, str):
            raise TypeError(
                f"run_id must be a string or None, got "
                f"{type(run_id).__name__}. It is a ledger key carried into the "
                f"transcript verbatim; this module never reads it."
            )
        if context is not None and not isinstance(context, str):
            raise TypeError(
                f"context must be a string or None, got "
                f"{type(context).__name__}. It is the application's "
                f"instruction-and-nonce string -- build one with "
                f"qsecure.protocol.signature.fresh_context(instruction, nonce) "
                f"-- and it is hashed into every round identifier this session "
                f"announces, so it is fixed here and not at signing time."
            )

        self._params = params
        # ``params`` itself unless it reserves check rounds (identity, not equality,
        # so seams receive the caller's own object).
        self._scored_params = (
            params.sifted() if params.has_check_rounds else params
        )
        self._resource_factory = resource_factory
        self._payload_map = payload_map
        self._channel_monitor = channel_monitor
        self._distributor: Distributor = (
            distribute_public_key_with_checks
            if distributor is None
            else distributor
        )
        self._symmetriser: Symmetriser = (
            symmetrise_records if symmetriser is None else symmetriser
        )
        self._signer: Signer = honest_signer if signer is None else signer
        self._signer_sees_recipient_logs = signer_sees_recipient_logs
        self._count_exchange: CountExchange = (
            exchange_matched_counts
            if count_exchange is None
            else count_exchange
        )
        self._count_exchange_timing = count_exchange_timing
        self._forwarder: Forwarder = (
            honest_forwarder if forwarder is None else forwarder
        )
        # Only a forwarder that *requires* a view is built one.
        self._forwarder_wants_view = _forwarder_wants_view(self._forwarder)
        self._run_id = run_id
        self._context = context
        # One draw, three derived streams; the caller's generator is not retained so
        # no seam can rewind it.
        material = resolve_rng(rng).bytes(_STREAM_MATERIAL_BYTES)
        self._alice_rng = _derive_stream(material, _ALICE_STREAM_LABEL)
        self._recipient_rng = _derive_stream(material, _RECIPIENT_STREAM_LABEL)
        self._binding_rng = _derive_stream(material, _BINDING_STREAM_LABEL)

        self._keys: tuple[PrivateKey, PrivateKey] | None = None
        self._signing_keys: tuple[PrivateKey, PrivateKey] | None = None
        self._plans: dict[int, CheckRoundPlan] = {}
        self._check_logs: dict[tuple[Party, int], CheckLog] = {}
        self._channel: list[ChannelSample] = []
        self._openings: dict[int, str] = {}
        self._session_ids: dict[int, str] = {}
        self._raw_records: dict[int, dict[Party, RecipientRecord]] = {}
        self._records: dict[int, dict[Party, RecipientRecord]] = {}
        self._signature: Signature | None = None
        self._forwarded: Signature | None = None
        self._pooled: PooledMatchedCounts | None = None
        self._counts_compared = False
        self._results: dict[Party, VerificationResult] = {}
        self._aborts: dict[Party, VerificationAbort] = {}
        # Counted, not recorded as an outcome: a replay refusal must not displace the
        # verdict that spent the round.
        self._replay_refusals: dict[Party, int] = {}
        # One ledger per verifier: Bob's history must not reach Charlie's decision.
        self._ledgers: dict[Party, ConsumedRecords] = {
            party: ConsumedRecords(party) for party in VERIFIERS
        }

    def __repr__(self) -> str:
        """Return a debugging representation naming the phase reached."""
        if self._signature is None:
            stage = "distributed" if self._keys is not None else "new"
        else:
            reached = "".join(
                party.value[0]
                for party in VERIFIERS
                if party in self._results
            )
            refused = ", ".join(
                party.value for party in VERIFIERS if party in self._aborts
            )
            stage = (
                f"signed bit {self._signature.message_bit}"
                + (f", verified by {reached}" if reached else "")
                + (f", no verdict from {refused}" if refused else "")
            )
        return (
            f"QDSSession(L={self._params.key_length}, "
            f"s_a={self._params.s_a}, s_v={self._params.s_v}, {stage})"
        )

    # -- read-only state ---------------------------------------------------- #

    @property
    def params(self) -> ProtocolParams:
        """ProtocolParams: The parameter set this run executes under."""
        return self._params

    @property
    def scored_params(self) -> ProtocolParams:
        """ProtocolParams: The set every record and verdict is checked against."""
        return self._scored_params

    @property
    def check_plans(self) -> dict[int, CheckRoundPlan]:
        """dict: The check-round plan for each message bit, keyed by bit."""
        return dict(self._plans)

    @property
    def check_logs(self) -> dict[tuple[Party, int], CheckLog]:
        """dict: The published check observations, keyed by ``(party, bit)``."""
        return dict(self._check_logs)

    @property
    def channel(self) -> tuple[ChannelSample, ...]:
        """tuple of ChannelSample: One summary per check round, in hop order."""
        return tuple(self._channel)

    @property
    def is_distributed(self) -> bool:
        """bool: ``True`` once Phase A has run."""
        return self._keys is not None

    @property
    def is_signed(self) -> bool:
        """bool: ``True`` once Phase B has run."""
        return self._signature is not None

    @property
    def is_complete(self) -> bool:
        """bool: ``True`` once both verifiers have reached a verdict."""
        return self.is_signed and all(
            party in self._results for party in VERIFIERS
        )

    @property
    def keys(self) -> tuple[PrivateKey, PrivateKey]:
        """tuple of PrivateKey: Alice's committed pair ``(k_0, k_1)``."""
        if self._keys is None:
            raise self._not_yet(
                "no keys have been drawn", "session.distribute()"
            )
        return self._keys

    @property
    def signing_keys(self) -> tuple[PrivateKey, PrivateKey]:
        """tuple of PrivateKey: The pair Alice may actually declare."""
        if self._signing_keys is None:
            raise self._not_yet(
                "no keys have been drawn", "session.distribute()"
            )
        return self._signing_keys

    @property
    def records(self) -> dict[int, dict[Party, RecipientRecord]]:
        """dict: Recipients' logs, keyed by message bit then by party."""
        if not self._records:
            raise self._not_yet(
                "nothing has been distributed", "session.distribute()"
            )
        return {bit: dict(byparty) for bit, byparty in self._records.items()}

    @property
    def raw_records(self) -> dict[int, dict[Party, RecipientRecord]]:
        """dict: The recipients' logs *before* the symmetrisation exchange."""
        if not self._raw_records:
            raise self._not_yet(
                "nothing has been distributed", "session.distribute()"
            )
        return {bit: dict(byparty) for bit, byparty in self._raw_records.items()}

    @property
    def signature(self) -> Signature:
        """Signature: The declaration under verification."""
        if self._signature is None:
            raise self._not_yet(
                "nothing has been signed", "session.sign(message_bit)"
            )
        return self._signature

    @property
    def results(self) -> dict[Party, VerificationResult]:
        """dict: Verdicts so far, keyed by party, in the order reached."""
        return dict(self._results)

    @property
    def aborts(self) -> dict[Party, VerificationAbort]:
        """dict: Verifiers who were asked and reached no verdict, keyed by
        party.
        """
        return dict(self._aborts)

    @property
    def pooled(self) -> PooledMatchedCounts | None:
        """PooledMatchedCounts or None: what Phase C' produced."""
        return self._pooled

    @property
    def counts_compared(self) -> bool:
        """bool: ``True`` once Phase C' has been attempted."""
        return self._counts_compared

    @property
    def session_ids(self) -> dict[int, str]:
        """dict: The identifier of each distribution round, keyed by message
        bit.
        """
        if not self._session_ids:
            raise self._not_yet(
                "no distribution round has been opened", "session.distribute()"
            )
        return dict(self._session_ids)

    def opening_for(self, message_bit: int) -> str:
        """Return the secret opening of one round, for a dispute."""
        bit = _as_message_bit(message_bit)
        if not self._openings:
            raise self._not_yet(
                "no distribution round has been opened", "session.distribute()"
            )
        return self._openings[bit]

    def ledger_for(self, party: Party | str) -> ConsumedRecords:
        """Return one verifier's ledger of rounds he has already decided."""
        resolved = _as_party(party)
        if resolved not in self._ledgers:
            raise ValueError(
                f"{resolved.value} keeps no consumed-records ledger: only the "
                f"verifiers reach verdicts, so only they have rounds to spend. "
                f"Ask for Party.BOB or Party.CHARLIE."
            )
        return self._ledgers[resolved]

    # -- Phase A ------------------------------------------------------------ #

    def distribute(self) -> dict[int, dict[Party, RecipientRecord]]:
        """Run Phase A: draw both keys and teleport both public keys to both.

        >>> import numpy as np
        >>> from qsecure.protocol.params import ProtocolParams
        >>> from qsecure.protocol.session import QDSSession
        >>> records = QDSSession(
        ...     ProtocolParams(key_length=12), rng=np.random.default_rng(8)
        ... ).distribute()
        >>> sorted(records), sorted(records[0])
        ([0, 1], [<Party.BOB: 'Bob'>, <Party.CHARLIE: 'Charlie'>])
        """
        if self._keys is not None:
            raise ValueError(
                "this session has already distributed its public keys. A "
                "session is single-use: the key states are consumed on receipt, "
                "and redistributing under the same session would either reuse a "
                "key whose states are gone or silently replace evidence the "
                "verifiers have already been scored against. Build a new "
                "QDSSession for the next message."
            )

        keys = generate_key_pair(self._params, rng=self._alice_rng)
        signing_keys: list[PrivateKey] = []
        openings: dict[int, str] = {}
        session_ids: dict[int, str] = {}
        raw_records: dict[int, dict[Party, RecipientRecord]] = {}
        records: dict[int, dict[Party, RecipientRecord]] = {}
        for bit in MESSAGE_BITS:
            # Check plan from the recipients' stream, which Alice's seams never see.
            # Nothing is drawn when no check rounds are reserved.
            plan = (
                draw_check_plan(self._params, rng=self._recipient_rng)
                if self._params.has_check_rounds
                else None
            )
            if plan is not None:
                self._plans[bit] = plan
            # The round's opening, kept here until Phase B reveals it.
            openings[bit] = fresh_opening(rng=self._binding_rng)
            session_ids[bit] = session_identifier(
                bit,
                self._scored_params.key_length,
                opening=openings[bit],
                context=self._context,
            )
            returned = self._distributor(
                keys[bit],
                self._params,
                parties=VERIFIERS,
                # Tapped on every hop of a checked run; untouched otherwise.
                resource_factory=(
                    self._resource_factory
                    if plan is None
                    else _ChannelTap(
                        self._resource_factory,
                        plan=plan,
                        monitor=self._channel_monitor,
                        sink=self._channel,
                    )
                ),
                # Alice's stream only: cloning it reveals nothing about the coins.
                rng=self._alice_rng,
                # Passed only when in force, so older seams are called as before.
                **({} if plan is None else {"check_plan": plan}),
                **(
                    {}
                    if self._payload_map is None
                    else {"payload_map": self._payload_map}
                ),
            )
            signing_keys.append(
                keys[bit] if plan is None else plan.sift_key(keys[bit])
            )
            raw = {
                party: record.with_session_id(session_ids[bit])
                for party, record in self._check_distribution(
                    returned, bit, collect_logs=True
                ).items()
            }
            # Phase A': the recipients' step, on whatever the distributor produced.
            exchanged = self._symmetriser(raw, rng=self._recipient_rng)
            raw_records[bit] = raw
            # Re-stamped after the exchange so a symmetriser cannot strip the binding.
            records[bit] = {
                party: record.with_session_id(session_ids[bit])
                for party, record in self._check_distribution(
                    exchanged, bit
                ).items()
            }

        self._keys = keys
        # Same tuple object on an unchecked run, so ``declared_key is keys[b]`` holds.
        self._signing_keys = (
            keys if not self._plans else (signing_keys[0], signing_keys[1])
        )
        self._openings = openings
        self._session_ids = session_ids
        self._raw_records = raw_records
        self._records = records
        return self.records

    def _check_distribution(
        self, returned: Any, message_bit: int, *, collect_logs: bool = False
    ) -> dict[Party, RecipientRecord]:
        """Validate one distributor call's return value."""
        if not isinstance(returned, Mapping):
            raise TypeError(
                f"the distributor seam must return a mapping of Party to "
                f"RecipientRecord or to RecipientDistribution, got "
                f"{type(returned).__name__} for message bit {message_bit}; "
                f"distribute_public_key_with_checks returns exactly that."
            )
        checked: dict[Party, RecipientRecord] = {}
        for party, value in returned.items():
            resolved = _as_party(party)
            record = value
            if isinstance(value, RecipientDistribution):
                record = value.record
                if collect_logs and value.log.round_count:
                    if (
                        value.log.party is not resolved
                        or value.log.message_bit != message_bit
                    ):
                        raise ValueError(
                            f"the distributor seam filed a check log for "
                            f"{value.log.party.value} on message bit "
                            f"{value.log.message_bit} under "
                            f"{resolved.value} on bit {message_bit}. A check "
                            f"log is a published statement about one link, so a "
                            f"swap would attribute one recipient's channel to "
                            f"the other -- which is exactly the attribution a "
                            f"one-sided attack turns on."
                        )
                    # Empty logs are dropped so unchecked runs serialise unchanged.
                    self._check_logs[(resolved, message_bit)] = value.log
            if not isinstance(record, RecipientRecord):
                raise TypeError(
                    f"the distributor seam returned "
                    f"{type(value).__name__} for {resolved.value} on message "
                    f"bit {message_bit}; a RecipientRecord, or a "
                    f"RecipientDistribution holding one, is required."
                )
            if record.party is not resolved:
                raise ValueError(
                    f"the distributor seam keyed a record by "
                    f"{resolved.value!r} that is tagged {record.party.value!r}. "
                    f"The key selects the acceptance threshold, so a swap would "
                    f"score one verifier's evidence against the other's cut."
                )
            if record.message_bit != message_bit:
                raise ValueError(
                    f"the distributor seam returned a record for message bit "
                    f"{record.message_bit} while distributing for bit "
                    f"{message_bit}. Each bit gets its own key and its own "
                    f"distribution; crossing them would verify a declaration "
                    f"against states that were never sent for it."
                )
            # Against the sifted set: a checked record is already shorter.
            record.check_against(self._scored_params)
            checked[resolved] = record

        missing = [party.value for party in VERIFIERS if party not in checked]
        if missing:
            raise ValueError(
                f"the distributor seam produced no record for {missing} on "
                f"message bit {message_bit}. Both verifiers are required: "
                f"transferability and non-repudiation are statements about Bob "
                f"and Charlie together, and a session without Charlie cannot "
                f"express either."
            )
        return checked

    # -- Phase B ------------------------------------------------------------ #

    def sign(self, message_bit: int) -> Signature:
        """Run Phase B: declare a key for ``message_bit``.

        >>> import numpy as np
        >>> from qsecure.protocol.params import ProtocolParams
        >>> from qsecure.protocol.session import QDSSession
        >>> session = QDSSession(
        ...     ProtocolParams(key_length=12), rng=np.random.default_rng(9)
        ... )
        >>> _ = session.distribute()
        >>> session.sign(1).declared_key is session.keys[1]
        True
        """
        if self._keys is None:
            raise self._not_yet(
                "there is nothing to sign against", "session.distribute()"
            )
        if self._signature is not None:
            raise ValueError(
                f"this session has already signed message bit "
                f"{self._signature.message_bit}. One distribution supports one "
                f"signature: signing the other bit as well would hand a "
                f"verifier two declarations scored against logs drawn from the "
                f"same run, and the independence every bound assumes would be "
                f"gone. Build a new QDSSession."
            )
        bit = _as_message_bit(message_bit)

        # No recipient logs by default; the opt-in passes the raw logs only.
        signature = self._signer(
            bit,
            self.signing_keys,
            self._scored_params,
            records=(
                self.raw_records
                if self._signer_sees_recipient_logs
                else NO_RECIPIENT_LOGS
            ),
        )
        if not isinstance(signature, Signature):
            raise TypeError(
                f"the signer seam must return a Signature, got "
                f"{type(signature).__name__}. Even a forgery is a Signature -- "
                f"it is a declaration, and what makes it a forgery is that the "
                f"key inside it is not the one whose states were distributed."
            )
        if signature.message_bit != bit:
            raise ValueError(
                f"the signer seam was asked for message bit {bit} and returned "
                f"a signature for bit {signature.message_bit}. The bit selects "
                f"which distribution the verifiers score against, so the two "
                f"must agree; a signer that wants to attack the other bit "
                f"should be asked to sign that bit."
            )
        signature.check_against(self._scored_params)
        signature = self._bind_to_round(signature)

        self._signature = signature
        return signature

    def _bind_to_round(self, signature: Signature) -> Signature:
        """Attach this run's opening and context to a declaration."""
        opening = self._openings.get(signature.message_bit)
        if opening is None:
            return signature
        return Signature(
            message_bit=signature.message_bit,
            declared_key=signature.declared_key,
            session_opening=opening,
            context=self._context,
        )

    # -- Phase C' ----------------------------------------------------------- #

    def exchange_counts(self) -> PooledMatchedCounts | None:
        """Run Phase C': Bob and Charlie compare how much evidence each holds.

        >>> import numpy as np
        >>> from qsecure.protocol.params import ProtocolParams
        >>> from qsecure.protocol.session import QDSSession
        >>> session = QDSSession(
        ...     ProtocolParams(key_length=600), rng=np.random.default_rng(4)
        ... )
        >>> _ = session.distribute()
        >>> _ = session.sign(0)
        >>> pooled = session.exchange_counts()
        >>> pooled.pooled == pooled.bob_count + pooled.charlie_count
        True
        >>> pooled.meets_every_floor
        True
        """
        if self._signature is None:
            raise self._not_yet(
                "there is no declaration for the recipients to count against",
                "session.sign(message_bit)",
            )
        after = self._count_exchange_timing == COUNTS_AFTER_FORWARDING
        if after and self._forwarded is None:
            raise self._not_yet(
                "this session counts against the declaration each recipient "
                "received, and the hop has not happened yet, so Charlie is not "
                "holding one",
                "session.forward()",
            )
        if self._counts_compared:
            return self._pooled

        bit = self._signature.message_bit
        # Each message is built from that recipient's own log only.
        counted = self._declaration_counted()
        messages: dict[Party, MatchedCountMessage] = {
            party: matched_count_message(
                counted, self._records[bit][party], self._scored_params
            )
            for party in VERIFIERS
        }
        pooled = self._count_exchange(messages, self._scored_params)
        if pooled is not None and not isinstance(pooled, PooledMatchedCounts):
            raise TypeError(
                f"the count_exchange seam must return a PooledMatchedCounts or "
                f"None, got {type(pooled).__name__}. Returning None means the "
                f"recipients did not compare counts, which is what "
                f"no_count_exchange does; anything else is a wiring error."
            )
        self._pooled = pooled
        self._counts_compared = True
        return pooled

    def _declaration_counted(self) -> Signature:
        """Return the declaration Phase C' counts against, under either timing."""
        assert self._signature is not None  # callers check
        if (
            self._count_exchange_timing == COUNTS_AFTER_FORWARDING
            and self._forwarded is not None
        ):
            return self._forwarded
        return self._signature

    def forward(self) -> Signature:
        """Run the Bob-to-Charlie hop, without verifying at the far end.

        >>> import numpy as np
        >>> from qsecure.protocol.params import Party, ProtocolParams
        >>> from qsecure.protocol.session import (
        ...     COUNTS_AFTER_FORWARDING, QDSSession)
        >>> session = QDSSession(
        ...     ProtocolParams(key_length=192),
        ...     count_exchange_timing=COUNTS_AFTER_FORWARDING,
        ...     rng=np.random.default_rng(7),
        ... )
        >>> _ = session.distribute()
        >>> _ = session.sign(0)
        >>> forwarded = session.forward()
        >>> session.forward() is forwarded
        True
        >>> pooled = session.exchange_counts()
        >>> pooled.meets_every_floor
        True
        >>> session.verify(Party.BOB).accepted, session.transfer().accepted
        (True, True)
        """
        if self._signature is None:
            raise self._not_yet(
                "there is no declaration to forward", "session.sign(message_bit)"
            )
        if self._forwarded is not None:
            return self._forwarded
        forwarded = self._call_forwarder(self._signature)
        if not isinstance(forwarded, Signature):
            raise TypeError(
                f"the forwarder seam must return a Signature, got "
                f"{type(forwarded).__name__}. Charlie scores a declaration; an "
                f"attack on the forwarding hop alters that declaration, it does "
                f"not remove it."
            )
        if forwarded.message_bit != self._signature.message_bit:
            raise ValueError(
                f"the forwarder seam returned a signature for message bit "
                f"{forwarded.message_bit} while forwarding one for bit "
                f"{self._signature.message_bit}. The bit selects which "
                f"distribution Charlie scores against, so changing it would "
                f"verify against states that were never sent for this run."
            )
        forwarded.check_against(self._scored_params)
        # Bound to this round so the hop cannot unname it.
        self._forwarded = self._bind_to_round(forwarded)
        return self._forwarded

    # -- Phase C ------------------------------------------------------------ #

    def verify(self, party: Party | str = Party.BOB) -> VerificationResult:
        """Run Phase C for one verifier against his own log."""
        if self._signature is None:
            raise self._not_yet(
                "there is no signature to verify", "session.sign(message_bit)"
            )
        resolved = _as_party(party)
        if resolved is Party.ALICE:
            raise ValueError(
                "Alice cannot verify: she is the signer, holds no measurement "
                "record and has no acceptance threshold. Verify at Party.BOB "
                "(cut at s_a) or Party.CHARLIE (cut at s_v)."
            )

        record = self._records[self._signature.message_bit][resolved]
        declaration = self._signature
        if resolved is Party.CHARLIE and self._forwarded is not None:
            # Charlie scores what the forwarding hop actually delivered, which
            # on an honest run is the same object Bob scored.
            declaration = self._forwarded
        # Phase C' lazily, so distribute/sign/verify/transfer also gets the pooled rule.
        pooled = self.exchange_counts()
        counterpart = None if pooled is None else pooled.counterpart_of(resolved)
        try:
            # Module-level verify() with this verifier's own ledger.
            result = verify(
                declaration,
                record,
                self._scored_params,
                counterpart_matched=counterpart,
                ledger=self._ledgers[resolved],
            )
        except MatchedSetTooSmall as too_small:
            # Record the refusal before raising, unless it refuses a replay:
            # that must not erase the verdict that spent the round.
            if too_small.abort.reason is AbortReason.RECORD_ALREADY_VERIFIED:
                self._replay_refusals[resolved] = (
                    self._replay_refusals.get(resolved, 0) + 1
                )
            if (
                too_small.abort.reason is AbortReason.RECORD_ALREADY_VERIFIED
                and resolved in self._results
            ):
                raise
            self._results.pop(resolved, None)
            self._aborts[resolved] = too_small.abort
            raise
        # One current outcome per verifier: a new verdict replaces a refusal.
        self._aborts.pop(resolved, None)
        self._results[resolved] = result
        return result

    def transfer(self) -> VerificationResult:
        """Forward the signature from Bob to Charlie and verify it there."""
        if Party.BOB not in self._results and Party.BOB not in self._aborts:
            raise self._not_yet(
                "Bob has not run Phase C, so there is nothing for him to "
                "forward",
                "session.verify(Party.BOB)",
            )
        assert self._signature is not None  # implied by Bob's verdict existing
        # forward() is idempotent, so the seam never gets a second chance.
        self.forward()
        return self.verify(Party.CHARLIE)

    def _call_forwarder(self, signature: Signature) -> Signature:
        """Run the Bob-to-Charlie hop, building Bob's view only if it is
        wanted.
        """
        if not self._forwarder_wants_view:
            return self._forwarder(signature, self._scored_params)
        verdict = self._results.get(Party.BOB)
        view = RecipientView.for_party(
            Party.BOB,
            signature.message_bit,
            raw_records=self._raw_records,
            records=self._records,
            matched_count=None if verdict is None else verdict.matched_count,
        )
        return self._forwarder(signature, self._scored_params, view=view)

    # -- the whole run ------------------------------------------------------ #

    def run(self, message_bit: int) -> SessionTranscript:
        """Execute the entire protocol for one message bit and report.

        >>> import numpy as np
        >>> from qsecure.protocol.params import ProtocolParams
        >>> from qsecure.protocol.session import QDSSession
        >>> params = ProtocolParams(key_length=24)
        >>> first = QDSSession(params, rng=np.random.default_rng(3)).run(0)
        >>> second = QDSSession(params, rng=np.random.default_rng(3)).run(0)
        >>> first == second
        True
        >>> first.transferable, first.aborted
        (True, False)
        """
        self.distribute()
        self.sign(message_bit)
        if self._count_exchange_timing == COUNTS_AFTER_FORWARDING:
            # Charlie must hold a declaration before counting, so the hop comes first.
            self.forward()
        self.exchange_counts()
        # Refusals are already recorded on the session; still ask both verifiers.
        try:
            self.verify(Party.BOB)
        except MatchedSetTooSmall:
            pass
        try:
            self.transfer()
        except MatchedSetTooSmall:
            pass
        return self.transcript()

    def transcript(self) -> SessionTranscript:
        """Freeze what has happened so far into a :class:`SessionTranscript`."""
        if self._signature is None:
            raise self._not_yet(
                "nothing has been signed, so there is no run to report",
                "session.sign(message_bit)",
            )
        records = tuple(
            self._records[bit][party]
            for bit in MESSAGE_BITS
            for party in VERIFIERS
        )
        forwarded = self._forwarded
        return SessionTranscript(
            params=self._params,
            message_bit=self._signature.message_bit,
            signature=self._signature,
            records=records,
            results=tuple(self._results.values()),
            aborts=tuple(self._aborts.values()),
            pooled=self._pooled,
            # None when the hop was the identity.
            forwarded_signature=(
                None
                if forwarded is None or forwarded == self._signature
                else forwarded
            ),
            run_id=self._run_id,
            # Sorted so two runs of one seed compare equal.
            check_logs=tuple(
                self._check_logs[key]
                for key in sorted(
                    self._check_logs, key=lambda item: (item[1], item[0].value)
                )
            ),
            channel=tuple(self._channel),
            signer_saw_recipient_logs=self._signer_sees_recipient_logs,
            count_exchange_timing=self._count_exchange_timing,
            # Sorted: the ledger is a set.
            spent_rounds=tuple(
                sorted(
                    (party.value, session_id, bit)
                    for party, ledger in self._ledgers.items()
                    for session_id, bit in ledger.spent_rounds()
                )
            ),
            replay_refusals=tuple(
                sorted(
                    (party.value, count)
                    for party, count in self._replay_refusals.items()
                )
            ),
        )

    # -- helpers ------------------------------------------------------------ #

    @staticmethod
    def _not_yet(problem: str, call: str) -> ValueError:
        """Build the out-of-order error, which always names the fix."""
        return ValueError(
            f"the session is not far enough along: {problem}. The phases run "
            f"in order -- distribute, sign, verify(Party.BOB), transfer -- so "
            f"call {call} first, or run(message_bit) for all of them."
        )


def _check_result_against(
    result: VerificationResult,
    params: ProtocolParams,
    message_bit: int,
) -> None:
    """Check one verdict against the parameter set the transcript claims."""
    expected_threshold = params.threshold_for(result.party)
    if result.threshold != expected_threshold:
        raise ValueError(
            f"{result.party.value}'s verdict carries threshold "
            f"{result.threshold!r} but this transcript's parameters put his cut "
            f"at {expected_threshold!r} (s_a={params.s_a!r}, "
            f"s_v={params.s_v!r}). The threshold is a property of *whose* log "
            f"was scored, never a free field: a verdict holding the other "
            f"verifier's cut is internally consistent and still wrong, and it "
            f"would make this transcript report transferability for a run the "
            f"verifier rejected."
        )
    if result.key_length != params.key_length:
        raise ValueError(
            f"{result.party.value}'s verdict reports key_length "
            f"{result.key_length} but this transcript's parameters say "
            f"{params.key_length}. The verdict's discarded-position count, and "
            f"every detection statistic derived from it, would be computed "
            f"against the wrong denominator."
        )
    if result.message_bit != message_bit:
        raise ValueError(
            f"{result.party.value}'s verdict is for message bit "
            f"{result.message_bit} but this transcript is tagged with bit "
            f"{message_bit}. The bit selects which distribution was scored."
        )


def _check_abort_against(
    abort: VerificationAbort,
    params: ProtocolParams,
    message_bit: int,
) -> None:
    """Check one refusal to score against the parameters the transcript claims."""
    if abort.key_length != params.key_length:
        raise ValueError(
            f"{abort.party.value}'s refusal reports key_length "
            f"{abort.key_length} but this transcript's parameters say "
            f"{params.key_length}. The shortfall it records would be measured "
            f"against the wrong number of positions."
        )
    expected_floor = minimum_matched_count(params)
    if abort.minimum_matched != expected_floor:
        raise ValueError(
            f"{abort.party.value}'s refusal quotes a matched-count floor of "
            f"{abort.minimum_matched} but this transcript's parameters put it "
            f"at {expected_floor} (L={params.key_length}, "
            f"|B|={len(params.bases)}). The floor is derived from the "
            f"parameter set, never a free field: a refusal carrying a floor "
            f"nobody applied would make a scored run look starved, or a "
            f"starved one look scored."
        )
    if abort.expected_matched != params.expected_matched:
        raise ValueError(
            f"{abort.party.value}'s refusal reports an honest mean of "
            f"{abort.expected_matched!r} but this transcript's parameters give "
            f"L/|B| = {params.expected_matched!r}. That number is what makes "
            f"the shortfall legible, so it has to be this run's."
        )
    if abort.message_bit != message_bit:
        raise ValueError(
            f"{abort.party.value}'s refusal is for message bit "
            f"{abort.message_bit} but this transcript is tagged with bit "
            f"{message_bit}. The bit selects which distribution was scored."
        )


def _check_pooled_against(
    pooled: PooledMatchedCounts,
    params: ProtocolParams,
    message_bit: int,
) -> None:
    """Check one count exchange against the parameters the transcript claims."""
    if pooled.key_length != params.key_length:
        raise ValueError(
            f"the recorded count exchange reports key_length "
            f"{pooled.key_length} but this transcript's parameters say "
            f"{params.key_length}. Both floors are derived from the key "
            f"length, so the two cannot be from the same run."
        )
    for name, recorded, expected in (
        (
            "per-verifier",
            pooled.minimum_matched,
            minimum_matched_count(params),
        ),
        (
            "pooled",
            pooled.minimum_pooled,
            minimum_pooled_matched_count(params),
        ),
    ):
        if recorded != expected:
            raise ValueError(
                f"the recorded count exchange quotes a {name} matched-count "
                f"floor of {recorded} but this transcript's parameters put it "
                f"at {expected} (L={params.key_length}, "
                f"|B|={len(params.bases)}). A floor is derived from the "
                f"parameter set, never a free field: an exchange carrying a "
                f"floor nobody applied would make a run that failed the rule "
                f"look like one that passed it."
            )
    if pooled.message_bit != message_bit:
        raise ValueError(
            f"the recorded count exchange is for message bit "
            f"{pooled.message_bit} but this transcript is tagged with bit "
            f"{message_bit}. The bit selects which distribution was counted."
        )


def forwarder_wants_view(forwarder: Forwarder) -> bool:
    """Return ``True`` when ``forwarder`` *requires* a keyword-only ``view``."""
    try:
        signature = inspect.signature(forwarder)
    except (TypeError, ValueError):
        return False
    probe = object()
    try:
        signature.bind(probe, probe)
    except TypeError:
        pass
    else:
        return False
    try:
        signature.bind(probe, probe, view=probe)
    except TypeError:
        raise TypeError(
            f"forwarder must be callable either as forwarder(signature, "
            f"params) or as forwarder(signature, params, *, view), but "
            f"{forwarder!r} accepts neither: its signature is {signature}. The "
            f"view is the forwarding recipient's own RecipientView -- his two "
            f"logs and his matched count, and nothing of the counterpart's; a "
            f"forwarder that does not want it should take two arguments."
        ) from None
    return True


_forwarder_wants_view = forwarder_wants_view
"""Deprecated private alias of :func:`forwarder_wants_view`."""


def _check_sample_against(
    sample: ChannelSample,
    published: Mapping[tuple[Party, int], CheckLog],
    params: ProtocolParams,
) -> None:
    """Check one channel sample really describes a check round of this run."""
    if sample.position >= params.key_length:
        raise ValueError(
            f"a channel sample sits at position {sample.position} but this "
            f"run has {params.key_length} positions. Check positions are "
            f"indexed in the unsifted key, 0 .. L-1."
        )
    log = published.get((sample.party, sample.message_bit))
    if log is None:
        return
    if sample.position not in log.positions:
        raise ValueError(
            f"a channel sample claims position {sample.position} of "
            f"{sample.party.value}'s message bit {sample.message_bit}, but "
            f"that position is not a check round of this run: the published "
            f"log recorded {len(log.positions)} of them and this is not one. "
            f"Only check positions may be published per position -- a key "
            f"position's channel is precisely what sampling exists not to "
            f"report."
        )
    arm = (
        CheckRole.QBER
        if sample.position in {entry.position for entry in log.qber}
        else CheckRole.CHSH
    )
    if sample.role is not arm:
        raise ValueError(
            f"a channel sample at position {sample.position} of "
            f"{sample.party.value}'s message bit {sample.message_bit} is "
            f"tagged {sample.role.value!r}, but the published log recorded that "
            f"round in the {arm.value!r} arm. The role says which statistic the "
            f"round feeds, so a mismatch would file a pair's diagnostics under "
            f"the wrong estimator."
        )


def _check_seam(seam: Any, name: str, expected: str) -> None:
    """Reject a non-callable seam at construction rather than mid-run."""
    if seam is not None and not callable(seam):
        raise TypeError(
            f"{name} must be None or {expected}, got "
            f"{type(seam).__name__}. See the seams section of "
            f"qsecure.protocol.session for what each seam replaces."
        )
