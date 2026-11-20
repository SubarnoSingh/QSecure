"""Randomness isolation, made structural: an adversary owns its randomness.

A behavioural check that an adversary's decisions move with its own generator
(check (b)) and never with the session's seed (check (a)).

.. _check-a-channel:

How check (a) reaches the candidate
-----------------------------------
:func:`check_attack_isolation` installs a :class:`SessionEnvironment` around
every build and probe call and varies it with the session seed, opening the
three routes a real experiment leaks through: the harness's seed
(:func:`active_session`), ambient global randomness, and an adversary generator
derived from the session's seed.

.. _deterministic-mode:

When check (b) cannot apply
---------------------------
A candidate that is deterministic by construction (the optimal recipient
forger) cannot pass check (b). Prefer a sampled realisation; otherwise check a
randomised sibling sharing the same call path and waive (b) with
``deterministic="..."`` only alongside it.

Examples
--------
>>> import numpy as np
>>> from qsecure.protocol.params import Party, ProtocolParams
>>> from qsecure.protocol.session import QDSSession
>>> params = ProtocolParams(key_length=60)
>>> seed = 20260141
>>> honest = QDSSession(params, rng=np.random.default_rng(seed))
>>> _ = honest.distribute()
>>> rebuilt = QDSSession(params, rng=np.random.default_rng(seed))
>>> _ = rebuilt.distribute()
>>> rebuilt.records == honest.records
True

>>> def coins(session):
...     # Position i is True where Bob ended up holding Charlie's raw entry.
...     return [
...         session.records[bit][Party.BOB].entries[index]
...         == session.raw_records[bit][Party.CHARLIE].entries[index]
...         for bit in (0, 1)
...         for index in range(params.key_length)
...     ]
>>> predicted, actual = coins(rebuilt), coins(honest)
>>> sum(a == b for a, b in zip(predicted, actual)), len(actual)
(120, 120)
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import random
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Protocol

import numpy as np

from qsecure.protocol.keys import PrivateKey
from qsecure.protocol.params import Party, ProtocolParams
from qsecure.protocol.records import (
    RecipientRecord,
    RecipientView,
    recipient_views,
)
from qsecure.protocol.session import (
    NO_RECIPIENT_LOGS,
    QDSSession,
    forwarder_wants_view,
    honest_signer,
)

__all__ = [
    "DEFAULT_ATTACK_SEEDS",
    "DEFAULT_SESSION_SEEDS",
    "MIN_JUSTIFICATION",
    "SCENARIO_PARAMS",
    "SCENARIO_SEED",
    "SESSION_MATERIAL_BYTES",
    "AttackBuilder",
    "AttackIsolationError",
    "DecisionProbe",
    "IsolationReport",
    "SessionEnvironment",
    "SignerScenario",
    "active_session",
    "assert_attack_isolated",
    "canonical",
    "check_attack_isolation",
    "derived_from_seed",
    "forwarder_probe",
    "require_distinct_streams",
    "same_stream",
    "session_environment",
    "signer_probe",
    "signer_scenario",
    "stream_fingerprint",
]


DEFAULT_SESSION_SEEDS: Final[tuple[int, ...]] = (901, 902, 903, 904, 905)
"""Session seeds check (a) varies over, with the attack's generator held fixed.
"""

DEFAULT_ATTACK_SEEDS: Final[tuple[int, ...]] = (101, 102, 103, 104, 105)
"""Attack-generator seeds check (b) varies over, with the session seed fixed.
"""

SCENARIO_SEED: Final[int] = 20260141
"""Seed of the one honest run :func:`signer_scenario` freezes."""

SCENARIO_PARAMS: Final[ProtocolParams] = ProtocolParams(key_length=24)
"""Parameters for the frozen scenario -- small, because it is run per probe
set.
"""

MIN_JUSTIFICATION: Final[int] = 40
"""Shortest ``deterministic=`` justification :func:`check_attack_isolation`
takes.
"""

SESSION_MATERIAL_BYTES: Final[int] = 32
"""Bytes :class:`~qsecure.protocol.session.QDSSession` draws from its
generator.
"""

_MAX_CANONICAL_DEPTH: Final[int] = 16
"""Recursion limit for :func:`canonical`, so a cyclic trace fails loudly."""

_OPAQUE_REPR: Final[re.Pattern[str]] = re.compile(
    r"^<.* object at 0x[0-9a-fA-F]+>$"
)
"""Matches the default :func:`repr`, whose address makes every trace differ."""


class AttackBuilder(Protocol):
    """Callable returning a fresh adversary, built from a generator of its own.
    """

    def __call__(self, *, rng: np.random.Generator) -> Any:
        """Return a new adversary drawing its randomness from ``rng``."""
        ...


class DecisionProbe(Protocol):
    """Callable that exercises one adversary and returns what it decided."""

    def __call__(self, attack: Any, session_seed: int) -> Any:
        """Run ``attack`` and return the decisions it made."""
        ...


class AttackIsolationError(AssertionError):
    """Raised by :func:`assert_attack_isolated` when a candidate fails a check.
    """


@dataclass(frozen=True)
class SessionEnvironment:
    """The session randomness in force for one probe call, deliberately reachable.

    Parameters
    ----------
    seed : int
        The seed the session under attack was built from -- what a harness
        passes as ``rng=numpy.random.default_rng(seed)``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.isolation import SessionEnvironment
    >>> environment = SessionEnvironment(seed=901)

    >>> environment.stream_material() == np.random.default_rng(901).bytes(32)
    True
    >>> len(environment.stream_material())
    32
    """

    seed: int

    def __post_init__(self) -> None:
        """Coerce and range-check the seed, so a bool or a float fails here."""
        if isinstance(self.seed, bool) or not isinstance(
            self.seed, (int, np.integer)
        ):
            raise TypeError(
                f"seed must be an int, got {type(self.seed).__name__}. This is "
                f"the integer a harness passes to numpy.random.default_rng, "
                f"not the generator it returns."
            )
        if int(self.seed) < 0:
            raise ValueError(f"seed must be non-negative, got {int(self.seed)}")
        object.__setattr__(self, "seed", int(self.seed))

    def session_rng(self) -> np.random.Generator:
        """Return a fresh copy of the generator the session was given.

        Examples
        --------
        >>> from qsecure.attacks.isolation import SessionEnvironment
        >>> environment = SessionEnvironment(seed=7)
        >>> environment.session_rng().bytes(4) == (
        ...     environment.session_rng().bytes(4)
        ... )
        True
        """
        return np.random.default_rng(self.seed)

    def stream_material(self) -> bytes:
        """Return the session's seed material: its one draw from that generator.

        Returns
        -------
        bytes
            :data:`SESSION_MATERIAL_BYTES` bytes.

        Examples
        --------
        >>> from qsecure.attacks.isolation import SessionEnvironment
        >>> SessionEnvironment(seed=1).stream_material() != (
        ...     SessionEnvironment(seed=2).stream_material()
        ... )
        True
        """
        return self.session_rng().bytes(SESSION_MATERIAL_BYTES)


_ACTIVE_SESSION: SessionEnvironment | None = None
"""The environment installed by :func:`session_environment`, or ``None``."""


def active_session() -> SessionEnvironment | None:
    """Return the :class:`SessionEnvironment` in force, or ``None``.

    Returns
    -------
    SessionEnvironment or None
        ``None`` outside :func:`check_attack_isolation`, so ordinary use of an
        adversary is unaffected by the existence of this hook.

    Examples
    --------
    >>> from qsecure.attacks.isolation import active_session, session_environment
    >>> active_session() is None
    True
    >>> with session_environment(901) as environment:
    ...     active_session() is environment, active_session().seed
    (True, 901)
    >>> active_session() is None
    True
    """
    return _ACTIVE_SESSION


@contextlib.contextmanager
def session_environment(seed: int) -> Iterator[SessionEnvironment]:
    """Install one session's randomness for the duration of a probe call.

    Parameters
    ----------
    seed : int
        Non-negative.

    Yields
    ------
    SessionEnvironment
        The environment installed, also reachable via :func:`active_session`.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.isolation import session_environment

    >>> with session_environment(901):
    ...     first = float(np.random.random())
    >>> with session_environment(901):
    ...     again = float(np.random.random())
    >>> with session_environment(902):
    ...     other = float(np.random.random())
    >>> first == again, first == other
    (True, False)

    >>> before = np.random.get_state()[1][:4].tolist()
    >>> with session_environment(903):
    ...     _ = np.random.random()
    >>> np.random.get_state()[1][:4].tolist() == before
    True
    """
    global _ACTIVE_SESSION
    environment = SessionEnvironment(seed=seed)
    previous = _ACTIVE_SESSION
    numpy_state = np.random.get_state()
    stdlib_state = random.getstate()
    _ACTIVE_SESSION = environment
    np.random.seed(environment.seed % (2**32))
    random.seed(environment.seed)
    try:
        yield environment
    finally:
        _ACTIVE_SESSION = previous
        np.random.set_state(numpy_state)
        random.setstate(stdlib_state)


def stream_fingerprint(rng: np.random.Generator) -> tuple[Any, ...]:
    """Return a content-comparable identity for a generator's stream.

    Parameters
    ----------
    rng : numpy.random.Generator
        The generator to fingerprint.

    Returns
    -------
    tuple
        ``(derivation, state)``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.isolation import stream_fingerprint

    >>> left, right = np.random.default_rng(7), np.random.default_rng(7)
    >>> left is right
    False
    >>> stream_fingerprint(left) == stream_fingerprint(right)
    True
    >>> stream_fingerprint(left) == stream_fingerprint(
    ...     np.random.default_rng(8)
    ... )
    False

    >>> _ = left.random()
    >>> stream_fingerprint(left)[1] == stream_fingerprint(right)[1]
    False
    >>> stream_fingerprint(left)[0] == stream_fingerprint(right)[0]
    True
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator, got "
            f"{type(rng).__name__}"
        )
    bit_generator = rng.bit_generator
    sequence = getattr(bit_generator, "seed_seq", None)
    derivation: Any = None
    if sequence is not None:
        entropy = getattr(sequence, "entropy", None)
        if entropy is not None:
            derivation = (
                type(bit_generator).__name__,
                canonical(entropy),
                canonical(tuple(getattr(sequence, "spawn_key", ()))),
            )
    return (derivation, canonical(bit_generator.state))


def same_stream(left: np.random.Generator, right: np.random.Generator) -> bool:
    """Return whether two generators draw from the same stream.

    Parameters
    ----------
    left, right : numpy.random.Generator
        The two generators.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.isolation import same_stream

    >>> shared = np.random.default_rng(11)
    >>> same_stream(shared, shared)
    True
    >>> same_stream(np.random.default_rng(11), np.random.default_rng(11))
    True
    >>> same_stream(np.random.default_rng(11), np.random.default_rng(12))
    False

    >>> advanced = np.random.default_rng(11)
    >>> _ = advanced.random(1000)
    >>> same_stream(advanced, np.random.default_rng(11))
    True

    >>> parent = np.random.default_rng(11)
    >>> first, second = parent.spawn(2)
    >>> same_stream(first, second)
    False
    """
    if left is right:
        return True
    left_derivation, left_state = stream_fingerprint(left)
    right_derivation, right_state = stream_fingerprint(right)
    if left_state == right_state:
        return True
    return left_derivation is not None and left_derivation == right_derivation


def derived_from_seed(rng: np.random.Generator, seed: int) -> bool:
    """Return whether ``rng`` is the stream ``default_rng(seed)`` produces.

    Parameters
    ----------
    rng : numpy.random.Generator
        The adversary's generator.
    seed : int
        A session seed the harness will build a generator from.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.isolation import derived_from_seed
    >>> derived_from_seed(np.random.default_rng(500_000), 500_000)
    True
    >>> derived_from_seed(np.random.default_rng(4242), 500_000)
    False
    """
    return same_stream(rng, SessionEnvironment(seed=seed).session_rng())


def require_distinct_streams(
    left: np.random.Generator,
    right: np.random.Generator,
    *,
    left_name: str,
    right_name: str,
    detail: str = "",
) -> None:
    """Raise unless two generators are independent streams.

    Parameters
    ----------
    left, right : numpy.random.Generator
        The two generators, in the order the message should name them.
    left_name, right_name : str
        Argument names, quoted in the message.
    detail : str, optional
        One extra sentence about what each generator is for in the calling
        function, appended to the standard explanation.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.isolation import require_distinct_streams
    >>> require_distinct_streams(
    ...     np.random.default_rng(1),
    ...     np.random.default_rng(2),
    ...     left_name="rng",
    ...     right_name="session_rng",
    ... ) is None
    True
    >>> try:
    ...     require_distinct_streams(
    ...         np.random.default_rng(1),
    ...         np.random.default_rng(1),
    ...         left_name="rng",
    ...         right_name="session_rng",
    ...     )
    ... except ValueError as error:
    ...     print(str(error).splitlines()[0])
    rng and session_rng are the same stream.
    """
    if not isinstance(detail, str):
        raise TypeError(
            f"detail must be a string, got {type(detail).__name__}"
        )
    if not same_stream(left, right):
        return
    context = f"{detail.strip()}\n" if detail.strip() else ""
    raise ValueError(
        f"{left_name} and {right_name} are the same stream.\n"
        f"They need not be the same object to be the same stream, and here "
        f"they are not being compared as objects: two generators built from "
        f"one seed produce identical bytes for ever. QDSSession consumes its "
        f"caller's generator as a single 32-byte draw and derives Alice's "
        f"stream, the recipients' stream and the binding stream from that "
        f"material, so an adversary holding the same stream rebuilds the whole "
        f"run and predicts every private symmetrisation coin. Every rate "
        f"measured that way is fiction while the transcript looks entirely "
        f"normal.\n"
        f"{context}"
        f"Seed the two independently, and not from two arithmetic neighbours "
        f"of one constant either; use two unrelated seeds, or parent.spawn(2), "
        f"whose children this guard accepts."
    )


def canonical(value: Any) -> Any:
    """Return a hashable, content-comparable normal form of a decision trace.

    Parameters
    ----------
    value : object
        Any decision trace: primitives, numpy scalars and arrays, sequences,
        mappings, sets, and dataclass instances, nested arbitrarily.

    Returns
    -------
    object
        A hashable value equal to the canonical form of any equal-content
        input.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.isolation import canonical
    >>> canonical(np.array([1, 0, 1])) == canonical(np.array([1, 0, 1]))
    True
    >>> canonical({"b": (1, 2), "a": 3}) == canonical({"a": 3, "b": [1, 2]})
    True
    >>> canonical(float("nan")) == canonical(float("nan"))
    True
    >>> hash(canonical({"flips": np.array([2, 5])})) is not None
    True
    """
    return _canonical(value, 0)


def _canonical(value: Any, depth: int) -> Any:
    """Recursive worker for :func:`canonical`; ``depth`` guards cycles."""
    if depth > _MAX_CANONICAL_DEPTH:
        raise ValueError(
            f"decision trace nests deeper than {_MAX_CANONICAL_DEPTH} levels, "
            f"which usually means it contains a cycle. A probe should return a "
            f"flat, content-comparable summary of what the adversary decided, "
            f"not a live object graph."
        )
    if value is None or isinstance(value, (bool, int, bytes)):
        return value
    if isinstance(value, str):
        # StrEnum members (Party, PauliBasis) land here and compare by value.
        return str(value)
    if isinstance(value, (float, complex)):
        return (type(value).__name__, repr(value))
    if isinstance(value, np.generic):
        return _canonical(value.item(), depth + 1)
    if isinstance(value, np.ndarray):
        return (
            "ndarray",
            tuple(value.shape),
            str(value.dtype),
            _canonical(value.tolist(), depth + 1),
        )
    if isinstance(value, Mapping):
        items = tuple(
            (_canonical(key, depth + 1), _canonical(item, depth + 1))
            for key, item in value.items()
        )
        return ("mapping", tuple(sorted(items, key=repr)))
    if isinstance(value, (set, frozenset)):
        members = tuple(_canonical(item, depth + 1) for item in value)
        return ("set", tuple(sorted(members, key=repr)))
    if isinstance(value, Sequence):
        return ("sequence", tuple(_canonical(item, depth + 1) for item in value))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            "dataclass",
            type(value).__qualname__,
            tuple(
                (field.name, _canonical(getattr(value, field.name), depth + 1))
                for field in dataclasses.fields(value)
            ),
        )
    text = repr(value)
    if _OPAQUE_REPR.match(text):
        raise ValueError(
            f"a probe returned {type(value).__qualname__}, whose repr carries "
            f"its memory address, so two identical runs would compare unequal "
            f"and every candidate would be reported as reading the session. "
            f"Return what the adversary *decided* -- a Signature, a tuple of "
            f"positions, an array of flips -- rather than the adversary itself "
            f"or a live protocol object."
        )
    return ("repr", type(value).__qualname__, text)


@dataclass(frozen=True)
class IsolationReport:
    """The outcome of both halves of the check, with the evidence kept.

    Parameters
    ----------
    attack : str
        The candidate's name, for messages.
    session_seeds : tuple of int
        The seeds check (a) varied over, in order.
    attack_seeds : tuple of int
        The seeds check (b) varied over, in order.
    session_seed_offered : bool
        Whether the builder accepted a ``session_seed`` argument.
    decisions_across_session_seeds : tuple
        Canonicalised decisions under ``attack_seeds[0]``, one per session
        seed.
    decisions_across_attack_seeds : tuple
        Canonicalised decisions under ``session_seeds[0]``, one per attack
        seed.
    deterministic_justification : str or None, optional
        The caller's stated reason that check (b) is inapplicable to this
        candidate, or ``None`` for the ordinary two-sided check.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.isolation import check_attack_isolation
    >>> class Coin:
    ...     '''An adversary whose only choice comes from its own generator.'''
    ...     def __init__(self, *, rng: np.random.Generator) -> None:
    ...         self.face = int(rng.integers(1000))
    >>> report = check_attack_isolation(Coin, lambda attack, seed: attack.face)
    >>> report.isolated, report.reads_the_session
    (True, False)
    >>> report.session_seed_offered
    False
    """

    attack: str
    session_seeds: tuple[int, ...]
    attack_seeds: tuple[int, ...]
    session_seed_offered: bool
    decisions_across_session_seeds: tuple[Any, ...]
    decisions_across_attack_seeds: tuple[Any, ...]
    deterministic_justification: str | None = None

    @property
    def reads_the_session(self) -> bool:
        """bool: ``True`` when check (a) failed -- decisions moved with the
        seed.
        """
        return len(set(self.decisions_across_session_seeds)) > 1

    @property
    def uses_its_own_generator(self) -> bool:
        """bool: ``True`` when check (b) passed -- decisions moved with
        ``rng``.
        """
        return len(set(self.decisions_across_attack_seeds)) > 1

    @property
    def deterministic_accepted(self) -> bool:
        """bool: ``True`` when check (b) was waived by a stated justification.
        """
        return self.deterministic_justification is not None

    @property
    def isolated(self) -> bool:
        """bool: ``True`` when check (a) passed and check (b) passed or is
        waived.
        """
        if self.reads_the_session:
            return False
        return self.uses_its_own_generator or self.deterministic_accepted

    @property
    def offending_session_seeds(self) -> tuple[int, ...]:
        """tuple of int: Session seeds whose decisions differ from the first.
        """
        if not self.decisions_across_session_seeds:
            return ()
        anchor = self.decisions_across_session_seeds[0]
        return tuple(
            seed
            for seed, decision in zip(
                self.session_seeds,
                self.decisions_across_session_seeds,
                strict=True,
            )
            if decision != anchor
        )

    @property
    def distinct_decisions(self) -> int:
        """int: How many distinct decisions the attack seeds produced."""
        return len(set(self.decisions_across_attack_seeds))

    def summary(self) -> str:
        """Return a one-block human-readable verdict.

        Returns
        -------
        str
            Four lines: the candidate, each check with its verdict, and the
            overall result.

        Examples
        --------
        >>> import numpy as np
        >>> from qsecure.attacks.isolation import check_attack_isolation
        >>> class Coin:
        ...     def __init__(self, *, rng: np.random.Generator) -> None:
        ...         self.face = int(rng.integers(1000))
        >>> report = check_attack_isolation(
        ...     Coin, lambda attack, seed: attack.face
        ... )
        >>> print(report.summary())
        Coin: ISOLATED
          (a) vary session seed, fix own rng: 1 distinct decision (want 1)
          (b) vary own rng, fix session seed: 5 distinct decisions (want >1)
          session_seed offered to the builder: no
        """
        across_session = len(set(self.decisions_across_session_seeds))
        verdict = "ISOLATED" if self.isolated else "NOT ISOLATED"
        if self.deterministic_accepted:
            verdict = (
                "ISOLATED (check (b) waived)"
                if self.isolated
                else "NOT ISOLATED"
            )
        waiver = (
            ""
            if not self.deterministic_accepted
            else f"\n  (b) waived, deterministic by construction: "
            f"{self.deterministic_justification}"
        )
        return (
            f"{self.attack}: {verdict}\n"
            f"  (a) vary session seed, fix own rng: {across_session} distinct "
            f"{'decision' if across_session == 1 else 'decisions'} (want 1)\n"
            f"  (b) vary own rng, fix session seed: "
            f"{self.distinct_decisions} distinct "
            f"{'decision' if self.distinct_decisions == 1 else 'decisions'} "
            f"(want >1)\n"
            f"  session_seed offered to the builder: "
            f"{'yes' if self.session_seed_offered else 'no'}"
            f"{waiver}"
        )


def _as_justification(value: Any) -> str | None:
    """Validate a ``deterministic=`` waiver: ``None`` or a real sentence."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            f"deterministic must be None or a string explaining why check (b) "
            f"cannot apply, got {type(value).__name__}. It is not a flag: "
            f"passing True would switch off half the check without leaving a "
            f"reason in the report."
        )
    reason = value.strip()
    if len(reason) < MIN_JUSTIFICATION:
        raise ValueError(
            f"deterministic must be at least {MIN_JUSTIFICATION} characters of "
            f"actual justification, got {len(reason)}. Say why this adversary "
            f"has no randomness for check (b) to vary -- for instance that "
            f"declaring his own raw log strictly dominates every randomised "
            f"alternative at every position -- because that sentence is the "
            f"argument replacing the check, and it is what the next reader "
            f"audits."
        )
    return reason


def _as_seed_tuple(seeds: Any, name: str) -> tuple[int, ...]:
    """Validate a seed list: at least two distinct non-negative integers."""
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise TypeError(
            f"{name} must be a sequence of integer seeds, got "
            f"{type(seeds).__name__}"
        )
    values = []
    for position, seed in enumerate(seeds):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(
                f"{name}[{position}] must be an int, got "
                f"{type(seed).__name__}. These are seeds the check builds "
                f"generators from, not generators."
            )
        if int(seed) < 0:
            raise ValueError(
                f"{name}[{position}] must be non-negative, got {int(seed)}"
            )
        values.append(int(seed))
    if len(set(values)) < 2:
        raise ValueError(
            f"{name} must hold at least two distinct seeds, got {values}. The "
            f"check works by varying one thing at a time; with a single seed "
            f"there is nothing to vary and the verdict would be meaningless."
        )
    return tuple(values)


def _builder_offers_session_seed(build: Any, name: str) -> bool:
    """Say whether ``build`` takes ``session_seed``; raise if it refuses
    ``rng``.
    """
    if not callable(build):
        raise TypeError(
            f"{name} must be a callable returning a fresh adversary -- normally "
            f"the adversary's own class -- got {type(build).__name__}"
        )
    try:
        signature = inspect.signature(build)
    except (TypeError, ValueError):
        # A C-level callable exposes no signature. Assume the minimum.
        return False

    takes_rng = False
    takes_seed = False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            takes_rng = True
            takes_seed = True
        elif parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            if parameter.name == "rng":
                takes_rng = True
            elif parameter.name == "session_seed":
                takes_seed = True
    if not takes_rng:
        raise TypeError(
            f"{name} must accept a keyword argument 'rng'; its signature is "
            f"{signature}. every adversary takes its OWN "
            f"numpy.random.Generator as a constructor argument, so that its "
            f"behaviour cannot depend on the seed the harness gives the "
            f"session. An adversary built from an integer seed, or from no "
            f"randomness at all, cannot be checked -- and an attack that "
            f"reuses the harness seed silently predicts every symmetrisation "
            f"coin."
        )
    return takes_seed


def check_attack_isolation(
    build: AttackBuilder,
    probe: DecisionProbe,
    *,
    session_seeds: Sequence[int] = DEFAULT_SESSION_SEEDS,
    attack_seeds: Sequence[int] = DEFAULT_ATTACK_SEEDS,
    name: str | None = None,
    deterministic: str | None = None,
) -> IsolationReport:
    """Run both halves of the isolation check against one candidate adversary.

    Parameters
    ----------
    build : AttackBuilder
        Callable returning a fresh adversary.
    probe : DecisionProbe
        Callable ``(attack, session_seed) -> decisions``.
    session_seeds : sequence of int, optional
        Varied in check (a) with ``attack_seeds[0]`` held fixed.
    attack_seeds : sequence of int, optional
        Varied in check (b) with ``session_seeds[0]`` held fixed.
    name : str or None, optional
        Label for the report; defaults to the builder's ``__name__``.
    deterministic : str or None, optional
        A written justification that check (b) is inapplicable because the
        candidate is deterministic *by construction*.

    Returns
    -------
    IsolationReport
        Both halves' evidence and their verdicts.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.isolation import check_attack_isolation

    >>> class OwnCoins:
    ...     def __init__(self, *, rng: np.random.Generator) -> None:
    ...         self.flips = rng.integers(0, 2, size=8)
    >>> probe = lambda attack, session_seed: attack.flips
    >>> check_attack_isolation(OwnCoins, probe).isolated
    True

    >>> class SeedPeeker:
    ...     def __init__(
    ...         self, *, rng: np.random.Generator, session_seed: int
    ...     ) -> None:
    ...         self.flips = rng.integers(0, 2, size=8) ^ np.random.default_rng(
    ...             session_seed
    ...         ).integers(0, 2, size=8)
    >>> report = check_attack_isolation(SeedPeeker, probe)
    >>> report.isolated, report.reads_the_session
    (False, True)
    >>> report.session_seed_offered
    True
    >>> len(report.offending_session_seeds)
    4

    >>> class Deaf:
    ...     def __init__(self, *, rng: np.random.Generator) -> None:
    ...         self.flips = np.zeros(8, dtype=int)
    >>> report = check_attack_isolation(Deaf, probe)
    >>> report.reads_the_session, report.uses_its_own_generator
    (False, False)
    >>> report.isolated
    False
    """
    sessions = _as_seed_tuple(session_seeds, "session_seeds")
    attacks = _as_seed_tuple(attack_seeds, "attack_seeds")
    justification = _as_justification(deterministic)
    offered = _builder_offers_session_seed(build, "build")
    if not callable(probe):
        raise TypeError(
            f"probe must be a callable (attack, session_seed) -> decisions, got "
            f"{type(probe).__name__}"
        )
    label = name if name is not None else getattr(build, "__name__", None)
    if label is None:
        label = type(build).__name__

    def observe(attack_seed: int, session_seed: int) -> Any:
        """Build a fresh adversary for one seed pair and read its decisions."""
        with session_environment(session_seed):
            generator = np.random.default_rng(attack_seed)
            if offered:
                keywords = {"rng": generator, "session_seed": session_seed}
                attack = build(**keywords)  # type: ignore[call-arg]
            else:
                attack = build(rng=generator)
            return canonical(probe(attack, session_seed))

    anchor = observe(attacks[0], sessions[0])
    across_session = (anchor,) + tuple(
        observe(attacks[0], seed) for seed in sessions[1:]
    )
    across_attack = (anchor,) + tuple(
        observe(seed, sessions[0]) for seed in attacks[1:]
    )
    return IsolationReport(
        attack=label,
        session_seeds=sessions,
        attack_seeds=attacks,
        session_seed_offered=offered,
        decisions_across_session_seeds=across_session,
        decisions_across_attack_seeds=across_attack,
        deterministic_justification=justification,
    )


def assert_attack_isolated(
    build: AttackBuilder,
    probe: DecisionProbe,
    *,
    session_seeds: Sequence[int] = DEFAULT_SESSION_SEEDS,
    attack_seeds: Sequence[int] = DEFAULT_ATTACK_SEEDS,
    name: str | None = None,
    deterministic: str | None = None,
) -> IsolationReport:
    """Run :func:`check_attack_isolation` and raise unless the candidate passes.

    Parameters
    ----------
    build : AttackBuilder
        As :func:`check_attack_isolation`.
    probe : DecisionProbe
        As :func:`check_attack_isolation`.
    session_seeds : sequence of int, optional
        As :func:`check_attack_isolation`.
    attack_seeds : sequence of int, optional
        As :func:`check_attack_isolation`.
    name : str or None, optional
        As :func:`check_attack_isolation`.
    deterministic : str or None, optional
        As :func:`check_attack_isolation`.

    Returns
    -------
    IsolationReport
        The passing report.

    Raises
    ------
    AttackIsolationError
        If either half failed.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.isolation import (
    ...     AttackIsolationError, assert_attack_isolated)
    >>> class OwnCoins:
    ...     def __init__(self, *, rng: np.random.Generator) -> None:
    ...         self.flips = rng.integers(0, 2, size=8)
    >>> probe = lambda attack, session_seed: attack.flips
    >>> assert_attack_isolated(OwnCoins, probe).isolated
    True

    >>> class Deaf:
    ...     def __init__(self, *, rng: np.random.Generator) -> None:
    ...         self.flips = np.zeros(8, dtype=int)
    >>> try:
    ...     assert_attack_isolated(Deaf, probe)
    ... except AttackIsolationError as error:
    ...     print(str(error).splitlines()[0])
    Deaf ignored the generator it was handed.
    """
    report = check_attack_isolation(
        build,
        probe,
        session_seeds=session_seeds,
        attack_seeds=attack_seeds,
        name=name,
        deterministic=deterministic,
    )
    if report.reads_the_session:
        raise AttackIsolationError(
            f"{report.attack} reads the session's randomness.\n"
            f"Its own generator was held fixed at seed "
            f"{report.attack_seeds[0]} and only the session seed changed, yet "
            f"its decisions moved at session "
            f"{'seeds' if len(report.offending_session_seeds) != 1 else 'seed'} "
            f"{list(report.offending_session_seeds)} (anchor: session seed "
            f"{report.session_seeds[0]}).\n"
            f"an adversary's behaviour must be a function of its "
            f"own generator and of what the threat model says it can observe. "
            f"An attack derived from the session seed predicts every "
            f"symmetrisation coin and publishes repudiation rates that are "
            f"fiction. Draw the adversary's choices from the rng it was "
            f"constructed with, and never from the harness seed.\n"
            f"Three routes reach the seed and any of them produces this "
            f"symptom: a constructor argument, a draw from ambient global "
            f"randomness (numpy.random.* or random.*, never allowed for "
            f"exactly this reason), and an own generator the harness derived "
            f"from the session's seed. See the check-a-channel section of qsecure.attacks.isolation.\n"
            f"If the decisions moved because the *probe* handed the adversary "
            f"different records per seed, fix the probe instead: it must hold "
            f"everything the adversary legitimately observes fixed.\n"
            f"{report.summary()}"
        )
    if not report.uses_its_own_generator and not report.deterministic_accepted:
        raise AttackIsolationError(
            f"{report.attack} ignored the generator it was handed.\n"
            f"The session seed was held fixed at {report.session_seeds[0]} and "
            f"the attack's own generator was varied over "
            f"{list(report.attack_seeds)}, and every run decided the same "
            f"thing.\n"
            f"This is not a pass with a caveat, it is a failed check: a "
            f"constant is independent of everything, so the first half of the "
            f"check proved nothing about this candidate. Either the adversary "
            f"is not drawing from its rng -- in which case its randomness isolation is "
            f"untested -- or the probe is not looking at the part of it that is "
            f"random.\n"
            f"If the candidate is deterministic *by construction* -- its "
            f"optimal move is a fixed function of what it observes, so there "
            f"is no randomness to vary and never was -- pass "
            f"deterministic='...' with the argument for why, and read "
            f"the deterministic-mode section of qsecure.attacks.isolation first: the waiver is only sound when a "
            f"randomised variant of the same adversary is checked alongside "
            f"it.\n"
            f"{report.summary()}"
        )
    return report


@dataclass(frozen=True)
class SignerScenario:
    """One honest Phase A outcome, frozen, to be replayed into every probe.

    Parameters
    ----------
    params : ProtocolParams
        The parameter set the scenario was produced under.
    message_bit : int
        The bit the probe will ask the candidate to sign.
    keys : tuple of PrivateKey
        Alice's committed ``(k_0, k_1)``.
    raw_records : mapping
        Pre-exchange logs, keyed by message bit then party.
    records : mapping
        Post-exchange logs, same shape, likewise read-only.

    Examples
    --------
    >>> from qsecure.attacks.isolation import signer_scenario
    >>> scenario = signer_scenario()
    >>> scenario.message_bit, len(scenario.keys)
    (0, 2)
    >>> sorted(scenario.records)
    [0, 1]
    >>> scenario.records[0][Party.BOB].symmetrised
    True
    """

    params: ProtocolParams
    message_bit: int
    keys: tuple[PrivateKey, PrivateKey]
    raw_records: Mapping[int, Mapping[Party, RecipientRecord]]
    records: Mapping[int, Mapping[Party, RecipientRecord]]

    def __post_init__(self) -> None:
        """Replace both nested mappings with read-only views of themselves."""
        for name in ("raw_records", "records"):
            nested = getattr(self, name)
            object.__setattr__(
                self,
                name,
                MappingProxyType(
                    {
                        bit: MappingProxyType(dict(by_party))
                        for bit, by_party in nested.items()
                    }
                ),
            )

    @property
    def views(self) -> dict[Party, RecipientView]:
        """dict: One :class:`RecipientView` per verifier, Bob then Charlie."""
        return recipient_views(
            self.message_bit,
            raw_records=self.raw_records,
            records=self.records,
        )


_SCENARIO_CACHE: dict[tuple[int, int, int], SignerScenario] = {}


def signer_scenario(
    params: ProtocolParams = SCENARIO_PARAMS,
    *,
    message_bit: int = 0,
    seed: int = SCENARIO_SEED,
) -> SignerScenario:
    """Run one honest session and freeze it as the probe's fixed observations.

    Parameters
    ----------
    params : ProtocolParams, optional
        Defaults to :data:`SCENARIO_PARAMS`.
    message_bit : int, optional
        ``0`` or ``1``.
    seed : int, optional
        The scenario's own seed, deliberately unrelated to the seeds the check
        varies.

    Examples
    --------
    >>> from qsecure.attacks.isolation import signer_scenario
    >>> scenario = signer_scenario()
    >>> len(scenario.records[0][Party.BOB]) == scenario.params.key_length
    True
    >>> signer_scenario() is scenario
    True
    """
    if not isinstance(params, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(params).__name__}"
        )
    if isinstance(message_bit, bool) or message_bit not in (0, 1):
        raise ValueError(
            f"message_bit must be 0 or 1, got {message_bit!r}. A scenario "
            f"covers the one distribution run the candidate will sign against."
        )
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if int(seed) < 0:
        raise ValueError(f"seed must be non-negative, got {int(seed)}")

    key = (params.key_length, int(message_bit), int(seed))
    cached = _SCENARIO_CACHE.get(key)
    if cached is not None:
        return cached
    session = QDSSession(params, rng=np.random.default_rng(int(seed)))
    session.distribute()
    scenario = SignerScenario(
        params=params,
        message_bit=int(message_bit),
        keys=session.keys,
        raw_records=session.raw_records,
        records=session.records,
    )
    _SCENARIO_CACHE[key] = scenario
    return scenario


def signer_probe(scenario: SignerScenario | None = None) -> DecisionProbe:
    """Return a :class:`DecisionProbe` for the ``Signer`` seam.

    Parameters
    ----------
    scenario : SignerScenario or None, optional
        The frozen observations.

    Returns
    -------
    DecisionProbe
        ``(attack, session_seed) -> Signature``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.isolation import assert_attack_isolated, signer_probe
    >>> from qsecure.protocol.keys import KeyElement, PrivateKey, key_from_record
    >>> from qsecure.protocol.params import Party
    >>> from qsecure.protocol.signature import Signature

    >>> class ForgingBob:
    ...     def __init__(self, *, rng: np.random.Generator) -> None:
    ...         self.rng = rng
    ...     def __call__(self, message_bit, keys, params, *, records):
    ...         own = records[message_bit][Party.BOB]
    ...         elements = list(key_from_record(own, message_bit=message_bit))
    ...         where = int(self.rng.integers(len(elements)))
    ...         flipped = elements[where]
    ...         elements[where] = KeyElement(
    ...             flipped.basis, -flipped.eigenvalue
    ...         )
    ...         return Signature(
    ...             message_bit, PrivateKey(message_bit, elements)
    ...         )
    >>> assert_attack_isolated(ForgingBob, signer_probe()).isolated
    True
    """
    resolved = signer_scenario() if scenario is None else scenario
    if not isinstance(resolved, SignerScenario):
        raise TypeError(
            f"scenario must be a SignerScenario or None, got "
            f"{type(resolved).__name__}. Build one with signer_scenario()."
        )

    def probe(attack: Any, session_seed: int) -> Any:
        """Declare once against the frozen scenario; return what was said."""
        del session_seed
        return attack(
            resolved.message_bit,
            resolved.keys,
            resolved.params,
            records=resolved.raw_records,
        )

    return probe


def forwarder_probe(
    scenario: SignerScenario | None = None,
    *,
    party: Party = Party.BOB,
    matched_count: int | None = None,
) -> DecisionProbe:
    """Return a :class:`DecisionProbe` for the ``forwarder`` seam.

    Parameters
    ----------
    scenario : SignerScenario or None, optional
        The frozen observations.
    party : Party, optional
        Whose view the hop is given; Bob owns this hop, so
        :attr:`~qsecure.protocol.params.Party.BOB` is the only value a shipped
        run produces and the default.
    matched_count : int or None, optional
        The matched count to put on the view, standing in for the verdict the
        forwarding party has just reached.

    Returns
    -------
    DecisionProbe
        ``(attack, session_seed) -> Signature``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.isolation import (
    ...     assert_attack_isolated, forwarder_probe)
    >>> from qsecure.protocol.keys import KeyElement, PrivateKey
    >>> from qsecure.protocol.signature import Signature

    >>> class CoinFlippingHop:
    ...     def __init__(self, *, rng: np.random.Generator) -> None:
    ...         self.rng = rng
    ...     def __call__(self, signature, params, *, view):
    ...         elements = [
    ...             KeyElement(
    ...                 params.bases[int(self.rng.integers(len(params.bases)))],
    ...                 int(self.rng.choice((-1, 1))),
    ...             )
    ...             for _ in range(params.key_length)
    ...         ]
    ...         return Signature(
    ...             signature.message_bit,
    ...             PrivateKey(signature.message_bit, elements),
    ...         )
    >>> assert_attack_isolated(CoinFlippingHop, forwarder_probe()).isolated
    True

    >>> seen = {}
    >>> class Peeking:
    ...     def __init__(self, *, rng: np.random.Generator) -> None:
    ...         self.rng = rng
    ...     def __call__(self, signature, params, *, view):
    ...         seen["party"] = view.party
    ...         return signature
    >>> _ = forwarder_probe()(Peeking(rng=np.random.default_rng(0)), 901)
    >>> seen["party"] is Party.BOB
    True
    """
    resolved = signer_scenario() if scenario is None else scenario
    if not isinstance(resolved, SignerScenario):
        raise TypeError(
            f"scenario must be a SignerScenario or None, got "
            f"{type(resolved).__name__}. Build one with signer_scenario()."
        )
    if not isinstance(party, Party):
        raise TypeError(
            f"party must be a Party, got {type(party).__name__}"
        )
    if party is Party.ALICE:
        raise ValueError(
            "Alice does not forward: the hop this probe models is the one "
            "between the two recipients, and she holds no measurement record "
            "to build a RecipientView from. Pass Party.BOB, who owns it."
        )
    if matched_count is not None and (
        isinstance(matched_count, bool)
        or not isinstance(matched_count, (int, np.integer))
    ):
        raise TypeError(
            f"matched_count must be an int or None, got "
            f"{type(matched_count).__name__}"
        )

    view = RecipientView.for_party(
        party,
        resolved.message_bit,
        raw_records=resolved.raw_records,
        records=resolved.records,
        matched_count=None if matched_count is None else int(matched_count),
    )
    declaration = honest_signer(
        resolved.message_bit,
        resolved.keys,
        resolved.params,
        records=NO_RECIPIENT_LOGS,
    )

    def probe(attack: Any, session_seed: int) -> Any:
        """Forward once against the frozen scenario; return what was passed on.
        """
        del session_seed
        if forwarder_wants_view(attack):
            return attack(declaration, resolved.params, view=view)
        return attack(declaration, resolved.params)

    return probe
