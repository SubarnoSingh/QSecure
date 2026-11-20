"""Impersonation: Mallory stands where Alice stands, on one seam or on both.

Mallory runs the protocol honestly with a key pair of her own; how far that
gets depends on how many of Alice's seams she holds.

Examples
--------
>>> import numpy as np
>>> from qsecure.attacks.impersonation import (
...     ImpersonationScope, run_impersonation)
>>> from qsecure.protocol.params import ProtocolParams
>>> demo = ProtocolParams(key_length=48)
>>> control = run_impersonation(
...     ImpersonationScope.NONE, demo,
...     session_seed=4, attack_rng=np.random.default_rng(0))
>>> control.accepted_by_both, control.bob_rate
(True, 0.0)

>>> attacked = run_impersonation(
...     ImpersonationScope.SIGNING, demo,
...     session_seed=4, attack_rng=np.random.default_rng(0))
>>> attacked.accepted_by_both, attacked.transferable
(False, False)
>>> attacked.bob_rate > 0.25
True

>>> whole = run_impersonation(
...     ImpersonationScope.FULL, demo,
...     session_seed=4, attack_rng=np.random.default_rng(0))
>>> whole.accepted_by_both, whole.bob_rate, whole.charlie_rate
(True, 0.0, 0.0)
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from qsecure.attacks.isolation import derived_from_seed
from qsecure.attacks.statistics import two_sided_z, wilson_bounds
from qsecure.protocol.checkrounds import CheckRoundPlan, Interval
from qsecure.protocol.distribute import (
    PayloadMap,
    RecipientDistribution,
    ResourceFactory,
    distribute_public_key_with_checks,
)
from qsecure.protocol.keys import PrivateKey, generate_private_key
from qsecure.protocol.params import VERIFIERS, Party, ProtocolParams
from qsecure.protocol.records import RecipientRecord
from qsecure.protocol.session import QDSSession, SessionTranscript
from qsecure.protocol.signature import Signature

__all__ = [
    "DEMO_MEASUREMENT_PARAMS",
    "MATCHED_IDENTICAL_TRIALS",
    "MEASURED",
    "MEASUREMENT_PARAMS",
    "ImpersonationMeasurement",
    "ImpersonationScope",
    "ImpersonationTrial",
    "Impersonator",
    "MeasuredRun",
    "detector_signals",
    "distributor_probe",
    "impersonation_seams",
    "matched_sets_identical",
    "measure_impersonation",
    "run_impersonation",
    "session_seeds",
    "shipped_summary",
    "signing_probe",
    "wilson_interval",
]


CONFIDENCE: Final[float] = 0.95

MEASUREMENT_PARAMS: Final[ProtocolParams] = ProtocolParams(key_length=192)

DEMO_MEASUREMENT_PARAMS: Final[ProtocolParams] = ProtocolParams(key_length=48)

FIRST_SESSION_SEED: Final[int] = 900_000


class ImpersonationScope(enum.StrEnum):
    """Which of Alice's two seams Mallory holds.

    Attributes
    ----------
    NONE
        Neither.
    DISTRIBUTION
        Phase A only: Mallory sends states for her own key, Alice declares
        hers.
    SIGNING
        Phase B only: Alice's states, Mallory's declaration.
    FULL
        Both.

    Examples
    --------
    >>> from qsecure.attacks.impersonation import ImpersonationScope
    >>> ImpersonationScope.FULL.is_partial, ImpersonationScope.FULL.in_model
    (False, False)
    >>> ImpersonationScope.SIGNING.is_partial
    True
    >>> sorted(str(scope) for scope in ImpersonationScope if scope.is_partial)
    ['distribution', 'signing']
    """

    NONE = "none"
    DISTRIBUTION = "distribution"
    SIGNING = "signing"
    FULL = "full"

    @property
    def takes_distribution(self) -> bool:
        """bool: True when Mallory drives Phase A."""
        return self in (ImpersonationScope.DISTRIBUTION, ImpersonationScope.FULL)

    @property
    def takes_signing(self) -> bool:
        """bool: True when Mallory declares in Phase B."""
        return self in (ImpersonationScope.SIGNING, ImpersonationScope.FULL)

    @property
    def is_partial(self) -> bool:
        """bool: True for exactly one seam -- the in-model, caught-cold cases.
        """
        return self.takes_distribution != self.takes_signing

    @property
    def in_model(self) -> bool:
        """bool: False only for :attr:`FULL`, which (AUTH) assumes away."""
        return self is not ImpersonationScope.FULL


def _as_scope(scope: ImpersonationScope | str) -> ImpersonationScope:
    """Return ``scope`` as an :class:`ImpersonationScope`."""
    if isinstance(scope, ImpersonationScope):
        return scope
    try:
        return ImpersonationScope(str(scope).lower())
    except ValueError:
        known = ", ".join(str(member) for member in ImpersonationScope)
        raise ValueError(
            f"unknown impersonation scope {scope!r}; expected one of {known}. "
            f"The scope names which of Alice's two seams Mallory holds, and "
            f"the answer changes the result completely."
        ) from None


def _as_verifier(party: Party | str) -> Party:
    """Return ``party`` as a verifier, refusing Alice."""
    try:
        resolved = Party(str(party))
    except ValueError:
        raise ValueError(
            f"unknown party {party!r}; expected Bob or Charlie"
        ) from None
    if not resolved.is_verifier:
        raise ValueError(
            f"{resolved!s} keeps no record and reaches no verdict, so there is "
            f"no rate to report for her. Ask about Bob or Charlie."
        )
    return resolved


def _rate(mismatches: int | None, matched: int | None) -> float | None:
    """Return ``mismatches / matched``, or ``None`` when there was no verdict.
    """
    if mismatches is None or matched is None or matched == 0:
        return None
    return mismatches / matched


class Impersonator:
    """Mallory: she runs the protocol correctly, with a key pair of her own.

    Parameters
    ----------
    rng : numpy.random.Generator
        Mallory's own generator.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.impersonation import Impersonator
    >>> from qsecure.protocol.params import ProtocolParams
    >>> params = ProtocolParams(key_length=12)
    >>> mallory = Impersonator(rng=np.random.default_rng(11))
    >>> declaration = mallory.sign(0, (), params, records={})
    >>> len(declaration), declaration.message_bit
    (12, 0)

    >>> mallory.key_for(0, params) is declaration.declared_key
    True
    >>> sorted(mallory.keys)
    [(0, 12)]

    >>> twin = Impersonator(rng=np.random.default_rng(11))
    >>> twin.key_for(0, params) == mallory.key_for(0, params)
    True
    >>> other = Impersonator(rng=np.random.default_rng(12))
    >>> other.key_for(0, params) == mallory.key_for(0, params)
    False
    """

    def __init__(self, *, rng: np.random.Generator) -> None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                f"rng must be a numpy.random.Generator, got "
                f"{type(rng).__name__}. an adversary owns its "
                f"randomness and is constructed from a generator of its own. "
                f"An integer seed is refused because the natural next step is "
                f"to reuse the harness seed, which would let this adversary "
                f"rebuild the session's private streams."
            )
        self._rng = rng
        self._keys: dict[tuple[int, int], PrivateKey] = {}

    def __repr__(self) -> str:
        """Return a debugging representation naming what she has drawn."""
        drawn = ", ".join(
            f"bit {bit} at L={length}" for bit, length in sorted(self._keys)
        )
        return f"Impersonator({drawn or 'nothing drawn yet'})"

    @property
    def keys(self) -> Mapping[tuple[int, int], PrivateKey]:
        """Mapping: the keys drawn so far, keyed by ``(message_bit, length)``.
        """
        return dict(self._keys)

    def key_for(self, message_bit: int, params: ProtocolParams) -> PrivateKey:
        """Return Mallory's key for one bit and length, drawing it if needed.

        Parameters
        ----------
        message_bit : int
            ``0`` or ``1``.
        params : ProtocolParams
            Supplies the length and the basis alphabet.

        Returns
        -------
        PrivateKey
            The same object on every later call for the same ``(message_bit,
            key_length)``.

        Examples
        --------
        >>> import numpy as np
        >>> from qsecure.attacks.impersonation import Impersonator
        >>> from qsecure.protocol.params import ProtocolParams
        >>> mallory = Impersonator(rng=np.random.default_rng(3))
        >>> key = mallory.key_for(1, ProtocolParams(key_length=9))
        >>> len(key), key.message_bit
        (9, 1)
        >>> mallory.key_for(1, ProtocolParams(key_length=9)) is key
        True
        """
        if not isinstance(params, ProtocolParams):
            raise TypeError(
                f"params must be a ProtocolParams, got {type(params).__name__}"
            )
        if isinstance(message_bit, bool) or message_bit not in (0, 1):
            raise ValueError(
                f"message_bit must be 0 or 1, got {message_bit!r}. Mallory "
                f"commits to one key per future message bit exactly as Alice "
                f"does."
            )
        cache_key = (int(message_bit), params.key_length)
        cached = self._keys.get(cache_key)
        if cached is None:
            cached = generate_private_key(params, int(message_bit), rng=self._rng)
            self._keys[cache_key] = cached
        return cached

    def distribute(
        self,
        key: PrivateKey,
        params: ProtocolParams,
        *,
        parties: Sequence[Party | str] = VERIFIERS,
        resource_factory: ResourceFactory | None = None,
        rng: np.random.Generator | None = None,
        check_plan: CheckRoundPlan | None = None,
        payload_map: PayloadMap | None = None,
    ) -> dict[Party, RecipientDistribution]:
        """Phase A, run honestly -- on Mallory's key instead of Alice's.

        Parameters
        ----------
        key : PrivateKey
            Alice's key for this bit.
        params : ProtocolParams
            The full parameter set, at the unsifted length.
        parties : sequence of Party or str, optional
            Passed through.
        resource_factory : callable or None, optional
            Passed through, tap and all.
        rng : numpy.random.Generator or None, optional
            The session's **Alice-side** stream, used only to drive the
            teleportation it is meant to drive.
        check_plan : CheckRoundPlan or None, optional
            Executed as given, and used to sift her key so that :meth:`sign`
            declares the right one.
        payload_map : callable or None, optional
            Passed through.

        Returns
        -------
        dict of Party to RecipientDistribution
            Whatever
            :func:`~qsecure.protocol.distribute.distribute_public_key_with_checks`
            returned for her key.

        Examples
        --------
        >>> import numpy as np
        >>> from qsecure.attacks.impersonation import Impersonator
        >>> from qsecure.protocol.keys import generate_private_key
        >>> from qsecure.protocol.params import Party, ProtocolParams
        >>> params = ProtocolParams(key_length=6)
        >>> alice = generate_private_key(params, 0, rng=np.random.default_rng(1))
        >>> mallory = Impersonator(rng=np.random.default_rng(2))
        >>> sent = mallory.distribute(
        ...     alice, params, rng=np.random.default_rng(3))
        >>> sorted(sent) == [Party.BOB, Party.CHARLIE]
        True
        >>> mallory.key_for(0, params) == alice
        False
        """
        mine = self.key_for(key.message_bit, params)
        if check_plan is not None:
            sifted = check_plan.sift_key(mine)
            self._keys[(int(key.message_bit), len(sifted))] = sifted
        return distribute_public_key_with_checks(
            mine,
            params,
            parties=parties,
            resource_factory=resource_factory,
            rng=rng,
            check_plan=check_plan,
            payload_map=payload_map,
        )

    def sign(
        self,
        message_bit: int,
        keys: tuple[PrivateKey, PrivateKey] | tuple[()],
        params: ProtocolParams,
        *,
        records: Mapping[int, Mapping[Party, RecipientRecord]],
    ) -> Signature:
        """Phase B, run honestly -- on Mallory's key instead of Alice's.

        Parameters
        ----------
        message_bit : int
            The bit to declare.
        keys : tuple
            Alice's committed pair.
        params : ProtocolParams
            The session's *scored* parameters, which fix the declared length.
        records : mapping
            **Ignored**: no impersonator in the threat model holds a
            recipient's log, and this attack does not need one.

        Returns
        -------
        Signature
            Her key for ``message_bit`` at ``params.key_length``, unbound; the
            session attaches the round identifier itself
            (:meth:`~qsecure.protocol.session.QDSSession._bind_to_round`), so
            the declaration is scored on its contents and never rejected as a
            replay.

        Examples
        --------
        >>> import numpy as np
        >>> from qsecure.attacks.impersonation import Impersonator
        >>> from qsecure.protocol.params import ProtocolParams
        >>> params = ProtocolParams(key_length=9)
        >>> mallory = Impersonator(rng=np.random.default_rng(4))
        >>> declaration = mallory.sign(1, (), params, records={})
        >>> declaration.message_bit, len(declaration)
        (1, 9)
        """
        del keys, records
        return Signature(
            message_bit=int(message_bit),
            declared_key=self.key_for(message_bit, params),
        )


def impersonation_seams(
    mallory: Impersonator, scope: ImpersonationScope | str
) -> dict[str, Any]:
    """Return the :class:`~qsecure.protocol.session.QDSSession` keywords for a
    scope.

    Parameters
    ----------
    mallory : Impersonator
        The adversary whose bound methods go into the seams.
    scope : ImpersonationScope or str
        Which seams she holds.

    Returns
    -------
    dict
        Keyword arguments, ready to splat.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.impersonation import (
    ...     ImpersonationScope, Impersonator, impersonation_seams)
    >>> mallory = Impersonator(rng=np.random.default_rng(0))
    >>> sorted(impersonation_seams(mallory, ImpersonationScope.FULL))
    ['distributor', 'signer']
    >>> sorted(impersonation_seams(mallory, "signing"))
    ['signer']
    >>> impersonation_seams(mallory, ImpersonationScope.NONE)
    {}
    """
    if not isinstance(mallory, Impersonator):
        raise TypeError(
            f"mallory must be an Impersonator, got {type(mallory).__name__}"
        )
    resolved = _as_scope(scope)
    seams: dict[str, Any] = {}
    if resolved.takes_distribution:
        seams["distributor"] = mallory.distribute
    if resolved.takes_signing:
        seams["signer"] = mallory.sign
    return seams


@dataclass(frozen=True)
class ImpersonationTrial:
    """One session, reduced to the numbers an impersonation experiment needs.

    Parameters
    ----------
    scope : ImpersonationScope
        Which seams Mallory held.
    session_seed : int
        The harness's seed for this run.
    key_length : int
        ``L``.
    message_bit : int
        The bit signed.
    bob_matched, charlie_matched : int or None
        ``|M_R|`` per verifier; ``None`` if that verifier aborted.
    bob_mismatches, charlie_mismatches : int or None
        ``e_R`` per verifier; ``None`` if that verifier aborted.
    bob_accepted, charlie_accepted : bool or None
        The verdicts; ``None`` if that verifier aborted.
    transferable : bool
        The transcript's own verdict on the run as a whole.

    Examples
    --------
    >>> from qsecure.attacks.impersonation import (
    ...     ImpersonationScope, ImpersonationTrial)
    >>> trial = ImpersonationTrial(
    ...     scope=ImpersonationScope.SIGNING, session_seed=1, key_length=48,
    ...     message_bit=0, bob_matched=16, charlie_matched=18,
    ...     bob_mismatches=8, charlie_mismatches=9,
    ...     bob_accepted=False, charlie_accepted=False, transferable=False)
    >>> trial.bob_rate, trial.accepted_by_both
    (0.5, False)
    >>> trial.reached_verdicts
    True
    """

    scope: ImpersonationScope
    session_seed: int
    key_length: int
    message_bit: int
    bob_matched: int | None
    charlie_matched: int | None
    bob_mismatches: int | None
    charlie_mismatches: int | None
    bob_accepted: bool | None
    charlie_accepted: bool | None
    transferable: bool

    @property
    def bob_rate(self) -> float | None:
        """float or None: ``r_Bob``, or ``None`` if Bob reached no verdict."""
        return _rate(self.bob_mismatches, self.bob_matched)

    @property
    def charlie_rate(self) -> float | None:
        """float or None: ``r_Charlie``, or ``None`` if he reached no verdict.
        """
        return _rate(self.charlie_mismatches, self.charlie_matched)

    @property
    def reached_verdicts(self) -> bool:
        """bool: True when both verifiers scored rather than aborting."""
        return self.bob_accepted is not None and self.charlie_accepted is not None

    @property
    def accepted_by_both(self) -> bool:
        """bool: True only when both verifiers accepted."""
        return bool(self.bob_accepted) and bool(self.charlie_accepted)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping of every field."""
        return {
            "scope": str(self.scope),
            "session_seed": self.session_seed,
            "key_length": self.key_length,
            "message_bit": self.message_bit,
            "bob_matched": self.bob_matched,
            "charlie_matched": self.charlie_matched,
            "bob_mismatches": self.bob_mismatches,
            "charlie_mismatches": self.charlie_mismatches,
            "bob_accepted": self.bob_accepted,
            "charlie_accepted": self.charlie_accepted,
            "transferable": self.transferable,
        }


def _refuse_shared_stream(
    attack_rng: np.random.Generator, seeds: Sequence[int], *, where: str
) -> None:
    """Refuse an adversary generator built from a session seed.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.impersonation import _refuse_shared_stream
    >>> _refuse_shared_stream(
    ...     np.random.default_rng(4242), (900_000, 900_001), where="demo"
    ... ) is None
    True
    >>> try:
    ...     _refuse_shared_stream(
    ...         np.random.default_rng(900_001), (900_000, 900_001), where="demo"
    ...     )
    ... except ValueError as error:
    ...     print(str(error).splitlines()[0])
    demo: attack_rng is the stream session seed 900001 produces.
    """
    for seed in seeds:
        if derived_from_seed(attack_rng, int(seed)):
            raise ValueError(
                f"{where}: attack_rng is the stream session seed {int(seed)} "
                f"produces.\n"
                f"They are different objects and the same randomness: "
                f"numpy.random.default_rng({int(seed)}) gives identical bytes "
                f"every time it is called. QDSSession derives Alice's stream, "
                f"the recipients' stream and the binding stream from one "
                f"32-byte draw off that generator, so an adversary holding it "
                f"predicts every private symmetrisation coin and the "
                f"acceptance rates measured here mean nothing.\n"
                f"Seed Mallory from something unrelated to the session seeds "
                f"-- see FIRST_SESSION_SEED, which is 900_000 so that no "
                f"plausible adversary seed collides with one."
            )


def session_seeds(trials: int, *, first: int = FIRST_SESSION_SEED) -> tuple[int, ...]:
    """Return ``trials`` consecutive session seeds.

    Parameters
    ----------
    trials : int
        How many, at least ``1``.
    first : int, optional
        The first seed.

    Examples
    --------
    >>> from qsecure.attacks.impersonation import session_seeds
    >>> session_seeds(3, first=10)
    (10, 11, 12)
    """
    if isinstance(trials, bool) or not isinstance(trials, (int, np.integer)):
        raise TypeError(f"trials must be an int, got {type(trials).__name__}")
    if int(trials) < 1:
        raise ValueError(f"trials must be at least 1, got {int(trials)}")
    if isinstance(first, bool) or not isinstance(first, (int, np.integer)):
        raise TypeError(f"first must be an int, got {type(first).__name__}")
    if int(first) < 0:
        raise ValueError(f"first must be non-negative, got {int(first)}")
    return tuple(int(first) + index for index in range(int(trials)))


def run_impersonation(
    scope: ImpersonationScope | str,
    params: ProtocolParams = MEASUREMENT_PARAMS,
    *,
    session_seed: int,
    attack_rng: np.random.Generator,
    message_bit: int = 0,
) -> ImpersonationTrial:
    """Run one session under one scope and reduce it to an
    :class:`ImpersonationTrial`.

    Parameters
    ----------
    scope : ImpersonationScope or str
        Which seams Mallory holds.
    params : ProtocolParams, optional
        Defaults to :data:`MEASUREMENT_PARAMS`.
    session_seed : int
        The harness's seed, hers to know and never Mallory's.
    attack_rng : numpy.random.Generator
        Mallory's own generator.
    message_bit : int, optional
        ``0`` or ``1``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.impersonation import (
    ...     ImpersonationScope, run_impersonation, DEMO_MEASUREMENT_PARAMS)
    >>> trial = run_impersonation(
    ...     ImpersonationScope.DISTRIBUTION, DEMO_MEASUREMENT_PARAMS,
    ...     session_seed=7, attack_rng=np.random.default_rng(1))
    >>> trial.scope, trial.key_length, trial.accepted_by_both
    (<ImpersonationScope.DISTRIBUTION: 'distribution'>, 48, False)
    """
    if not isinstance(params, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(params).__name__}"
        )
    resolved = _as_scope(scope)
    if not isinstance(attack_rng, np.random.Generator):
        raise TypeError(
            f"attack_rng must be a numpy.random.Generator, got "
            f"{type(attack_rng).__name__}"
        )
    _refuse_shared_stream(
        attack_rng, (int(session_seed),), where="run_impersonation"
    )
    mallory = Impersonator(rng=attack_rng)
    session = QDSSession(
        params,
        rng=np.random.default_rng(int(session_seed)),
        **impersonation_seams(mallory, resolved),
    )
    transcript = session.run(int(message_bit))
    return _reduce(transcript, resolved, int(session_seed), params)


def _reduce(
    transcript: SessionTranscript,
    scope: ImpersonationScope,
    session_seed: int,
    params: ProtocolParams,
) -> ImpersonationTrial:
    """Turn one transcript into one trial row."""
    scored = params.sifted() if params.has_check_rounds else params
    bob, charlie = transcript.bob, transcript.charlie
    return ImpersonationTrial(
        scope=scope,
        session_seed=session_seed,
        key_length=scored.key_length,
        message_bit=transcript.message_bit,
        bob_matched=None if bob is None else bob.matched_count,
        charlie_matched=None if charlie is None else charlie.matched_count,
        bob_mismatches=None if bob is None else bob.mismatches,
        charlie_mismatches=None if charlie is None else charlie.mismatches,
        bob_accepted=None if bob is None else bob.accepted,
        charlie_accepted=None if charlie is None else charlie.accepted,
        transferable=transcript.transferable,
    )


@dataclass(frozen=True)
class ImpersonationMeasurement:
    """A whole experimental arm: many trials at one scope, with intervals.

    Parameters
    ----------
    key_length : int
        ``L``, the same for every trial in the arm.

    Examples
    --------
    >>> from qsecure.attacks.impersonation import (
    ...     ImpersonationMeasurement, ImpersonationScope, ImpersonationTrial)
    >>> rows = tuple(
    ...     ImpersonationTrial(
    ...         scope=ImpersonationScope.SIGNING, session_seed=seed,
    ...         key_length=48, message_bit=0, bob_matched=16,
    ...         charlie_matched=16, bob_mismatches=8, charlie_mismatches=8,
    ...         bob_accepted=False, charlie_accepted=False, transferable=False)
    ...     for seed in range(10)
    ... )
    >>> arm = ImpersonationMeasurement(
    ...     ImpersonationScope.SIGNING, 48, rows)
    >>> arm.successes, arm.trial_count, arm.success_rate
    (0, 10, 0.0)
    >>> arm.mismatch_rate("Bob")
    0.5
    """

    scope: ImpersonationScope
    key_length: int
    trials: tuple[ImpersonationTrial, ...]

    def __post_init__(self) -> None:
        """Coerce the fields and refuse an arm that mixes scopes or lengths."""
        object.__setattr__(self, "scope", _as_scope(self.scope))
        object.__setattr__(self, "trials", tuple(self.trials))
        if not self.trials:
            raise ValueError(
                "an ImpersonationMeasurement needs at least one trial: a rate "
                "with no denominator is not a measurement."
            )
        for trial in self.trials:
            if not isinstance(trial, ImpersonationTrial):
                raise TypeError(
                    f"every trial must be an ImpersonationTrial, got "
                    f"{type(trial).__name__}"
                )
            if trial.scope is not self.scope:
                raise ValueError(
                    f"trial at session seed {trial.session_seed} has scope "
                    f"{trial.scope!s}, but this arm is {self.scope!s}. Arms are "
                    f"compared against each other, so mixing scopes inside one "
                    f"would average two different experiments into a number "
                    f"describing neither."
                )
            if trial.key_length != self.key_length:
                raise ValueError(
                    f"trial at session seed {trial.session_seed} ran at "
                    f"L = {trial.key_length}, but this arm is L = "
                    f"{self.key_length}. Every rate here is conditional on L."
                )

    @property
    def trial_count(self) -> int:
        """int: how many sessions this arm is."""
        return len(self.trials)

    @property
    def successes(self) -> int:
        """int: sessions accepted by **both** verifiers."""
        return sum(1 for trial in self.trials if trial.accepted_by_both)

    @property
    def success_rate(self) -> float:
        """float: :attr:`successes` over :attr:`trial_count`."""
        return self.successes / self.trial_count

    @property
    def success_interval(self) -> Interval:
        """Interval: Wilson score interval for :attr:`success_rate`."""
        return wilson_interval(self.successes, self.trial_count)

    @property
    def aborts(self) -> int:
        """int: sessions in which some verifier reached no verdict at all."""
        return sum(1 for trial in self.trials if not trial.reached_verdicts)

    def accepted_by(self, party: Party | str) -> int:
        """Return how many sessions one verifier accepted.

        Parameters
        ----------
        party : Party or str
            ``Bob`` or ``Charlie``.
        """
        chosen = _as_verifier(party)
        return sum(
            1
            for trial in self.trials
            if (
                trial.bob_accepted
                if chosen is Party.BOB
                else trial.charlie_accepted
            )
        )

    def matched_total(self, party: Party | str) -> int:
        """Return one verifier's pooled ``sum |M_R|`` over the arm."""
        chosen = _as_verifier(party)
        return sum(
            (trial.bob_matched if chosen is Party.BOB else trial.charlie_matched)
            or 0
            for trial in self.trials
        )

    def mismatch_total(self, party: Party | str) -> int:
        """Return one verifier's pooled ``sum e_R`` over the arm."""
        chosen = _as_verifier(party)
        return sum(
            (
                trial.bob_mismatches
                if chosen is Party.BOB
                else trial.charlie_mismatches
            )
            or 0
            for trial in self.trials
        )

    def mismatch_rate(self, party: Party | str) -> float:
        """Return one verifier's **pooled** mismatch rate over the arm."""
        matched = self.matched_total(party)
        if matched == 0:
            raise ValueError(
                f"no positions were scored for {_as_verifier(party)!s} in this "
                f"arm, so there is no mismatch rate to report. Every trial "
                f"aborted before a verdict; look at .aborts."
            )
        return self.mismatch_total(party) / matched

    def mismatch_interval(self, party: Party | str) -> Interval:
        """Return the Wilson interval for :meth:`mismatch_rate`."""
        return wilson_interval(
            self.mismatch_total(party), self.matched_total(party)
        )

    def summary(self) -> str:
        """Return a one-line human summary naming the scope, ``L`` and the
        rate.
        """
        return (
            f"{self.scope!s} at L={self.key_length}: "
            f"{self.successes}/{self.trial_count} accepted by both, "
            f"r_Bob={self.mismatch_rate(Party.BOB):.4f}, "
            f"r_Charlie={self.mismatch_rate(Party.CHARLIE):.4f}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping of the arm's headline numbers."""
        return {
            "scope": str(self.scope),
            "key_length": self.key_length,
            "trials": self.trial_count,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "success_interval": self.success_interval.to_dict(),
            "aborts": self.aborts,
            "bob": {
                "accepted": self.accepted_by(Party.BOB),
                "matched_total": self.matched_total(Party.BOB),
                "mismatch_total": self.mismatch_total(Party.BOB),
                "mismatch_rate": self.mismatch_rate(Party.BOB),
                "mismatch_interval": self.mismatch_interval(Party.BOB).to_dict(),
            },
            "charlie": {
                "accepted": self.accepted_by(Party.CHARLIE),
                "matched_total": self.matched_total(Party.CHARLIE),
                "mismatch_total": self.mismatch_total(Party.CHARLIE),
                "mismatch_rate": self.mismatch_rate(Party.CHARLIE),
                "mismatch_interval": self.mismatch_interval(
                    Party.CHARLIE
                ).to_dict(),
            },
        }


def measure_impersonation(
    scope: ImpersonationScope | str,
    params: ProtocolParams = MEASUREMENT_PARAMS,
    *,
    seeds: Sequence[int] | None = None,
    trials: int = 200,
    attack_rng: np.random.Generator,
    message_bit: int = 0,
) -> ImpersonationMeasurement:
    """Run one arm of the experiment and return it with its intervals.

    Parameters
    ----------
    scope : ImpersonationScope or str
        Which seams Mallory holds.
    params : ProtocolParams, optional
        Defaults to :data:`MEASUREMENT_PARAMS`.
    seeds : sequence of int or None, optional
        The session seeds, one per trial.
    trials : int, optional
        How many seeds to generate when ``seeds`` is ``None``.
    attack_rng : numpy.random.Generator
        Mallory's generator, advanced across the loop so each trial gets a
        fresh key pair.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.impersonation import (
    ...     DEMO_MEASUREMENT_PARAMS, ImpersonationScope, measure_impersonation)
    >>> arm = measure_impersonation(
    ...     ImpersonationScope.FULL, DEMO_MEASUREMENT_PARAMS, trials=4,
    ...     attack_rng=np.random.default_rng(0))
    >>> arm.successes, arm.trial_count, arm.mismatch_rate("Bob")
    (4, 4, 0.0)
    """
    resolved = _as_scope(scope)
    chosen = session_seeds(trials) if seeds is None else tuple(int(s) for s in seeds)
    if not chosen:
        raise ValueError("seeds must hold at least one session seed")
    if not isinstance(attack_rng, np.random.Generator):
        raise TypeError(
            f"attack_rng must be a numpy.random.Generator, got "
            f"{type(attack_rng).__name__}"
        )
    _refuse_shared_stream(attack_rng, chosen, where="measure_impersonation")
    rows = tuple(
        run_impersonation(
            resolved,
            params,
            session_seed=seed,
            attack_rng=attack_rng,
            message_bit=message_bit,
        )
        for seed in chosen
    )
    scored = params.sifted() if params.has_check_rounds else params
    return ImpersonationMeasurement(resolved, scored.key_length, rows)


def matched_sets_identical(
    control: ImpersonationMeasurement, attacked: ImpersonationMeasurement
) -> bool:
    """Return whether two paired arms agree on every matched count, trial by
    trial.

    Parameters
    ----------
    control : ImpersonationMeasurement
        Normally the :attr:`ImpersonationScope.NONE` arm.
    attacked : ImpersonationMeasurement
        Another arm at the same seeds.

    Returns
    -------
    bool
        True when every paired trial reports the same ``|M_Bob|`` and
        ``|M_Charlie|``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.impersonation import (
    ...     DEMO_MEASUREMENT_PARAMS, ImpersonationScope, matched_sets_identical,
    ...     measure_impersonation, session_seeds)
    >>> seeds = session_seeds(3)
    >>> control = measure_impersonation(
    ...     ImpersonationScope.NONE, DEMO_MEASUREMENT_PARAMS, seeds=seeds,
    ...     attack_rng=np.random.default_rng(0))
    >>> swapped = measure_impersonation(
    ...     ImpersonationScope.DISTRIBUTION, DEMO_MEASUREMENT_PARAMS,
    ...     seeds=seeds, attack_rng=np.random.default_rng(0))
    >>> matched_sets_identical(control, swapped)
    True
    """
    for name, arm in (("control", control), ("attacked", attacked)):
        if not isinstance(arm, ImpersonationMeasurement):
            raise TypeError(
                f"{name} must be an ImpersonationMeasurement, got "
                f"{type(arm).__name__}"
            )
    if control.trial_count != attacked.trial_count:
        raise ValueError(
            f"the arms are not paired: {control.trial_count} control trials "
            f"against {attacked.trial_count} attacked. Run both over the same "
            f"seed sequence -- see session_seeds."
        )
    control_seeds = [trial.session_seed for trial in control.trials]
    attacked_seeds = [trial.session_seed for trial in attacked.trials]
    if control_seeds != attacked_seeds:
        raise ValueError(
            "the arms are not paired: they used different session seeds, or "
            "the same seeds in a different order. A matched-count comparison "
            "is only meaningful trial by trial."
        )
    return all(
        left.bob_matched == right.bob_matched
        and left.charlie_matched == right.charlie_matched
        for left, right in zip(control.trials, attacked.trials, strict=True)
    )


def wilson_interval(
    successes: int, trials: int, *, confidence: float = CONFIDENCE
) -> Interval:
    """Return the Wilson score interval for a binomial proportion.

    Parameters
    ----------
    successes : int
        ``0 <= successes <= trials``.
    trials : int
        At least ``1``.
    confidence : float, optional
        In ``(0, 1)``.

    Returns
    -------
    Interval
        Tagged ``"wilson"``, clipped to ``[0, 1]``.

    Examples
    --------
    >>> from qsecure.attacks.impersonation import wilson_interval
    >>> interval = wilson_interval(0, 200)
    >>> interval.low, round(interval.high, 4), interval.method
    (0.0, 0.0188, 'wilson')
    >>> perfect = wilson_interval(200, 200)
    >>> round(perfect.low, 4), perfect.high
    (0.9812, 1.0)
    >>> half = wilson_interval(6400, 12800)
    >>> round(half.low, 4), round(half.high, 4)
    (0.4913, 0.5087)
    """
    if isinstance(successes, bool) or not isinstance(
        successes, (int, np.integer)
    ):
        raise TypeError(
            f"successes must be an int, got {type(successes).__name__}"
        )
    if isinstance(trials, bool) or not isinstance(trials, (int, np.integer)):
        raise TypeError(f"trials must be an int, got {type(trials).__name__}")
    hits, total = int(successes), int(trials)
    if total < 1:
        raise ValueError(f"trials must be at least 1, got {total}")
    if not 0 <= hits <= total:
        raise ValueError(
            f"successes must satisfy 0 <= successes <= trials, got "
            f"{hits} of {total}"
        )
    if isinstance(confidence, bool) or not isinstance(
        confidence, (int, float, np.floating)
    ):
        raise TypeError(
            f"confidence must be a real number, got {type(confidence).__name__}"
        )
    level = float(confidence)
    if not 0.0 < level < 1.0:
        raise ValueError(
            f"confidence must lie strictly inside (0, 1), got {level}"
        )
    low, high = wilson_bounds(hits, total, z=_two_sided_z(level))
    return Interval(low, high, level, "wilson")


def _two_sided_z(confidence: float) -> float:
    """Return the two-sided normal quantile for ``confidence``."""
    return two_sided_z(confidence)


@dataclass(frozen=True)
class MeasuredRun:
    """One published row of :data:`MEASURED`, with its own arithmetic checked.

    Parameters
    ----------
    key_length : int
        ``L``.
    trials : int
        Sessions run.
    successes : int
        Sessions accepted by **both** verifiers.
    bob_mismatches, bob_matched : int
        Pooled ``sum e_Bob`` and ``sum |M_Bob|`` over the arm.
    charlie_mismatches, charlie_matched : int
        The same for Charlie.

    Examples
    --------
    >>> from qsecure.attacks.impersonation import MEASURED, ImpersonationScope
    >>> row = MEASURED[ImpersonationScope.FULL]
    >>> row.key_length, row.trials, row.successes
    (192, 200, 200)
    >>> row.success_rate, row.bob_rate, row.charlie_rate
    (1.0, 0.0, 0.0)
    """

    scope: ImpersonationScope
    key_length: int
    trials: int
    successes: int
    bob_mismatches: int
    bob_matched: int
    charlie_mismatches: int
    charlie_matched: int

    def __post_init__(self) -> None:
        """Coerce the scope and refuse a row whose counts cannot be real."""
        object.__setattr__(self, "scope", _as_scope(self.scope))
        if self.key_length < 1:
            raise ValueError(f"key_length must be positive, got {self.key_length}")
        if self.trials < 1:
            raise ValueError(f"trials must be at least 1, got {self.trials}")
        if not 0 <= self.successes <= self.trials:
            raise ValueError(
                f"successes must satisfy 0 <= successes <= trials, got "
                f"{self.successes} of {self.trials}"
            )
        for errors, matched, who in (
            (self.bob_mismatches, self.bob_matched, "Bob"),
            (self.charlie_mismatches, self.charlie_matched, "Charlie"),
        ):
            if matched < 1:
                raise ValueError(
                    f"{who} must have scored at least one position for a rate "
                    f"to exist, got {matched}"
                )
            if not 0 <= errors <= matched:
                raise ValueError(
                    f"{who}'s mismatches must satisfy 0 <= e <= |M|, got "
                    f"{errors} of {matched}"
                )
            if matched > self.trials * self.key_length:
                raise ValueError(
                    f"{who} cannot have scored {matched} positions in "
                    f"{self.trials} trials of length {self.key_length}"
                )

    @property
    def success_rate(self) -> float:
        """float: acceptance by both verifiers, over the arm."""
        return self.successes / self.trials

    @property
    def success_interval(self) -> Interval:
        """Interval: Wilson interval for :attr:`success_rate`."""
        return wilson_interval(self.successes, self.trials)

    @property
    def bob_rate(self) -> float:
        """float: pooled ``r_Bob = sum e / sum |M|``."""
        return self.bob_mismatches / self.bob_matched

    @property
    def charlie_rate(self) -> float:
        """float: pooled ``r_Charlie``."""
        return self.charlie_mismatches / self.charlie_matched

    def rate_for(self, party: Party | str) -> float:
        """Return the pooled mismatch rate for one verifier."""
        return (
            self.bob_rate
            if _as_verifier(party) is Party.BOB
            else self.charlie_rate
        )

    def interval_for(self, party: Party | str) -> Interval:
        """Return the Wilson interval for one verifier's pooled rate."""
        if _as_verifier(party) is Party.BOB:
            return wilson_interval(self.bob_mismatches, self.bob_matched)
        return wilson_interval(self.charlie_mismatches, self.charlie_matched)

    def summary(self) -> str:
        """Return the row as one quotable line, ``L`` included."""
        return (
            f"{self.scope!s} L={self.key_length} n={self.trials}: "
            f"accepted {self.successes}/{self.trials}, "
            f"r_Bob={self.bob_rate:.4f}, r_Charlie={self.charlie_rate:.4f}"
        )


def _row(
    scope: ImpersonationScope,
    trials: int,
    successes: int,
    bob: tuple[int, int],
    charlie: tuple[int, int],
) -> MeasuredRun:
    """Build one :data:`MEASURED` row at :data:`MEASUREMENT_PARAMS`."""
    return MeasuredRun(
        scope=scope,
        key_length=MEASUREMENT_PARAMS.key_length,
        trials=trials,
        successes=successes,
        bob_mismatches=bob[0],
        bob_matched=bob[1],
        charlie_mismatches=charlie[0],
        charlie_matched=charlie[1],
    )


MEASURED: Final[Mapping[ImpersonationScope, MeasuredRun]] = {
    ImpersonationScope.NONE: _row(
        ImpersonationScope.NONE, 200, 200, (0, 12774), (0, 12696)
    ),
    ImpersonationScope.FULL: _row(
        ImpersonationScope.FULL, 200, 200, (0, 12706), (0, 12798)
    ),
    ImpersonationScope.SIGNING: _row(
        ImpersonationScope.SIGNING, 200, 0, (6420, 12870), (6484, 12870)
    ),
    ImpersonationScope.DISTRIBUTION: _row(
        ImpersonationScope.DISTRIBUTION, 200, 0, (6426, 12774), (6344, 12696)
    ),
}

MATCHED_IDENTICAL_TRIALS: Final[Mapping[ImpersonationScope, int]] = {
    ImpersonationScope.NONE: 200,
    ImpersonationScope.FULL: 0,
    ImpersonationScope.SIGNING: 0,
    ImpersonationScope.DISTRIBUTION: 200,
}


def shipped_summary() -> tuple[str, ...]:
    """Return :data:`MEASURED` as four quotable lines, one per scope.

    Returns
    -------
    tuple of str
        One line per scope, in the order control, full, signing, distribution.

    Examples
    --------
    >>> from qsecure.attacks.impersonation import shipped_summary
    >>> for line in shipped_summary():
    ...     print(line)
    none L=192 n=200: accepted 200/200, r_Bob=0.0000, r_Charlie=0.0000
    full L=192 n=200: accepted 200/200, r_Bob=0.0000, r_Charlie=0.0000
    signing L=192 n=200: accepted 0/200, r_Bob=0.4988, r_Charlie=0.5038
    distribution L=192 n=200: accepted 0/200, r_Bob=0.5031, r_Charlie=0.4997

    >>> from qsecure.attacks.impersonation import MEASURED, ImpersonationScope
    >>> row = MEASURED[ImpersonationScope.SIGNING]
    >>> interval = row.interval_for("Charlie")
    >>> f"[{interval.low:.4f}, {interval.high:.4f}]", interval.covers(0.5)
    ('[0.4952, 0.5124]', True)

    >>> whole = MEASURED[ImpersonationScope.FULL].success_interval
    >>> half = MEASURED[ImpersonationScope.SIGNING].success_interval
    >>> f"{whole.low:.4f}", f"{half.high:.4f}"
    ('0.9812', '0.0188')

    >>> from qsecure.attacks.impersonation import MATCHED_IDENTICAL_TRIALS
    >>> (MATCHED_IDENTICAL_TRIALS[ImpersonationScope.DISTRIBUTION],
    ...  MATCHED_IDENTICAL_TRIALS[ImpersonationScope.SIGNING])
    (200, 0)
    """
    order = (
        ImpersonationScope.NONE,
        ImpersonationScope.FULL,
        ImpersonationScope.SIGNING,
        ImpersonationScope.DISTRIBUTION,
    )
    return tuple(MEASURED[scope].summary() for scope in order)


def detector_signals() -> dict[str, dict[str, Any]]:
    """Return what a Detector could read, and whether it moves.

    Returns
    -------
    dict
        Statistic name to a mapping with keys ``"honest"``, ``"partial"``,
        ``"full"``, ``"reachable"`` and ``"note"``.

    Examples
    --------
    >>> from qsecure.attacks.impersonation import detector_signals
    >>> signals = detector_signals()
    >>> signals["mismatch_rate"]["partial"], signals["mismatch_rate"]["full"]
    ('~0.5', '0.0')
    >>> all(entry["reachable"] for entry in signals.values())
    True
    >>> sorted(
    ...     name for name, entry in signals.items()
    ...     if entry["partial"] == entry["honest"]
    ... )
    ['channel_qber', 'matched_count', 'pooled_matched_count', 'signer_identity']
    """
    return {
        "mismatch_rate": {
            "honest": "~0.0",
            "partial": "~0.5",
            "full": "0.0",
            "reachable": True,
            "note": (
                "VerificationResult.rate, per verifier, on the transcript. The "
                "only statistic that moves under partial impersonation, and it "
                "moves the whole way."
            ),
        },
        "matched_count": {
            "honest": "~L/3",
            "partial": "~L/3",
            "full": "~L/3",
            "reachable": True,
            "note": (
                "VerificationResult.matched_count. Binomial(L, 1/3) in every "
                "scope: an impersonator substitutes values, never the sampling "
                "law. Under the DISTRIBUTION scope it is identical position "
                "for position to the control at the same session seed -- see "
                "matched_sets_identical."
            ),
        },
        "pooled_matched_count": {
            "honest": "~2L/3",
            "partial": "~2L/3",
            "full": "~2L/3",
            "reachable": True,
            "note": (
                "SessionTranscript.pooled_matched_count, from Phase C'. The "
                "floors it feeds are an evidence-liveness control, not an "
                "impersonation detector, and this row is why."
            ),
        },
        "channel_qber": {
            "honest": "~0.0",
            "partial": "~0.0",
            "full": "~0.0",
            "reachable": True,
            "note": (
                "Check-round QBER, when the parameter set reserves check "
                "rounds. An impersonating distributor teleports faithfully over "
                "an ideal resource, so the channel looks perfect while the key "
                "is somebody else's: the channel monitor and the mismatch rate "
                "are measuring different things and this attack separates them."
            ),
        },
        "signer_identity": {
            "honest": "absent",
            "partial": "absent",
            "full": "absent",
            "reachable": True,
            "note": (
                "Nothing in Signature, RecipientRecord, VerificationResult or "
                "SessionTranscript names or binds a signer; the round "
                "identifier binds a declaration to a distribution, not to a "
                "person. Reachable in the sense that its absence is checkable, "
                "which is the honest reading of assumption (AUTH). This is why "
                "the FULL scope is undetectable rather than merely hard."
            ),
        },
    }


def signing_probe() -> Any:
    """Return a :class:`~qsecure.attacks.isolation.DecisionProbe` for
    :meth:`Impersonator.sign`.

    Returns
    -------
    DecisionProbe
        ``(mallory, session_seed) -> Signature``.

    Examples
    --------
    >>> from qsecure.attacks.impersonation import Impersonator, signing_probe
    >>> from qsecure.attacks.isolation import assert_attack_isolated
    >>> assert_attack_isolated(Impersonator, signing_probe()).isolated
    True
    """
    from qsecure.attacks.isolation import signer_probe

    inner = signer_probe()

    def probe(attack: Any, session_seed: int) -> Any:
        """Declare through the candidate's signing seam and return what it
        said.
        """
        return inner(attack.sign, session_seed)

    return probe


def distributor_probe(
    params: ProtocolParams = ProtocolParams(key_length=12),
) -> Any:
    """Return a :class:`~qsecure.attacks.isolation.DecisionProbe` for
    :meth:`Impersonator.distribute`.

    Parameters
    ----------
    params : ProtocolParams, optional
        A small parameter set; the probe runs a real distribution, so keep it
        cheap.

    Returns
    -------
    DecisionProbe
        ``(mallory, session_seed) -> tuple of str``, her declared key labels.

    Examples
    --------
    >>> from qsecure.attacks.impersonation import Impersonator, distributor_probe
    >>> from qsecure.attacks.isolation import assert_attack_isolated
    >>> assert_attack_isolated(Impersonator, distributor_probe()).isolated
    True
    """
    if not isinstance(params, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(params).__name__}"
        )

    def probe(attack: Any, session_seed: int) -> Any:
        """Distribute once through the candidate; report the key it
        substituted.
        """
        alice = generate_private_key(params, 0, rng=np.random.default_rng(1_234))
        attack.distribute(
            alice, params, rng=np.random.default_rng(int(session_seed))
        )
        return attack.key_for(0, params).labels

    return probe
