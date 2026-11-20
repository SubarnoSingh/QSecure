"""The replay adversary, measured against the round-binding and ledger defence.

Each attack is measured with the defence enabled and disabled; measurement
functions assemble the pairs directly and call
:func:`~qsecure.protocol.verify.verify_or_abort`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from qsecure.attacks.isolation import require_distinct_streams
from qsecure.attacks.statistics import wilson_bounds
from qsecure.protocol.analysis import forgery_probability
from qsecure.protocol.keys import KeyElement, PrivateKey
from qsecure.protocol.params import Party, ProtocolParams
from qsecure.protocol.records import RecipientRecord, RecipientView
from qsecure.protocol.session import QDSSession
from qsecure.protocol.signature import Signature, session_identifier
from qsecure.protocol.tally import (
    exchange_matched_counts,
    matched_count_message,
)
from qsecure.protocol.verify import (
    AbortReason,
    ConsumedRecords,
    VerificationAbort,
    VerificationResult,
    verify_or_abort,
)

__all__ = [
    "CAPTURE_PARAMS",
    "CAPTURE_SEED",
    "COUNTS_AS_RECEIVED",
    "COUNTS_AS_SIGNED",
    "COUNTS_NONE",
    "AttackOutcome",
    "ReplayCapture",
    "ReplayingForwarder",
    "measure_cross_session_pairing",
    "measure_identifier_preimage",
    "measure_ledger_denial_of_service",
    "measure_ledger_poisoning",
    "measure_shared_identifier",
    "measure_shared_identifier_self_denial",
    "measure_straight_replay",
    "replay_capture",
    "replay_probe",
    "wilson_interval",
]


CAPTURE_SEED: Final[int] = 41200626
"""Seed of the one finished run :func:`replay_capture` freezes."""

CAPTURE_PARAMS: Final[ProtocolParams] = ProtocolParams(key_length=24)
"""Parameters for the frozen capture -- small, because it is built on demand.
"""

_Z_95: Final[float] = 1.959963984540054
"""Standard normal two-sided 95% quantile, for :func:`wilson_interval`."""

_OPENING_HEX_BYTES: Final[int] = 16
"""Width of an invented opening, matching the one the protocol draws."""

COUNTS_NONE: Final[str] = "none"
"""Phase C' did not run: the pre-pooled variant, per-verifier floors only."""

COUNTS_AS_RECEIVED: Final[str] = "as-received"
"""Each recipient counted against the declaration he actually received."""

COUNTS_AS_SIGNED: Final[str] = "as-signed"
"""Both counts named the declaration Alice signed, whatever Charlie then got.
"""

_COUNTS_MODES: Final[tuple[str, ...]] = (
    COUNTS_NONE,
    COUNTS_AS_RECEIVED,
    COUNTS_AS_SIGNED,
)
"""Every Phase C' ordering :func:`_as_counts_mode` will accept."""


def _as_nonnegative_int(value: Any, name: str) -> int:
    """Coerce a count, refusing bools and negatives with a named message."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{name} must be an int, got {type(value).__name__}; counts of "
            f"trials and successes are integers, not rates."
        )
    count = int(value)
    if count < 0:
        raise ValueError(f"{name} must be non-negative, got {count}")
    return count


def _as_generator(value: Any, name: str) -> np.random.Generator:
    """Require an explicit generator; never fall back to a default."""
    if not isinstance(value, np.random.Generator):
        raise TypeError(
            f"{name} must be a numpy.random.Generator, got "
            f"{type(value).__name__}. Every generator in this module is passed "
            f"explicitly, and the adversary's own is separate from the "
            f"world's on purpose."
        )
    return value


def _distinct_generators(
    world: Any, attack: Any
) -> tuple[np.random.Generator, np.random.Generator]:
    """Validate both generators and refuse a shared stream."""
    checked_world = _as_generator(world, "rng")
    checked_attack = _as_generator(attack, "attack_rng")
    require_distinct_streams(
        checked_world,
        checked_attack,
        left_name="rng",
        right_name="attack_rng",
        detail=(
            "rng is the world's and builds the sessions; attack_rng is the "
            "adversary's own and nothing he does may be a function of the "
            "first."
        ),
    )
    return checked_world, checked_attack


def wilson_interval(
    successes: int, trials: int, *, z: float = _Z_95
) -> tuple[float, float]:
    """Return a Wilson score interval for a measured success rate.

    Parameters
    ----------
    successes : int
        Number of successful trials, in ``0 .. trials``.
    trials : int
        Number of trials.
    z : float, optional
        Normal quantile; the default is two-sided 95%.

    Returns
    -------
    tuple of (float, float)
        Lower and upper bounds, clipped into ``[0, 1]``.

    Examples
    --------
    >>> from qsecure.attacks.replay import wilson_interval
    >>> low, high = wilson_interval(0, 200)
    >>> low, round(high, 4)
    (0.0, 0.0188)
    >>> low, high = wilson_interval(200, 200)
    >>> round(low, 4), high
    (0.9812, 1.0)
    >>> low, high = wilson_interval(25, 2000)
    >>> round(low, 5), round(high, 5)
    (0.00848, 0.01839)
    """
    trial_count = _as_nonnegative_int(trials, "trials")
    success_count = _as_nonnegative_int(successes, "successes")
    if trial_count < 1:
        raise ValueError(
            f"trials must be at least 1, got {trial_count}; a rate measured "
            f"over no trials is not a rate."
        )
    if success_count > trial_count:
        raise ValueError(
            f"successes ({success_count}) exceeds trials ({trial_count})"
        )
    if isinstance(z, bool) or not isinstance(z, (int, float, np.floating)):
        raise TypeError(f"z must be a real number, got {type(z).__name__}")
    if float(z) < 0.0:
        raise ValueError(f"z must be non-negative, got {float(z)}")

    return wilson_bounds(success_count, trial_count, z=float(z))


@dataclass(frozen=True)
class AttackOutcome:
    """One measured attack rate, with the conditions that produced it.

    Parameters
    ----------
    label : str
        What was measured, in a few words.
    defended : bool
        Whether the replay defence was enabled for these trials.
    trials : int
        How many independent attempts were made.
    successes : int
        How many achieved the adversary's goal.
    refusals : int
        How many trials ended in a
        :class:`~qsecure.protocol.verify.VerificationAbort` rather than a
        verdict.
    key_length : int
        ``L`` the trials ran at.
    note : str, optional
        One line of context: the analytic prediction, the mechanism, or the
        caveat.

    Examples
    --------
    >>> from qsecure.attacks.replay import AttackOutcome
    >>> outcome = AttackOutcome(
    ...     label="straight replay", defended=True, trials=200,
    ...     successes=0, refusals=200, key_length=24,
    ... )
    >>> outcome.rate
    0.0
    >>> print(outcome.summary())
    straight replay [defended, L=24]: 0/200 = 0.0000, 95% CI [0.0000, 0.0188]
      200 of 200 trials reached no verdict at all.
    """

    label: str
    defended: bool
    trials: int
    successes: int
    refusals: int
    key_length: int
    note: str = ""

    def __post_init__(self) -> None:
        """Validate every field; an unvalidated outcome is a wrong table."""
        if not isinstance(self.label, str) or not self.label:
            raise ValueError(
                f"label must be a non-empty string, got {self.label!r}"
            )
        if not isinstance(self.defended, bool):
            raise TypeError(
                f"defended must be a bool, got {type(self.defended).__name__}. "
                f"It says which arm of the before/after this number is, and "
                f"there is no third arm."
            )
        if not isinstance(self.note, str):
            raise TypeError(
                f"note must be a string, got {type(self.note).__name__}"
            )
        object.__setattr__(
            self, "trials", _as_nonnegative_int(self.trials, "trials")
        )
        object.__setattr__(
            self, "successes", _as_nonnegative_int(self.successes, "successes")
        )
        object.__setattr__(
            self, "refusals", _as_nonnegative_int(self.refusals, "refusals")
        )
        object.__setattr__(
            self,
            "key_length",
            _as_nonnegative_int(self.key_length, "key_length"),
        )
        if self.trials < 1:
            raise ValueError(
                f"trials must be at least 1, got {self.trials}; an outcome "
                f"over no trials reports nothing."
            )
        if self.successes > self.trials:
            raise ValueError(
                f"successes ({self.successes}) exceeds trials ({self.trials})"
            )
        if self.refusals > self.trials:
            raise ValueError(
                f"refusals ({self.refusals}) exceeds trials ({self.trials})"
            )

    @property
    def rate(self) -> float:
        """float: ``successes / trials``."""
        return self.successes / self.trials

    @property
    def interval(self) -> tuple[float, float]:
        """tuple of (float, float): Wilson 95% interval for :attr:`rate`."""
        return wilson_interval(self.successes, self.trials)

    def summary(self) -> str:
        """Return a human-readable statement of the result.

        Returns
        -------
        str
            The rate with its interval, the refusal count when there were any,
            and the note when there is one.

        Examples
        --------
        >>> from qsecure.attacks.replay import AttackOutcome
        >>> print(AttackOutcome(
        ...     label="cross-session pairing", defended=False, trials=1000,
        ...     successes=13, refusals=0, key_length=24,
        ...     note="predicted 0.012520",
        ... ).summary())
        cross-session pairing [undefended, L=24]: 13/1000 = 0.0130, 95% CI [0.0076, 0.0221]
          predicted 0.012520
        """
        low, high = self.interval
        arm = "defended" if self.defended else "undefended"
        lines = [
            f"{self.label} [{arm}, L={self.key_length}]: "
            f"{self.successes}/{self.trials} = {self.rate:.4f}, "
            f"95% CI [{low:.4f}, {high:.4f}]"
        ]
        if self.refusals:
            lines.append(
                f"  {self.refusals} of {self.trials} trials reached no verdict "
                f"at all."
            )
        if self.note:
            lines.append(f"  {self.note}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ReplayCapture:
    """A finished round, as an adversary who watched it would hold it.

    Parameters
    ----------
    params : ProtocolParams
        What the captured round ran under.
    signature : Signature
        The declaration, with its opening revealed exactly as Phase B reveals
        it.
    records : dict
        The two verifiers' post-symmetrisation logs, keyed by
        :class:`~qsecure.protocol.params.Party`.

    Examples
    --------
    >>> from qsecure.attacks.replay import replay_capture
    >>> from qsecure.protocol.params import Party
    >>> capture = replay_capture()
    >>> capture.signature.session_id == capture.records[Party.BOB].session_id
    True
    >>> capture.signature.session_opening is not None
    True
    """

    params: ProtocolParams
    signature: Signature
    records: dict[Party, RecipientRecord]


_CAPTURE_CACHE: dict[tuple[ProtocolParams, int, int], ReplayCapture] = {}


def replay_capture(
    params: ProtocolParams = CAPTURE_PARAMS,
    *,
    message_bit: int = 0,
    seed: int = CAPTURE_SEED,
) -> ReplayCapture:
    """Run one honest round to completion and freeze it as an adversary's loot.

    Parameters
    ----------
    params : ProtocolParams, optional
        Defaults to :data:`CAPTURE_PARAMS`.
    message_bit : int, optional
        ``0`` or ``1``.
    seed : int, optional
        Defaults to :data:`CAPTURE_SEED`, which is unrelated to every seed a
        measurement or the isolation check varies.

    Examples
    --------
    >>> from qsecure.attacks.replay import replay_capture
    >>> capture = replay_capture()
    >>> len(capture.signature) == capture.params.key_length
    True
    >>> replay_capture() is capture
    True
    """
    if not isinstance(params, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(params).__name__}"
        )
    if isinstance(message_bit, bool) or message_bit not in (0, 1):
        raise ValueError(
            f"message_bit must be 0 or 1, got {message_bit!r}; a capture "
            f"covers the one round the adversary watched."
        )
    seed_value = _as_nonnegative_int(seed, "seed")

    cache_key = (params, int(message_bit), seed_value)
    cached = _CAPTURE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    session = QDSSession(params, rng=np.random.default_rng(seed_value))
    session.distribute()
    signature = session.sign(int(message_bit))
    capture = ReplayCapture(
        params=params,
        signature=signature,
        records=dict(session.records[int(message_bit)]),
    )
    _CAPTURE_CACHE[cache_key] = capture
    return capture


class ReplayingForwarder:
    """The Bob-to-Charlie hop, re-presenting a declaration captured earlier.

    Parameters
    ----------
    rng : numpy.random.Generator
        The adversary's **own**.
    captures : sequence of Signature or None, optional
        The declarations this adversary is holding.
    replay_probability : float, optional
        In ``[0, 1]``.
    relabel : bool, optional
        When ``True`` (the default) a substituted declaration is rebuilt with
        an opening drawn from this adversary's generator -- an attempt to make
        a stale round name a fresh one, which succeeds only on a preimage
        (:func:`measure_identifier_preimage`).

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.replay import ReplayingForwarder, replay_capture
    >>> capture = replay_capture()
    >>> attack = ReplayingForwarder(
    ...     rng=np.random.default_rng(11), replay_probability=1.0
    ... )
    >>> forwarded = attack(capture.signature, capture.params)
    >>> attack.replays
    1

    >>> forwarded.session_id == capture.signature.session_id
    False
    >>> forwarded.declared_key is capture.signature.declared_key
    True

    >>> quiet = ReplayingForwarder(
    ...     rng=np.random.default_rng(11), replay_probability=0.0
    ... )
    >>> quiet(capture.signature, capture.params) is capture.signature
    True
    """

    __slots__ = (
        "_captures",
        "_rng",
        "relabel",
        "replay_probability",
        "replays",
    )

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        captures: Sequence[Signature] | None = None,
        replay_probability: float = 0.5,
        relabel: bool = True,
    ) -> None:
        self._rng: Final[np.random.Generator] = _as_generator(rng, "rng")
        if captures is None:
            resolved: tuple[Signature, ...] = tuple(
                replay_capture(message_bit=bit).signature for bit in (0, 1)
            )
        else:
            if isinstance(captures, (str, bytes)) or not isinstance(
                captures, Sequence
            ):
                raise TypeError(
                    f"captures must be a sequence of Signature, got "
                    f"{type(captures).__name__}"
                )
            for position, item in enumerate(captures):
                if not isinstance(item, Signature):
                    raise TypeError(
                        f"captures[{position}] must be a Signature, got "
                        f"{type(item).__name__}. An adversary replays a "
                        f"declaration, which is the only thing Phase B made "
                        f"public."
                    )
            resolved = tuple(captures)
            if not resolved:
                raise ValueError(
                    "captures must hold at least one Signature; an adversary "
                    "with nothing captured has no replay to mount."
                )
        self._captures: Final[tuple[Signature, ...]] = resolved
        if isinstance(replay_probability, bool) or not isinstance(
            replay_probability, (int, float, np.floating)
        ):
            raise TypeError(
                f"replay_probability must be a real number, got "
                f"{type(replay_probability).__name__}"
            )
        probability = float(replay_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"replay_probability must lie in [0, 1], got {probability}"
            )
        if not isinstance(relabel, bool):
            raise TypeError(
                f"relabel must be a bool, got {type(relabel).__name__}"
            )
        self.replay_probability: float = probability
        self.relabel: bool = relabel
        self.replays: int = 0

    def __repr__(self) -> str:
        """Return a debugging representation naming the loot and the rate."""
        return (
            f"ReplayingForwarder({len(self._captures)} capture(s), "
            f"p={self.replay_probability}, relabel={self.relabel}, "
            f"{self.replays} replay(s) so far)"
        )

    def __call__(
        self,
        signature: Signature,
        params: ProtocolParams,
        *,
        view: RecipientView | None = None,
    ) -> Signature:
        """Return the declaration Charlie will be given."""
        del params, view
        if not isinstance(signature, Signature):
            raise TypeError(
                f"signature must be a Signature, got "
                f"{type(signature).__name__}"
            )
        coin = float(self._rng.random())
        choice = int(self._rng.integers(len(self._captures)))
        opening = self._rng.bytes(_OPENING_HEX_BYTES).hex()
        if coin >= self.replay_probability:
            return signature
        eligible = [
            stale
            for stale in self._captures
            if stale.message_bit == signature.message_bit
            and len(stale) == len(signature)
        ]
        if not eligible:
            return signature
        stale = eligible[choice % len(eligible)]
        self.replays += 1
        if not self.relabel:
            return stale
        return Signature(
            message_bit=stale.message_bit,
            declared_key=stale.declared_key,
            session_opening=opening,
            context=stale.context,
        )


def replay_probe(capture: ReplayCapture | None = None) -> Any:
    """Return a :class:`~qsecure.attacks.isolation.DecisionProbe` for this seam.

    Parameters
    ----------
    capture : ReplayCapture or None, optional
        The frozen observations.

    Returns
    -------
    DecisionProbe
        ``(attack, session_seed) -> Signature``.

    Examples
    --------
    >>> from qsecure.attacks.isolation import assert_attack_isolated
    >>> from qsecure.attacks.replay import ReplayingForwarder, replay_probe
    >>> assert_attack_isolated(ReplayingForwarder, replay_probe()).isolated
    True
    """
    resolved = replay_capture() if capture is None else capture
    if not isinstance(resolved, ReplayCapture):
        raise TypeError(
            f"capture must be a ReplayCapture or None, got "
            f"{type(resolved).__name__}. Build one with replay_capture()."
        )

    def probe(attack: Any, session_seed: int) -> Any:
        """Forward once against the frozen capture; return what was passed on.
        """
        del session_seed
        return attack(resolved.signature, resolved.params)

    return probe


def _as_params_arg(value: Any) -> ProtocolParams:
    """Require an explicit parameter set; there is no default worth guessing.
    """
    if not isinstance(value, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(value).__name__}. "
            f"Every measurement here is a rate at one key length, so the key "
            f"length is named rather than inherited."
        )
    return value


def _as_flag(value: Any, name: str) -> bool:
    """Require a real bool; a truthy value would silently pick an arm."""
    if not isinstance(value, bool):
        raise TypeError(
            f"{name} must be a bool, got {type(value).__name__}. It selects "
            f"which arm of the before/after comparison is being measured, and "
            f"a truthy value would pick one silently."
        )
    return value


def _as_trials(value: Any) -> int:
    """Coerce a trial count and refuse zero: a rate needs a denominator."""
    count = _as_nonnegative_int(value, "trials")
    if count < 1:
        raise ValueError(
            f"trials must be at least 1, got {count}; an attack measured over "
            f"no trials reports nothing, and reporting nothing as a zero rate "
            f"is how a defence gets credit it never earned."
        )
    return count


def _as_verifier(value: Any) -> Party:
    """Resolve a verifier, refusing Alice, who holds no record to attack."""
    party = value if isinstance(value, Party) else Party(value)
    if party is Party.ALICE:
        raise ValueError(
            "party must be Party.BOB or Party.CHARLIE. Alice signs; she holds "
            "no measurement record and reaches no verdict, so there is nothing "
            "at her end for a replay to be presented to."
        )
    return party


def _random_key(
    params: ProtocolParams, message_bit: int, rng: np.random.Generator
) -> PrivateKey:
    """Draw a key uniformly, from the **adversary's** generator."""
    bases = tuple(params.bases)
    basis_choices = rng.integers(len(bases), size=params.key_length)
    signs = rng.integers(2, size=params.key_length)
    return PrivateKey(
        message_bit,
        tuple(
            KeyElement(bases[int(where)], 1 if int(sign) == 0 else -1)
            for where, sign in zip(basis_choices, signs, strict=True)
        ),
    )


def _accepted(outcome: VerificationResult | VerificationAbort) -> bool:
    """Return whether an outcome is a verdict *and* an accepting one."""
    return isinstance(outcome, VerificationResult) and outcome.accepted


def _refused(outcome: VerificationResult | VerificationAbort) -> bool:
    """Return whether an outcome is a refusal to score rather than a verdict.
    """
    return isinstance(outcome, VerificationAbort)


def _describe(outcome: VerificationResult | VerificationAbort) -> str:
    """Return a short phrase naming what a verifier did, for a message.

    Examples
    --------
    >>> from qsecure.attacks.replay import _describe
    >>> from qsecure.protocol.params import Party
    >>> from qsecure.protocol.verify import AbortReason, VerificationAbort
    >>> _describe(
    ...     VerificationAbort(
    ...         party=Party.CHARLIE,
    ...         reason=AbortReason.BELOW_FLOOR,
    ...         matched_count=1,
    ...         minimum_matched=3,
    ...         expected_matched=16.0,
    ...         key_length=24,
    ...         message_bit=0,
    ...     )
    ... )
    "no verdict, reason 'matched-count-below-floor'"
    """
    if isinstance(outcome, VerificationAbort):
        return f"no verdict, reason {str(outcome.reason.value)!r}"
    return f"a verdict, accepted={outcome.accepted}"


def _round_already_spent(
    outcome: VerificationResult | VerificationAbort,
) -> bool:
    """Return whether an outcome is the ledger refusing a spent round.

    Examples
    --------
    >>> from qsecure.attacks.replay import _round_already_spent
    >>> from qsecure.protocol.params import Party
    >>> from qsecure.protocol.verify import AbortReason, VerificationAbort
    >>> spent = VerificationAbort(
    ...     party=Party.CHARLIE,
    ...     reason=AbortReason.RECORD_ALREADY_VERIFIED,
    ...     matched_count=0,
    ...     minimum_matched=1,
    ...     expected_matched=16.0,
    ...     key_length=24,
    ...     message_bit=0,
    ... )
    >>> _round_already_spent(spent)
    True

    >>> starved = VerificationAbort(
    ...     party=Party.CHARLIE,
    ...     reason=AbortReason.BELOW_FLOOR,
    ...     matched_count=1,
    ...     minimum_matched=3,
    ...     expected_matched=16.0,
    ...     key_length=24,
    ...     message_bit=0,
    ... )
    >>> _round_already_spent(starved)
    False
    """
    return (
        isinstance(outcome, VerificationAbort)
        and outcome.reason is AbortReason.RECORD_ALREADY_VERIFIED
    )


def _signed_round(
    params: ProtocolParams, message_bit: int, rng: np.random.Generator
) -> tuple[Signature, dict[Party, RecipientRecord]]:
    """Run one honest round and return its declaration and both logs."""
    session = QDSSession(params, rng=rng)
    session.distribute()
    signature = session.sign(message_bit)
    return signature, dict(session.records[message_bit])


def _present(
    record: RecipientRecord, *, defended: bool
) -> RecipientRecord:
    """Return the log as the verifier holds it in one arm of the comparison."""
    return record if defended else record.with_session_id(None)


def _ledger(party: Party, *, defended: bool) -> ConsumedRecords | None:
    """Return the verifier's ledger, or ``None`` for the undefended arm."""
    return ConsumedRecords(party) if defended else None


def _as_counts_mode(value: Any) -> str:
    """Resolve which Phase C' ordering a measurement is to run under."""
    if not isinstance(value, str):
        raise TypeError(
            f"counts must be a string naming a Phase C' ordering, got "
            f"{type(value).__name__}"
        )
    if value not in _COUNTS_MODES:
        raise ValueError(
            f"counts must be one of {list(_COUNTS_MODES)}, got {value!r}. It "
            f"selects which declaration the recipients counted against, and "
            f"that single choice decides whether the denial-of-service "
            f"succeeds -- so it is named rather than defaulted silently."
        )
    return value


def _counterpart_count(
    declaration: Signature,
    records: dict[Party, RecipientRecord],
    params: ProtocolParams,
    *,
    ordering: str,
) -> Any:
    """Return what Charlie receives from Bob over Phase C', or ``None``."""
    if ordering == COUNTS_NONE:
        return None
    pooled = exchange_matched_counts(
        {
            party: matched_count_message(declaration, records[party], params)
            for party in (Party.BOB, Party.CHARLIE)
        },
        params,
    )
    return pooled.counterpart_of(Party.CHARLIE)


def measure_straight_replay(
    *,
    params: ProtocolParams,
    trials: int,
    rng: np.random.Generator,
    defended: bool,
    party: Party | str = Party.BOB,
) -> AttackOutcome:
    """Measure attack (a): re-present a captured declaration to its own verifier.

    Parameters
    ----------
    params : ProtocolParams
        The parameter set every trial runs at.
    trials : int
        Positive.
    rng : numpy.random.Generator
        The world's generator; each session draws from it, so consecutive
        rounds are independent without any seed arithmetic here.
    defended : bool
        ``True`` stamps the round on the verifier's log and gives him a
        :class:`~qsecure.protocol.verify.ConsumedRecords`; ``False`` is the
        state of the package before the defence, with an unstamped log and no
        ledger.
    party : Party or str, optional
        Which verifier is replayed at.

    Returns
    -------
    AttackOutcome
        ``successes`` counts trials whose **second** presentation was accepted.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.replay import measure_straight_replay
    >>> from qsecure.protocol.params import ProtocolParams
    >>> params = ProtocolParams(key_length=24)

    >>> before = measure_straight_replay(
    ...     params=params, trials=8, rng=np.random.default_rng(4), defended=False
    ... )
    >>> before.successes, before.trials
    (8, 8)

    >>> after = measure_straight_replay(
    ...     params=params, trials=8, rng=np.random.default_rng(4), defended=True
    ... )
    >>> after.successes, after.refusals
    (0, 8)
    """
    checked = _as_params_arg(params)
    count = _as_trials(trials)
    world = _as_generator(rng, "rng")
    _as_flag(defended, "defended")
    verifier = _as_verifier(party)

    successes = 0
    refusals = 0
    for _ in range(count):
        signature, records = _signed_round(checked, 0, world)
        record = _present(records[verifier], defended=defended)
        ledger = _ledger(verifier, defended=defended)
        first = verify_or_abort(signature, record, checked, ledger=ledger)
        again = verify_or_abort(signature, record, checked, ledger=ledger)
        if _refused(again):
            refusals += 1
        if _accepted(first) and _accepted(again):
            successes += 1
    return AttackOutcome(
        label=f"straight re-verification at {verifier.value}",
        defended=defended,
        trials=count,
        successes=successes,
        refusals=refusals,
        key_length=checked.key_length,
        note=(
            "refused as RECORD_ALREADY_VERIFIED by the consumed-records ledger"
            if defended
            else "the scoring rule is pure and cannot tell a second call from "
            "a first"
        ),
    )


def measure_cross_session_pairing(
    *,
    params: ProtocolParams,
    trials: int,
    rng: np.random.Generator,
    defended: bool,
    party: Party | str = Party.BOB,
) -> AttackOutcome:
    """Measure attack (b): run A's declaration against run B's records.

    Parameters
    ----------
    trials : int
        Positive.
    rng : numpy.random.Generator
        The world's generator.
    defended : bool
        Whether the verifier's log carries the round it was made in.
    party : Party or str, optional
        Which verifier is targeted, and therefore which threshold applies --
        ``s_a = 1/64`` for Bob, ``s_v = 1/16`` for Charlie.

    Returns
    -------
    AttackOutcome
        ``successes`` counts accepted pairings; ``refusals`` counts trials that
        reached no verdict, which under the defence is all of them.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.replay import measure_cross_session_pairing
    >>> from qsecure.protocol.params import ProtocolParams
    >>> params = ProtocolParams(key_length=24)
    >>> after = measure_cross_session_pairing(
    ...     params=params, trials=6, rng=np.random.default_rng(17),
    ...     defended=True,
    ... )
    >>> after.successes, after.refusals
    (0, 6)

    >>> before = measure_cross_session_pairing(
    ...     params=params, trials=6, rng=np.random.default_rng(17),
    ...     defended=False,
    ... )
    >>> before.refusals
    0
    """
    checked = _as_params_arg(params)
    count = _as_trials(trials)
    world = _as_generator(rng, "rng")
    _as_flag(defended, "defended")
    verifier = _as_verifier(party)

    successes = 0
    refusals = 0
    for _ in range(count):
        stale_signature, _ = _signed_round(checked, 0, world)
        _, live_records = _signed_round(checked, 0, world)
        record = _present(live_records[verifier], defended=defended)
        outcome = verify_or_abort(
            stale_signature,
            record,
            checked,
            ledger=_ledger(verifier, defended=defended),
        )
        if _refused(outcome):
            refusals += 1
        elif _accepted(outcome):
            successes += 1
    predicted = forgery_probability(checked, party=verifier)
    return AttackOutcome(
        label=f"cross-session pairing at {verifier.value}",
        defended=defended,
        trials=count,
        successes=successes,
        refusals=refusals,
        key_length=checked.key_length,
        note=(
            "refused as SESSION_MISMATCH before any position is counted"
            if defended
            else f"predicted {predicted:.6g} by "
            f"analysis.forgery_probability: an unrelated key agrees at 1/2 "
            f"per matched position, which is the outside forger's law exactly"
        ),
    )


def measure_ledger_denial_of_service(
    *,
    params: ProtocolParams,
    trials: int,
    rng: np.random.Generator,
    attack_rng: np.random.Generator,
    defended: bool,
    counts: str = COUNTS_AS_RECEIVED,
) -> AttackOutcome:
    """Measure attack (c): burn Charlie's round so the genuine signature dies.

    Parameters
    ----------
    trials : int
        Positive.
    rng : numpy.random.Generator
        The world's generator, which builds the rounds.
    attack_rng : numpy.random.Generator
        A **different stream**: the adversary's own, from which the forged
        key is drawn.
    defended : bool
        ``True`` gives Charlie the ledger and the stamped log.
    counts : str, optional
        Which declaration Phase C' counted against.

    Returns
    -------
    AttackOutcome
        ``successes`` counts trials where the honest declaration was refused
        because the round had already been spent; ``refusals`` counts every
        no-verdict, so the two are equal exactly when the ledger was the only
        thing refusing.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.replay import (
    ...     COUNTS_AS_SIGNED, measure_ledger_denial_of_service)
    >>> from qsecure.protocol.params import ProtocolParams
    >>> params = ProtocolParams(key_length=24)

    >>> after = measure_ledger_denial_of_service(
    ...     params=params, trials=8, rng=np.random.default_rng(31),
    ...     attack_rng=np.random.default_rng(77), defended=True,
    ... )
    >>> after.successes, after.refusals
    (8, 8)

    >>> before = measure_ledger_denial_of_service(
    ...     params=params, trials=8, rng=np.random.default_rng(31),
    ...     attack_rng=np.random.default_rng(77), defended=False,
    ... )
    >>> before.successes
    0

    >>> saved = measure_ledger_denial_of_service(
    ...     params=params, trials=8, rng=np.random.default_rng(31),
    ...     attack_rng=np.random.default_rng(77), defended=True,
    ...     counts=COUNTS_AS_SIGNED,
    ... )
    >>> saved.successes
    0
    """
    checked = _as_params_arg(params)
    count = _as_trials(trials)
    world, adversary = _distinct_generators(rng, attack_rng)
    _as_flag(defended, "defended")
    ordering = _as_counts_mode(counts)

    successes = 0
    refusals = 0
    for trial in range(count):
        signature, records = _signed_round(checked, 0, world)
        record = _present(records[Party.CHARLIE], defended=defended)
        ledger = _ledger(Party.CHARLIE, defended=defended)
        forged = Signature(
            message_bit=signature.message_bit,
            declared_key=_random_key(
                checked, signature.message_bit, adversary
            ),
            session_opening=signature.session_opening,
            context=signature.context,
        )
        counted_against = signature if ordering == COUNTS_AS_SIGNED else forged
        verify_or_abort(
            forged,
            record,
            checked,
            counterpart_matched=_counterpart_count(
                counted_against, records, checked, ordering=ordering
            ),
            ledger=ledger,
        )
        honest = verify_or_abort(
            signature,
            record,
            checked,
            counterpart_matched=_counterpart_count(
                signature, records, checked, ordering=ordering
            ),
            ledger=ledger,
        )
        if _refused(honest):
            refusals += 1
        if _round_already_spent(honest):
            successes += 1
        elif not _accepted(honest):
            raise ValueError(
                f"trial {trial} of {count}: the honest declaration was neither "
                f"accepted nor denied by the ledger. It came back as "
                f"{_describe(honest)}, which is not this attack's mechanism: "
                f"success here is the round having been *spent* by the "
                f"forgery, so a trial whose honest declaration would not have "
                f"been accepted anyway measures nothing at all. Counting such "
                f"a trial as a success -- which scoring on 'not accepted' did "
                f"-- turns a published rate into a statement about this seed "
                f"rather than about the attack, and counting it as a failure "
                f"would understate a defence that never got to act.\n"
                f"L = {checked.key_length}: below 137 the pooled matched-count "
                f"floor degenerates and an honest run can fail on its own. "
                f"Raise the key length, or investigate the reason above."
            )
    return AttackOutcome(
        label=(
            f"hop burns Charlie's round before the genuine declaration "
            f"[counts {ordering}]"
        ),
        defended=defended,
        trials=count,
        successes=successes,
        refusals=refusals,
        key_length=checked.key_length,
        note=(
            "the count-provenance check refuses the forgery, so nothing is "
            "spent and the genuine declaration still lands"
            if ordering == COUNTS_AS_SIGNED
            else (
                "the ledger makes the denial permanent: "
                "RECORD_ALREADY_VERIFIED"
                if defended
                else "without the ledger the denial is recoverable on the "
                "next presentation"
            )
        ),
    )


def measure_ledger_poisoning(
    *,
    params: ProtocolParams,
    trials: int,
    rng: np.random.Generator,
    attack_rng: np.random.Generator,
) -> AttackOutcome:
    """Measure attack (c), the other half: add an entry nobody earned.

    Parameters
    ----------
    trials : int
        Positive.
    rng : numpy.random.Generator
        The world's generator.
    attack_rng : numpy.random.Generator
        A different object.

    Returns
    -------
    AttackOutcome
        ``successes`` counts trials in which any route added an entry;
        ``refusals`` counts trials in which **all three** routes were refused,
        which is the only reading under which a partial trial cannot look like
        a pass.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.replay import measure_ledger_poisoning
    >>> from qsecure.protocol.params import ProtocolParams
    >>> outcome = measure_ledger_poisoning(
    ...     params=ProtocolParams(key_length=24), trials=6,
    ...     rng=np.random.default_rng(5), attack_rng=np.random.default_rng(6),
    ... )
    >>> outcome.successes, outcome.refusals
    (0, 6)
    """
    checked = _as_params_arg(params)
    count = _as_trials(trials)
    world, adversary = _distinct_generators(rng, attack_rng)

    successes = 0
    refusals = 0
    for _ in range(count):
        signature, records = _signed_round(checked, 0, world)
        record = records[Party.CHARLIE]
        ledger = ConsumedRecords(Party.CHARLIE)

        # Route 1: a declaration that names a round nobody ran.
        elsewhere = Signature(
            message_bit=signature.message_bit,
            declared_key=signature.declared_key,
            session_opening=adversary.bytes(_OPENING_HEX_BYTES).hex(),
            context=signature.context,
        )
        wrong_round = verify_or_abort(elsewhere, record, checked, ledger=ledger)

        # Route 2: a counterpart count that cannot carry a verdict.
        starved = verify_or_abort(
            signature, record, checked, counterpart_matched=0, ledger=ledger
        )

        # Route 3: the other verifier's evidence, offered to this ledger.
        blocked = False
        try:
            ledger.spend(records[Party.BOB])
        except ValueError:
            blocked = True

        if _refused(wrong_round) and _refused(starved) and blocked:
            refusals += 1

        if len(ledger) > 0:
            successes += 1
    return AttackOutcome(
        label="poison a verifier's ledger from outside",
        defended=True,
        trials=count,
        successes=successes,
        refusals=refusals,
        key_length=checked.key_length,
        note=(
            "a refusal spends nothing and a cross-party record is refused, so "
            "the only way into a ledger is a verdict the verifier reached"
        ),
    )


def measure_shared_identifier(
    *,
    params: ProtocolParams,
    trials: int,
    rng: np.random.Generator,
    attack_rng: np.random.Generator,
) -> AttackOutcome:
    """Measure attack (d): a signer gives two rounds one identifier.

    Parameters
    ----------
    trials : int
        Positive.
    rng : numpy.random.Generator
        The world's generator.
    attack_rng : numpy.random.Generator
        A different object.

    Returns
    -------
    AttackOutcome
        ``successes`` counts accepted migrations.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.replay import measure_shared_identifier
    >>> from qsecure.protocol.params import ProtocolParams
    >>> outcome = measure_shared_identifier(
    ...     params=ProtocolParams(key_length=24), trials=6,
    ...     rng=np.random.default_rng(23), attack_rng=np.random.default_rng(24),
    ... )
    >>> outcome.successes, outcome.refusals
    (0, 0)
    """
    checked = _as_params_arg(params)
    count = _as_trials(trials)
    world, adversary = _distinct_generators(rng, attack_rng)

    successes = 0
    refusals = 0
    for _ in range(count):
        opening = adversary.bytes(_OPENING_HEX_BYTES).hex()
        shared = session_identifier(0, checked.key_length, opening=opening)
        stale_signature, _ = _signed_round(checked, 0, world)
        _, live_records = _signed_round(checked, 0, world)
        relabelled = Signature(
            message_bit=0,
            declared_key=stale_signature.declared_key,
            session_opening=opening,
        )
        record = live_records[Party.BOB].with_session_id(shared)
        outcome = verify_or_abort(
            relabelled, record, checked, ledger=ConsumedRecords(Party.BOB)
        )
        if _refused(outcome):
            refusals += 1
        elif _accepted(outcome):
            successes += 1
    predicted = forgery_probability(checked, party=Party.BOB)
    return AttackOutcome(
        label="declaration migrates between two rounds sharing an identifier",
        defended=True,
        trials=count,
        successes=successes,
        refusals=refusals,
        key_length=checked.key_length,
        note=(
            f"the binding passes by construction and buys nothing: scored at "
            f"the outside forger's law, predicted {predicted:.6g} by "
            f"analysis.forgery_probability"
        ),
    )


def measure_shared_identifier_self_denial(
    *,
    params: ProtocolParams,
    trials: int,
    rng: np.random.Generator,
    attack_rng: np.random.Generator,
) -> AttackOutcome:
    """Measure what reusing an opening actually costs the signer who does it.

    Parameters
    ----------
    trials : int
        Positive.
    rng : numpy.random.Generator
        The world's generator.
    attack_rng : numpy.random.Generator
        A different object.

    Returns
    -------
    AttackOutcome
        ``successes`` counts trials whose honest second round was denied.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.replay import measure_shared_identifier_self_denial
    >>> from qsecure.protocol.params import ProtocolParams
    >>> outcome = measure_shared_identifier_self_denial(
    ...     params=ProtocolParams(key_length=24), trials=6,
    ...     rng=np.random.default_rng(23), attack_rng=np.random.default_rng(24),
    ... )
    >>> outcome.successes, outcome.refusals
    (6, 6)
    """
    checked = _as_params_arg(params)
    count = _as_trials(trials)
    world, adversary = _distinct_generators(rng, attack_rng)

    successes = 0
    refusals = 0
    for _ in range(count):
        opening = adversary.bytes(_OPENING_HEX_BYTES).hex()
        shared = session_identifier(0, checked.key_length, opening=opening)
        first_signature, first_records = _signed_round(checked, 0, world)
        second_signature, second_records = _signed_round(checked, 0, world)
        ledger = ConsumedRecords(Party.BOB)

        def present(
            signature: Signature,
            records: dict[Party, RecipientRecord],
            *,
            opening: str = opening,
            shared: str = shared,
            ledger: ConsumedRecords = ledger,
        ) -> VerificationResult | VerificationAbort:
            """Score one of the two rounds, both announcing ``shared``."""
            return verify_or_abort(
                Signature(
                    message_bit=0,
                    declared_key=signature.declared_key,
                    session_opening=opening,
                ),
                records[Party.BOB].with_session_id(shared),
                checked,
                ledger=ledger,
            )

        present(first_signature, first_records)
        outcome = present(second_signature, second_records)
        if _refused(outcome):
            refusals += 1
        if not _accepted(outcome):
            successes += 1
    return AttackOutcome(
        label="second round denied by an identifier its signer reused",
        defended=True,
        trials=count,
        successes=successes,
        refusals=refusals,
        key_length=checked.key_length,
        note=(
            "two rounds sharing an identifier are one round to every verifier; "
            "only the signer can arrange it and it costs her the second round"
        ),
    )


def measure_identifier_preimage(
    *,
    target: str,
    params: ProtocolParams,
    attempts: int,
    rng: np.random.Generator,
    message_bit: int = 0,
) -> AttackOutcome:
    """Measure attack (d): forge an opening so a stale declaration looks fresh.

    Parameters
    ----------
    target : str
        The identifier to hit -- normally the live round's, read off a
        verifier's log.
    params : ProtocolParams
        Supplies the key length bound into every identifier.
    attempts : int
        Positive.
    rng : numpy.random.Generator
        The **adversary's own** generator; no session is built here, so
        there is no world generator to keep it away from.
    message_bit : int, optional
        ``0`` or ``1``.

    Returns
    -------
    AttackOutcome
        ``successes`` counts openings that hashed to ``target``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.replay import (
    ...     measure_identifier_preimage, replay_capture)
    >>> capture = replay_capture()
    >>> outcome = measure_identifier_preimage(
    ...     target=capture.signature.session_id, params=capture.params,
    ...     attempts=2000, rng=np.random.default_rng(99),
    ... )
    >>> outcome.successes
    0
    >>> round(outcome.interval[1], 5)
    0.00192
    """
    if not isinstance(target, str):
        raise TypeError(
            f"target must be a string identifier, got {type(target).__name__}"
        )
    if not target:
        raise ValueError(
            "target must be non-empty: there is no round named by the empty "
            "identifier, so a search for it would report a meaningless zero."
        )
    checked = _as_params_arg(params)
    count = _as_trials(attempts)
    adversary = _as_generator(rng, "rng")
    if isinstance(message_bit, bool) or message_bit not in (0, 1):
        raise ValueError(f"message_bit must be 0 or 1, got {message_bit!r}")

    successes = 0
    for _ in range(count):
        opening = adversary.bytes(_OPENING_HEX_BYTES).hex()
        if (
            session_identifier(
                int(message_bit), checked.key_length, opening=opening
            )
            == target
        ):
            successes += 1
    return AttackOutcome(
        label="preimage on a round identifier",
        defended=True,
        trials=count,
        successes=successes,
        refusals=0,
        key_length=checked.key_length,
        note=(
            "analytic rate 2**-128 = 2.9e-39 per attempt; a bounded search can "
            "only bound it, and this one bounds it at the Wilson upper limit"
        ),
    )
