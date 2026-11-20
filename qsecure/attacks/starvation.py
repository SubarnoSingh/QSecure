"""Count starvation: a verifier denies every verdict with one integer.

The counterpart matched count exchanged in Phase C' is unauthenticated, and
the pooled and counterpart floors are applied to whatever was declared.

.. _starvation-headroom:

Denial headroom
---------------
A verifier ``R`` needs ``|M_R| >= m_min``, ``|M_R| + c >= M_min`` and
``c >= m_min``, so any declared count ``c`` at or below
``h = max(m_min - 1, M_min - |M_B| - 1)`` (:func:`denial_headroom`) denies the
run with probability ``1``.

Examples
--------
>>> from qsecure.attacks.starvation import least_implausible_z
>>> from qsecure.protocol.params import DEFAULT_PARAMS, ProtocolParams
>>> f"{least_implausible_z(ProtocolParams(key_length=600)):.4f}"
'-11.6047'
>>> f"{least_implausible_z(DEFAULT_PARAMS):.4f}"
'-11.5375'
>>> import math
>>> f"{-math.sqrt(3 * 64 * math.log(2)):.4f}"          # the L-independent limit
'-11.5362'

>>> from qsecure.protocol.analysis import matched_shortfall_probability
>>> from qsecure.protocol.verify import minimum_matched_count
>>> for parameters in (ProtocolParams(key_length=600), DEFAULT_PARAMS):
...     floor = minimum_matched_count(parameters)
...     tail = matched_shortfall_probability(parameters, minimum_matched=floor)
...     print(parameters.key_length, floor, f"{tail:.1e}")
600 67 4.1e-37
115200 36555 2.5e-31

>>> from qsecure.attacks.starvation import pooled_branch_requirement
>>> threshold, probability = pooled_branch_requirement(DEFAULT_PARAMS)
>>> threshold, f"{probability:.1e}"
(37635, '8.3e-07')
"""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from qsecure.attacks.statistics import wilson_bounds
from qsecure.attacks.isolation import DecisionProbe, derived_from_seed
from qsecure.protocol.analysis import (
    matched_shortfall_probability,
    matched_statistics,
)
from qsecure.protocol.params import VERIFIERS, Party, ProtocolParams
from qsecure.protocol.session import QDSSession, SessionTranscript
from qsecure.protocol.tally import (
    MatchedCountMessage,
    PooledMatchedCounts,
    exchange_matched_counts,
)
from qsecure.protocol.verify import (
    MatchedSetTooSmall,
    minimum_matched_count,
    minimum_pooled_matched_count,
)

__all__ = [
    "PROBE_PARAMS",
    "CountStarver",
    "StarvationDecision",
    "StarvationMeasurement",
    "StarvationMode",
    "declaration_z_score",
    "declared_versus_scored",
    "denial_headroom",
    "least_implausible_z",
    "measure_starvation",
    "pooled_branch_requirement",
    "probe_messages",
    "starvation_probe",
    "wilson_interval",
]


PROBE_PARAMS: Final[ProtocolParams] = ProtocolParams(key_length=600)
"""Parameters the isolation probe's frozen messages are stated under."""

_HONEST_PROBE_COUNTS: Final[tuple[int, int]] = (198, 203)
"""``(m_B, m_C)`` the probe freezes: two ordinary draws near ``mu = 200``."""

_PROBE_DIGEST: Final[str] = "a1b2c3d4"
"""A stand-in declaration fingerprint."""


class StarvationMode(enum.StrEnum):
    """How far below the headroom a starving declaration goes.

    Attributes
    ----------
    ZERO
        Declare ``0``.
    LEAST_IMPLAUSIBLE
        Declare the largest count that still denies -- :func:`denial_headroom`,
        less a small draw from the adversary's own generator so that a run of
        them is not the constant ``m_min - 1``.
    """

    ZERO = "zero"
    LEAST_IMPLAUSIBLE = "least-implausible"


@dataclass(frozen=True)
class StarvationDecision:
    """What the starver did on one Phase C' exchange, and what it could see.

    Parameters
    ----------
    party : Party
        The starving verifier.
    message_bit : int
        The bit whose declaration was being counted.
    true_count : int
        What this verifier's honest message said, i.e. ``|M_R|``.
    declared_count : int
        What was put on the wire instead.
    counterpart_count : int
        The other verifier's count, passed through untouched.
    headroom : int
        :func:`denial_headroom` for this run -- the largest declaration that
        would still deny.
    starved : bool
        Whether this call chose to deny.

    Examples
    --------
    >>> from qsecure.attacks.starvation import StarvationDecision
    >>> from qsecure.protocol.params import Party
    >>> decision = StarvationDecision(
    ...     party=Party.CHARLIE, message_bit=0, true_count=203,
    ...     declared_count=64, counterpart_count=198, headroom=66,
    ...     starved=True,
    ... )
    >>> decision.denies, decision.understatement
    (True, 139)
    """

    party: Party
    message_bit: int
    true_count: int
    declared_count: int
    counterpart_count: int
    headroom: int
    starved: bool

    @property
    def denies(self) -> bool:
        """bool: ``True`` iff the declared count is at or below the headroom.
        """
        return self.declared_count <= self.headroom

    @property
    def understatement(self) -> int:
        """int: ``|M_R|`` minus what was declared."""
        return self.true_count - self.declared_count


def denial_headroom(counterpart_matched: int, params: ProtocolParams) -> int:
    """Return the largest own-count declaration that still denies the run.

    Parameters
    ----------
    counterpart_matched : int
        ``|M_B|`` -- what the *other* verifier holds, which is what the pooled
        branch is measured against.
    params : ProtocolParams
        The parameter set the run executes under.

    Returns
    -------
    int
        In ``[0, key_length]``.

    Examples
    --------
    >>> from qsecure.attacks.starvation import denial_headroom
    >>> from qsecure.protocol.params import DEMO_PARAMS, ProtocolParams
    >>> params = ProtocolParams(key_length=600)          # m_min 67, M_min 212
    >>> denial_headroom(200, params)                     # the ordinary case
    66
    >>> denial_headroom(140, params)                     # Bob starved himself
    71
    >>> denial_headroom(64, DEMO_PARAMS)                 # L = 192: m_min is 1
    0
    """
    checked = _as_params(params)
    counterpart = _as_count(counterpart_matched, "counterpart_matched")
    if counterpart > checked.key_length:
        raise ValueError(
            f"counterpart_matched ({counterpart}) cannot exceed key_length "
            f"({checked.key_length}); a matched set is a subset of the key "
            f"positions."
        )
    own_branch = minimum_matched_count(checked) - 1
    pooled_branch = minimum_pooled_matched_count(checked) - counterpart - 1
    return max(0, min(checked.key_length, max(own_branch, pooled_branch)))


def declaration_z_score(count: int, params: ProtocolParams) -> float:
    """Return how many honest standard deviations a declared count is off.

    Parameters
    ----------
    count : int
        The declared matched count.
    params : ProtocolParams
        The parameter set, supplying ``mu = L/|B|`` and ``sqrt(mu(1 - 1/|B|))``
        through :func:`~qsecure.protocol.analysis.matched_statistics`.

    Returns
    -------
    float
        ``(count - mu) / sd``.

    Examples
    --------
    >>> from qsecure.attacks.starvation import declaration_z_score
    >>> from qsecure.protocol.params import ProtocolParams
    >>> params = ProtocolParams(key_length=600)
    >>> f"{declaration_z_score(200, params):.2f}"        # the honest mean
    '0.00'
    >>> f"{declaration_z_score(66, params):.2f}"         # the best a starver can do
    '-11.60'
    >>> f"{declaration_z_score(0, params):.2f}"          # the audit's variant
    '-17.32'
    """
    checked = _as_params(params)
    value = _as_count(count, "count")
    statistics = matched_statistics(checked)
    return (value - statistics.expected) / math.sqrt(statistics.variance)


def least_implausible_z(params: ProtocolParams) -> float:
    """Return the z-score of the quietest declaration that still denies.

    Parameters
    ----------
    params : ProtocolParams
        The parameter set.

    Returns
    -------
    float
        Negative, and bounded away from ``0`` uniformly in ``L``.

    Examples
    --------
    >>> from qsecure.attacks.starvation import least_implausible_z
    >>> from qsecure.protocol.params import (
    ...     DEFAULT_PARAMS, DEMO_PARAMS, ProtocolParams
    ... )
    >>> f"{least_implausible_z(ProtocolParams(key_length=600)):.4f}"
    '-11.6047'
    >>> f"{least_implausible_z(ProtocolParams(key_length=1200)):.4f}"
    '-11.5738'
    >>> f"{least_implausible_z(DEFAULT_PARAMS):.4f}"
    '-11.5375'
    >>> f"{least_implausible_z(DEMO_PARAMS):.4f}"     # m_min degenerate, no claim
    '-9.7980'
    """
    checked = _as_params(params)
    return declaration_z_score(
        max(0, minimum_matched_count(checked) - 1), checked
    )


def pooled_branch_requirement(params: ProtocolParams) -> tuple[int, float]:
    """Return what the counterpart's own count must be for the second branch.

    Parameters
    ----------
    params : ProtocolParams
        The parameter set.

    Returns
    -------
    threshold : int
        ``M_min - m_min``.
    probability : float
        How often an honest counterpart lands there, from
        :func:`~qsecure.protocol.analysis.matched_shortfall_probability`.

    Examples
    --------
    >>> from qsecure.attacks.starvation import pooled_branch_requirement
    >>> from qsecure.protocol.params import DEFAULT_PARAMS, ProtocolParams
    >>> threshold, probability = pooled_branch_requirement(
    ...     ProtocolParams(key_length=600)
    ... )
    >>> threshold, f"{probability:.1e}"
    (145, '4.0e-07')
    >>> threshold, probability = pooled_branch_requirement(DEFAULT_PARAMS)
    >>> threshold, f"{probability:.1e}"
    (37635, '8.3e-07')
    """
    checked = _as_params(params)
    threshold = minimum_pooled_matched_count(checked) - minimum_matched_count(
        checked
    )
    if threshold < 1:
        return threshold, 0.0
    return threshold, matched_shortfall_probability(
        checked, minimum_matched=threshold
    )


class CountStarver:
    """A verifier who under-reports his Phase C' count to deny the verdict.

    Parameters
    ----------
    rng : numpy.random.Generator
        The adversary's **own**.
    party : Party or str, optional
        Which verifier is dishonest.
    mode : StarvationMode or str, optional
        Defaults to :attr:`~StarvationMode.LEAST_IMPLAUSIBLE`, the strongest
        variant.
    denial_probability : float, optional
        In ``[0, 1]``.
    target_bits : iterable of int or None, optional
        Message bits eligible for denial.
    jitter : int, optional
        Non-negative.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.starvation import CountStarver
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> params = ProtocolParams(key_length=600)
    >>> starver = CountStarver(rng=np.random.default_rng(2026))
    >>> transcript = QDSSession(
    ...     params, rng=np.random.default_rng(7), count_exchange=starver
    ... ).run(0)

    >>> transcript.is_complete, transcript.transferable
    (False, False)
    >>> [(abort.party.value, str(abort.reason)) for abort in transcript.aborts]
    [('Bob', 'counterpart-matched-count-below-floor')]

    >>> declared, scored = transcript.pooled.charlie_count, [
    ...     result.matched_count
    ...     for result in transcript.results
    ...     if result.party.value == "Charlie"
    ... ][0]
    >>> declared < 67 <= scored
    True
    """

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        party: Party | str = Party.CHARLIE,
        mode: StarvationMode | str = StarvationMode.LEAST_IMPLAUSIBLE,
        denial_probability: float = 1.0,
        target_bits: Iterable[int] | None = None,
        jitter: int = 8,
    ) -> None:
        """Store the adversary's own generator and its policy."""
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                f"rng must be a numpy.random.Generator, got "
                f"{type(rng).__name__}. an adversary owns its "
                f"randomness and never derives it from the seed the harness "
                f"gave QDSSession."
            )
        self._rng = rng
        self.party = _as_verifier(party)
        self.mode = StarvationMode(mode)
        if isinstance(denial_probability, bool) or not isinstance(
            denial_probability, (int, float, np.floating, np.integer)
        ):
            raise TypeError(
                f"denial_probability must be a real number in [0, 1], got "
                f"{type(denial_probability).__name__}"
            )
        probability = float(denial_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"denial_probability must lie in [0, 1], got {probability}. It "
                f"is the per-exchange coin the selective attack is defined by."
            )
        self.denial_probability = probability
        if target_bits is None:
            resolved_bits = frozenset({0, 1})
        else:
            resolved_bits = frozenset(
                _as_message_bit(bit) for bit in target_bits
            )
            if not resolved_bits:
                raise ValueError(
                    "target_bits must name at least one message bit, got an "
                    "empty collection. Pass None for both bits, or omit the "
                    "adversary entirely for an honest run."
                )
        self.target_bits = resolved_bits
        span = _as_count(jitter, "jitter")
        self.jitter = span
        self._log: list[StarvationDecision] = []

    @property
    def log(self) -> tuple[StarvationDecision, ...]:
        """tuple of StarvationDecision: every exchange, in call order."""
        return tuple(self._log)

    def __call__(
        self,
        messages: Mapping[Party | str, MatchedCountMessage],
        params: ProtocolParams,
    ) -> PooledMatchedCounts:
        """Run Phase C' with this verifier's count replaced."""
        checked = _as_params(params)
        counterpart_party = _counterpart_of(self.party)
        own = _message_for(messages, self.party)
        counterpart = _message_for(messages, counterpart_party)
        headroom = denial_headroom(counterpart.matched_count, checked)
        coin = float(self._rng.random())
        starving = (
            own.message_bit in self.target_bits
            and coin < self.denial_probability
        )
        declared = own.matched_count
        if starving:
            declared = self._starving_count(own.matched_count, headroom)
        self._log.append(
            StarvationDecision(
                party=self.party,
                message_bit=own.message_bit,
                true_count=own.matched_count,
                declared_count=declared,
                counterpart_count=counterpart.matched_count,
                headroom=headroom,
                starved=starving,
            )
        )
        doctored = dataclasses.replace(own, matched_count=declared)
        return exchange_matched_counts(
            {self.party: doctored, counterpart_party: counterpart}, checked
        )

    def _starving_count(self, true_count: int, headroom: int) -> int:
        """Return the count to declare on a call that has decided to deny."""
        ceiling = min(true_count, headroom)
        if self.mode is StarvationMode.ZERO or ceiling <= 0:
            return 0
        if self.jitter == 0:
            return ceiling
        return max(0, ceiling - int(self._rng.integers(0, self.jitter + 1)))


def probe_messages(
    params: ProtocolParams = PROBE_PARAMS,
    *,
    message_bit: int = 0,
    counts: tuple[int, int] = _HONEST_PROBE_COUNTS,
) -> dict[Party, MatchedCountMessage]:
    """Return the frozen pair of messages the isolation probe replays.

    Parameters
    ----------
    params : ProtocolParams, optional
        Defaults to :data:`PROBE_PARAMS`.
    message_bit : int, optional
        ``0`` or ``1``.
    counts : tuple of int, optional
        ``(m_B, m_C)``.

    Returns
    -------
    dict of Party to MatchedCountMessage
        Keyed by :attr:`~qsecure.protocol.params.Party.BOB` then
        :attr:`~qsecure.protocol.params.Party.CHARLIE`.

    Examples
    --------
    >>> from qsecure.attacks.starvation import probe_messages
    >>> from qsecure.protocol.params import Party
    >>> messages = probe_messages()
    >>> messages[Party.BOB].matched_count, messages[Party.CHARLIE].matched_count
    (198, 203)
    >>> messages[Party.BOB].declaration_digest == messages[
    ...     Party.CHARLIE
    ... ].declaration_digest
    True
    """
    checked = _as_params(params)
    bit = _as_message_bit(message_bit)
    bob_count, charlie_count = (
        _as_count(counts[0], "counts[0]"),
        _as_count(counts[1], "counts[1]"),
    )
    return {
        Party.BOB: MatchedCountMessage(
            Party.BOB, bit, bob_count, checked.key_length, _PROBE_DIGEST
        ),
        Party.CHARLIE: MatchedCountMessage(
            Party.CHARLIE, bit, charlie_count, checked.key_length, _PROBE_DIGEST
        ),
    }


def starvation_probe(
    params: ProtocolParams = PROBE_PARAMS, *, repeats: int = 24
) -> DecisionProbe:
    """Return the :class:`~qsecure.attacks.isolation.DecisionProbe` for this
    attack.

    Parameters
    ----------
    params : ProtocolParams, optional
        Defaults to :data:`PROBE_PARAMS`.
    repeats : int, optional
        How many exchanges to observe per probe call.

    Returns
    -------
    DecisionProbe
        Ready for :func:`~qsecure.attacks.isolation.assert_attack_isolated`.

    Examples
    --------
    >>> from qsecure.attacks.isolation import assert_attack_isolated
    >>> from qsecure.attacks.starvation import CountStarver, starvation_probe
    >>> report = assert_attack_isolated(CountStarver, starvation_probe())
    >>> report.isolated, report.session_seed_offered
    (True, False)
    >>> report.distinct_decisions > 1
    True
    """
    checked = _as_params(params)
    count = _as_count(repeats, "repeats")
    if count < 1:
        raise ValueError(
            f"repeats must be at least 1, got {count}; the probe has to see at "
            f"least one decision."
        )
    frozen = probe_messages(checked)

    def probe(attack: Any, session_seed: int) -> tuple[int, ...]:
        """Declare ``repeats`` times against the frozen pair; return the
        counts.
        """
        del session_seed
        return tuple(
            attack(frozen, checked).count_for(_starver_party(attack))
            for _ in range(count)
        )

    return probe


def wilson_interval(
    successes: int, trials: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Return a Wilson score interval for a measured rate.

    Parameters
    ----------
    successes : int
        Number of successes, ``0 <= successes <= trials``.
    trials : int
        Number of trials, at least ``1``.
    z : float, optional
        Standard normal quantile.

    Returns
    -------
    tuple of float
        ``(low, high)``, both in ``[0, 1]``.

    Examples
    --------
    >>> from qsecure.attacks.starvation import wilson_interval
    >>> low, high = wilson_interval(60, 60)
    >>> f"{low:.4f}", f"{high:.4f}"
    ('0.9398', '1.0000')
    >>> low, high = wilson_interval(0, 60)
    >>> f"{low:.4f}", f"{high:.4f}"
    ('0.0000', '0.0602')
    """
    hits = _as_count(successes, "successes")
    total = _as_count(trials, "trials")
    if total < 1:
        raise ValueError(
            f"trials must be at least 1, got {total}; a rate needs a "
            f"denominator."
        )
    if hits > total:
        raise ValueError(
            f"successes ({hits}) cannot exceed trials ({total})."
        )
    if isinstance(z, bool) or not isinstance(
        z, (int, float, np.floating, np.integer)
    ):
        raise TypeError(f"z must be a real number, got {type(z).__name__}")
    quantile = float(z)
    if quantile <= 0.0:
        raise ValueError(f"z must be positive, got {quantile}")
    return wilson_bounds(hits, total, z=quantile)


@dataclass(frozen=True)
class StarvationMeasurement:
    """The outcome of one arm of the starvation experiment.

    Parameters
    ----------
    label : str
        The arm's name.
    key_length : int
        ``L`` the arm was run at.
    trials : int
        Sessions run.
    denied : int
        Runs where at least one verifier reached **no verdict**.
    complete : int
        Runs where both verifiers reached a verdict.
    transferable : int
        Runs where both verifiers accepted.
    starved_exchanges : int
        Exchanges on which the adversary chose to deny.
    accepted_by_starver : int
        Runs where the starving verifier himself accepted.
    declared_counts : tuple of int
        What the starving verifier put on the wire, per run.
    scored_counts : tuple of int
        What the starving verifier's own verdict recorded as his true
        ``|M_R|``, per run in which he reached one.
    abort_reasons : tuple of str
        Every abort reason recorded across the arm, in run order.

    Examples
    --------
    >>> from qsecure.attacks.starvation import StarvationMeasurement
    >>> arm = StarvationMeasurement(
    ...     label="zero", key_length=192, trials=40, denied=40, complete=0,
    ...     transferable=0, starved_exchanges=40, accepted_by_starver=40,
    ...     declared_counts=(0,) * 40, scored_counts=(64,) * 40,
    ...     abort_reasons=("pooled-matched-count-below-floor",) * 40,
    ... )
    >>> arm.denial_rate
    1.0
    >>> low, high = arm.denial_interval
    >>> f"{low:.4f}"
    '0.9124'
    """

    label: str
    key_length: int
    trials: int
    denied: int
    complete: int
    transferable: int
    starved_exchanges: int
    accepted_by_starver: int
    declared_counts: tuple[int, ...]
    scored_counts: tuple[int, ...]
    abort_reasons: tuple[str, ...]

    @property
    def denial_rate(self) -> float:
        """float: fraction of runs in which some verifier reached no verdict.
        """
        return self.denied / self.trials

    @property
    def denial_interval(self) -> tuple[float, float]:
        """tuple of float: 95% Wilson interval for :attr:`denial_rate`."""
        return wilson_interval(self.denied, self.trials)

    @property
    def completion_rate(self) -> float:
        """float: fraction of runs in which both verifiers reached a verdict.
        """
        return self.complete / self.trials

    def summary(self) -> str:
        """Return a one-line account of the arm.

        Examples
        --------
        >>> from qsecure.attacks.starvation import StarvationMeasurement
        >>> StarvationMeasurement(
        ...     "zero", 192, 40, 40, 0, 0, 40, 40, (), (), ()
        ... ).summary()
        'zero (L=192): denied 40/40 = 1.000 [0.912, 1.000], complete 0/40, \
transferable 0/40, starving exchanges 40.'
        """
        low, high = self.denial_interval
        return (
            f"{self.label} (L={self.key_length}): denied {self.denied}/"
            f"{self.trials} = {self.denial_rate:.3f} [{low:.3f}, {high:.3f}], "
            f"complete {self.complete}/{self.trials}, transferable "
            f"{self.transferable}/{self.trials}, starving exchanges "
            f"{self.starved_exchanges}."
        )


def declared_versus_scored(
    transcript: SessionTranscript, party: Party | str
) -> tuple[int, int] | None:
    """Return ``(declared, scored)`` for one verifier from a single transcript.

    Parameters
    ----------
    transcript : SessionTranscript
        A finished or partial run.
    party : Party or str
        The verifier to look at.

    Returns
    -------
    tuple of int or None
        ``(declared, scored)``, or ``None`` when the pair is not available:
        either the recipients did not exchange counts
        (:attr:`~qsecure.protocol.session.SessionTranscript.counts_exchanged`
        ``False``) or this verifier reached no verdict, so no true count was
        published.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.starvation import (
    ...     CountStarver, declared_versus_scored
    ... )
    >>> from qsecure.protocol.params import Party, ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> transcript = QDSSession(
    ...     ProtocolParams(key_length=600),
    ...     rng=np.random.default_rng(7),
    ...     count_exchange=CountStarver(rng=np.random.default_rng(2026)),
    ... ).run(0)
    >>> declared, scored = declared_versus_scored(transcript, Party.CHARLIE)
    >>> declared < scored
    True
    >>> declared_versus_scored(transcript, Party.BOB) is None   # Bob aborted
    True
    """
    if not isinstance(transcript, SessionTranscript):
        raise TypeError(
            f"transcript must be a SessionTranscript, got "
            f"{type(transcript).__name__}"
        )
    resolved = _as_verifier(party)
    if transcript.pooled is None:
        return None
    for result in transcript.results:
        if result.party is resolved:
            return (
                transcript.pooled.count_for(resolved),
                result.matched_count,
            )
    return None


def measure_starvation(
    params: ProtocolParams,
    trials: int,
    *,
    label: str,
    starver: CountStarver | None = None,
    message_bits: Sequence[int] = (0,),
    session_seed_start: int = 500_000,
) -> StarvationMeasurement:
    """Run one arm of the experiment and count what the sessions did.

    Parameters
    ----------
    params : ProtocolParams
        The parameter set every session in the arm runs under.
    trials : int
        Sessions to run, at least ``1``.
    label : str
        Name for the arm.
    starver : CountStarver or None, optional
        ``None`` runs the honest control arm with the shipped
        :func:`~qsecure.protocol.tally.exchange_matched_counts`.
    message_bits : sequence of int, optional
        Cycled over the trials, so an arm can mix bits and a bit-selective
        starver can be seen to let one of them through.
    session_seed_start : int, optional
        Session seeds are ``session_seed_start + i``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.starvation import CountStarver, measure_starvation
    >>> from qsecure.protocol.params import DEMO_PARAMS
    >>> arm = measure_starvation(
    ...     DEMO_PARAMS, 3, label="demo",
    ...     starver=CountStarver(rng=np.random.default_rng(11)),
    ... )
    >>> arm.denied, arm.complete, arm.starved_exchanges
    (3, 0, 3)
    >>> set(arm.declared_counts)                # m_min is 1 at L = 192
    {0}
    """
    checked = _as_params(params)
    total = _as_count(trials, "trials")
    if total < 1:
        raise ValueError(f"trials must be at least 1, got {total}")
    if not isinstance(label, str) or not label:
        raise ValueError(
            f"label must be a non-empty string naming the arm, got {label!r}"
        )
    if starver is not None and not isinstance(starver, CountStarver):
        raise TypeError(
            f"starver must be a CountStarver or None, got "
            f"{type(starver).__name__}; None runs the honest control arm."
        )
    bits = tuple(_as_message_bit(bit) for bit in message_bits)
    if not bits:
        raise ValueError(
            "message_bits must name at least one bit; it is cycled over the "
            "trials."
        )
    start = _as_count(session_seed_start, "session_seed_start")
    if starver is not None:
        for offset in range(total):
            if derived_from_seed(starver._rng, start + offset):  # noqa: SLF001
                raise ValueError(
                    f"starver's generator is the stream session seed "
                    f"{start + offset} produces. The arm would measure an "
                    f"adversary correlated with the very runs it is denying, "
                    f"and its denial rate would be a statement about one seed "
                    f"rather than about the attack. session_seed_start is "
                    f"{start} and the arm runs {total} consecutive seeds from "
                    f"there; seed the CountStarver from something unrelated to "
                    f"that range."
                )

    denied = complete = transferable = accepted_by_starver = 0
    declared: list[int] = []
    scored: list[int] = []
    reasons: list[str] = []
    starved_before = 0 if starver is None else len(starver.log)

    for index in range(total):
        session = QDSSession(
            checked,
            rng=np.random.default_rng(start + index),
            count_exchange=exchange_matched_counts
            if starver is None
            else starver,
        )
        try:
            transcript = session.run(bits[index % len(bits)])
        except MatchedSetTooSmall:  # pragma: no cover - run() catches these
            raise
        if not transcript.is_complete:
            denied += 1
        else:
            complete += 1
        if transcript.transferable:
            transferable += 1
        reasons.extend(str(abort.reason) for abort in transcript.aborts)
        if starver is not None and transcript.pooled is not None:
            declared.append(transcript.pooled.count_for(starver.party))
            pair = declared_versus_scored(transcript, starver.party)
            if pair is not None:
                scored.append(pair[1])
                accepted_by_starver += sum(
                    1
                    for result in transcript.results
                    if result.party is starver.party and result.accepted
                )

    starved = (
        0
        if starver is None
        else sum(
            1 for decision in starver.log[starved_before:] if decision.starved
        )
    )
    return StarvationMeasurement(
        label=label,
        key_length=checked.key_length,
        trials=total,
        denied=denied,
        complete=complete,
        transferable=transferable,
        starved_exchanges=starved,
        accepted_by_starver=accepted_by_starver,
        declared_counts=tuple(declared),
        scored_counts=tuple(scored),
        abort_reasons=tuple(reasons),
    )


def _as_params(params: Any) -> ProtocolParams:
    """Return ``params`` unchanged, refusing anything else."""
    if not isinstance(params, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(params).__name__}"
        )
    return params


def _as_count(value: Any, name: str) -> int:
    """Return ``value`` as a non-negative :class:`int`."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{name} must be an int, got {type(value).__name__}"
        )
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative, got {number}")
    return number


def _as_message_bit(value: Any) -> int:
    """Return ``value`` as ``0`` or ``1``."""
    if isinstance(value, bool) or value not in (0, 1):
        raise ValueError(
            f"message bits are 0 or 1, got {value!r}. A run signs one bit and "
            f"the two distributions are independent."
        )
    return int(value)


def _as_verifier(party: Any) -> Party:
    """Return ``party`` as Bob or Charlie."""
    if isinstance(party, Party):
        resolved = party
    elif isinstance(party, str):
        try:
            resolved = Party(party)
        except ValueError as unknown:
            raise ValueError(
                f"party must name a verifier, got {party!r}; the verifiers are "
                f"{[member.value for member in VERIFIERS]}."
            ) from unknown
    else:
        raise TypeError(
            f"party must be a Party or a string, got {type(party).__name__}"
        )
    if resolved is Party.ALICE:
        raise ValueError(
            "Alice is not in Phase C': she holds no measurement record and "
            "reports no matched count. Count starvation is a recipient's "
            "attack -- pass Party.BOB or Party.CHARLIE."
        )
    return resolved


def _message_for(
    messages: Mapping[Party | str, MatchedCountMessage], party: Party
) -> MatchedCountMessage:
    """Return one verifier's message from a mapping keyed either way."""
    for key in (party, party.value):
        if key in messages:
            message = messages[key]
            if not isinstance(message, MatchedCountMessage):
                raise TypeError(
                    f"messages[{key!r}] must be a MatchedCountMessage, got "
                    f"{type(message).__name__}; the seam is called with the "
                    f"pair the session computed."
                )
            return message
    raise ValueError(
        f"the count exchange needs a message from {party.value}, and the "
        f"mapping has none. Phase C' is defined pairwise over both verifiers."
    )


def _counterpart_of(party: Party) -> Party:
    """Return the other verifier."""
    return Party.CHARLIE if party is Party.BOB else Party.BOB


def _starver_party(attack: Any) -> Party:
    """Return the party a probed candidate reports for."""
    party = getattr(attack, "party", Party.CHARLIE)
    return party if isinstance(party, Party) else _as_verifier(party)
