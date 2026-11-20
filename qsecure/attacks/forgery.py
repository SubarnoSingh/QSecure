"""Forgery: the outside adversary who knows nothing, and the recipient who does.

:class:`OutsideForger` declares a fresh random key; the forging recipient
declares from his own record. Both are measured against
:func:`~qsecure.protocol.analysis.forgery_probability`.

Examples
--------
>>> import numpy as np
>>> from qsecure.attacks.forgery import OutsideForger, measure_outside_forgery
>>> from qsecure.protocol.params import Party, ProtocolParams
>>> measurement = measure_outside_forgery(
...     ProtocolParams(key_length=24),
...     trials=24,
...     rng=np.random.default_rng(11),
...     session_rng=np.random.default_rng(12),
... )
>>> measurement[Party.CHARLIE].mismatch_rate_within(0.5, 0.08)
True

>>> from qsecure.attacks.forgery import measure_recipient_forgery
>>> bound = measure_recipient_forgery(
...     ProtocolParams(key_length=48),
...     trials=24,
...     rng=np.random.default_rng(13),
...     session_rng=np.random.default_rng(14),
... )
>>> bound.mismatch_rate_within(1 / 12, 0.03)
True
>>> bound.scored_fraction_within(2 / 3, 0.05)
True

>>> stripped = measure_recipient_forgery(
...     ProtocolParams(key_length=48),
...     trials=24,
...     rng=np.random.default_rng(15),
...     session_rng=np.random.default_rng(16),
...     symmetrised=False,
... )
>>> stripped.mismatch_rate_within(1 / 3, 0.06)
True
>>> stripped.scored_fraction_within(1 / 3, 0.05)
True
>>> round(bound.mismatch_rate, 3) < round(stripped.mismatch_rate, 3)
True
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from qsecure.attacks.isolation import (
    DecisionProbe,
    SignerScenario,
    require_distinct_streams,
    signer_scenario,
)
from qsecure.attacks.statistics import wilson_bounds
from qsecure.core.rng import resolve_rng
from qsecure.protocol.analysis import (
    forgery_probability,
    recipient_forgery_probability,
)
from qsecure.protocol.keys import (
    KeyElement,
    PrivateKey,
    generate_private_key,
    key_from_record,
)
from qsecure.protocol.params import VERIFIERS, Party, ProtocolParams
from qsecure.protocol.records import RecipientView
from qsecure.protocol.session import QDSSession
from qsecure.protocol.signature import Signature
from qsecure.protocol.symmetrise import no_symmetrisation
from qsecure.protocol.tally import no_count_exchange

__all__ = [
    "OUTSIDE_FORGER_MISMATCH_RATE",
    "RECIPIENT_FORGER_MISMATCH_RATE",
    "UNSYMMETRISED_FORGER_MISMATCH_RATE",
    "Z_95",
    "ForgeryMeasurement",
    "OutsideForger",
    "RecipientForger",
    "forwarder_probe",
    "measure_outside_forgery",
    "measure_recipient_forgery",
    "wilson_interval",
]


Z_95: Final[float] = 1.959963984540054
"""float: The normal quantile behind every interval this module reports."""

OUTSIDE_FORGER_MISMATCH_RATE: Final[float] = 0.5
"""float: Eve's mismatch rate among scored positions -- a fair coin."""

RECIPIENT_FORGER_MISMATCH_RATE: Final[float] = 1.0 / 12.0
"""float: Bob's mismatch rate among scored positions, ``(n-1)/(2n(n+1))``."""

UNSYMMETRISED_FORGER_MISMATCH_RATE: Final[float] = 1.0 / 3.0
"""float: The same recipient's mismatch rate with Phase A' removed."""

_GUESS_LABEL: Final[str] = "guess_probability"
"""Name quoted in :class:`RecipientForger`'s validation messages."""


def wilson_interval(
    successes: int, trials: int, *, z: float = Z_95
) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion.

    Parameters
    ----------
    successes : int
        Number of successes, ``0 <= successes <= trials``.
    trials : int
        Number of trials.
    z : float, optional
        Normal quantile; defaults to :data:`Z_95`.

    Returns
    -------
    tuple of float
        ``(low, high)``, both clipped into ``[0, 1]``.

    Examples
    --------
    >>> from qsecure.attacks.forgery import wilson_interval
    >>> low, high = wilson_interval(0, 100)
    >>> low, round(high, 4)
    (0.0, 0.037)
    >>> low, high = wilson_interval(50, 100)
    >>> round(low, 4), round(high, 4)
    (0.4038, 0.5962)
    """
    checked_trials = _as_count(trials, "trials")
    checked_successes = _as_count(successes, "successes")
    if checked_trials <= 0:
        raise ValueError(
            f"trials must be positive, got {checked_trials}. An interval over "
            f"zero trials is not a wide measurement, it is the absence of one; "
            f"report the trial count instead of a rate."
        )
    if checked_successes > checked_trials:
        raise ValueError(
            f"successes ({checked_successes}) exceeds trials "
            f"({checked_trials}); a proportion above one is a counting error "
            f"upstream, not a rate."
        )
    if isinstance(z, bool) or not isinstance(z, (int, float, np.floating)):
        raise TypeError(f"z must be a real number, got {type(z).__name__}")
    quantile = float(z)
    if not math.isfinite(quantile) or quantile <= 0.0:
        raise ValueError(
            f"z must be positive and finite, got {quantile}. It is a normal "
            f"quantile; Z_95 is the default."
        )

    return wilson_bounds(checked_successes, checked_trials, z=quantile)


def _as_count(value: Any, name: str) -> int:
    """Coerce a non-negative integer count, refusing bools and floats."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{name} must be an int, got {type(value).__name__}. Counts of "
            f"runs are integers; a float here is usually a rate that has "
            f"already been divided once."
        )
    count = int(value)
    if count < 0:
        raise ValueError(f"{name} must be non-negative, got {count}")
    return count


def _as_generator(value: Any, name: str) -> np.random.Generator:
    """Resolve one generator argument, keeping :func:`resolve_rng`'s rules."""
    if value is not None and not isinstance(value, np.random.Generator):
        raise TypeError(
            f"{name} must be a numpy.random.Generator or None, got "
            f"{type(value).__name__}. Integer seeds are refused throughout the "
            f"project; build a generator with numpy.random.default_rng."
        )
    return resolve_rng(value)


@dataclass(frozen=True)
class ForgeryMeasurement:
    """What one block of forgery trials produced, with its evidence kept.

    Parameters
    ----------
    label : str
        Which experiment this was, for a report row.
    party : Party
        The verifier being attacked.
    key_length : int
        The scored key length, i.e. ``params.signing_length``.
    trials : int
        Runs attempted.
    accepted : int
        Runs on which the target accepted the forged declaration.
    no_verdict : int
        Runs on which the target reached no verdict at all -- a matched-count
        floor refused the evidence base, or Phase C' refused the declaration's
        provenance.
    scored_positions : int
        Total matched positions summed over the runs that reached a verdict.
    mismatches : int
        Total disagreements within those matched positions.
    predicted : float or None
        The analytic acceptance probability for this experiment, or ``None``
        when no closed form applies.
    predicted_mismatch_rate : float or None
        The analytic per-scored-position mismatch rate.
    symmetrised : bool
        Whether Phase A' ran.
    bob_accepted : int
        How many runs the *forwarding* recipient's own verification accepted,
        i.e. how many runs were genuine forgeries rather than repudiations.

    Examples
    --------
    >>> from qsecure.attacks.forgery import ForgeryMeasurement
    >>> from qsecure.protocol.params import Party
    >>> block = ForgeryMeasurement(
    ...     label="outside forger",
    ...     party=Party.CHARLIE,
    ...     key_length=30,
    ...     trials=2000,
    ...     accepted=9,
    ...     no_verdict=0,
    ...     scored_positions=20000,
    ...     mismatches=10000,
    ...     predicted=0.004211,
    ...     predicted_mismatch_rate=0.5,
    ... )
    >>> round(block.rate, 5), block.agrees_with_prediction
    (0.0045, True)
    >>> block.mismatch_rate
    0.5
    """

    label: str
    party: Party
    key_length: int
    trials: int
    accepted: int
    no_verdict: int = 0
    scored_positions: int = 0
    mismatches: int = 0
    predicted: float | None = None
    predicted_mismatch_rate: float | None = None
    symmetrised: bool = True
    bob_accepted: int = 0

    def __post_init__(self) -> None:
        """Coerce the counts and refuse a self-contradictory block."""
        if not isinstance(self.label, str):
            raise TypeError(
                f"label must be a string, got {type(self.label).__name__}"
            )
        if not isinstance(self.party, (Party, str)):
            raise TypeError(
                f"party must be a Party or its label, got "
                f"{type(self.party).__name__}"
            )
        try:
            resolved = Party(self.party)
        except ValueError as error:
            raise ValueError(
                f"party names nobody in the protocol: {self.party!r}. The "
                f"verifiers are "
                f"{[member.value for member in VERIFIERS]}."
            ) from error
        if resolved is Party.ALICE:
            raise ValueError(
                "a forgery measurement is about a verifier's decision, and "
                "Alice reaches none. Pass Party.BOB or Party.CHARLIE."
            )
        object.__setattr__(self, "party", resolved)
        for name in (
            "key_length",
            "trials",
            "accepted",
            "no_verdict",
            "scored_positions",
            "mismatches",
            "bob_accepted",
        ):
            object.__setattr__(self, name, _as_count(getattr(self, name), name))
        if self.trials <= 0:
            raise ValueError(
                f"trials must be positive, got {self.trials}. A measurement "
                f"block with no runs in it has no rate to report."
            )
        if self.accepted + self.no_verdict > self.trials:
            raise ValueError(
                f"accepted ({self.accepted}) plus no_verdict "
                f"({self.no_verdict}) exceeds trials ({self.trials}). A run "
                f"ends in exactly one of accepted, rejected or no-verdict; "
                f"counting an aborted run as a rejection as well is the error "
                f"this check exists to catch."
            )
        if self.mismatches > self.scored_positions:
            raise ValueError(
                f"mismatches ({self.mismatches}) exceeds scored_positions "
                f"({self.scored_positions}); mismatches are counted within the "
                f"matched set only."
            )
        for name in ("predicted", "predicted_mismatch_rate"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.floating)
            ):
                raise TypeError(
                    f"{name} must be a float or None, got "
                    f"{type(value).__name__}"
                )
            number = float(value)
            if not 0.0 <= number <= 1.0:
                raise ValueError(
                    f"{name} must be a probability in [0, 1], got {number}"
                )
            object.__setattr__(self, name, number)
        if not isinstance(self.symmetrised, bool):
            raise TypeError(
                f"symmetrised must be a bool, got "
                f"{type(self.symmetrised).__name__}. It names which protocol "
                f"variant was run, not a rate."
            )

    @property
    def rejected(self) -> int:
        """int: Runs on which the target scored the forgery and refused it."""
        return self.trials - self.accepted - self.no_verdict

    @property
    def rate(self) -> float:
        """float: Acceptance rate, ``accepted / trials``."""
        return self.accepted / self.trials

    @property
    def interval(self) -> tuple[float, float]:
        """tuple of float: Two-sided 95% Wilson interval for :attr:`rate`."""
        return wilson_interval(self.accepted, self.trials)

    @property
    def agrees_with_prediction(self) -> bool:
        """bool: Whether :attr:`predicted` lies inside :attr:`interval`."""
        if self.predicted is None:
            return False
        low, high = self.interval
        return low <= self.predicted <= high

    @property
    def scored_runs(self) -> int:
        """int: Runs on which the target reached a verdict."""
        return self.trials - self.no_verdict

    @property
    def mismatch_rate(self) -> float:
        """float: Disagreements per scored position, ``e / |M|``."""
        if self.scored_positions == 0:
            return float("nan")
        return self.mismatches / self.scored_positions

    @property
    def mismatch_interval(self) -> tuple[float, float]:
        """tuple of float: 95% Wilson interval for :attr:`mismatch_rate`."""
        if self.scored_positions == 0:
            return (0.0, 1.0)
        return wilson_interval(self.mismatches, self.scored_positions)

    @property
    def scored_fraction(self) -> float:
        """float: Matched positions per key position, ``|M| / L``."""
        positions = self.scored_runs * self.key_length
        if positions == 0:
            return float("nan")
        return self.scored_positions / positions

    def mismatch_rate_within(self, target: float, tolerance: float) -> bool:
        """Say whether :attr:`mismatch_rate` is within ``tolerance`` of
        ``target``.

        Parameters
        ----------
        target : float
            The predicted rate, e.g. :data:`RECIPIENT_FORGER_MISMATCH_RATE`.
        tolerance : float
            Non-negative absolute tolerance.

        Returns
        -------
        bool
            ``False`` when nothing was scored, since ``nan`` is within nothing.

        Examples
        --------
        >>> from qsecure.attacks.forgery import ForgeryMeasurement
        >>> from qsecure.protocol.params import Party
        >>> block = ForgeryMeasurement(
        ...     "demo", Party.CHARLIE, 48, 10, 0,
        ...     scored_positions=320, mismatches=27,
        ... )
        >>> block.mismatch_rate_within(1 / 12, 0.02)
        True
        """
        return _within(self.mismatch_rate, target, tolerance)

    def scored_fraction_within(self, target: float, tolerance: float) -> bool:
        """Say whether :attr:`scored_fraction` is within ``tolerance``.

        Parameters
        ----------
        target : float
            The predicted fraction, ``2/3`` or ``1/3``.
        tolerance : float
            Non-negative absolute tolerance.

        Examples
        --------
        >>> from qsecure.attacks.forgery import ForgeryMeasurement
        >>> from qsecure.protocol.params import Party
        >>> block = ForgeryMeasurement(
        ...     "demo", Party.CHARLIE, 48, 10, 0, scored_positions=320
        ... )
        >>> block.scored_fraction_within(2 / 3, 0.02)
        True
        """
        return _within(self.scored_fraction, target, tolerance)

    def summary(self) -> str:
        """Return a compact human-readable block, one experiment per call.

        Returns
        -------
        str
            Five lines: the label, the acceptance rate with its interval and
            the prediction, the mismatch rate with its interval and prediction,
            the scored fraction, and the no-verdict count.

        Examples
        --------
        >>> from qsecure.attacks.forgery import ForgeryMeasurement
        >>> from qsecure.protocol.params import Party
        >>> print(ForgeryMeasurement(
        ...     "recipient forger", Party.CHARLIE, 48, 100, 34,
        ...     scored_positions=3200, mismatches=267,
        ...     predicted=0.371, predicted_mismatch_rate=1 / 12,
        ... ).summary())
        recipient forger vs Charlie, L=48, symmetrised, 100 trials
          accepted   0.34000  95% CI [0.25462, 0.43722]  predicted 0.371 (agrees)
          mismatch   0.08344  95% CI [0.07435, 0.09353]  predicted 0.083333 (agrees)
          scored     0.66667 of positions
          no verdict 0 runs, 66 rejections
        """
        low, high = self.interval
        mlow, mhigh = self.mismatch_interval
        accepted_note = (
            "no prediction"
            if self.predicted is None
            else (
                f"predicted {self.predicted:g} "
                f"({'agrees' if self.agrees_with_prediction else 'DISAGREES'})"
            )
        )
        if self.scored_positions == 0:
            mismatch_note = "nothing was scored"
        elif self.predicted_mismatch_rate is None:
            mismatch_note = "no prediction"
        else:
            inside = mlow <= self.predicted_mismatch_rate <= mhigh
            mismatch_note = (
                f"predicted {self.predicted_mismatch_rate:.5g} "
                f"({'agrees' if inside else 'DISAGREES'})"
            )
        return (
            f"{self.label} vs {self.party.value}, L={self.key_length}, "
            f"{'symmetrised' if self.symmetrised else 'NOT symmetrised'}, "
            f"{self.trials} trials\n"
            f"  accepted   {self.rate:.5f}  95% CI [{low:.5f}, {high:.5f}]  "
            f"{accepted_note}\n"
            f"  mismatch   {self.mismatch_rate:.5f}  95% CI "
            f"[{mlow:.5f}, {mhigh:.5f}]  {mismatch_note}\n"
            f"  scored     {self.scored_fraction:.5f} of positions\n"
            f"  no verdict {self.no_verdict} runs, {self.rejected} rejections"
        )


def _within(value: float, target: float, tolerance: float) -> bool:
    """Return ``abs(value - target) <= tolerance``, ``False`` at ``nan``."""
    if isinstance(tolerance, bool) or not isinstance(
        tolerance, (int, float, np.floating)
    ):
        raise TypeError(
            f"tolerance must be a real number, got {type(tolerance).__name__}"
        )
    if float(tolerance) < 0.0:
        raise ValueError(
            f"tolerance must be non-negative, got {float(tolerance)}. A "
            f"negative tolerance makes every comparison fail and reads as a "
            f"passing test that never passes."
        )
    if math.isnan(value):
        return False
    return abs(value - float(target)) <= float(tolerance)


class OutsideForger:
    """Eve: draws a fresh key, declares it, and knows nothing.

    Parameters
    ----------
    rng : numpy.random.Generator or None
        Her own generator, and the only randomness she uses.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.forgery import OutsideForger
    >>> from qsecure.attacks.isolation import assert_attack_isolated, signer_probe
    >>> assert_attack_isolated(OutsideForger, signer_probe()).isolated
    True

    >>> from qsecure.protocol.keys import generate_key_pair
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import NO_RECIPIENT_LOGS
    >>> params = ProtocolParams(key_length=8)
    >>> alice = generate_key_pair(params, rng=np.random.default_rng(1))
    >>> other = generate_key_pair(params, rng=np.random.default_rng(2))
    >>> first = OutsideForger(rng=np.random.default_rng(5))(
    ...     0, alice, params, records=NO_RECIPIENT_LOGS
    ... )
    >>> second = OutsideForger(rng=np.random.default_rng(5))(
    ...     0, other, params, records=NO_RECIPIENT_LOGS
    ... )
    >>> first == second, first.declared_key == alice[0]
    (True, False)
    """

    def __init__(self, *, rng: np.random.Generator | None = None) -> None:
        self._rng = _as_generator(rng, "rng")
        self.calls = 0

    def __repr__(self) -> str:
        """Return a debugging representation naming the call count."""
        return f"OutsideForger(calls={self.calls})"

    def __call__(
        self,
        message_bit: int,
        keys: Any,
        params: ProtocolParams,
        *,
        records: Any = None,
    ) -> Signature:
        """Declare a uniformly random key for ``message_bit``.

        Examples
        --------
        >>> import numpy as np
        >>> from qsecure.attacks.forgery import OutsideForger
        >>> from qsecure.protocol.params import ProtocolParams
        >>> from qsecure.protocol.session import NO_RECIPIENT_LOGS
        >>> forger = OutsideForger(rng=np.random.default_rng(0))
        >>> declaration = forger(
        ...     1, None, ProtocolParams(key_length=6),
        ...     records=NO_RECIPIENT_LOGS,
        ... )
        >>> declaration.message_bit, len(declaration), forger.calls
        (1, 6, 1)
        """
        del keys, records  # She holds neither.
        if not isinstance(params, ProtocolParams):
            raise TypeError(
                f"params must be a ProtocolParams, got {type(params).__name__}."
                f" The signer seam is handed the session's scored parameter "
                f"set; it supplies the declaration's length and alphabet."
            )
        self.calls += 1
        return Signature(
            message_bit,
            generate_private_key(params, message_bit, rng=self._rng),
        )


class RecipientForger:
    """Bob, forging to Charlie on the forwarding hop.

    Parameters
    ----------
    rng : numpy.random.Generator or None
        His own generator.
    guess_probability : float, optional
        Probability that a position is declared from an independent draw
        instead of his log; default ``0.0``.

    Examples
    --------
    >>> import numpy as np
    >>> from functools import partial
    >>> from qsecure.attacks.forgery import RecipientForger, forwarder_probe
    >>> from qsecure.attacks.isolation import (
    ...     assert_attack_isolated, check_attack_isolation
    ... )

    >>> randomised = partial(RecipientForger, guess_probability=0.25)
    >>> assert_attack_isolated(
    ...     randomised, forwarder_probe(), name="RecipientForger"
    ... ).isolated
    True

    >>> report = check_attack_isolation(RecipientForger, forwarder_probe())
    >>> report.reads_the_session, report.distinct_decisions
    (False, 1)
    """

    def __init__(
        self,
        *,
        rng: np.random.Generator | None = None,
        guess_probability: float = 0.0,
    ) -> None:
        self._rng = _as_generator(rng, "rng")
        if isinstance(guess_probability, bool) or not isinstance(
            guess_probability, (int, float, np.floating)
        ):
            raise TypeError(
                f"{_GUESS_LABEL} must be a real number, got "
                f"{type(guess_probability).__name__}"
            )
        probability = float(guess_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"{_GUESS_LABEL} must be a probability in [0, 1], got "
                f"{probability}. It is the fraction of positions the forger "
                f"throws away and replaces with an independent draw; 0.0 is "
                f"the optimal forger and the only value a measured rate should "
                f"be taken at."
            )
        self.guess_probability = probability
        self.calls = 0
        self.guessed_positions = 0

    def __repr__(self) -> str:
        """Return a debugging representation naming the knob and the traffic.
        """
        return (
            f"RecipientForger({_GUESS_LABEL}={self.guess_probability!r}, "
            f"calls={self.calls}, guessed={self.guessed_positions})"
        )

    def __call__(
        self,
        signature: Signature,
        params: ProtocolParams,
        *,
        view: RecipientView,
    ) -> Signature:
        """Substitute a declaration built from his own raw log.

        Examples
        --------
        >>> import numpy as np
        >>> from qsecure.attacks.forgery import RecipientForger
        >>> from qsecure.attacks.isolation import signer_scenario
        >>> from qsecure.protocol.params import Party
        >>> from qsecure.protocol.signature import Signature
        >>> scenario = signer_scenario()
        >>> view = scenario.views[Party.BOB]
        >>> honest = Signature(scenario.message_bit, scenario.keys[0])
        >>> forger = RecipientForger(rng=np.random.default_rng(3))
        >>> forged = forger(honest, scenario.params, view=view)
        >>> forged.message_bit == honest.message_bit
        True
        >>> forged.declared_key == honest.declared_key
        False

        >>> all(
        ...     element.basis == entry.basis
        ...     and element.eigenvalue == entry.eigenvalue
        ...     for element, entry in zip(
        ...         forged.declared_key, view.raw_record.entries
        ...     )
        ... )
        True
        """
        if not isinstance(signature, Signature):
            raise TypeError(
                f"signature must be a Signature, got "
                f"{type(signature).__name__}. The forwarder seam is handed the "
                f"declaration Bob received; a forgery replaces it with another "
                f"declaration, it does not remove it."
            )
        if not isinstance(params, ProtocolParams):
            raise TypeError(
                f"params must be a ProtocolParams, got {type(params).__name__}"
            )
        if not isinstance(view, RecipientView):
            raise TypeError(
                f"view must be a RecipientView, got {type(view).__name__}. "
                f"This adversary is a recipient and holds one recipient's "
                f"evidence; a mapping of both logs would put the target's own "
                f"record in his hands and measure a different adversary."
            )
        if view.party is not Party.BOB:
            raise ValueError(
                f"view belongs to {view.party.value}, but the forwarding hop "
                f"is Bob's: he is the party who receives Alice's declaration "
                f"and passes it on. A Charlie-flavoured forger attacks Bob and "
                f"needs a route of his own, not this seam."
            )
        if view.message_bit != signature.message_bit:
            raise ValueError(
                f"view is for message bit {view.message_bit} but the "
                f"declaration is for bit {signature.message_bit}. The two "
                f"distribution runs share no key and no evidence, so a forgery "
                f"built from the wrong run's log is scored against states that "
                f"were never sent for it."
            )

        elements = list(
            key_from_record(view.raw_record, message_bit=signature.message_bit)
        )
        if self.guess_probability > 0.0:
            elements = self._blend(elements, params)
        self.calls += 1
        return Signature(
            signature.message_bit,
            PrivateKey(signature.message_bit, elements),
        )

    def _blend(
        self, elements: list[KeyElement], params: ProtocolParams
    ) -> list[KeyElement]:
        """Replace a random subset of ``elements`` with independent draws."""
        bases = params.bases
        coins = self._rng.random(len(elements))
        blended = list(elements)
        for index, coin in enumerate(coins):
            if coin >= self.guess_probability:
                continue
            basis = bases[int(self._rng.integers(len(bases)))]
            eigenvalue = 1 if int(self._rng.integers(2)) == 0 else -1
            blended[index] = KeyElement(basis, eigenvalue)
            self.guessed_positions += 1
        return blended


def forwarder_probe(scenario: SignerScenario | None = None) -> DecisionProbe:
    """Return a :class:`~qsecure.attacks.isolation.DecisionProbe` for the hop.

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
    >>> from functools import partial
    >>> from qsecure.attacks.forgery import RecipientForger, forwarder_probe
    >>> from qsecure.attacks.isolation import assert_attack_isolated
    >>> assert_attack_isolated(
    ...     partial(RecipientForger, guess_probability=0.5),
    ...     forwarder_probe(),
    ...     name="RecipientForger",
    ... ).isolated
    True
    """
    resolved = signer_scenario() if scenario is None else scenario
    if not isinstance(resolved, SignerScenario):
        raise TypeError(
            f"scenario must be a SignerScenario or None, got "
            f"{type(resolved).__name__}. Build one with signer_scenario()."
        )
    view = resolved.views[Party.BOB]
    declaration = Signature(
        resolved.message_bit, resolved.keys[resolved.message_bit]
    )

    def probe(attack: Any, session_seed: int) -> Any:
        """Forward once against the frozen view; return what was declared."""
        del session_seed
        return attack(declaration, resolved.params, view=view)

    return probe


def _check_streams(
    rng: np.random.Generator | None, session_rng: np.random.Generator | None
) -> tuple[np.random.Generator, np.random.Generator]:
    """Resolve the two generators and refuse a shared stream."""
    attack = _as_generator(rng, "rng")
    session = _as_generator(session_rng, "session_rng")
    if rng is not None and session_rng is not None:
        require_distinct_streams(
            attack,
            session,
            left_name="rng",
            right_name="session_rng",
            detail=(
                "rng is the forger's own and session_rng builds the sessions "
                "he is measured against; nothing about him may be a function "
                "of the second."
            ),
        )
    return attack, session


def _as_trials(trials: Any) -> int:
    """Coerce and range-check a trial count."""
    count = _as_count(trials, "trials")
    if count <= 0:
        raise ValueError(
            f"trials must be positive, got {count}. A block of no runs "
            f"produces no rate and no interval; ask for at least one."
        )
    return count


def measure_outside_forgery(
    params: ProtocolParams,
    *,
    trials: int,
    rng: np.random.Generator | None = None,
    session_rng: np.random.Generator | None = None,
) -> dict[Party, ForgeryMeasurement]:
    """Run :class:`OutsideForger` on the signer seam and report both verifiers.

    Parameters
    ----------
    params : ProtocolParams
        The parameter set.
    trials : int
        Number of runs.
    rng : numpy.random.Generator or None, optional
        **The adversary's** generator.
    session_rng : numpy.random.Generator or None, optional
        The harness's generator, from which each
        :class:`~qsecure.protocol.session.QDSSession` draws its own material.

    Returns
    -------
    dict of Party to ForgeryMeasurement
        One block per verifier, keyed by
        :data:`~qsecure.protocol.params.VERIFIERS`, each carrying the exact
        prediction from :func:`~qsecure.protocol.analysis.forgery_probability`
        for that party's threshold.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.forgery import measure_outside_forgery
    >>> from qsecure.protocol.params import Party, ProtocolParams
    >>> blocks = measure_outside_forgery(
    ...     ProtocolParams(key_length=24),
    ...     trials=40,
    ...     rng=np.random.default_rng(21),
    ...     session_rng=np.random.default_rng(22),
    ... )
    >>> sorted(blocks) == list(VERIFIERS)
    True

    >>> blocks[Party.BOB].mismatch_rate_within(0.5, 0.1)
    True
    >>> blocks[Party.CHARLIE].scored_fraction_within(1 / 3, 0.05)
    True
    """
    if not isinstance(params, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(params).__name__}"
        )
    count = _as_trials(trials)
    attack_rng, harness_rng = _check_streams(rng, session_rng)

    scored = params.sifted() if params.has_check_rounds else params
    accepted = {party: 0 for party in VERIFIERS}
    no_verdict = {party: 0 for party in VERIFIERS}
    positions = {party: 0 for party in VERIFIERS}
    mismatches = {party: 0 for party in VERIFIERS}

    for _ in range(count):
        session = QDSSession(
            params,
            signer=OutsideForger(rng=attack_rng),
            rng=harness_rng,
        )
        transcript = session.run(0)
        for party in VERIFIERS:
            result = transcript.results_by_party.get(party)
            if result is None:
                no_verdict[party] += 1
                continue
            accepted[party] += int(result.accepted)
            positions[party] += result.matched_count
            mismatches[party] += result.mismatches

    return {
        party: ForgeryMeasurement(
            label="outside forger",
            party=party,
            key_length=scored.key_length,
            trials=count,
            accepted=accepted[party],
            no_verdict=no_verdict[party],
            scored_positions=positions[party],
            mismatches=mismatches[party],
            predicted=forgery_probability(scored, party=party),
            predicted_mismatch_rate=OUTSIDE_FORGER_MISMATCH_RATE,
        )
        for party in VERIFIERS
    }


def measure_recipient_forgery(
    params: ProtocolParams,
    *,
    trials: int,
    rng: np.random.Generator | None = None,
    session_rng: np.random.Generator | None = None,
    symmetrised: bool = True,
    pooled_counts: bool = False,
    guess_probability: float = 0.0,
) -> ForgeryMeasurement:
    """Run :class:`RecipientForger` on the forwarding hop and report Charlie.

    Parameters
    ----------
    params : ProtocolParams
        The parameter set.
    trials : int
        Number of runs.
    rng : numpy.random.Generator or None, optional
        **The adversary's** generator.
    session_rng : numpy.random.Generator or None, optional
        The harness's.
    symmetrised : bool, optional
        ``False`` runs the variant with Phase A' removed.
    pooled_counts : bool, optional
        ``True`` runs the shipped pooled rule, under which Charlie reaches no
        verdict on a substituted declaration.
    guess_probability : float, optional
        Passed to :class:`RecipientForger`.

    Returns
    -------
    ForgeryMeasurement
        Charlie's block, carrying the exact prediction from
        :func:`~qsecure.protocol.analysis.recipient_forgery_probability` for
        the arm that was run -- ``None`` when ``guess_probability`` is
        positive, because no closed form covers a deliberately degraded forger.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.forgery import measure_recipient_forgery
    >>> from qsecure.protocol.params import ProtocolParams
    >>> block = measure_recipient_forgery(
    ...     ProtocolParams(key_length=48),
    ...     trials=20,
    ...     rng=np.random.default_rng(31),
    ...     session_rng=np.random.default_rng(32),
    ... )
    >>> block.bob_accepted == block.trials
    True
    >>> block.mismatch_rate_within(1 / 12, 0.035)
    True

    >>> pooled = measure_recipient_forgery(
    ...     ProtocolParams(key_length=24),
    ...     trials=4,
    ...     rng=np.random.default_rng(33),
    ...     session_rng=np.random.default_rng(34),
    ...     pooled_counts=True,
    ... )
    >>> pooled.no_verdict == pooled.trials, pooled.accepted
    (True, 0)
    >>> pooled.predicted is None
    True
    """
    if not isinstance(params, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(params).__name__}"
        )
    for name, flag in (
        ("symmetrised", symmetrised),
        ("pooled_counts", pooled_counts),
    ):
        if not isinstance(flag, bool):
            raise TypeError(
                f"{name} must be a bool, got {type(flag).__name__}. It selects "
                f"which protocol variant is run, not a rate."
            )
    count = _as_trials(trials)
    attack_rng, harness_rng = _check_streams(rng, session_rng)

    scored = params.sifted() if params.has_check_rounds else params
    accepted = 0
    no_verdict = 0
    positions = 0
    mismatches = 0
    bob_accepted = 0

    for _ in range(count):
        session = QDSSession(
            params,
            forwarder=RecipientForger(
                rng=attack_rng, guess_probability=guess_probability
            ),
            symmetriser=None if symmetrised else no_symmetrisation,
            count_exchange=None if pooled_counts else no_count_exchange,
            rng=harness_rng,
        )
        transcript = session.run(0)
        bob = transcript.bob
        bob_accepted += int(bob is not None and bob.accepted)
        charlie = transcript.charlie
        if charlie is None:
            no_verdict += 1
            continue
        accepted += int(charlie.accepted)
        positions += charlie.matched_count
        mismatches += charlie.mismatches

    predictable = guess_probability <= 0.0 and not pooled_counts
    predicted = (
        recipient_forgery_probability(
            scored, party=Party.CHARLIE, symmetrised=symmetrised
        )
        if predictable
        else None
    )
    predicted_mismatch = (
        (
            RECIPIENT_FORGER_MISMATCH_RATE
            if symmetrised
            else UNSYMMETRISED_FORGER_MISMATCH_RATE
        )
        if predictable
        else None
    )
    return ForgeryMeasurement(
        label="recipient forger",
        party=Party.CHARLIE,
        key_length=scored.key_length,
        trials=count,
        accepted=accepted,
        no_verdict=no_verdict,
        scored_positions=positions,
        mismatches=mismatches,
        predicted=predicted,
        predicted_mismatch_rate=predicted_mismatch,
        symmetrised=symmetrised,
        bob_accepted=bob_accepted,
    )
