"""Unauthorised verification: three routes to a verdict nobody granted.

A party outside the round's authorised recipient set tries to reach a verdict
on Alice's declaration with a fabricated (U1), leaked (U2) or tapped (U3)
:class:`~qsecure.protocol.records.RecipientRecord`.

Examples
--------
>>> import numpy as np
>>> from qsecure.attacks.unauthorised import MEASUREMENT_PARAMS, run_unauthorised
>>> from qsecure.protocol.params import Party
>>> trials = run_unauthorised(
...     MEASUREMENT_PARAMS, session_seed=900_000, attack_rng=np.random.default_rng(7)
... )
>>> bob = trials[Party.BOB]
>>> bob.owner_accepted, bob.fabricated_accepted
(True, False)

>>> bob.fabricated_rate > 0.3
True
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from qsecure.attacks.channel import (
    PAULI_AXES,
    collapse_tensor,
    qber_from_tensor,
)
from qsecure.attacks.impersonation import session_seeds
from qsecure.attacks.isolation import DecisionProbe, derived_from_seed, same_stream
from qsecure.attacks.statistics import wilson_bounds
from qsecure.protocol.keys import generate_private_key
from qsecure.protocol.params import VERIFIERS, Party, ProtocolParams
from qsecure.protocol.records import RecipientRecord
from qsecure.protocol.session import QDSSession
from qsecure.protocol.signature import Signature, sign
from qsecure.protocol.verify import (
    MatchedSetTooSmall,
    VerificationResult,
    verify,
)

__all__ = [
    "ATTACK_SEED",
    "CONTROL_STREAM",
    "FABRICATED_MATCHED_FRACTION",
    "FABRICATED_MISMATCH_RATE",
    "MEASURED",
    "MEASUREMENT_PARAMS",
    "TAPPED_LINK_QBER",
    "UnauthorisedFabricator",
    "UnauthorisedHolder",
    "UnauthorisedMeasurement",
    "UnauthorisedTrial",
    "fabricator_probe",
    "measure_unauthorised",
    "run_unauthorised",
    "session_seeds",
    "shipped_summary",
    "tapped_record_reduction",
    "verdict_digest",
]


FABRICATED_MATCHED_FRACTION: Final[float] = 1.0 / 3.0
"""float: Share of positions a fabricated record is scored on, ``1 / |B|``."""

FABRICATED_MISMATCH_RATE: Final[float] = 0.5
"""float: A fabricated record's mismatch rate among scored positions."""

TAPPED_LINK_QBER: Final[float] = 1.0 / 3.0
"""float: QBER on a link whose travelling half was measured and re-sent."""

MEASUREMENT_PARAMS: Final[ProtocolParams] = ProtocolParams(key_length=192)
"""The parameter set :data:`MEASURED` was produced under."""

ATTACK_SEED: Final[int] = 4242
"""The attack generator's seed behind every row of :data:`MEASURED`."""

CONTROL_STREAM: Final[int] = 1
"""Stream label the harness folds in beside a session seed for its own draws.
"""


def _as_verifier(party: Party | str) -> Party:
    """Return ``party`` as one of the two recipients, refusing Alice."""
    try:
        resolved = Party(str(party))
    except ValueError:
        raise ValueError(
            f"unknown party {party!r}; expected Bob or Charlie. Those are the "
            f"only identities a record can carry, so an unauthorised verifier "
            f"has to claim one of them."
        ) from None
    if not resolved.is_verifier:
        raise ValueError(
            f"{resolved!s} keeps no record and reaches no verdict, so there is "
            f"no verification for anyone to attempt in her name."
        )
    return resolved


def _as_generator(value: Any, name: str) -> np.random.Generator:
    """Return ``value`` as a generator, refusing integer seeds."""
    if not isinstance(value, np.random.Generator):
        raise TypeError(
            f"{name} must be a numpy.random.Generator, got "
            f"{type(value).__name__}. an adversary owns its "
            f"randomness. An integer seed is refused because the natural next "
            f"step is to reuse the harness seed, and then every rate below is a "
            f"statement about one stream read twice."
        )
    return value


def _as_positive(value: Any, name: str) -> int:
    """Return ``value`` as a positive integer count."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    count = int(value)
    if count < 1:
        raise ValueError(
            f"{name} must be at least 1, got {count}. A rate with no "
            f"denominator is not a wide measurement, it is the absence of one."
        )
    return count


def _refuse_shared_stream(
    attack_rng: np.random.Generator, seeds: Sequence[int], *, where: str
) -> None:
    """Refuse an attack generator built from one of the session seeds.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.unauthorised import _refuse_shared_stream
    >>> _refuse_shared_stream(
    ...     np.random.default_rng(4242), (900_000,), where="demo"
    ... ) is None
    True
    >>> try:
    ...     _refuse_shared_stream(
    ...         np.random.default_rng(900_000), (900_000,), where="demo"
    ...     )
    ... except ValueError as error:
    ...     print(str(error).splitlines()[0])
    demo: attack_rng is the stream session seed 900000 produces.
    """
    for seed in seeds:
        if isinstance(seed, tuple):
            shared = same_stream(attack_rng, np.random.default_rng(seed))
        else:
            shared = derived_from_seed(attack_rng, int(seed))
        if shared:
            raise ValueError(
                f"{where}: attack_rng is the stream session seed {seed} "
                f"produces.\n"
                f"They are different objects and the same randomness. A "
                f"fabricator drawing from the session's stream can reproduce "
                f"the bases his record will be scored against, and the measured "
                f"mismatch rate would then describe a collusion rather than an "
                f"outsider.\n"
                f"Seed the adversary from something unrelated to the session "
                f"seeds; see ATTACK_SEED and session_seeds."
            )


def verdict_digest(result: VerificationResult) -> str:
    """Return a canonical text form of one verdict, for byte comparison.

    Parameters
    ----------
    result : VerificationResult
        A verdict.

    Returns
    -------
    str
        Sorted-key JSON of every field.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.unauthorised import verdict_digest
    >>> from qsecure.protocol.distribute import distribute_to_recipient
    >>> from qsecure.protocol.keys import generate_private_key
    >>> from qsecure.protocol.params import Party, ProtocolParams
    >>> from qsecure.protocol.signature import sign
    >>> from qsecure.protocol.verify import verify
    >>> params = ProtocolParams(key_length=24)
    >>> key = generate_private_key(params, 0, rng=np.random.default_rng(1))
    >>> record = distribute_to_recipient(
    ...     key, params, party=Party.BOB, rng=np.random.default_rng(2)
    ... )
    >>> digest = verdict_digest(verify(sign(0, key, params), record, params))
    >>> print(digest)  # doctest: +NORMALIZE_WHITESPACE
    {"accepted": true, "key_length": 24, "matched_count": 7, "message_bit": 0,
     "mismatches": 0, "party": "Bob", "rate": 0.0, "threshold": 0.015625}
    """
    if not isinstance(result, VerificationResult):
        raise TypeError(
            f"result must be a VerificationResult, got {type(result).__name__}"
        )
    return json.dumps(
        dataclasses.asdict(result), sort_keys=True, separators=(", ", ": ")
    )


class UnauthorisedFabricator:
    """The party who holds no distribution data and invents a record.

    Parameters
    ----------
    rng : numpy.random.Generator
        The fabricator's own generator.
    claims : Party or str, optional
        The identity he writes into the record, defaulting to
        :attr:`~qsecure.protocol.params.Party.BOB`.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.unauthorised import UnauthorisedFabricator
    >>> from qsecure.protocol.params import Party, ProtocolParams
    >>> params = ProtocolParams(key_length=12)
    >>> outsider = UnauthorisedFabricator(rng=np.random.default_rng(5))
    >>> record = outsider.fabricate(params, message_bit=0)
    >>> len(record), record.party, outsider.records_built
    (12, <Party.BOB: 'Bob'>, 1)

    >>> twin = UnauthorisedFabricator(rng=np.random.default_rng(5))
    >>> twin.fabricate(params, message_bit=0) == record
    True

    >>> impostor = UnauthorisedFabricator(
    ...     rng=np.random.default_rng(6), claims=Party.CHARLIE
    ... )
    >>> impostor.fabricate(params, message_bit=0).party
    <Party.CHARLIE: 'Charlie'>
    """

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        claims: Party | str = Party.BOB,
    ) -> None:
        self._rng = _as_generator(rng, "rng")
        self.claims = _as_verifier(claims)
        self.records_built = 0

    def __repr__(self) -> str:
        """Return a debugging representation naming the claim and the traffic.
        """
        return (
            f"UnauthorisedFabricator(claims={self.claims!s}, "
            f"records_built={self.records_built})"
        )

    def fabricate(
        self,
        params: ProtocolParams,
        *,
        message_bit: int,
        session_id: str | None = None,
    ) -> RecipientRecord:
        """Write one invented log at the scored length.

        Parameters
        ----------
        params : ProtocolParams
            The *scored* parameter set.
        message_bit : int
            ``0`` or ``1``.
        session_id : str or None, optional
            The round identifier he stamps the log with, or ``None`` for a log
            naming no round.

        Returns
        -------
        RecipientRecord
            Flagged ``symmetrised=True``, because a log claiming to have come
            through Phase A' is what a recipient of this round would hold and
            the flag is provenance he is free to write.

        Examples
        --------
        >>> import numpy as np
        >>> from qsecure.attacks.unauthorised import UnauthorisedFabricator
        >>> from qsecure.protocol.params import ProtocolParams
        >>> record = UnauthorisedFabricator(
        ...     rng=np.random.default_rng(2)
        ... ).fabricate(ProtocolParams(key_length=9), message_bit=1)
        >>> record.message_bit, record.symmetrised, record.session_id is None
        (1, True, True)
        """
        if not isinstance(params, ProtocolParams):
            raise TypeError(
                f"params must be a ProtocolParams, got {type(params).__name__}. "
                f"Pass the scored set: a record at the unsifted length is "
                f"refused by verify() before any counting happens."
            )
        if isinstance(message_bit, bool) or message_bit not in (0, 1):
            raise ValueError(
                f"message_bit must be 0 or 1, got {message_bit!r}. The two "
                f"distribution runs share no key, so a log has to claim one of "
                f"them."
            )
        bases = params.bases
        length = params.key_length
        chosen = self._rng.integers(len(bases), size=length)
        coins = self._rng.integers(2, size=length)
        self.records_built += 1
        return RecipientRecord.from_measurements(
            self.claims,
            int(message_bit),
            [bases[int(index)] for index in chosen],
            [1 if int(coin) == 0 else -1 for coin in coins],
            symmetrised=True,
            session_id=session_id,
        )


def fabricator_probe(
    params: ProtocolParams = ProtocolParams(key_length=12),
) -> DecisionProbe:
    """Return a :class:`~qsecure.attacks.isolation.DecisionProbe` for U1.

    Parameters
    ----------
    params : ProtocolParams, optional
        A small parameter set; the probe runs no session, so this only sets how
        many entries are compared.

    Returns
    -------
    DecisionProbe
        ``(attack, session_seed) -> tuple of tuple``, his fabricated entries.

    Examples
    --------
    >>> from qsecure.attacks.unauthorised import (
    ...     UnauthorisedFabricator, fabricator_probe)
    >>> from qsecure.attacks.isolation import assert_attack_isolated
    >>> assert_attack_isolated(
    ...     UnauthorisedFabricator, fabricator_probe()
    ... ).isolated
    True
    """
    if not isinstance(params, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(params).__name__}"
        )

    def probe(attack: Any, session_seed: int) -> Any:
        """Fabricate once; report the log."""
        del session_seed
        record = attack.fabricate(params, message_bit=0)
        return tuple(
            (entry.basis.value, entry.eigenvalue) for entry in record.entries
        )

    return probe


class UnauthorisedHolder:
    """The party who holds a genuine recipient's log, obtained outside the
    protocol.

    Parameters
    ----------
    record : RecipientRecord
        The stolen log, verbatim.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.unauthorised import UnauthorisedHolder, verdict_digest
    >>> from qsecure.protocol.distribute import distribute_to_recipient
    >>> from qsecure.protocol.keys import generate_private_key
    >>> from qsecure.protocol.params import Party, ProtocolParams
    >>> from qsecure.protocol.signature import sign
    >>> from qsecure.protocol.verify import verify
    >>> params = ProtocolParams(key_length=24)
    >>> key = generate_private_key(params, 0, rng=np.random.default_rng(1))
    >>> owned = distribute_to_recipient(
    ...     key, params, party=Party.BOB, rng=np.random.default_rng(2)
    ... )
    >>> declaration = sign(0, key, params)
    >>> thief = UnauthorisedHolder(owned)
    >>> stolen = thief.verify(declaration, params)
    >>> stolen.accepted, stolen.matched_count, stolen.mismatches
    (True, 7, 0)

    >>> verdict_digest(stolen) == verdict_digest(verify(declaration, owned, params))
    True

    >>> thief.claims
    <Party.BOB: 'Bob'>
    """

    def __init__(self, record: RecipientRecord) -> None:
        if not isinstance(record, RecipientRecord):
            raise TypeError(
                f"record must be a RecipientRecord, got {type(record).__name__}."
                f" This adversary is defined by holding a genuine one; a "
                f"fabricated log is UnauthorisedFabricator's route and has a "
                f"different answer."
            )
        self.record = record
        self.verifications = 0

    def __repr__(self) -> str:
        """Return a debugging representation naming whose log he holds."""
        return (
            f"UnauthorisedHolder(claims={self.claims!s}, "
            f"verifications={self.verifications})"
        )

    @property
    def claims(self) -> Party:
        """Party: The identity on the stolen record, which is its owner's."""
        return self.record.party

    def verify(
        self, signature: Signature, params: ProtocolParams
    ) -> VerificationResult:
        """Reach the owner's verdict on ``signature``.

        Parameters
        ----------
        signature : Signature
            The declaration to score.
        params : ProtocolParams
            The scored parameter set.

        Returns
        -------
        VerificationResult
            Field for field what the owner would have got.

        Raises
        ------
        TypeError, ValueError, MatchedSetTooSmall
            Exactly as :func:`~qsecure.protocol.verify.verify`, since that is
            what this is.
        """
        self.verifications += 1
        return verify(signature, self.record, params)


@dataclass(frozen=True)
class UnauthorisedTrial:
    """One session, one verifier, reduced to the numbers the three routes need.

    Parameters
    ----------
    party : Party
        The recipient whose identity is in play.
    session_seed : int
        The harness's seed for this run, recorded so a surprising trial replays
        exactly.
    key_length : int
        ``L``.
    message_bit : int
        The bit signed.
    owner_accepted : bool
        Whether the genuine record accepted the genuine declaration.
    fabricated_accepted : bool or None
        The fabricator's verdict, or ``None`` if his invented log did not clear
        the matched-count floor and he reached none.
    fabricated_matched, fabricated_mismatches : int or None
        ``|M|`` and ``e`` for the fabricated record against the genuine
        declaration.
    forged_matched, forged_mismatches : int or None
        The same fabricated record against a declaration nobody distributed
        states for, and ``None`` on the same terms: that arm can be refused the
        floor independently of the other, so it has its own refusal to report.

    Examples
    --------
    >>> from qsecure.attacks.unauthorised import UnauthorisedTrial
    >>> from qsecure.protocol.params import Party
    >>> trial = UnauthorisedTrial(
    ...     party=Party.BOB, session_seed=900_000, key_length=192,
    ...     message_bit=0, owner_accepted=True, fabricated_accepted=False,
    ...     fabricated_matched=64, fabricated_mismatches=32,
    ...     forged_matched=66, forged_mismatches=33,
    ... )
    >>> trial.fabricated_rate, trial.reached_verdict, trial.forged_scored
    (0.5, True, True)
    """

    party: Party
    session_seed: int
    key_length: int
    message_bit: int
    owner_accepted: bool
    fabricated_accepted: bool | None
    fabricated_matched: int | None
    fabricated_mismatches: int | None
    forged_matched: int | None
    forged_mismatches: int | None

    @property
    def reached_verdict(self) -> bool:
        """bool: Whether the fabricated log cleared the floor and was scored.
        """
        return self.fabricated_accepted is not None

    @property
    def forged_scored(self) -> bool:
        """bool: The same, for the arm scored against the forged declaration.
        """
        return self.forged_matched is not None

    @property
    def fabricated_rate(self) -> float | None:
        """float or None: ``e / |M|`` on the genuine declaration."""
        if not self.fabricated_matched:
            return None
        return (self.fabricated_mismatches or 0) / self.fabricated_matched

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping of every field."""
        row = dataclasses.asdict(self)
        row["party"] = str(self.party)
        return row


@dataclass(frozen=True)
class UnauthorisedMeasurement:
    """A whole arm: many trials against one verifier, pooled, with intervals.

    Parameters
    ----------
    party : Party
        The recipient whose identity is in play.
    key_length : int
        ``L``, the same for every trial in the arm.
    trials : int
        Sessions run.
    owner_accepted : int
        Trials on which the genuine record accepted the genuine declaration.
    fabricated_accepted : int
        Trials on which the fabricated record accepted the genuine declaration.
    fabricated_no_verdict : int
        Trials on which the fabricated log failed the matched-count floor.
    fabricated_matched, fabricated_mismatches : int
        Pooled over the trials that reached a verdict.
    forged_no_verdict : int
        The same refusal count for the forged-declaration arm, which is not
        implied by :attr:`fabricated_no_verdict`: the floor is a property of a
        declaration and a record together, so one arm can be refused where the
        other was scored.
    forged_matched, forged_mismatches : int
        Pooled the same way, against a declaration nobody distributed states
        for.

    Examples
    --------
    >>> from qsecure.attacks.unauthorised import MEASURED
    >>> from qsecure.protocol.params import Party
    >>> arm = MEASURED[Party.BOB]
    >>> arm.trials, arm.owner_accepted, arm.fabricated_accepted
    (200, 200, 0)
    >>> arm.verdict_independent_of_declaration
    True
    """

    party: Party
    key_length: int
    trials: int
    owner_accepted: int
    fabricated_accepted: int
    fabricated_no_verdict: int
    fabricated_matched: int
    fabricated_mismatches: int
    forged_no_verdict: int
    forged_matched: int
    forged_mismatches: int

    def __post_init__(self) -> None:
        """Coerce the party and refuse an arm whose counts cannot be real."""
        object.__setattr__(self, "party", _as_verifier(self.party))
        object.__setattr__(
            self, "key_length", _as_positive(self.key_length, "key_length")
        )
        object.__setattr__(self, "trials", _as_positive(self.trials, "trials"))
        for name in (
            "owner_accepted",
            "fabricated_accepted",
            "fabricated_no_verdict",
            "forged_no_verdict",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(
                    f"{name} must be an int, got {type(value).__name__}"
                )
            if not 0 <= int(value) <= self.trials:
                raise ValueError(
                    f"{name} must satisfy 0 <= n <= trials, got {int(value)} of "
                    f"{self.trials}"
                )
            object.__setattr__(self, name, int(value))
        for errors, matched, label in (
            (self.fabricated_mismatches, self.fabricated_matched, "fabricated"),
            (self.forged_mismatches, self.forged_matched, "forged"),
        ):
            if isinstance(matched, bool) or not isinstance(
                matched, (int, np.integer)
            ):
                raise TypeError(f"{label}_matched must be an int")
            if isinstance(errors, bool) or not isinstance(
                errors, (int, np.integer)
            ):
                raise TypeError(f"{label}_mismatches must be an int")
            if not 0 <= int(errors) <= int(matched):
                raise ValueError(
                    f"{label}_mismatches must satisfy 0 <= e <= |M|, got "
                    f"{int(errors)} of {int(matched)}; mismatches are counted "
                    f"within the matched set only."
                )
            if int(matched) > self.trials * self.key_length:
                raise ValueError(
                    f"{label}_matched is {int(matched)}, which is more "
                    f"positions than {self.trials} trials of length "
                    f"{self.key_length} contain."
                )

    @property
    def scored_trials(self) -> int:
        """int: Trials on which the fabricated log was scored."""
        return self.trials - self.fabricated_no_verdict

    @property
    def fabricated_acceptance_rate(self) -> float:
        """float: How often the fabricated record accepted, over all trials."""
        return self.fabricated_accepted / self.trials

    @property
    def fabricated_acceptance_interval(self) -> tuple[float, float]:
        """tuple of float: 95% Wilson interval for the acceptance rate."""
        return wilson_bounds(self.fabricated_accepted, self.trials)

    @property
    def matched_fraction(self) -> float:
        """float: ``|M| / L`` for the fabricated record."""
        positions = self.scored_trials * self.key_length
        if positions == 0:
            return float("nan")
        return self.fabricated_matched / positions

    @property
    def fabricated_rate(self) -> float:
        """float: Mismatch rate on the genuine declaration."""
        if self.fabricated_matched == 0:
            return float("nan")
        return self.fabricated_mismatches / self.fabricated_matched

    @property
    def fabricated_interval(self) -> tuple[float, float]:
        """tuple of float: 95% Wilson interval for :attr:`fabricated_rate`."""
        if self.fabricated_matched == 0:
            return (0.0, 1.0)
        return wilson_bounds(self.fabricated_mismatches, self.fabricated_matched)

    @property
    def forged_rate(self) -> float:
        """float: The same record's rate on a declaration Alice never made."""
        if self.forged_matched == 0:
            return float("nan")
        return self.forged_mismatches / self.forged_matched

    @property
    def forged_interval(self) -> tuple[float, float]:
        """tuple of float: 95% Wilson interval for :attr:`forged_rate`."""
        if self.forged_matched == 0:
            return (0.0, 1.0)
        return wilson_bounds(self.forged_mismatches, self.forged_matched)

    @property
    def verdict_independent_of_declaration(self) -> bool:
        """bool: Whether the two mismatch intervals overlap."""
        if self.fabricated_no_verdict or self.forged_no_verdict:
            return False
        if self.fabricated_matched == 0 or self.forged_matched == 0:
            return False
        low, high = self.fabricated_interval
        other_low, other_high = self.forged_interval
        return low <= other_high and other_low <= high

    def summary(self) -> str:
        """Return the arm as one quotable line, ``L`` and ``n`` included.

        Examples
        --------
        >>> from qsecure.attacks.unauthorised import MEASURED
        >>> from qsecure.protocol.params import Party
        >>> print(MEASURED[Party.CHARLIE].summary())
        Charlie L=192 n=200: owner 200/200, fabricated 0/200 accepted at r=0.5043 (forged r=0.5042)
        """
        return (
            f"{self.party!s} L={self.key_length} n={self.trials}: "
            f"owner {self.owner_accepted}/{self.trials}, "
            f"fabricated {self.fabricated_accepted}/{self.trials} accepted at "
            f"r={self.fabricated_rate:.4f} (forged r={self.forged_rate:.4f})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping of the arm's headline numbers."""
        row = dataclasses.asdict(self)
        row["party"] = str(self.party)
        row["matched_fraction"] = self.matched_fraction
        row["fabricated_rate"] = self.fabricated_rate
        row["fabricated_interval"] = list(self.fabricated_interval)
        row["forged_rate"] = self.forged_rate
        row["forged_interval"] = list(self.forged_interval)
        row["verdict_independent_of_declaration"] = (
            self.verdict_independent_of_declaration
        )
        return row


def run_unauthorised(
    params: ProtocolParams = MEASUREMENT_PARAMS,
    *,
    session_seed: int,
    attack_rng: np.random.Generator,
    message_bit: int = 0,
) -> dict[Party, UnauthorisedTrial]:
    """Run one honest session and attempt verification on it three ways.

    Parameters
    ----------
    params : ProtocolParams, optional
        Defaults to :data:`MEASUREMENT_PARAMS`.
    session_seed : int
        The harness's seed, and never the adversary's.
    attack_rng : numpy.random.Generator
        The fabricator's own generator, advanced by the call so that a loop
        gives each trial a fresh invented log.
    message_bit : int, optional
        ``0`` or ``1``.

    Returns
    -------
    dict of Party to UnauthorisedTrial
        One row per verifier, keyed by
        :data:`~qsecure.protocol.params.VERIFIERS`.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.unauthorised import MEASUREMENT_PARAMS, run_unauthorised
    >>> from qsecure.protocol.params import Party
    >>> rows = run_unauthorised(
    ...     MEASUREMENT_PARAMS, session_seed=900_001,
    ...     attack_rng=np.random.default_rng(4242),
    ... )
    >>> sorted(rows) == [Party.BOB, Party.CHARLIE]
    True
    >>> rows[Party.CHARLIE].owner_accepted
    True
    >>> rows[Party.CHARLIE].fabricated_accepted
    False
    """
    if not isinstance(params, ProtocolParams):
        raise TypeError(
            f"params must be a ProtocolParams, got {type(params).__name__}"
        )
    _as_generator(attack_rng, "attack_rng")
    if isinstance(message_bit, bool) or message_bit not in (0, 1):
        raise ValueError(f"message_bit must be 0 or 1, got {message_bit!r}")
    _refuse_shared_stream(
        attack_rng,
        (int(session_seed), (int(session_seed), CONTROL_STREAM)),
        where="run_unauthorised",
    )

    bit = int(message_bit)
    session = QDSSession(params, rng=np.random.default_rng(int(session_seed)))
    session.distribute()
    declaration = session.sign(bit)
    scored = session.scored_params
    forged = sign(
        bit,
        generate_private_key(
            scored,
            bit,
            rng=np.random.default_rng((int(session_seed), CONTROL_STREAM)),
        ),
        scored,
        session_opening=session.opening_for(bit),
    )

    rows: dict[Party, UnauthorisedTrial] = {}
    for party in VERIFIERS:
        owned = session.records[bit][party]
        control = verify(declaration, owned, scored)

        outsider = UnauthorisedFabricator(rng=attack_rng, claims=party)
        invented = outsider.fabricate(
            scored, message_bit=bit, session_id=declaration.session_id
        )
        genuine_arm = _score(declaration, invented, scored)
        forged_arm = _score(forged, invented, scored)

        rows[party] = UnauthorisedTrial(
            party=party,
            session_seed=int(session_seed),
            key_length=scored.key_length,
            message_bit=bit,
            owner_accepted=control.accepted,
            fabricated_accepted=genuine_arm[0],
            fabricated_matched=genuine_arm[1],
            fabricated_mismatches=genuine_arm[2],
            forged_matched=forged_arm[1],
            forged_mismatches=forged_arm[2],
        )
    return rows


def _score(
    declaration: Signature, record: RecipientRecord, params: ProtocolParams
) -> tuple[bool | None, int | None, int | None]:
    """Score one declaration against one record, reporting a refusal as
    ``None``.
    """
    try:
        result = verify(declaration, record, params)
    except MatchedSetTooSmall:
        return (None, None, None)
    return (result.accepted, result.matched_count, result.mismatches)


def measure_unauthorised(
    params: ProtocolParams = MEASUREMENT_PARAMS,
    *,
    seeds: Sequence[int] | None = None,
    trials: int = 200,
    attack_rng: np.random.Generator,
    message_bit: int = 0,
) -> dict[Party, UnauthorisedMeasurement]:
    """Run the whole experiment and return one pooled arm per verifier.

    Parameters
    ----------
    params : ProtocolParams, optional
        Defaults to :data:`MEASUREMENT_PARAMS`.
    seeds : sequence of int or None, optional
        The session seeds, one per trial.
    trials : int, optional
        How many seeds to generate when ``seeds`` is ``None``.
    attack_rng : numpy.random.Generator
        The fabricator's generator, advanced across the loop.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.attacks.unauthorised import MEASUREMENT_PARAMS, measure_unauthorised
    >>> from qsecure.protocol.params import Party
    >>> arms = measure_unauthorised(
    ...     MEASUREMENT_PARAMS, trials=4, attack_rng=np.random.default_rng(4242)
    ... )
    >>> bob = arms[Party.BOB]
    >>> bob.owner_accepted, bob.fabricated_accepted
    (4, 0)
    >>> bob.fabricated_no_verdict, bob.forged_no_verdict
    (0, 0)
    """
    if not isinstance(attack_rng, np.random.Generator):
        raise TypeError(
            f"attack_rng must be a numpy.random.Generator, got "
            f"{type(attack_rng).__name__}"
        )
    chosen = (
        session_seeds(trials)
        if seeds is None
        else tuple(int(seed) for seed in seeds)
    )
    if not chosen:
        raise ValueError("seeds must hold at least one session seed")
    _refuse_shared_stream(attack_rng, chosen, where="measure_unauthorised")

    rows: dict[Party, list[UnauthorisedTrial]] = {
        party: [] for party in VERIFIERS
    }
    for seed in chosen:
        for party, trial in run_unauthorised(
            params,
            session_seed=seed,
            attack_rng=attack_rng,
            message_bit=message_bit,
        ).items():
            rows[party].append(trial)
    return {party: _pool(party, trials) for party, trials in rows.items()}


def _pool(
    party: Party, trials: Sequence[UnauthorisedTrial]
) -> UnauthorisedMeasurement:
    """Add up one verifier's trials into an arm."""
    if not trials:
        raise ValueError(
            "an UnauthorisedMeasurement needs at least one trial: a rate with "
            "no denominator is not a measurement."
        )
    length = trials[0].key_length
    for trial in trials:
        if trial.key_length != length:
            raise ValueError(
                f"trial at session seed {trial.session_seed} ran at "
                f"L = {trial.key_length}, but this arm is L = {length}."
            )
    return UnauthorisedMeasurement(
        party=party,
        key_length=length,
        trials=len(trials),
        owner_accepted=sum(1 for row in trials if row.owner_accepted),
        fabricated_accepted=sum(
            1 for row in trials if row.fabricated_accepted
        ),
        fabricated_no_verdict=sum(
            1 for row in trials if not row.reached_verdict
        ),
        fabricated_matched=sum((row.fabricated_matched or 0) for row in trials),
        fabricated_mismatches=sum(
            (row.fabricated_mismatches or 0) for row in trials
        ),
        forged_no_verdict=sum(1 for row in trials if not row.forged_scored),
        forged_matched=sum((row.forged_matched or 0) for row in trials),
        forged_mismatches=sum((row.forged_mismatches or 0) for row in trials),
    )


MEASURED: Final[Mapping[Party, UnauthorisedMeasurement]] = {
    Party.BOB: UnauthorisedMeasurement(
        party=Party.BOB,
        key_length=192,
        trials=200,
        owner_accepted=200,
        fabricated_accepted=0,
        fabricated_no_verdict=0,
        fabricated_matched=12883,
        fabricated_mismatches=6434,
        forged_no_verdict=0,
        forged_matched=12864,
        forged_mismatches=6499,
    ),
    Party.CHARLIE: UnauthorisedMeasurement(
        party=Party.CHARLIE,
        key_length=192,
        trials=200,
        owner_accepted=200,
        fabricated_accepted=0,
        fabricated_no_verdict=0,
        fabricated_matched=12774,
        fabricated_mismatches=6442,
        forged_no_verdict=0,
        forged_matched=12649,
        forged_mismatches=6378,
    ),
}


def shipped_summary() -> tuple[str, ...]:
    """Return :data:`MEASURED` as two quotable lines, one per verifier.

    Returns
    -------
    tuple of str
        Bob's line, then Charlie's.

    Examples
    --------
    >>> from qsecure.attacks.unauthorised import shipped_summary
    >>> for line in shipped_summary():
    ...     print(line)
    Bob L=192 n=200: owner 200/200, fabricated 0/200 accepted at r=0.4994 (forged r=0.5052)
    Charlie L=192 n=200: owner 200/200, fabricated 0/200 accepted at r=0.5043 (forged r=0.5042)

    >>> from qsecure.attacks.unauthorised import MEASURED
    >>> from qsecure.protocol.params import Party
    >>> arm = MEASURED[Party.BOB]
    >>> low, high = arm.fabricated_interval
    >>> f"[{low:.4f}, {high:.4f}]", arm.verdict_independent_of_declaration
    ('[0.4908, 0.5081]', True)

    >>> f"{arm.matched_fraction:.4f}"
    '0.3355'
    """
    return tuple(MEASURED[party].summary() for party in VERIFIERS)


def tapped_record_reduction() -> dict[str, Any]:
    """Return the U3 argument as data, with its citations and its one number.

    Returns
    -------
    dict
        Keys ``"claim"``, ``"mechanism"``, ``"reduces_to"``, ``"detected_by"``,
        ``"measured"``, ``"argued"`` and ``"qber"``.

    Examples
    --------
    >>> from qsecure.attacks.unauthorised import tapped_record_reduction
    >>> reduction = tapped_record_reduction()
    >>> reduction["reduces_to"]
    'qsecure.attacks.channel.InterceptResend'
    >>> round(reduction["qber"], 6)
    0.333333

    >>> import textwrap
    >>> print(textwrap.fill(reduction["argued"], 70))
    No-cloning leaves no route to an eigenvalue except measuring for it,
    so a party building a record off the wire is an intercept-resend
    adversary. That step is an argument, not a measurement.
    """
    return {
        "claim": (
            "A party who taps Phase A to build his own record is already "
            "covered by the channel threshold family."
        ),
        "mechanism": (
            "Measuring one half of a Bell pair collapses both, so an "
            "eigenvalue learned off the wire is an eigenvalue destroyed on it."
        ),
        "reduces_to": "qsecure.attacks.channel.InterceptResend",
        "detected_by": (
            "check-round QBER and CHSH, qsecure.protocol.checkrounds, sized to "
            "resolve s_a = 1/64 to a quarter of itself"
        ),
        "measured": (
            "qber_from_tensor(collapse_tensor(PAULI_AXES)), the QBER the "
            "collapsed resource leaves on the tapped link"
        ),
        "argued": (
            "No-cloning leaves no route to an eigenvalue except measuring for "
            "it, so a party building a record off the wire is an "
            "intercept-resend adversary. That step is an argument, not a "
            "measurement."
        ),
        "qber": qber_from_tensor(collapse_tensor(PAULI_AXES)),
    }


def _self_check() -> None:
    """Run the arithmetic this module claims, cheaply, and raise on a surprise.
    """
    arms = measure_unauthorised(
        MEASUREMENT_PARAMS,
        trials=8,
        attack_rng=np.random.default_rng(ATTACK_SEED),
    )
    for party, arm in arms.items():
        assert arm.owner_accepted == arm.trials, party
        assert arm.fabricated_accepted == 0, party
        assert arm.fabricated_no_verdict == 0, party
        assert arm.forged_no_verdict == 0, party
        assert math.isclose(
            arm.fabricated_rate, FABRICATED_MISMATCH_RATE, abs_tol=0.12
        ), party
        assert math.isclose(
            arm.matched_fraction, FABRICATED_MATCHED_FRACTION, abs_tol=0.06
        ), party
    assert math.isclose(tapped_record_reduction()["qber"], TAPPED_LINK_QBER)
    print(*shipped_summary(), sep="\n")


if __name__ == "__main__":  # pragma: no cover
    _self_check()
