"""One request in, one scored run out: the dashboard's whole execution path.

An honest run, end to end, and the same request twice:
>>> from qsecure.web.driver import RunRequest, run_once
>>> request = RunRequest.from_mapping(
...     {"attack": "honest", "key_length": 96, "check_fraction": 0.25,
...      "seed": 7}
... )
>>> first = run_once(request)
>>> first.error is None, first.detection["detected"]
(True, False)
>>> second = run_once(request)
>>> first.detection == second.detection
True

>>> replayed = run_once(
...     RunRequest.from_mapping({"attack": "replay", "key_length": 96, "seed": 3})
... )
>>> replayed.ground_truth["replay"]["replays"]
1
>>> replayed.ground_truth["acted"], replayed.detection["detected"]
(True, True)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Mapping

import numpy as np

from qsecure.attacks.channel import (
    DepolarisingChannel,
    InterceptResend,
    KeptShareSwap,
    qber_from_tensor,
)
from qsecure.attacks.forgery import OutsideForger, RecipientForger
from qsecure.attacks.impersonation import Impersonator, impersonation_seams
from qsecure.attacks.isolation import require_distinct_streams
from qsecure.attacks.replay import ReplayingForwarder, replay_capture
from qsecure.attacks.starvation import CountStarver
from qsecure.detect import detect
from qsecure.detect.statistics import TranscriptStatistics
from qsecure.protocol.params import Party, ProtocolParams
from qsecure.protocol.session import QDSSession
from qsecure.web.catalogue import ATTACK_KEYS, AttackSpec, attack_spec
from qsecure.web.limits import (
    CHANNEL_ERROR_RATE_MAX,
    EPS_DEFAULT,
    NOISE_MAX,
    RequestRefused,
    check_check_fraction,
    check_eps,
    check_key_length,
    check_probability,
    check_seed,
    check_timing,
    safe_text,
)
from qsecure.web.payload import ground_truth_for, nulls_stated, run_facts


__all__ = [
    "MESSAGE_BIT",
    "RunRequest",
    "RunResult",
    "run_once",
]


#: Domain tag for the session's stream; distinct from the adversaries'.
_SESSION_DOMAIN: Final[int] = 0x51_9E_55_10

#: Domain tag for the adversaries' streams.
_ATTACK_DOMAIN: Final[int] = 0xAD_5E_A5_17

#: Seed of the captured replay round; fixed, independent of the request seed.
_CAPTURE_SEED: Final[int] = 0x_CA_97_00_1E

#: The bit every dashboard run signs (the request carries no message bit).
MESSAGE_BIT: Final[int] = 0

#: Per ordering: the two orderings answer a substituted declaration differently.
_TIMING_NOTES: Final[Mapping[str, str]] = {
    "before-forwarding": (
        "Under 'before-forwarding', the shipped ordering, Charlie's count was "
        "taken against the declaration Alice signed while he is holding "
        "another, so he refuses on provenance and reaches NO VERDICT. That is "
        "a denial of transfer, not a detection and not a rejection."
    ),
    "after-forwarding": (
        "Under 'after-forwarding' Charlie counted against the declaration that "
        "reached him, so he SCORES it. This is the arm whose acceptance rate "
        "the closed form predicts, and it is a different question from the "
        "other ordering -- runs under the two are never pooled."
    ),
}

#: How the link's true matched-position error rate is named on the screen.
_LINK_MODELS: Final[Mapping[str, str]] = {
    "ideal": "an ideal Phi+ resource on both links",
    "depolarising": "a Pauli twirl at strength `noise` on the travelling half",
    "intercept-resend": "Eve measures and re-sends, leaving a product state",
    "kept-share": "Eve copies the travelling half onto an ancilla and keeps it",
}


def _seed_streams(seed: int, children: int) -> tuple[
    np.random.Generator, tuple[np.random.Generator, ...]
]:
    """Derive the session's generator and ``children`` adversary generators.

    >>> from qsecure.web.driver import _seed_streams
    >>> session, adversaries = _seed_streams(11, 2)
    >>> len(adversaries)
    2
    >>> from qsecure.attacks.isolation import same_stream
    >>> same_stream(session, adversaries[0]) or same_stream(*adversaries)
    False

    The same seed gives the same streams, which is what makes a demo repeatable:
    >>> again, _ = _seed_streams(11, 2)
    >>> bool(session.random() == again.random())
    True
    """
    session = np.random.default_rng(
        np.random.SeedSequence([_SESSION_DOMAIN, int(seed)])
    )
    root = np.random.SeedSequence([_ATTACK_DOMAIN, int(seed)])
    adversaries = tuple(
        np.random.default_rng(child) for child in root.spawn(children)
    )
    for index, generator in enumerate(adversaries):
        require_distinct_streams(
            session,
            generator,
            left_name="the session's generator",
            right_name=f"adversary generator {index}",
            detail=(
                "An adversary that shares the session's stream can predict "
                "the private symmetrisation coins and every rate it publishes "
                "is fiction."
            ),
        )
        for other in range(index):
            require_distinct_streams(
                adversaries[other],
                generator,
                left_name=f"adversary generator {other}",
                right_name=f"adversary generator {index}",
            )
    return session, adversaries


def _capture_seed_for(signing_length: int) -> int:
    """Return the seed the loot for a run of this shape is minted under.

    >>> from qsecure.web.driver import _capture_seed_for
    >>> _capture_seed_for(96) == _capture_seed_for(96)
    True
    >>> _capture_seed_for(96) == _capture_seed_for(192)
    False
    """
    digest = hashlib.blake2b(
        f"{_CAPTURE_SEED}|{signing_length}".encode("ascii"), digest_size=4
    ).digest()
    return int.from_bytes(digest, "big")


def _capture_for(params: ProtocolParams) -> Any:
    """Mint the replayed declaration **at this run's own length**.

    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.web.driver import _capture_for
    >>> params = ProtocolParams(key_length=96, check_fraction=0.25)
    >>> params.sifted().key_length
    72
    >>> len(_capture_for(params).signature)
    72
    """
    scored = params.sifted() if params.has_check_rounds else params
    signing_length = scored.key_length
    capture = replay_capture(
        ProtocolParams(
            key_length=signing_length,
            s_a=params.s_a,
            s_v=params.s_v,
            bases=params.bases,
            allow_forgeable=params.allow_forgeable,
        ),
        message_bit=MESSAGE_BIT,
        seed=_capture_seed_for(signing_length),
    )
    if len(capture.signature) != signing_length:  # pragma: no cover - guard
        raise RuntimeError(
            f"the replay capture is {len(capture.signature)} positions long "
            f"and the live declaration is {signing_length}. The adversary "
            f"would decline it and forward honestly, and the arm would report "
            f"a clean run while looking as though it had attacked."
        )
    return capture


@dataclass(frozen=True)
class RunRequest:
    """One validated ``POST /api/run`` body.

    >>> from qsecure.web.driver import RunRequest
    >>> request = RunRequest.from_mapping({"attack": "honest"})
    >>> request.key_length, request.eps, request.seed
    (192, 1e-09, 20260141)
    """

    attack: str
    key_length: int
    check_fraction: float
    noise: float
    eps: float
    channel_error_rate: float
    tolerated_depolarising: float
    count_exchange_timing: str
    seed: int

    @classmethod
    def from_mapping(cls, body: Mapping[str, Any]) -> RunRequest:
        """Validate a request body and return it, or refuse with the cap.

        >>> from qsecure.web.driver import RunRequest
        >>> from qsecure.web.limits import RequestRefused
        >>> try:
        ...     RunRequest.from_mapping({"attack": "nope"})
        ... except RequestRefused as refusal:
        ...     print(refusal.field)
        attack

        >>> try:
        ...     RunRequest.from_mapping(
        ...         {"attack": "channel-kept-share", "noise": 0.1}
        ...     )
        ... except RequestRefused as refusal:
        ...     print(refusal.field, "resource line" in str(refusal))
        noise True
        """
        attack = body.get("attack", "honest")
        if not isinstance(attack, str) or attack not in ATTACK_KEYS:
            # `safe_text`, not `!r`: the field is caller-controlled and echoed twice.
            raise RequestRefused(
                "attack",
                f"attack={safe_text(attack)} is not one of "
                f"{list(ATTACK_KEYS)}. It is "
                f"refused rather than defaulted to the honest control: running "
                f"a different experiment than the one asked for and reporting "
                f"it under the requested label is how a dashboard lies.",
                attack,
                list(ATTACK_KEYS),
            )
        spec = attack_spec(attack)
        key_length = check_key_length(body.get("key_length", 192))
        check_fraction = check_check_fraction(
            body.get("check_fraction", 0.25), key_length
        )
        noise = check_probability(body.get("noise", 0.0), "noise", NOISE_MAX)
        if noise > 0.0 and not spec.accepts_noise:
            raise RequestRefused(
                "noise",
                f"attack={attack!r} occupies the resource line itself, so it "
                f"cannot also carry a background link noise of {noise!r}. "
                f"There is one resource line and one thing may stand on it; "
                f"composing two channels would be inventing physics this "
                f"project does not derive. Send noise=0.0, or choose "
                f"'channel-manipulation', whose strength IS this parameter.",
                noise,
                0.0,
            )
        return cls(
            attack=attack,
            key_length=key_length,
            check_fraction=check_fraction,
            noise=noise,
            eps=check_eps(body.get("eps", EPS_DEFAULT)),
            channel_error_rate=check_probability(
                body.get("channel_error_rate", 0.0),
                "channel_error_rate",
                CHANNEL_ERROR_RATE_MAX,
            ),
            tolerated_depolarising=check_probability(
                body.get("tolerated_depolarising", 0.0),
                "tolerated_depolarising",
                NOISE_MAX,
            ),
            count_exchange_timing=check_timing(
                body.get("count_exchange_timing", "before-forwarding")
            ),
            seed=check_seed(body.get("seed", 20260141)),
        )

    @property
    def spec(self) -> AttackSpec:
        """AttackSpec: The roster entry this request names."""
        return attack_spec(self.attack)

    def params(self) -> ProtocolParams:
        """Return the parameter set this request runs under.

        >>> from qsecure.web.driver import RunRequest
        >>> RunRequest.from_mapping(
        ...     {"key_length": 192, "check_fraction": 0.25}
        ... ).params().signing_length
        144
        """
        return ProtocolParams(
            key_length=self.key_length, check_fraction=self.check_fraction
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the request as it was resolved, defaults filled in."""
        return {
            "attack": self.attack,
            "key_length": self.key_length,
            "check_fraction": self.check_fraction,
            "noise": self.noise,
            "eps": self.eps,
            "channel_error_rate": self.channel_error_rate,
            "tolerated_depolarising": self.tolerated_depolarising,
            "count_exchange_timing": self.count_exchange_timing,
            "seed": self.seed,
            "message_bit": MESSAGE_BIT,
        }


@dataclass
class _Mounting:
    """The seams one arm occupies, and how to report what it did afterwards."""

    kwargs: dict[str, Any] = field(default_factory=dict)
    link: dict[str, Any] = field(default_factory=dict)
    targeted: list[list[Any]] = field(default_factory=list)
    extra: Callable[[], dict[str, Any]] = dict
    acted: Callable[[], bool] = lambda: True
    notes: str = ""


def _link_report(
    model: str,
    strength: float,
    target: Party | None,
    error_rate: float,
    channel_error_rate: float,
    tolerated_depolarising: float = 0.0,
) -> dict[str, Any]:
    """Describe the physical link, as ground truth the detector cannot read.

    >>> from qsecure.web.driver import _link_report
    >>> report = _link_report("depolarising", 0.03125, None, 0.015625, 0.0)
    >>> report["true_error_rate_by_party"]
    {'Bob': 0.015625, 'Charlie': 0.015625}
    >>> report["nulls_match_link"], report["rate_null_matches_link"]
    (False, False)

    >>> paired = _link_report(
    ...     "depolarising", 0.03125, None, 0.015625, 0.015625, 0.03125
    ... )
    >>> paired["rate_null_matches_link"], paired["channel_null_matches_link"]
    (True, True)
    >>> paired["nulls_match_link"]
    True
    """
    by_party = {
        party.value: (
            error_rate if target is None or target is party else 0.0
        )
        for party in (Party.BOB, Party.CHARLIE)
    }
    rate_matches = all(
        rate == channel_error_rate for rate in by_party.values()
    )
    # Only a depolarising link has a Werner p0; otherwise report that there is none.
    channel_matches = (
        model in ("ideal", "depolarising")
        and tolerated_depolarising == strength
    )
    return {
        "model": model,
        "description": _LINK_MODELS[model],
        "strength": strength,
        "targeted_party": None if target is None else target.value,
        "true_error_rate_by_party": by_party,
        "rate_null_given_to_detector": channel_error_rate,
        "channel_null_given_to_detector": tolerated_depolarising,
        "rate_null_matches_link": rate_matches,
        "channel_null_matches_link": channel_matches,
        "nulls_match_link": rate_matches and channel_matches,
        "note": (
            "The link's true error rate is the ADVERSARY's ground truth and "
            "the detector never reads it. detect() is told two nulls, both "
            "defaulting to a perfect link: channel_error_rate for the rate "
            "family and tolerated_depolarising for the channel family. When "
            "either disagrees with the link, an HONEST run departs from a law "
            "it was scored against and fires correctly -- that is the null "
            "being wrong about the link, not an attack, and both verifiers "
            "still accept."
        ),
    }


def _both_bits(party: Party) -> list[list[Any]]:
    """Return the ``[party, message_bit]`` pairs for one recipient's link.

    >>> from qsecure.protocol.params import Party
    >>> from qsecure.web.driver import _both_bits
    >>> _both_bits(Party.BOB)
    [['Bob', 0], ['Bob', 1]]
    """
    return [[party.value, bit] for bit in (0, 1)]


def _mount(request: RunRequest) -> _Mounting:
    """Build the session keywords for one arm, and its ground-truth reporters."""
    spec = request.spec
    params = request.params()
    wants_link_noise = request.noise > 0.0 and not spec.owns_resource_line
    needed = (1 if spec.seams else 0) + (1 if wants_link_noise else 0)
    session_rng, generators = _seed_streams(request.seed, needed)
    supply = iter(generators)

    mounting = _Mounting(kwargs={"rng": session_rng})
    mounting.link = _link_report(
        "ideal",
        0.0,
        None,
        0.0,
        request.channel_error_rate,
        request.tolerated_depolarising,
    )

    if wants_link_noise:
        channel = DepolarisingChannel(request.noise, rng=next(supply))
        mounting.kwargs["resource_factory"] = channel.resource
        mounting.link = _link_report(
            "depolarising",
            request.noise,
            None,
            qber_from_tensor(channel.tensor),
            request.channel_error_rate,
            request.tolerated_depolarising,
        )
        mounting.targeted = _both_bits(Party.BOB) + _both_bits(Party.CHARLIE)
        mounting.notes = (
            f"The link carries depolarising noise at strength {request.noise} "
            f"on BOTH recipients' hops. There is no adversary in the sense the "
            f"roster means -- this is the wire being imperfect."
        )
        mounting.extra = lambda attack=channel: {
            "link_hops": {
                "engaged": attack.engaged_count,
                "logged": len(attack.log),
            }
        }
        mounting.acted = lambda attack=channel: attack.engaged_count > 0

    key = spec.key
    if key == "honest":
        if not wants_link_noise:
            mounting.acted = lambda: False
            mounting.notes = "No adversary mounted, and a clean link."
        return mounting

    if key == "outside-forgery":
        forger = OutsideForger(rng=next(supply))
        mounting.notes = (
            "Eve declared a key drawn independently of every record."
        )
        mounting.kwargs["signer"] = forger
        mounting.extra = lambda attack=forger: {
            "forgery": {"declarations_substituted": attack.calls}
        }
        return mounting

    if key == "recipient-forgery":
        forger = RecipientForger(rng=next(supply))
        mounting.kwargs["forwarder"] = forger
        mounting.targeted = [[Party.CHARLIE.value, MESSAGE_BIT]]
        mounting.notes = (
            "Bob accepted Alice's declaration and handed Charlie one built "
            "from his own raw log. " + _TIMING_NOTES[request.count_exchange_timing]
        )
        mounting.extra = lambda attack=forger: {
            "forgery": {"declarations_substituted": attack.calls}
        }
        return mounting

    if key in ("impersonation-partial", "impersonation-full",
               "impersonation-distribution"):
        scope = {
            "impersonation-partial": "signing",
            "impersonation-full": "full",
            "impersonation-distribution": "distribution",
        }[key]
        mallory = Impersonator(rng=next(supply))
        mounting.kwargs.update(impersonation_seams(mallory, scope))
        mounting.notes = (
            "Mallory ran the whole protocol correctly with a key pair of her "
            "own. Every statistic is drawn from the honest law. This is not a "
            "miss: assumption (AUTH) puts the position out of model, and no "
            "transcript statistic reaches it."
            if scope == "full"
            else (
                f"Mallory held Alice's {scope} seam only, so her declaration "
                f"and the recipients' records were produced by different "
                f"parties. The transcript cannot separate this from an "
                f"outside forgery, and the detector names the group."
            )
        )
        mounting.extra = lambda attack=mallory, scope=scope: {
            "impersonation": {
                "scope": scope,
                "keys_drawn": len(attack.keys),
            }
        }
        return mounting

    if key == "replay":
        capture = _capture_for(params)
        forwarder = ReplayingForwarder(
            rng=next(supply),
            captures=(capture.signature,),
            replay_probability=1.0,
            relabel=True,
        )
        mounting.kwargs["forwarder"] = forwarder
        mounting.targeted = [[Party.CHARLIE.value, MESSAGE_BIT]]
        mounting.notes = (
            "The hop re-presented a declaration from a finished round, "
            "relabelled with an opening of its own invention. The session "
            "rebinds whatever the hop returns to the live round, so the stale "
            "declaration reaches Charlie carrying the LIVE identifier. "
            + _TIMING_NOTES[request.count_exchange_timing]
        )
        mounting.extra = lambda attack=forwarder, capture=capture: {
            "replay": {
                "replays": attack.replays,
                "replay_probability": attack.replay_probability,
                "relabelled": attack.relabel,
                "capture_key_length": len(capture.signature),
                "capture_session_id": capture.signature.session_id,
                "note": (
                    "The capture is minted at this run's signing length. A "
                    "capture of another shape is declined by the adversary, "
                    "which forwards honestly and reports zero replays while "
                    "appearing to have attacked."
                ),
            }
        }
        mounting.acted = lambda attack=forwarder: attack.replays > 0
        return mounting

    if spec.owns_resource_line:
        generator = next(supply)
        target = Party.BOB
        if key == "channel-manipulation":
            attack: Any = DepolarisingChannel(
                request.noise, rng=generator, target=target
            )
            model = "depolarising"
            strength = request.noise
            mounting.notes = (
                f"A Pauli twirl at strength {request.noise} on Bob's link "
                f"only. Charlie's link was never touched, and its statistics "
                f"are those of an honest run."
                if request.noise > 0.0
                else (
                    "Eve was mounted on the resource seam and passed every "
                    "pair through untouched. The transcript is an honest "
                    "transcript and is reported as one."
                )
            )
        elif key == "channel-intercept-resend":
            attack = InterceptResend(rng=generator, target=target)
            model = "intercept-resend"
            strength = 1.0
            mounting.notes = (
                "Eve measured the travelling half on Bob's link and re-sent "
                "what she found, leaving a product state."
            )
        else:
            attack = KeptShareSwap(rng=generator, target=target)
            model = "kept-share"
            strength = 1.0
            mounting.notes = (
                "Eve copied the travelling half onto an ancilla and kept it, "
                "on Bob's link only."
            )
        mounting.kwargs["resource_factory"] = attack.resource
        mounting.link = _link_report(
            model,
            strength,
            target,
            qber_from_tensor(attack.tensor),
            request.channel_error_rate,
            request.tolerated_depolarising,
        )
        mounting.extra = lambda attack=attack: {
            "channel": {
                "target": None
                if attack.target is None
                else attack.target.value,
                "hops_logged": len(attack.log),
                "hops_engaged": attack.engaged_count,
            }
        }
        mounting.acted = lambda attack=attack: attack.engaged_count > 0
        # Aimed at Bob, but only actually acted on where it engaged; the pairs
        # are the aim and `acted` is the fact.
        mounting.targeted = _both_bits(target)
        return mounting

    if key == "count-starvation":
        starver = CountStarver(rng=next(supply), party=Party.CHARLIE)
        mounting.kwargs["count_exchange"] = starver
        mounting.targeted = [[Party.CHARLIE.value, MESSAGE_BIT]]
        mounting.notes = (
            "Charlie put a number below the pooled floor on the wire. Bob "
            "reaches NO VERDICT as a result -- he did not reject the "
            "signature, he was denied the evidence to score it, and the "
            "starver keeps his own acceptance."
        )
        mounting.extra = lambda attack=starver: {
            "starvation": {
                "party": attack.party.value,
                "mode": str(attack.mode),
                "decisions": [
                    {
                        "party": decision.party.value,
                        "message_bit": decision.message_bit,
                        "true_count": decision.true_count,
                        "declared_count": decision.declared_count,
                        "counterpart_count": decision.counterpart_count,
                        "headroom": decision.headroom,
                        "starved": decision.starved,
                        "understatement": decision.understatement,
                        "denies": decision.denies,
                    }
                    for decision in attack.log
                ],
            }
        }
        mounting.acted = lambda attack=starver: any(
            decision.starved for decision in attack.log
        )
        return mounting

    raise KeyError(f"no mounting for {key!r}")  # pragma: no cover - guard


@dataclass(frozen=True)
class RunResult:
    """One scored run, in the four keys the API contract fixes -- plus ``error``."""

    request: dict[str, Any]
    detection: dict[str, Any] | None
    run: dict[str, Any] | None
    ground_truth: dict[str, Any]
    timings: dict[str, float]
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the response body."""
        return {
            "request": self.request,
            "detection": self.detection,
            "run": self.run,
            "ground_truth": self.ground_truth,
            "timings": self.timings,
            "error": self.error,
        }


def run_once(request: RunRequest) -> RunResult:
    """Execute one arm and score it, measuring both halves.

    >>> from qsecure.web.driver import RunRequest, run_once
    >>> result = run_once(RunRequest.from_mapping({"key_length": 96, "seed": 5}))
    >>> result.timings["session_ms"] > result.timings["detect_ms"]
    True

    >>> "ground_truth" in result.detection, "link" in result.ground_truth
    (False, True)

    >>> starved = run_once(
    ...     RunRequest.from_mapping(
    ...         {"attack": "count-starvation", "key_length": 96}
    ...     )
    ... )
    >>> starved.detection["outcomes"]["Bob"]
    'refused-to-score'
    >>> [
    ...     (row["party"], row["outcome"], row["rate"])
    ...     for row in starved.run["verifiers"]
    ... ]
    [('Bob', 'refused-to-score', None), ('Charlie', 'accepted', 0.0)]

    >>> quiet = run_once(
    ...     RunRequest.from_mapping(
    ...         {"attack": "channel-manipulation", "noise": 0.0,
    ...          "key_length": 96}
    ...     )
    ... )
    >>> quiet.ground_truth["acted"], quiet.ground_truth["identical_to_honest"]
    (False, True)
    >>> quiet.detection["detected"]
    False
    """
    spec = request.spec
    timings: dict[str, float] = {"session_ms": 0.0, "detect_ms": 0.0}
    truth = ground_truth_for(
        spec.key,
        spec.label,
        spec.detectable,
        spec.assumption,
        acted=False,
        seams_held=spec.seams,
        notes="",
        extra={
            "seed": request.seed,
            "message_bit": MESSAGE_BIT,
            "model_note": spec.model_note,
            "completed": False,
            "note": (
                "Everything under this key is the ADVERSARY's own log and the "
                "harness's. detect() reads a JSON transcript and nothing "
                "else, so none of it reached the verdict beside it."
            ),
        },
    )

    try:
        mounting = _mount(request)
        truth["link"] = mounting.link
        started = time.perf_counter()
        session = QDSSession(
            request.params(),
            count_exchange_timing=request.count_exchange_timing,
            **mounting.kwargs,
        )
        transcript = session.run(MESSAGE_BIT)
        text = transcript.to_json()
        timings["session_ms"] = (time.perf_counter() - started) * 1000.0

        acted = bool(mounting.acted())
        truth.update(mounting.extra())
        truth["acted"] = acted
        truth["identical_to_honest"] = spec.key != "honest" and not acted
        truth["targeted_links"] = [
            list(pair) for pair in (mounting.targeted if acted else [])
        ]
        truth["notes"] = mounting.notes

        started = time.perf_counter()
        stats = TranscriptStatistics.from_json(text)
        detection = detect(
            stats,
            eps=request.eps,
            channel_error_rate=request.channel_error_rate,
            tolerated_depolarising=request.tolerated_depolarising,
        ).to_dict()
        timings["detect_ms"] = (time.perf_counter() - started) * 1000.0

        facts = run_facts(
            stats,
            detection,
            json.loads(text),
            requested_key_length=request.key_length,
            check_fraction=request.check_fraction,
        )
        facts["transcript_bytes"] = len(text)
        # Which null was stated; `null_is_noiseless` covers only the rate family.
        facts["nulls"] = nulls_stated(
            request.channel_error_rate, request.tolerated_depolarising
        )
        # The transcript's summary: the only prose naming the per-run repudiation bound.
        facts["transcript_summary"] = transcript.summary()
    except Exception as failure:  # noqa: BLE001 - deliberately total
        return RunResult(
            request=request.to_dict(),
            detection=None,
            run=None,
            ground_truth=truth,
            timings=timings,
            error={
                "kind": "run-failed",
                "attack": request.attack,
                "exception": type(failure).__name__,
                # Bounded like every client-bound value: a failure report must not fail.
                "message": safe_text(str(failure), 2000),
                "detail": (
                    "The arm was mounted and the run did not complete. This is "
                    "not a clean run and it is not a detection: no verdict was "
                    "produced at all."
                ),
            },
        )

    truth["completed"] = True
    return RunResult(
        request=request.to_dict(),
        detection=detection,
        run=facts,
        ground_truth=truth,
        timings=timings,
    )
