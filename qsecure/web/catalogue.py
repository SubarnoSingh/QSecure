"""The roster of arms the dashboard can run, and what each one may claim.

>>> from qsecure.web.catalogue import ATTACKS, attack_spec, roster_payload
>>> [spec.key for spec in ATTACKS][:4]
['honest', 'outside-forgery', 'recipient-forgery', 'impersonation-partial']
>>> attack_spec("impersonation-full").detectable
'undetectable-by-construction'
>>> "(AUTH)" in attack_spec("impersonation-full").assumption
True
>>> attack_spec("honest").detectable
'not-an-attack'

Every key that names a detector hypothesis is one the detector actually knows:
>>> from qsecure.detect.detector import HYPOTHESES
>>> known = {str(item) for item in HYPOTHESES}
>>> sorted(
...     spec.key for spec in ATTACKS
...     if spec.hypothesis is not None and spec.hypothesis not in known
... )
[]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final


__all__ = [
    "ATTACKS",
    "ATTACK_KEYS",
    "DETECTABLE",
    "DETECTABLE_VALUES",
    "NOT_AN_ATTACK",
    "UNDETECTABLE_BY_CONSTRUCTION",
    "AttackSpec",
    "attack_spec",
    "roster_payload",
]


#: The honest control: nothing to detect, not a detector that missed.
NOT_AN_ATTACK: Final[str] = "not-an-attack"

#: A derived threshold exists whose null this adversary departs from.
DETECTABLE: Final[str] = "detectable"

#: Assumption (AUTH) excludes it; no transcript statistic can.
UNDETECTABLE_BY_CONSTRUCTION: Final[str] = "undetectable-by-construction"

#: The whole vocabulary, as the frontend's contract file states it.
DETECTABLE_VALUES: Final[tuple[str, ...]] = (
    NOT_AN_ATTACK,
    DETECTABLE,
    UNDETECTABLE_BY_CONSTRUCTION,
)


@dataclass(frozen=True)
class AttackSpec:
    """One arm of the dashboard: what it mounts, and what it may claim.

    >>> from qsecure.web.catalogue import attack_spec
    >>> spec = attack_spec("count-starvation")
    >>> spec.seams, spec.hypothesis
    (('count_exchange',), 'count-starvation')
    """

    key: str
    label: str
    summary: str
    detectable: str
    assumption: str | None
    model_note: str
    seams: tuple[str, ...] = ()
    hypothesis: str | None = None
    owns_resource_line: bool = False
    noise_is_its_strength: bool = False
    targets_one_link: bool = False

    def __post_init__(self) -> None:
        """Reject a spec that could render as an unknown or unexplained verdict."""
        if self.detectable not in DETECTABLE_VALUES:
            raise ValueError(
                f"detectable must be one of {list(DETECTABLE_VALUES)}, got "
                f"{self.detectable!r}. It is a string rather than a boolean so "
                f"that the (AUTH) case is carried explicitly instead of by "
                f"omission."
            )
        if self.detectable == UNDETECTABLE_BY_CONSTRUCTION and not (
            self.assumption or ""
        ).strip():
            raise ValueError(
                f"arm {self.key!r} claims to be undetectable by construction "
                f"and names no assumption. That claim IS the assumption: "
                f"without one the field says only that nothing fired, which is "
                f"a missed detection and the opposite of what is meant."
            )
        if not self.model_note.strip():
            raise ValueError(f"arm {self.key!r} has no model note")

    @property
    def accepts_noise(self) -> bool:
        """bool: Whether a positive ``noise`` may be sent with this arm."""
        return (not self.owns_resource_line) or self.noise_is_its_strength

    def to_dict(self) -> dict[str, Any]:
        """Return the five contract fields, plus what the controls need."""
        return {
            "key": self.key,
            "label": self.label,
            "summary": self.summary,
            "detectable": self.detectable,
            "assumption": self.assumption,
            "model_note": self.model_note,
            "seams": list(self.seams),
            "hypothesis": self.hypothesis,
            "owns_resource_line": self.owns_resource_line,
            "accepts_noise": self.accepts_noise,
        }


_D6: Final[str] = (
    "The adversary owns its randomness and never reads the session's, so "
    "its choices are reproducible from the request seed and independent of the "
    "run."
)

_SEAM: Final[str] = (
    "Mounted through a shipped seam: no protocol edit was needed, so an "
    "attacked run and a clean one are the same program. "
)


ATTACKS: Final[tuple[AttackSpec, ...]] = (
    AttackSpec(
        key="honest",
        label="Honest run",
        summary=(
            "No adversary. Every statistic is drawn from the law the nulls "
            "describe. This is the baseline the false-positive bound is a "
            "statement about -- and note what it is not: it is not a run the "
            "detector is guaranteed to pass. The default null is a NOISELESS "
            "link, so an honest run over a link with noise > 0 departs from it "
            "and fires, correctly, with both verifiers still accepting."
        ),
        detectable=NOT_AN_ATTACK,
        assumption=None,
        model_note=(
            "The honest run is the run every null in this package describes. "
            "Its false-positive bound is PROVEN under that null; it is not a "
            "measurement and not a promise that nothing will fire."
        ),
    ),
    AttackSpec(
        key="outside-forgery",
        label="Outside forgery",
        summary=(
            "Eve substitutes Alice's declaration outright, knowing nothing. "
            "Matched positions disagree with probability 1/2 each, so the "
            "mismatch count leaves the point mass at zero immediately. She "
            "cannot move the matched count -- that depends only on the "
            "declared bases and the recipients' own draws -- which is why the "
            "matched-count floors are an evidence-liveness control here and "
            "not a forgery detector."
        ),
        detectable=DETECTABLE,
        assumption=None,
        model_note=(
            _SEAM + "Eve stands on the classical channel from Alice and holds "
            "no recipient's log and no symmetrisation coin. " + _D6
        ),
        seams=("signer",),
        hypothesis="outside-forgery",
    ),
    AttackSpec(
        key="recipient-forgery",
        label="Recipient forgery",
        summary=(
            "Bob verifies Alice's genuine declaration, accepts it, and forwards "
            "Charlie a declaration built from his own raw log. The 1/12 floor "
            "he runs into -- against the 1/3 he would face without "
            "symmetrisation -- is the clearest single demonstration of what "
            "Phase A' buys. WATCH THE ORDERING: under 'before-forwarding', the "
            "shipped one, Charlie counted against the declaration Alice signed "
            "and is holding another, so he refuses on provenance and reaches "
            "NO VERDICT -- a denial of transfer, not a detection and not a "
            "rejection. Under 'after-forwarding' he scores it, and that is the "
            "arm the closed form predicts. Two questions, never pooled."
        ),
        detectable=DETECTABLE,
        assumption=None,
        model_note=(
            _SEAM + "He holds his own log, the declaration once it reaches "
            "him, and the single integer his counterpart announced -- never "
            "the other recipient's log and never the coins. " + _D6
        ),
        seams=("forwarder",),
        hypothesis="recipient-forgery",
    ),
    AttackSpec(
        key="impersonation-partial",
        label="Impersonation, one seam",
        summary=(
            "Mallory holds Alice's signing seam but not her distribution seam, "
            "so she declares a key the recipients never measured against. "
            "Accepted 0/200 when measured, at a mismatch rate of 1/2. It is the "
            "contrast with the full seizure below that makes assumption (AUTH) "
            "load-bearing rather than decorative."
        ),
        detectable=DETECTABLE,
        assumption=None,
        model_note=(
            "Seizing one of Alice's two seams is inside the threat model. The "
            "transcript cannot separate this from an outside forgery, and the "
            "detector says so rather than picking one. " + _D6
        ),
        seams=("signer",),
        hypothesis="impersonation-signing-seam",
    ),
    AttackSpec(
        key="impersonation-full",
        label="Impersonation, both seams",
        summary=(
            "Mallory holds both of Alice's seams and runs the protocol "
            "correctly with a key pair of her own. Every statistic on the "
            "transcript is drawn from the honest law, because she generated "
            "both halves of it. Measured, it was accepted 200/200. This is "
            "not a gap in the detector and not a threshold anyone forgot to "
            "derive: no transcript statistic separates her from Alice, and "
            "there cannot be one. Expect this run to look exactly like an "
            "honest run, because at the level of the transcript it is one."
        ),
        detectable=UNDETECTABLE_BY_CONSTRUCTION,
        assumption=(
            "(AUTH) -- the classical channel Alice authenticates over is "
            "assumed authentic. Seizing both seams is out of model by "
            "assumption; no transcript statistic separates it from an honest "
            "run, and this project derives no bound that pretends otherwise."
        ),
        model_note=(
            "The assumption is where the weight sits, and it is stated rather "
            "than hidden. " + _D6
        ),
        seams=("distributor", "signer"),
        hypothesis="impersonation-full",
    ),
    AttackSpec(
        key="replay",
        label="Replay",
        summary=(
            "The forwarding hop re-presents a declaration captured from a round "
            "that is over, relabelled with an opening of its own invention. The "
            "capture is minted at THIS run's signing length: a capture of "
            "another shape is declined by the adversary, and the arm would then "
            "forward honestly while looking as though it ran -- a wrong number "
            "this project has already paid for once. The session rebinds "
            "whatever the hop returns to the live round, so Charlie sees two "
            "declarations behind one count and refuses: NO VERDICT again, not a "
            "rejection."
        ),
        detectable=DETECTABLE,
        assumption=None,
        model_note=(
            _SEAM + "Everything replayed was public when it was captured: the "
            "declaration and its opening. " + _D6
        ),
        seams=("forwarder",),
        hypothesis="replay",
    ),
    AttackSpec(
        key="channel-manipulation",
        label="Channel manipulation",
        summary=(
            "Eve twirls the travelling half of each pair on one recipient's "
            "link, at strength `noise`. QBER rises and CHSH falls on that link "
            "only -- pooling the two links would report the average of two "
            "channels and detect neither. She owns the quantum channel and may "
            "degrade the delivery, but what she sees is a locally maximally "
            "mixed half and two uniform classical bits, so she cannot read the "
            "key off the wire. At noise = 0 she is mounted and passes every "
            "pair through untouched: the transcript is then an honest "
            "transcript and is reported as one."
        ),
        detectable=DETECTABLE,
        assumption=None,
        model_note=(
            "The channel adversary owns the quantum links and holds no private "
            "log and no coin, so he degrades but does not learn. Check logs are "
            "built inside distribution and never reach the symmetriser, which "
            "is why this is the one attack a detector can attribute to a named "
            "link -- given enough check rounds. " + _D6
        ),
        seams=("resource_factory",),
        hypothesis="channel-manipulation",
        owns_resource_line=True,
        noise_is_its_strength=True,
        targets_one_link=True,
    ),
    AttackSpec(
        key="count-starvation",
        label="Count starvation",
        summary=(
            "A recipient understates the integer he puts on the wire in "
            "Phase C', denying the other party a verdict. A DENIAL IS NOT A "
            "REJECTION: the denied verifier learned nothing about the "
            "signature, and counting this as a catch would be the single most "
            "likely way for this screen to lie. It is free, deterministic and "
            "selective -- no key material, no quantum resource, no computation "
            "-- and it cannot be done quietly: the quietest denying "
            "declaration sits about -11.5 honest standard deviations below the "
            "mean at every key length."
        ),
        detectable=DETECTABLE,
        assumption=None,
        model_note=(
            _SEAM + "Phase C' is the recipients' own step on their own "
            "authenticated channel, so this seam replaces what THEY do, never "
            "what Alice does. " + _D6
        ),
        seams=("count_exchange",),
        hypothesis="count-starvation",
    ),
    # -- beyond the frontend's fixture set, offered live ---------------------- #
    AttackSpec(
        key="impersonation-distribution",
        label="Impersonation, distribution seam only",
        summary=(
            "The mirror image of the partial seizure above: Mallory runs "
            "Alice's quantum phase with a key of her own while Alice's genuine "
            "declaration still reaches the verifiers. Accepted 0/200, again at "
            "a mismatch rate of 1/2."
        ),
        detectable=DETECTABLE,
        assumption=None,
        model_note=(
            "Seizing one of Alice's two seams is inside the threat model. "
            + _D6
        ),
        seams=("distributor",),
        hypothesis="impersonation-distribution-seam",
    ),
    AttackSpec(
        key="channel-intercept-resend",
        label="Channel: intercept and resend",
        summary=(
            "The textbook attack, and the one that destroys the most "
            "entanglement per unit of effort: Eve measures the travelling half "
            "in an axis of her own and re-sends what she found, leaving Alice "
            "and the recipient sharing a product state. `noise` does not apply "
            "-- this attack occupies the resource line entirely and has no "
            "strength parameter."
        ),
        detectable=DETECTABLE,
        assumption=None,
        model_note=(
            "The channel adversary owns the quantum links and holds no private "
            "log. " + _D6
        ),
        seams=("resource_factory",),
        hypothesis="channel-manipulation",
        owns_resource_line=True,
        targets_one_link=True,
    ),
    AttackSpec(
        key="channel-kept-share",
        label="Channel: Eve keeps a share",
        summary=(
            "Entanglement swapping in the form that matters: Eve copies the "
            "travelling qubit onto an ancilla and keeps it, joining the "
            "entanglement rather than relaying it. The CHSH statistic of what "
            "survives is sqrt(2) = 1.414, well under the classical bound of 2 "
            "-- so a detector thresholding on 'does this link still violate the "
            "classical bound' has 0.59 of margin here, not 0.00."
        ),
        detectable=DETECTABLE,
        assumption=None,
        model_note=(
            "The channel adversary owns the quantum links and holds no private "
            "log. " + _D6
        ),
        seams=("resource_factory",),
        hypothesis="channel-manipulation",
        owns_resource_line=True,
        targets_one_link=True,
    ),
)


#: Every valid ``attack`` value, in menu order.
ATTACK_KEYS: Final[tuple[str, ...]] = tuple(spec.key for spec in ATTACKS)

_BY_KEY: Final[dict[str, AttackSpec]] = {spec.key: spec for spec in ATTACKS}


def attack_spec(key: str) -> AttackSpec:
    """Return the arm named ``key``.

    >>> from qsecure.web.catalogue import attack_spec
    >>> attack_spec("honest").label
    'Honest run'
    """
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(
            f"no such attack {key!r}; valid keys are {list(ATTACK_KEYS)}"
        ) from None


def roster_payload() -> list[dict[str, Any]]:
    """Return the whole roster as ``GET /api/attacks`` serves it.

    >>> from qsecure.web.catalogue import DETECTABLE_VALUES, roster_payload
    >>> payload = roster_payload()
    >>> payload[0]["key"], payload[0]["detectable"]
    ('honest', 'not-an-attack')
    >>> {entry["detectable"] for entry in payload} <= set(DETECTABLE_VALUES)
    True

    >>> [
    ...     entry["key"] for entry in payload
    ...     if entry["detectable"] == "undetectable-by-construction"
    ... ]
    ['impersonation-full']
    """
    return [spec.to_dict() for spec in ATTACKS]
