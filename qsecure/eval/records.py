"""What one trial leaves behind, and the wall between truth and evidence.

>>> import numpy as np
>>> from qsecure.detect.detector import detect
>>> from qsecure.eval.records import GroundTruth, TrialRecord
>>> from qsecure.eval.seeds import trial_seeds
>>> from qsecure.protocol.params import ProtocolParams
>>> from qsecure.protocol.session import QDSSession
>>> seeds = trial_seeds("demo", "small", 0)
>>> params = ProtocolParams(key_length=96)
>>> transcript = QDSSession(params, rng=seeds.session_rng()).run(0)
>>> record = TrialRecord.build(
...     seeds=seeds,
...     params=params,
...     eps=1e-9,
...     transcript=transcript,
...     detection=detect(transcript.to_json(), eps=1e-9),
...     truth=GroundTruth(hypothesis="honest"),
...     wall_clock={"session_seconds": 0.4, "detect_seconds": 0.003},
... )
>>> record.truth.hypothesis, record.truth.engaged
('honest', False)
>>> record.detection["detected"]
False
>>> record.transcript_summary["count_exchange_timing"]
'before-forwarding'

The same trial timed differently is the same result:
>>> slower = TrialRecord.from_dict(
...     {**record.to_dict(),
...      "wall_clock": {"session_seconds": 91.0, "detect_seconds": 9.0}}
... )
>>> slower.fingerprint() == record.fingerprint()
True

>>> flagged = TrialRecord.from_dict(
...     {**record.to_dict(),
...      "detection": {**record.detection, "detected": True}}
... )
>>> flagged.fingerprint() == record.fingerprint()
False
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Final, Mapping

from qsecure.detect.detector import Detection
from qsecure.protocol.params import ProtocolParams
from qsecure.protocol.session import SessionTranscript

from .seeds import TrialSeeds, check_name

__all__ = [
    "GroundTruth",
    "link_summary",
    "NON_DETERMINISTIC_FIELDS",
    "RECORD_SCHEMA",
    "TrialRecord",
    "UNDETECTABLE_ASSUMPTION",
    "UNDETECTABLE_BY_CONSTRUCTION",
    "transcript_summary",
]

RECORD_SCHEMA: Final[str] = "qsecure/eval/record/v1"
"""str: Written into every record so a reduction can refuse a stale one."""

UNDETECTABLE_BY_CONSTRUCTION: Final[str] = "undetectable-by-construction"
"""str: What a results table prints for a hypothesis assumption (AUTH) excludes."""

UNDETECTABLE_ASSUMPTION: Final[str] = "full impersonation under (AUTH)"
"""str: The assumption the token rests on, named wherever the token is used."""

NON_DETERMINISTIC_FIELDS: Final[frozenset[str]] = frozenset({"wall_clock"})
"""frozenset: Top-level record fields excluded from :meth:`TrialRecord.fingerprint`."""


def _canonical_json(value: Any) -> str:
    """Serialise deterministically: sorted keys, no whitespace slack."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class GroundTruth:
    """The harness's label for a trial. The detector never sees it.

    >>> from qsecure.eval.records import GroundTruth
    >>> GroundTruth(hypothesis="honest").engaged
    False
    >>> aimed = GroundTruth(
    ...     hypothesis="channel-manipulation",
    ...     engaged=True,
    ...     engaged_count=57,
    ...     targeted_link="Bob",
    ... )
    >>> aimed.targeted_link, aimed.detectable
    ('Bob', True)

    >>> GroundTruth(hypothesis="replay", engaged=True, engaged_count=0)
    Traceback (most recent call last):
        ...
    ValueError: engaged=True with engaged_count=0...
    """

    hypothesis: str
    engaged: bool = False
    engaged_count: int | None = None
    targeted_link: str | None = None
    detectable: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate the label."""
        if not isinstance(self.hypothesis, str) or not self.hypothesis:
            raise TypeError("hypothesis must be a non-empty str")
        for name in ("engaged", "detectable"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        for name in ("targeted_link", "notes"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a str or None")
        if not isinstance(self.notes, str):
            raise TypeError("notes must be a str")
        count = self.engaged_count
        if count is not None:
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError("engaged_count must be an int or None")
            if count < 0:
                raise ValueError(f"engaged_count must be >= 0, got {count}")
            if self.engaged and count == 0:
                raise ValueError(
                    "engaged=True with engaged_count=0 is a contradiction. An "
                    "adversary that reports no action did not act, and a cell "
                    "labelling those runs as attacked publishes a detection "
                    "rate whose denominator counts runs where nothing "
                    "happened. An earlier replay arm reported a clean 0/40 "
                    "while having forwarded honestly throughout."
                )

    @property
    def attacked(self) -> bool:
        """bool: ``True`` only if an adversary acted on this run."""
        return self.engaged

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "hypothesis": self.hypothesis,
            "engaged": self.engaged,
            "engaged_count": self.engaged_count,
            "targeted_link": self.targeted_link,
            "detectable": self.detectable,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GroundTruth:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            hypothesis=data["hypothesis"],
            engaged=bool(data.get("engaged", False)),
            engaged_count=data.get("engaged_count"),
            targeted_link=data.get("targeted_link"),
            detectable=bool(data.get("detectable", True)),
            notes=data.get("notes", ""),
        )


def link_summary(stats: Any) -> dict[str, dict[str, Any]]:
    """Reduce the per-link check statistics, one entry per link, never pooled.

    >>> import numpy as np
    >>> from qsecure.detect.statistics import TranscriptStatistics
    >>> from qsecure.eval.records import link_summary
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> params = ProtocolParams(key_length=96, check_fraction=0.25)
    >>> stats = TranscriptStatistics.from_transcript(
    ...     QDSSession(params, rng=np.random.default_rng(11)).run(0)
    ... )
    >>> sorted(link_summary(stats))
    ['Bob/0', 'Bob/1', 'Charlie/0', 'Charlie/1']
    >>> link_summary(stats)["Bob/0"]["errors"]
    0

    With no check rounds there is nothing to report, and nothing is reported:
    >>> unmonitored = TranscriptStatistics.from_transcript(
    ...     QDSSession(ProtocolParams(key_length=96),
    ...                rng=np.random.default_rng(11)).run(0)
    ... )
    >>> link_summary(unmonitored)
    {}
    """
    summary: dict[str, dict[str, Any]] = {}
    for (party, message_bit), link in stats.links.items():
        estimate = link.qber
        summary[f"{party}/{message_bit}"] = {
            "rounds": int(link.errors.trials),
            "errors": int(link.errors.count),
            "qber": None if estimate is None else float(estimate.estimate),
            "qber_upper": (
                None if estimate is None else float(estimate.interval.high)
            ),
        }
    return summary


def transcript_summary(
    transcript: SessionTranscript, *, stats: Any = None
) -> dict[str, Any]:
    """Reduce a transcript to the fields the results tables need.

    >>> import numpy as np
    >>> from qsecure.eval.records import transcript_summary
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> params = ProtocolParams(key_length=96, check_fraction=0.25)
    >>> summary = transcript_summary(
    ...     QDSSession(params, rng=np.random.default_rng(11)).run(0)
    ... )
    >>> summary["verdicts"]
    {'Bob': 'accepted', 'Charlie': 'accepted'}
    >>> summary["aborted"], summary["is_complete"], summary["transferable"]
    (False, True, True)
    >>> sorted(summary["matched"])
    ['Bob', 'Charlie']
    >>> summary["links"]["Bob/0"]["rounds"] > 0
    True
    """
    if not isinstance(transcript, SessionTranscript):
        raise TypeError(
            f"transcript must be a SessionTranscript, got "
            f"{type(transcript).__name__}"
        )
    verdicts: dict[str, str] = {}
    matched: dict[str, int] = {}
    mismatches: dict[str, int] = {}
    thresholds: dict[str, float] = {}
    for party, result in transcript.results_by_party.items():
        verdicts[str(party)] = "accepted" if result.accepted else "rejected"
        matched[str(party)] = int(result.matched_count)
        mismatches[str(party)] = int(result.mismatches)
        thresholds[str(party)] = float(result.threshold)
    for party in transcript.aborts_by_party:
        verdicts[str(party)] = "refused"

    if stats is None:
        from qsecure.detect.statistics import TranscriptStatistics

        stats = TranscriptStatistics.from_transcript(transcript)

    pooled = transcript.pooled_matched_count
    guarantee = transcript.repudiation_guarantee
    return {
        "run_id": transcript.run_id,
        "message_bit": int(transcript.message_bit),
        "count_exchange_timing": str(transcript.count_exchange_timing),
        "verdicts": verdicts,
        "matched": matched,
        "mismatches": mismatches,
        "thresholds": thresholds,
        "pooled_matched_count": None if pooled is None else int(pooled),
        "links": link_summary(stats),
        "aborts": {
            str(party): str(abort.reason)
            for party, abort in transcript.aborts_by_party.items()
        },
        "aborted": bool(transcript.aborted),
        "is_complete": bool(transcript.is_complete),
        "transferable": bool(transcript.transferable),
        "repudiated": bool(transcript.repudiated),
        "symmetrised": bool(transcript.symmetrised),
        "counts_exchanged": bool(transcript.counts_exchanged),
        "channel_monitored": bool(transcript.channel_monitored),
        "session_coherent": bool(transcript.session_coherent),
        "forwarding_altered_signature": bool(
            transcript.forwarding_altered_signature
        ),
        "repudiation_guarantee": None if guarantee is None else float(guarantee),
    }


@dataclass(frozen=True)
class TrialRecord:
    """One trial's reduced result: seed in, verdict out, label alongside.

    Round trip:
    >>> import numpy as np
    >>> from qsecure.detect.detector import detect
    >>> from qsecure.eval.records import GroundTruth, TrialRecord
    >>> from qsecure.eval.seeds import trial_seeds
    >>> from qsecure.protocol.params import ProtocolParams
    >>> from qsecure.protocol.session import QDSSession
    >>> seeds = trial_seeds("demo", "small", 1)
    >>> params = ProtocolParams(key_length=96)
    >>> transcript = QDSSession(params, rng=seeds.session_rng()).run(1)
    >>> record = TrialRecord.build(
    ...     seeds=seeds, params=params, eps=1e-6, transcript=transcript,
    ...     detection=detect(transcript.to_json(), eps=1e-6),
    ...     truth=GroundTruth(hypothesis="honest"),
    ...     wall_clock={"session_seconds": 0.5},
    ... )
    >>> TrialRecord.from_dict(record.to_dict()) == record
    True
    >>> record.identity
    ('demo', 'small', 1)
    """

    schema: str
    experiment: str
    cell: str
    index: int
    seeds: dict[str, Any]
    params: dict[str, Any]
    eps: float
    truth: GroundTruth
    detection: dict[str, Any]
    transcript_summary: dict[str, Any]
    wall_clock: dict[str, float] = field(default_factory=dict)
    transcript_json: str | None = None

    @classmethod
    def build(
        cls,
        *,
        seeds: TrialSeeds,
        params: ProtocolParams,
        eps: float,
        transcript: SessionTranscript,
        detection: Detection,
        truth: GroundTruth,
        wall_clock: Mapping[str, float] | None = None,
        retain_transcript: bool = False,
        stats: Any = None,
    ) -> TrialRecord:
        """Assemble a record from the pieces a trial produced."""
        if not isinstance(seeds, TrialSeeds):
            raise TypeError(f"seeds must be TrialSeeds, got {type(seeds).__name__}")
        if not isinstance(params, ProtocolParams):
            raise TypeError(
                f"params must be ProtocolParams, got {type(params).__name__}"
            )
        if not isinstance(detection, Detection):
            raise TypeError(
                f"detection must be a Detection, got {type(detection).__name__}"
            )
        if not isinstance(truth, GroundTruth):
            raise TypeError(f"truth must be GroundTruth, got {type(truth).__name__}")
        return cls(
            schema=RECORD_SCHEMA,
            experiment=check_name(seeds.experiment, what="experiment"),
            cell=check_name(seeds.cell, what="cell"),
            index=seeds.index,
            seeds=seeds.to_dict(),
            params=params.to_dict(),
            eps=float(eps),
            truth=truth,
            detection=detection.to_dict(),
            transcript_summary=transcript_summary(transcript, stats=stats),
            wall_clock={str(k): float(v) for k, v in (wall_clock or {}).items()},
            transcript_json=transcript.to_json() if retain_transcript else None,
        )

    @property
    def identity(self) -> tuple[str, str, int]:
        """tuple: ``(experiment, cell, index)`` -- what the filename is built from."""
        return (self.experiment, self.cell, self.index)

    @property
    def count_exchange_timing(self) -> str:
        """str: The ordering this run used. A table column, never averaged over."""
        return str(self.transcript_summary["count_exchange_timing"])

    @property
    def flagged(self) -> bool:
        """bool: Whether any detector signal fired. Not a rejection."""
        return bool(self.detection["detected"])

    @property
    def null_is_noiseless(self) -> bool:
        """bool: Whether this trial was scored against a noiseless null."""
        return bool(self.detection["null_is_noiseless"])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        payload: dict[str, Any] = {
            "schema": self.schema,
            "experiment": self.experiment,
            "cell": self.cell,
            "index": self.index,
            "seeds": self.seeds,
            "params": self.params,
            "eps": self.eps,
            "truth": self.truth.to_dict(),
            "detection": self.detection,
            "transcript_summary": self.transcript_summary,
            "wall_clock": self.wall_clock,
        }
        if self.transcript_json is not None:
            payload["transcript_json"] = self.transcript_json
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrialRecord:
        """Rebuild from :meth:`to_dict` output."""
        schema = data.get("schema")
        if schema != RECORD_SCHEMA:
            raise ValueError(
                f"record schema {schema!r} is not {RECORD_SCHEMA!r}. Delete "
                f"the stale results or re-run the sweep; reducing across two "
                f"schemas mixes two meanings of the same column."
            )
        return cls(
            schema=str(schema),
            experiment=str(data["experiment"]),
            cell=str(data["cell"]),
            index=int(data["index"]),
            seeds=dict(data["seeds"]),
            params=dict(data["params"]),
            eps=float(data["eps"]),
            truth=GroundTruth.from_dict(data["truth"]),
            detection=dict(data["detection"]),
            transcript_summary=dict(data["transcript_summary"]),
            wall_clock={
                str(k): float(v) for k, v in dict(data.get("wall_clock", {})).items()
            },
            transcript_json=data.get("transcript_json"),
        )

    def deterministic_payload(self) -> dict[str, Any]:
        """Return the record with the non-deterministic fields removed."""
        return {
            key: value
            for key, value in self.to_dict().items()
            if key not in NON_DETERMINISTIC_FIELDS
        }

    def fingerprint(self) -> str:
        """Return a SHA-256 over everything that must not depend on scheduling.

        >>> import numpy as np
        >>> from qsecure.detect.detector import detect
        >>> from qsecure.eval.records import GroundTruth, TrialRecord
        >>> from qsecure.eval.seeds import trial_seeds
        >>> from qsecure.protocol.params import ProtocolParams
        >>> from qsecure.protocol.session import QDSSession
        >>> def make(seconds):
        ...     seeds = trial_seeds("demo", "small", 2)
        ...     params = ProtocolParams(key_length=48)
        ...     run = QDSSession(params, rng=seeds.session_rng()).run(0)
        ...     return TrialRecord.build(
        ...         seeds=seeds, params=params, eps=1e-9, transcript=run,
        ...         detection=detect(run.to_json(), eps=1e-9),
        ...         truth=GroundTruth(hypothesis="honest"),
        ...         wall_clock={"session_seconds": seconds},
        ...     )
        >>> make(0.1).fingerprint() == make(99.0).fingerprint()
        True
        >>> len(make(0.1).fingerprint())
        64
        """
        text = _canonical_json(self.deterministic_payload())
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
