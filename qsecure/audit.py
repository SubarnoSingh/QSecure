"""The security event log: one line per verification outcome, and nothing else.

Append-only, bounded, and recording the shipped four-valued verdict.

Examples
--------
>>> from qsecure.audit import AuditLog
>>> from qsecure.detect.thresholds_structural import RunOutcome
>>> from qsecure.protocol.verify import AbortReason
>>> log = AuditLog()
>>> event = log.record(
...     party="Bob",
...     verdict=RunOutcome.ACCEPTED,
...     session_id="round-1",
...     key_length=192,
...     check_fraction=0.25,
...     hypotheses=["honest"],
... )
>>> event.seq, event.verdict
(0, <RunOutcome.ACCEPTED: 'accepted'>)
>>> log.record(
...     party="Charlie",
...     verdict=RunOutcome.REFUSED,
...     session_id="round-1",
...     abort_reason=AbortReason.BELOW_FLOOR,
...     key_length=192,
...     check_fraction=0.25,
... ).seq
1

>>> other = AuditLog()
>>> _ = other.record(
...     party="Bob",
...     verdict=RunOutcome.ACCEPTED,
...     session_id="round-1",
...     key_length=192,
...     check_fraction=0.25,
...     hypotheses=["honest"],
... )
>>> other.events()[0] == log.events()[0]
True

>>> import json
>>> lines = log.to_jsonl().splitlines()
>>> len(lines), json.loads(lines[0])["party"]
(2, 'Bob')
>>> json.loads(lines[1])["verdict"], json.loads(lines[1])["abort_reason"]
('refused-to-score', 'matched-count-below-floor')

>>> log.record(
...     party="Mallory",
...     verdict=RunOutcome.REJECTED,
...     key_length=192,
...     check_fraction=0.25,
... ).party
'Mallory'

>>> small = AuditLog(capacity=2)
>>> for index in range(5):
...     _ = small.record(
...         party="Bob",
...         verdict=RunOutcome.ACCEPTED,
...         key_length=96,
...         check_fraction=0.25,
...     )
>>> len(small), small.dropped, [event.seq for event in small.events()]
(2, 3, [3, 4])
"""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from qsecure.detect.thresholds_structural import RunOutcome

__all__ = ["DEFAULT_CAPACITY", "AuditLog", "SecurityEvent"]


DEFAULT_CAPACITY: Final[int] = 512
"""int: How many events a log retains before the oldest is dropped."""


@dataclass(frozen=True)
class SecurityEvent:
    """One verification outcome, as the log records it.

    Attributes
    ----------
    seq : int
        Position in the log that produced it, from zero.
    session_id : str or None
        Which round the event belongs to, as the caller names it.
    party : str
        Who reached the verdict, **as declared**.
    verdict : RunOutcome
        The shipped four-valued outcome, never a boolean.
    abort_reason : str or None
        Why no verdict was reached, where none was.
    key_length : int
        The parameter the thresholds were derived at.
    check_fraction : float
        The other parameter that moves those thresholds.
    hypotheses : tuple of str
        :attr:`~qsecure.detect.detector.Detection.named`, the hypotheses the
        evidence supported.
    timestamp : str or float or None
        Whatever the caller passed, unread and unvalidated.

    Examples
    --------
    >>> from qsecure.audit import SecurityEvent
    >>> from qsecure.detect.thresholds_structural import RunOutcome
    >>> event = SecurityEvent(
    ...     seq=0,
    ...     session_id=None,
    ...     party="Bob",
    ...     verdict=RunOutcome.ACCEPTED,
    ...     abort_reason=None,
    ...     key_length=192,
    ...     check_fraction=0.25,
    ...     hypotheses=("honest",),
    ... )
    >>> event.to_dict()["verdict"], event.to_dict()["hypotheses"]
    ('accepted', ['honest'])
    """

    seq: int
    session_id: str | None
    party: str
    verdict: RunOutcome
    abort_reason: str | None
    key_length: int
    check_fraction: float
    hypotheses: tuple[str, ...] = ()
    timestamp: str | float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the event as one JSON object.

        Returns
        -------
        dict
            Passes to :func:`json.dumps` unchanged: ``verdict`` and
            ``abort_reason`` are :class:`enum.StrEnum` members or ``None``, and
            the tuple becomes a list because JSON has no tuple.
        """
        return {
            "seq": self.seq,
            "session_id": self.session_id,
            "party": self.party,
            "verdict": str(self.verdict),
            "abort_reason": (
                None if self.abort_reason is None else str(self.abort_reason)
            ),
            "key_length": self.key_length,
            "check_fraction": self.check_fraction,
            "hypotheses": list(self.hypotheses),
            "timestamp": self.timestamp,
        }


class AuditLog:
    """An append-only, bounded record of verification outcomes.

    Parameters
    ----------
    capacity : int, optional
        How many events to retain, :data:`DEFAULT_CAPACITY` by default.
    path : pathlib.Path or str or None, optional
        Where to append the JSON Lines, or ``None`` -- the default -- to keep
        the log in memory only.

    Examples
    --------
    >>> from qsecure.audit import AuditLog
    >>> from qsecure.detect.thresholds_structural import RunOutcome
    >>> log = AuditLog()
    >>> log.path is None, log.capacity
    (True, 512)
    >>> _ = log.record(
    ...     party="Bob",
    ...     verdict=RunOutcome.REJECTED,
    ...     key_length=96,
    ...     check_fraction=0.25,
    ... )
    >>> len(log), log.recorded, log.dropped
    (1, 1, 0)

    >>> log.record(
    ...     party="Bob",
    ...     verdict="maybe",
    ...     key_length=96,
    ...     check_fraction=0.25,
    ... )
    Traceback (most recent call last):
        ...
    ValueError: 'maybe' is not a valid RunOutcome
    """

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        path: Path | str | None = None,
    ) -> None:
        self._capacity = int(capacity)
        self._path = None if path is None else Path(path)
        self._events: deque[SecurityEvent] = deque(maxlen=self._capacity)
        self._recorded = 0
        self._lock = threading.Lock()
        if self._path is not None:
            # Fail here on an unwritable path, not four minutes into a demo.
            with self._path.open("a", encoding="utf-8"):
                pass

    @property
    def capacity(self) -> int:
        """int: How many events are retained."""
        return self._capacity

    @property
    def path(self) -> Path | None:
        """pathlib.Path or None: Where events are appended, if anywhere."""
        return self._path

    @property
    def recorded(self) -> int:
        """int: Events recorded over the log's whole life, drops included."""
        return self._recorded

    @property
    def dropped(self) -> int:
        """int: Events the cap discarded."""
        with self._lock:
            return self._recorded - len(self._events)

    def __len__(self) -> int:
        """int: How many events are retained right now."""
        return len(self._events)

    def record(
        self,
        *,
        party: str,
        verdict: RunOutcome | str,
        key_length: int,
        check_fraction: float,
        session_id: str | None = None,
        abort_reason: str | None = None,
        hypotheses: Iterable[str] = (),
        timestamp: str | float | None = None,
    ) -> SecurityEvent:
        """Append one verification outcome.

        Parameters
        ----------
        party : str
            Who reached the verdict, as declared.
        verdict : RunOutcome or str
            Coerced through
            :class:`~qsecure.detect.thresholds_structural.RunOutcome`, so an
            outcome outside the four raises rather than being stored.
        abort_reason : str or None, optional
            An :class:`~qsecure.protocol.verify.AbortReason` member is accepted
            directly.
        timestamp : str or float or None, optional
            Passed through untouched; nothing here generates one.

        Returns
        -------
        SecurityEvent
            The event as stored, with its sequence number assigned.
        """
        event_verdict = RunOutcome(verdict)
        with self._lock:
            event = SecurityEvent(
                seq=self._recorded,
                session_id=None if session_id is None else str(session_id),
                party=str(party),
                verdict=event_verdict,
                abort_reason=(
                    None if abort_reason is None else str(abort_reason)
                ),
                key_length=int(key_length),
                check_fraction=float(check_fraction),
                hypotheses=tuple(str(name) for name in hypotheses),
                timestamp=timestamp,
            )
            self._recorded += 1
            self._events.append(event)
            if self._path is not None:
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event.to_dict()) + "\n")
        return event

    def events(self) -> tuple[SecurityEvent, ...]:
        """Return the retained events, oldest first.

        Returns
        -------
        tuple of SecurityEvent
            A snapshot.
        """
        with self._lock:
            return tuple(self._events)

    def snapshot(self) -> tuple[tuple[SecurityEvent, ...], int, int]:
        """Return the retained events, the lifetime count and the drops.

        Returns
        -------
        tuple
            ``(events, recorded, dropped)``, all three taken under one lock, so
            they describe the log at a single instant.
        """
        with self._lock:
            events = tuple(self._events)
            return events, self._recorded, self._recorded - len(events)

    def to_jsonl(self) -> str:
        """Return the retained events as JSON Lines.

        Returns
        -------
        str
            One JSON object per line, each line terminated.
        """
        return "".join(
            json.dumps(event.to_dict()) + "\n" for event in self.events()
        )
