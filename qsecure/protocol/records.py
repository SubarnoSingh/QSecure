"""The recipients' classical measurement logs: ``(index, chosen_basis, outcome_eigenvalue)``.

Each entry stores the basis the recipient chose; verification counts only
positions where it matches the declared basis (counting unmatched ones drives
rates to ``1/3``). Records carry provenance (``symmetrised``, ``session_id``)
but never name a signer: channel authentication is assumed.

>>> from qsecure.protocol.analysis import (
...     averaged_repudiation_bound, repudiation_bound)
>>> from qsecure.protocol.params import (
...     DEFAULT_PARAMS, UNSYMMETRISED_FORGER_RATE_THREE_BASIS)
>>> from qsecure.protocol.records import RecipientRecord

The unmatched-position bug rate sits above both thresholds:

>>> UNSYMMETRISED_FORGER_RATE_THREE_BASIS == 1 / 3
True
>>> UNSYMMETRISED_FORGER_RATE_THREE_BASIS > DEFAULT_PARAMS.s_v
True
>>> DEFAULT_PARAMS.s_a, DEFAULT_PARAMS.s_v
(0.015625, 0.0625)

A partial impersonator's matched QBER is ``1/2``:

>>> 0.5 > DEFAULT_PARAMS.s_v > DEFAULT_PARAMS.s_a > 0
True

Measured impersonation rows (full impersonation is accepted; it is outside the
authenticated-channel assumption):

>>> from qsecure.attacks.impersonation import shipped_summary
>>> for row in shipped_summary():
...     print(row)
none L=192 n=200: accepted 200/200, r_Bob=0.0000, r_Charlie=0.0000
full L=192 n=200: accepted 200/200, r_Bob=0.0000, r_Charlie=0.0000
signing L=192 n=200: accepted 0/200, r_Bob=0.4988, r_Charlie=0.5038
distribution L=192 n=200: accepted 0/200, r_Bob=0.5031, r_Charlie=0.4997

The ``M``-averaged figure vs the per-run figure at the expected ``M``:

>>> averaged = averaged_repudiation_bound(
...     DEFAULT_PARAMS, signer_sees_recipient_bases=False)
>>> f"{averaged:.1e}"
'6.9e-10'
>>> f"{repudiation_bound(DEFAULT_PARAMS, matched_records=76800):.1e}"
'6.9e-10'

Provenance fields:

>>> record = RecipientRecord.from_measurements("Bob", 0, ["X", "Z"], [1, -1])
>>> record.symmetrised
False
>>> sorted(record.to_dict())
['entries', 'message_bit', 'party', 'session_id', 'symmetrised']
>>> record.session_id is None
True
>>> record.with_session_id("9f3c").session_id
'9f3c'

.. _recipient-view:

:class:`RecipientView` is one recipient's holdings for one bit; nothing of the
counterpart is reachable from it:

>>> from qsecure.protocol.params import Party
>>> from qsecure.protocol.records import RecipientRecord, recipient_views
>>> raw = {
...     0: {
...         Party.BOB: RecipientRecord.from_measurements(
...             "Bob", 0, ["X", "Z"], [1, -1]),
...         Party.CHARLIE: RecipientRecord.from_measurements(
...             "Charlie", 0, ["Y", "Y"], [-1, 1]),
...     }
... }
>>> post = {
...     0: {
...         Party.BOB: RecipientRecord.from_measurements(
...             "Bob", 0, ["X", "Y"], [1, 1], symmetrised=True),
...         Party.CHARLIE: RecipientRecord.from_measurements(
...             "Charlie", 0, ["Y", "Z"], [-1, -1], symmetrised=True),
...     }
... }
>>> views = recipient_views(0, raw_records=raw, records=post)
>>> bob = views[Party.BOB]
>>> bob.party, len(bob), bob.matched_count
(<Party.BOB: 'Bob'>, 2, None)
>>> {record.party for record in (bob.raw_record, bob.record)}
{<Party.BOB: 'Bob'>}
>>> import dataclasses
>>> sorted(field.name for field in dataclasses.fields(bob))
['matched_count', 'message_bit', 'party', 'raw_record', 'record']
"""

from __future__ import annotations

import numbers
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, overload

from qsecure.core.paulis import PauliBasis
from qsecure.protocol.params import (
    VERIFIERS,
    Party,
    ProtocolParams,
    _as_basis,
    _as_eigenvalue,
    _as_message_bit,
    _as_party,
)

__all__ = [
    "RecordEntry",
    "RecipientRecord",
    "RecipientView",
    "recipient_views",
]


@dataclass(frozen=True)
class RecordEntry:
    """One log line: key index, the basis the recipient chose, and the measured eigenvalue.

    Examples
    --------
    >>> from qsecure.protocol.records import RecordEntry
    >>> entry = RecordEntry(3, "X", -1)
    >>> entry.bit
    1
    """

    index: int
    basis: PauliBasis
    eigenvalue: int

    def __post_init__(self) -> None:
        """Validate the index and coerce the basis and eigenvalue."""
        if isinstance(self.index, bool) or not isinstance(
            self.index, numbers.Integral
        ):
            raise ValueError(
                f"index must be a non-negative integer position in the key, got "
                f"{self.index!r} of type {type(self.index).__name__}"
            )
        object.__setattr__(self, "index", int(self.index))
        if self.index < 0:
            raise ValueError(
                f"index must be non-negative, got {self.index}. It is the "
                f"position of this qubit in the private key, counted from 0."
            )
        object.__setattr__(self, "basis", _as_basis(self.basis, name="basis"))
        object.__setattr__(self, "eigenvalue", _as_eigenvalue(self.eigenvalue))

    @property
    def bit(self) -> int:
        """int: The classical bit, ``(1 - eigenvalue) // 2``."""
        return (1 - self.eigenvalue) // 2

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "index": self.index,
            "basis": self.basis,
            "eigenvalue": self.eigenvalue,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RecordEntry:
        """Rebuild an entry from :meth:`to_dict` output."""
        return cls(
            index=data["index"],
            basis=data["basis"],
            eigenvalue=data["eigenvalue"],
        )


@dataclass(frozen=True)
class RecipientRecord:
    """One recipient's immutable log for one message bit.

    Entries must be indexed ``0 .. L-1`` in order, since verification reads the log
    positionally. ``symmetrised`` records whether Phase A' ran (a raw pair supports
    no non-repudiation); ``session_id`` is the round announced in Phase A.

    Raises
    ------
    ValueError
        If ``party`` is Alice, the bit is invalid, ``entries`` is empty or misindexed,
        or ``session_id`` is ``""``.
    TypeError
        If ``entries``, ``symmetrised`` or ``session_id`` has the wrong type.

    Examples
    --------
    >>> from qsecure.protocol.params import Party
    >>> from qsecure.protocol.records import RecipientRecord
    >>> record = RecipientRecord.from_measurements(
    ...     Party.BOB, 0, ["X", "Z", "Y"], [1, -1, 1]
    ... )
    >>> len(record), record.party
    (3, <Party.BOB: 'Bob'>)
    """

    party: Party
    message_bit: int
    entries: tuple[RecordEntry, ...]
    symmetrised: bool = False
    session_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the party and bit, then freeze and check the entry indices."""
        resolved = _as_party(self.party)
        if resolved is Party.ALICE:
            raise ValueError(
                "Alice holds no measurement record: she prepares and teleports "
                "the public-key states and keeps the private key, but never "
                "measures. A RecipientRecord belongs to Party.BOB or "
                "Party.CHARLIE."
            )
        object.__setattr__(self, "party", resolved)
        object.__setattr__(
            self, "message_bit", _as_message_bit(self.message_bit)
        )

        entries = self.entries
        if isinstance(entries, (str, bytes)) or not isinstance(
            entries, Sequence
        ):
            raise TypeError(
                f"entries must be a sequence of RecordEntry, got "
                f"{type(entries).__name__}"
            )
        frozen = tuple(entries)
        for position, entry in enumerate(frozen):
            if not isinstance(entry, RecordEntry):
                raise TypeError(
                    f"entries[{position}] must be a RecordEntry, got "
                    f"{type(entry).__name__}. Build the record with "
                    f"RecipientRecord.from_measurements(...) if you have plain "
                    f"basis and eigenvalue sequences."
                )
        if not frozen:
            raise ValueError(
                "a recipient record must have at least one entry: an empty log "
                "has no matched positions, so the mismatch rate is 0/0 and no "
                "accept/reject decision exists."
            )
        for position, entry in enumerate(frozen):
            if entry.index != position:
                raise ValueError(
                    f"entries must be indexed 0..{len(frozen) - 1} in ascending "
                    f"order, one per distributed qubit; entries[{position}] has "
                    f"index {entry.index}. The verifier reads this log "
                    f"positionally against the private key, so a gap, a "
                    f"duplicate or a reordering shifts every later comparison "
                    f"and turns an honest signature into a near-chance mismatch "
                    f"rate."
                )
        object.__setattr__(self, "entries", frozen)
        if not isinstance(self.symmetrised, bool):
            raise TypeError(
                f"symmetrised must be a bool, got "
                f"{type(self.symmetrised).__name__}. It is provenance -- "
                f"whether this log has been through symmetrise_records -- not a "
                f"count or a rate."
            )
        if self.session_id is not None and not isinstance(self.session_id, str):
            raise TypeError(
                f"session_id must be a string or None, got "
                f"{type(self.session_id).__name__}. It is the identifier Alice "
                f"announced with the distribution -- see "
                f"qsecure.protocol.signature.session_identifier -- not a count "
                f"or an index."
            )
        if self.session_id == "":
            raise ValueError(
                "session_id must be non-empty or None. An empty identifier "
                "would match no declaration and yet claim to name a round, so "
                "every verification against this log would refuse; a log that "
                "names no round says so with None and is scored as before."
            )


    @classmethod
    def from_measurements(
        cls,
        party: Party | str,
        message_bit: int,
        bases: Sequence[PauliBasis | str],
        eigenvalues: Sequence[int],
        *,
        symmetrised: bool = False,
        session_id: str | None = None,
    ) -> RecipientRecord:
        """Build a record from parallel basis and eigenvalue sequences (indices ``0 .. L-1``).

        Examples
        --------
        >>> from qsecure.protocol.records import RecipientRecord
        >>> record = RecipientRecord.from_measurements(
        ...     "Charlie", 1, ("Z", "Z"), (-1, 1)
        ... )
        >>> record.eigenvalues
        (-1, 1)
        """
        basis_list = list(bases)
        value_list = list(eigenvalues)
        if len(basis_list) != len(value_list):
            raise ValueError(
                f"bases and eigenvalues must be the same length, got "
                f"{len(basis_list)} and {len(value_list)}. They are two columns "
                f"of one log: entry i is the basis chosen for qubit i and the "
                f"eigenvalue it produced."
            )
        entries = tuple(
            RecordEntry(index=index, basis=basis, eigenvalue=eigenvalue)
            for index, (basis, eigenvalue) in enumerate(
                zip(basis_list, value_list, strict=True)
            )
        )
        return cls(
            party=party,
            message_bit=message_bit,
            entries=entries,
            symmetrised=symmetrised,
            session_id=session_id,
        )

    def with_session_id(self, session_id: str | None) -> RecipientRecord:
        """Return a copy stamped with ``session_id`` (entries shared by reference).

        Examples
        --------
        >>> from qsecure.protocol.records import RecipientRecord
        >>> raw = RecipientRecord.from_measurements("Bob", 0, ["X"], [1])
        >>> stamped = raw.with_session_id("9f3c")
        >>> stamped.session_id, raw.session_id
        ('9f3c', None)
        >>> stamped.entries is raw.entries
        True
        """
        return RecipientRecord(
            party=self.party,
            message_bit=self.message_bit,
            entries=self.entries,
            symmetrised=self.symmetrised,
            session_id=session_id,
        )


    def __len__(self) -> int:
        """int: The number of logged qubits (``L``)."""
        return len(self.entries)

    @overload
    def __getitem__(self, index: int) -> RecordEntry: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[RecordEntry, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> RecordEntry | tuple[RecordEntry, ...]:
        """Return entry ``index``, or a tuple for a slice (``record[i].index == i``)."""
        return self.entries[index]

    def __iter__(self) -> Iterator[RecordEntry]:
        """Iterate over the entries in key order."""
        return iter(self.entries)


    @property
    def length(self) -> int:
        """int: The number of logged qubits."""
        return len(self.entries)

    @property
    def bases(self) -> tuple[PauliBasis, ...]:
        """tuple of PauliBasis: The bases the recipient chose, in key order."""
        return tuple(entry.basis for entry in self.entries)

    @property
    def eigenvalues(self) -> tuple[int, ...]:
        """tuple of int: The measured eigenvalues, in key order."""
        return tuple(entry.eigenvalue for entry in self.entries)

    @property
    def bits(self) -> tuple[int, ...]:
        """tuple of int: The measured outcomes as classical bits."""
        return tuple(entry.bit for entry in self.entries)

    def check_against(self, params: ProtocolParams) -> None:
        """Raise ``ValueError`` if the log length or bases do not match ``params``.

        Examples
        --------
        >>> from qsecure.protocol.params import ProtocolParams
        >>> from qsecure.protocol.records import RecipientRecord
        >>> record = RecipientRecord.from_measurements("Bob", 0, ["X", "Y"], [1, 1])
        >>> record.check_against(ProtocolParams(key_length=2)) is None
        True
        """
        if len(self.entries) != params.key_length:
            raise ValueError(
                f"{self.party.value}'s record has {len(self.entries)} entries "
                f"but params.key_length is {params.key_length}. One qubit is "
                f"distributed and measured per key element, so the two must "
                f"agree exactly."
            )
        allowed = set(params.bases)
        stray = sorted({b.value for b in self.bases if b not in allowed})
        if stray:
            raise ValueError(
                f"{self.party.value}'s record uses basis/bases {stray}, which "
                f"are not in params.bases "
                f"{[b.value for b in params.bases]}. The recipient draws "
                f"uniformly from the declared alphabet; anything else changes "
                f"the match probability the security bounds assume."
            )


    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the whole log."""
        return {
            "party": self.party,
            "message_bit": self.message_bit,
            "entries": [entry.to_dict() for entry in self.entries],
            "symmetrised": self.symmetrised,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RecipientRecord:
        """Rebuild a record from :meth:`to_dict` output (provenance keys optional)."""
        return cls(
            party=data["party"],
            message_bit=data["message_bit"],
            entries=tuple(
                RecordEntry.from_dict(item) for item in data["entries"]
            ),
            symmetrised=data.get("symmetrised", False),
            session_id=data.get("session_id"),
        )


def _as_matched_count(value: Any, length: int) -> int:
    """Coerce and range-check a matched count to ``0 .. length``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(
            f"matched_count must be a non-negative integer or None, got "
            f"{value!r} of type {type(value).__name__}. It is |M_R|, the number "
            f"of positions this recipient may score, and None means no "
            f"declaration has arrived yet."
        )
    count = int(value)
    if not 0 <= count <= length:
        raise ValueError(
            f"matched_count must lie in 0..{length}, got {count}. The matched "
            f"set is a subset of this recipient's own {length} logged "
            f"positions, so a count outside that range describes a different "
            f"run."
        )
    return count


def _narrow(
    records: Mapping[int, Mapping[Party | str, RecipientRecord]],
    message_bit: int,
    party: Party,
    *,
    name: str,
) -> RecipientRecord:
    """Pull one party's log for one bit out of a ``{bit: {party: record}}`` mapping."""
    if not isinstance(records, Mapping):
        raise TypeError(
            f"{name} must be a mapping of message bit to (mapping of party to "
            f"RecipientRecord), got {type(records).__name__}. Pass "
            f"session.raw_records or session.records straight through."
        )
    if message_bit not in records:
        raise ValueError(
            f"{name} has no entry for message bit {message_bit}; it holds "
            f"{sorted(records)}. A run distributes for both bits, so a mapping "
            f"missing one came from a partial or hand-built distribution."
        )
    by_party = records[message_bit]
    if not isinstance(by_party, Mapping):
        raise TypeError(
            f"{name}[{message_bit}] must be a mapping of party to "
            f"RecipientRecord, got {type(by_party).__name__}"
        )
    resolved = {_as_party(key): value for key, value in by_party.items()}
    if party not in resolved:
        raise ValueError(
            f"{name}[{message_bit}] has no log for {party.value}; it holds "
            f"{sorted(member.value for member in resolved)}. A view is one "
            f"recipient's holdings, so the log it is built from has to be "
            f"present."
        )
    return resolved[party]


@dataclass(frozen=True)
class RecipientView:
    """Everything one recipient holds for one message bit, and nothing of the counterpart's.

    ``raw_record`` is his pre-exchange log (what a forging recipient declares);
    ``record`` is what he is scored on; ``matched_count`` is ``None`` until a
    declaration arrives. The counterpart's Phase C' count is deliberately not a field.

    Raises
    ------
    ValueError
        If ``party`` is Alice, a record belongs to another party, bit or length,
        ``raw_record`` is symmetrised, or ``matched_count`` is out of range.
    TypeError
        If either record is not a :class:`RecipientRecord`.

    Examples
    --------
    >>> from qsecure.protocol.params import Party
    >>> from qsecure.protocol.records import RecipientRecord, RecipientView
    >>> raw = RecipientRecord.from_measurements("Bob", 0, ["X", "Z"], [1, -1])
    >>> post = RecipientRecord.from_measurements(
    ...     "Bob", 0, ["X", "Y"], [1, 1], symmetrised=True
    ... )
    >>> view = RecipientView(Party.BOB, 0, raw, post)
    >>> view.party, len(view), view.symmetrised, view.matched_count
    (<Party.BOB: 'Bob'>, 2, True, None)
    >>> view.with_matched_count(1).matched_count
    1
    """

    party: Party
    message_bit: int
    raw_record: RecipientRecord
    record: RecipientRecord
    matched_count: int | None = None

    def __post_init__(self) -> None:
        """Resolve the party and check both logs really are that party's."""
        resolved = _as_party(self.party)
        if not resolved.is_verifier:
            raise ValueError(
                f"a RecipientView belongs to a verifier, "
                f"{[member.value for member in VERIFIERS]}, not to "
                f"{resolved.value}. Alice prepares, teleports and signs; she "
                f"measures nothing and holds no log, so there is no view of the "
                f"run from where she stands."
            )
        object.__setattr__(self, "party", resolved)
        object.__setattr__(
            self, "message_bit", _as_message_bit(self.message_bit)
        )

        for name, record in (
            ("raw_record", self.raw_record),
            ("record", self.record),
        ):
            if not isinstance(record, RecipientRecord):
                raise TypeError(
                    f"{name} must be a RecipientRecord, got "
                    f"{type(record).__name__}. Build a view from the logs a "
                    f"session hands back, or with "
                    f"RecipientView.for_party(party, bit, raw_records=..., "
                    f"records=...)."
                )
            if record.party is not resolved:
                raise ValueError(
                    f"{name} is {record.party.value}'s log but this view is "
                    f"{resolved.value}'s. A view holds one recipient's evidence "
                    f"and nothing of the other's; filing the counterpart's log "
                    f"under this party would hand a {resolved.value}-flavoured "
                    f"adversary exactly the evidence the threat model says he "
                    f"does not have."
                )
            if record.message_bit != self.message_bit:
                raise ValueError(
                    f"{name} is tagged with message bit {record.message_bit} "
                    f"but this view is for bit {self.message_bit}. The two "
                    f"distribution runs share no key and no evidence, so a "
                    f"mixed view scores one run's declaration against the "
                    f"other's log."
                )

        if len(self.raw_record) != len(self.record):
            raise ValueError(
                f"raw_record has {len(self.raw_record)} entries and record has "
                f"{len(self.record)}. Phase A' re-assigns the recipients' "
                f"copies position by position and creates none, so the two logs "
                f"are the same length by construction; a difference means they "
                f"came from different runs."
            )
        if self.raw_record.symmetrised:
            raise ValueError(
                "raw_record is flagged symmetrised, so it is not a raw log. "
                "The raw log is what this recipient measured for himself, "
                "straight off the channel and before the private exchange; it "
                "is what a forging recipient declares, and passing the "
                "post-exchange log twice would silently give him the half of "
                "the evidence the counterpart does *not* hold."
            )

        if self.matched_count is not None:
            object.__setattr__(
                self,
                "matched_count",
                _as_matched_count(self.matched_count, len(self.record)),
            )


    @classmethod
    def for_party(
        cls,
        party: Party | str,
        message_bit: int,
        *,
        raw_records: Mapping[int, Mapping[Party | str, RecipientRecord]],
        records: Mapping[int, Mapping[Party | str, RecipientRecord]],
        matched_count: int | None = None,
    ) -> RecipientView:
        """Narrow a run's ``raw_records``/``records`` mappings to one recipient's view.

        Examples
        --------
        >>> import numpy as np
        >>> from qsecure.protocol.params import Party, ProtocolParams
        >>> from qsecure.protocol.records import RecipientView
        >>> from qsecure.protocol.session import QDSSession
        >>> session = QDSSession(
        ...     ProtocolParams(key_length=12), rng=np.random.default_rng(3)
        ... )
        >>> _ = session.distribute()
        >>> view = RecipientView.for_party(
        ...     Party.BOB,
        ...     0,
        ...     raw_records=session.raw_records,
        ...     records=session.records,
        ... )
        >>> view.party, len(view), view.symmetrised
        (<Party.BOB: 'Bob'>, 12, True)
        """
        resolved = _as_party(party)
        bit = _as_message_bit(message_bit)
        return cls(
            party=resolved,
            message_bit=bit,
            raw_record=_narrow(raw_records, bit, resolved, name="raw_records"),
            record=_narrow(records, bit, resolved, name="records"),
            matched_count=matched_count,
        )

    def with_matched_count(self, matched_count: int | None) -> RecipientView:
        """Return a copy carrying ``matched_count``.

        Examples
        --------
        >>> from qsecure.protocol.records import RecipientRecord, RecipientView
        >>> raw = RecipientRecord.from_measurements("Bob", 1, ["X"], [1])
        >>> view = RecipientView("Bob", 1, raw, raw)
        >>> view.with_matched_count(1).matched_count, view.matched_count
        (1, None)
        """
        return RecipientView(
            party=self.party,
            message_bit=self.message_bit,
            raw_record=self.raw_record,
            record=self.record,
            matched_count=matched_count,
        )


    def __len__(self) -> int:
        """int: The number of logged positions (``L``)."""
        return len(self.record)

    @property
    def length(self) -> int:
        """int: The key length ``L``."""
        return len(self.record)

    @property
    def symmetrised(self) -> bool:
        """bool: Whether Phase A' ran on :attr:`record`."""
        return self.record.symmetrised

    @property
    def has_matched_count(self) -> bool:
        """bool: Whether ``|M_R|`` is known yet (distinct from a count of 0)."""
        return self.matched_count is not None


    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of one recipient's holdings."""
        return {
            "party": self.party,
            "message_bit": self.message_bit,
            "raw_record": self.raw_record.to_dict(),
            "record": self.record.to_dict(),
            "matched_count": self.matched_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RecipientView:
        """Rebuild a view from :meth:`to_dict` output (``matched_count`` optional)."""
        return cls(
            party=data["party"],
            message_bit=data["message_bit"],
            raw_record=RecipientRecord.from_dict(data["raw_record"]),
            record=RecipientRecord.from_dict(data["record"]),
            matched_count=data.get("matched_count"),
        )


def recipient_views(
    message_bit: int,
    *,
    raw_records: Mapping[int, Mapping[Party | str, RecipientRecord]],
    records: Mapping[int, Mapping[Party | str, RecipientRecord]],
    matched_counts: Mapping[Party | str, int] | None = None,
) -> dict[Party, RecipientView]:
    """Split one run's logs into one :class:`RecipientView` per verifier.

    ``matched_counts`` holds each verifier's own count, if known.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.protocol.params import Party, ProtocolParams
    >>> from qsecure.protocol.records import recipient_views
    >>> from qsecure.protocol.session import QDSSession
    >>> session = QDSSession(
    ...     ProtocolParams(key_length=12), rng=np.random.default_rng(4)
    ... )
    >>> _ = session.distribute()
    >>> views = recipient_views(
    ...     1, raw_records=session.raw_records, records=session.records
    ... )
    >>> [party.value for party in views]
    ['Bob', 'Charlie']
    >>> views[Party.BOB].record is views[Party.CHARLIE].record
    False
    """
    bit = _as_message_bit(message_bit)
    if matched_counts is None:
        counts: dict[Party, int] = {}
    elif isinstance(matched_counts, Mapping):
        counts = {
            _as_party(key): value for key, value in matched_counts.items()
        }
    else:
        raise TypeError(
            f"matched_counts must be a mapping of party to int, or None, got "
            f"{type(matched_counts).__name__}"
        )
    return {
        party: RecipientView.for_party(
            party,
            bit,
            raw_records=raw_records,
            records=records,
            matched_count=counts.get(party),
        )
        for party in VERIFIERS
    }
