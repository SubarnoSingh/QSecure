"""Phase A': Bob and Charlie privately swap their records on independent fair coins.

Without this, Alice can send Bob the declared eigenstate and Charlie its
orthogonal partner and repudiate with probability 1. With it, ``e_B + e_C`` and
``M = m_B + m_C`` are coin-independent, and Hoeffding gives
``exp(-M (s_v - s_a)**2 / 8)`` for every Alice strategy. The cost is a forger
floor of ``1/12`` instead of ``1/3``.

Key figures:

>>> import math
>>> from qsecure.protocol.analysis import (
...     averaged_repudiation_bound, repudiation_bound)
>>> from qsecure.protocol.params import (
...     DEFAULT_PARAMS, UNSYMMETRISED_FORGER_RATE_THREE_BASIS)
>>> from qsecure.protocol.verify import (
...     enforced_repudiation_bound,
...     minimum_matched_count,
...     minimum_pooled_matched_count,
... )

Cost: forger floor falls by a factor of four.

>>> UNSYMMETRISED_FORGER_RATE_THREE_BASIS == 1 / 3
True
>>> DEFAULT_PARAMS.forger_floor == 1 / 12
True
>>> UNSYMMETRISED_FORGER_RATE_THREE_BASIS / DEFAULT_PARAMS.forger_floor
4.0
>>> DEFAULT_PARAMS.key_length
115200

The averaged bound assumes the signer cannot see the logged bases; a signer who
does pins ``M = 13``:

>>> averaged = averaged_repudiation_bound(
...     DEFAULT_PARAMS, signer_sees_recipient_bases=False)
>>> f"{averaged:.1e}"
'6.9e-10'
>>> f"{repudiation_bound(DEFAULT_PARAMS, matched_records=13):.4f}"
'0.9964'

The unconditional figure needs the per-verifier floor, the pooled floor and the
joint abort together; the per-verifier floor alone gives a weaker number:

>>> m_min = minimum_matched_count(DEFAULT_PARAMS)
>>> M_min = minimum_pooled_matched_count(DEFAULT_PARAMS)
>>> m_min, M_min, M_min > 2 * m_min
(36555, 74190, True)
>>> f"{enforced_repudiation_bound(DEFAULT_PARAMS):.4e}"
'1.4139e-09'
>>> f"{math.exp(-2 * m_min * DEFAULT_PARAMS.gap ** 2 / 8):.4e}"
'1.9022e-09'

Examples
--------
>>> import numpy as np
>>> from qsecure.protocol.distribute import distribute_public_key
>>> from qsecure.protocol.keys import generate_private_key
>>> from qsecure.protocol.params import Party, ProtocolParams
>>> from qsecure.protocol.symmetrise import symmetrise_records
>>> params = ProtocolParams(key_length=12)
>>> key = generate_private_key(params, 0, rng=np.random.default_rng(1))
>>> raw = distribute_public_key(key, params, rng=np.random.default_rng(2))
>>> swapped = symmetrise_records(raw, rng=np.random.default_rng(3))
>>> swapped[Party.BOB].symmetrised
True
>>> sorted(swapped[Party.BOB].bases + swapped[Party.CHARLIE].bases) == sorted(
...     raw[Party.BOB].bases + raw[Party.CHARLIE].bases
... )
True
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np

from qsecure.core.rng import resolve_rng
from qsecure.protocol.params import VERIFIERS, Party, _as_party
from qsecure.protocol.records import RecipientRecord

__all__ = [
    "Symmetriser",
    "symmetrise_records",
    "no_symmetrisation",
]


class Symmetriser(Protocol):
    """Callable running Phase A' on one message bit's pair of raw records."""

    def __call__(
        self,
        records: Mapping[Party | str, RecipientRecord],
        *,
        rng: np.random.Generator | None = None,
    ) -> dict[Party, RecipientRecord]:
        """Return the recipients' post-exchange records, keyed by party."""
        ...


def _checked_pair(
    records: Mapping[Party | str, RecipientRecord],
) -> tuple[RecipientRecord, RecipientRecord]:
    """Validate the mapping and return ``(Bob's record, Charlie's record)``.

    Raises
    ------
    TypeError
        If ``records`` is not a mapping of parties to records.
    ValueError
        If a verifier is missing or misfiled, the logs differ in length or message
        bit, or either is already symmetrised.
    """
    if not isinstance(records, Mapping):
        raise TypeError(
            f"records must be a mapping of Party to RecipientRecord, got "
            f"{type(records).__name__}; distribute_public_key returns exactly "
            f"that."
        )
    resolved: dict[Party, RecipientRecord] = {}
    for party, record in records.items():
        key = _as_party(party)
        if not isinstance(record, RecipientRecord):
            raise TypeError(
                f"records[{key.value!r}] must be a RecipientRecord, got "
                f"{type(record).__name__}"
            )
        if record.party is not key:
            raise ValueError(
                f"records is keyed by {key.value!r} but the record stored "
                f"there belongs to {record.party.value!r}. The exchange moves "
                f"entries between two named verifiers, so a mislabelled key "
                f"would hand each of them the wrong half."
            )
        resolved[key] = record

    missing = [party.value for party in VERIFIERS if party not in resolved]
    if missing:
        raise ValueError(
            f"symmetrisation needs both verifiers' records, missing "
            f"{missing}. The step *is* an exchange between Bob and Charlie: "
            f"with one recipient there is nothing to exchange, and with one "
            f"recipient the protocol has no transferability and no "
            f"non-repudiation to protect in the first place."
        )
    extra = sorted(set(resolved) - set(VERIFIERS))
    if extra:
        raise ValueError(
            f"symmetrisation takes exactly the two verifiers' records, got "
            f"extra entries for {[party.value for party in extra]}. The "
            f"exchange is defined pairwise."
        )

    bob, charlie = resolved[Party.BOB], resolved[Party.CHARLIE]
    if bob.message_bit != charlie.message_bit:
        raise ValueError(
            f"the two records are from different distribution runs: Bob's is "
            f"for message bit {bob.message_bit} and Charlie's for bit "
            f"{charlie.message_bit}. Only the two copies of the *same* key "
            f"position may be exchanged; swapping across bits would hand a "
            f"verifier evidence about a key he will never be shown."
        )
    if len(bob) != len(charlie):
        raise ValueError(
            f"the two records have different lengths, {len(bob)} and "
            f"{len(charlie)}. The exchange is positional -- entry i of one log "
            f"trades places with entry i of the other -- so the two logs must "
            f"cover the same key positions."
        )
    for record in (bob, charlie):
        if record.symmetrised:
            raise ValueError(
                f"{record.party.value}'s record has already been symmetrised. "
                f"A second exchange is not harmless: it re-randomises an "
                f"assignment Alice has already been committed against, and "
                f"the bound conditions on one round of coins. Symmetrise the "
                f"raw records from distribution exactly once."
            )
    return bob, charlie


def symmetrise_records(
    records: Mapping[Party | str, RecipientRecord],
    *,
    rng: np.random.Generator | None = None,
) -> dict[Party, RecipientRecord]:
    """Swap Bob's and Charlie's entry at each position on an independent fair coin.

    One ``L``-long coin array is drawn per call. Every entry is preserved at its
    index, so ``e_B + e_C`` and ``m_B + m_C`` are unchanged; the coins decide only
    the split. Validation errors are as for :func:`_checked_pair`.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.protocol.params import Party
    >>> from qsecure.protocol.records import RecipientRecord
    >>> from qsecure.protocol.symmetrise import symmetrise_records
    >>> raw = {
    ...     Party.BOB: RecipientRecord.from_measurements(
    ...         "Bob", 0, ["X", "X"], [1, 1]
    ...     ),
    ...     Party.CHARLIE: RecipientRecord.from_measurements(
    ...         "Charlie", 0, ["Z", "Z"], [-1, -1]
    ...     ),
    ... }
    >>> swapped = symmetrise_records(raw, rng=np.random.default_rng(1))
    >>> [b.value for b in swapped[Party.BOB].bases]      # position 1 swapped
    ['X', 'Z']
    >>> [b.value for b in swapped[Party.CHARLIE].bases]
    ['Z', 'X']
    """
    bob, charlie = _checked_pair(records)
    generator = resolve_rng(rng)
    length = len(bob)

    coins = generator.integers(0, 2, size=length).astype(bool)

    bob_entries = [
        charlie.entries[index] if swap else bob.entries[index]
        for index, swap in enumerate(coins)
    ]
    charlie_entries = [
        bob.entries[index] if swap else charlie.entries[index]
        for index, swap in enumerate(coins)
    ]

    return {
        Party.BOB: RecipientRecord(
            party=Party.BOB,
            message_bit=bob.message_bit,
            entries=tuple(bob_entries),
            symmetrised=True,
        ),
        Party.CHARLIE: RecipientRecord(
            party=Party.CHARLIE,
            message_bit=charlie.message_bit,
            entries=tuple(charlie_entries),
            symmetrised=True,
        ),
    }


def no_symmetrisation(
    records: Mapping[Party | str, RecipientRecord],
    *,
    rng: np.random.Generator | None = None,
) -> dict[Party, RecipientRecord]:
    """Return the raw records unchanged. **Destroys non-repudiation.**

    For demonstrating the attack symmetrisation defends against. ``rng`` is unused.
    The records stay unsymmetrised, so ``verify_all`` refuses them unless
    ``require_symmetrised=False``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.protocol.params import Party
    >>> from qsecure.protocol.records import RecipientRecord
    >>> from qsecure.protocol.symmetrise import no_symmetrisation
    >>> raw = {
    ...     Party.BOB: RecipientRecord.from_measurements("Bob", 0, ["X"], [1]),
    ...     Party.CHARLIE: RecipientRecord.from_measurements(
    ...         "Charlie", 0, ["Z"], [-1]
    ...     ),
    ... }
    >>> kept = no_symmetrisation(raw)
    >>> kept[Party.BOB] is raw[Party.BOB], kept[Party.BOB].symmetrised
    (True, False)
    """
    del rng
    bob, charlie = _checked_pair(records)
    return {Party.BOB: bob, Party.CHARLIE: charlie}
