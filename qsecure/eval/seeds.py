"""Where every trial's randomness comes from, written down so it can be checked.

>>> from qsecure.eval.seeds import trial_seed, trial_seeds
>>> trial_seed("honest", "l192", 0)
15226155242793161573
>>> trial_seed("honest", "l192", 1)
10002029503400317596
>>> trial_seed("honest", "l192", 0, role="adversary")
7898412172115910904

The pair a trial actually receives:
>>> pair = trial_seeds("honest", "l192", 0)
>>> pair.session == trial_seed("honest", "l192", 0)
True
>>> pair.adversary == trial_seed("honest", "l192", 0, role="adversary")
True
>>> pair.session == pair.adversary
False
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from qsecure.core.rng import seed_to_generator

__all__ = [
    "ADVERSARY_ROLE",
    "NAME_PATTERN",
    "SEED_DOMAIN",
    "SEED_PERSON",
    "SESSION_ROLE",
    "TrialSeeds",
    "check_name",
    "trial_seed",
    "trial_seeds",
]

SEED_DOMAIN: Final[bytes] = b"qsecure/eval/seed/v1"
"""bytes: Domain separator, versioned. Bumping it re-seeds every experiment."""

SEED_PERSON: Final[bytes] = b"qsecure-eval"
"""bytes: BLAKE2b personalisation, at most 16 bytes."""

SESSION_ROLE: Final[str] = "session"
"""str: The role whose generator drives the protocol run itself."""

ADVERSARY_ROLE: Final[str] = "adversary"
"""str: The role whose generator drives the attack, and only the attack."""

NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"\A[a-z0-9][a-z0-9._-]{0,63}\Z")
"""re.Pattern: Legal experiment, cell and role names."""


def check_name(value: Any, *, what: str = "name") -> str:
    """Return ``value`` if it is a legal identity component, else raise.

    >>> from qsecure.eval.seeds import check_name
    >>> check_name("qber-sweep")
    'qber-sweep'
    >>> check_name("Two Words")
    Traceback (most recent call last):
        ...
    ValueError: name 'Two Words' is not a legal identity component...
    """
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a str, got {type(value).__name__}")
    if NAME_PATTERN.match(value) is None:
        raise ValueError(
            f"{what} {value!r} is not a legal identity component: it must "
            f"match {NAME_PATTERN.pattern}. Names are separated by a NUL byte "
            f"inside the seed material and are used verbatim as directory "
            f"components, so anything looser makes two trials collide or "
            f"escape the results directory."
        )
    return value


def trial_seed(
    experiment: str, cell: str, index: int, *, role: str = SESSION_ROLE
) -> int:
    """Derive one trial's 64-bit seed from its identity alone.

    >>> from qsecure.eval.seeds import trial_seed
    >>> seed = trial_seed("demo", "cell-a", 3)
    >>> 0 <= seed < 2 ** 64
    True

    Identity is the whole input, so every component moves the seed:
    >>> len({
    ...     trial_seed("demo", "cell-a", 3),
    ...     trial_seed("demo", "cell-b", 3),
    ...     trial_seed("other", "cell-a", 3),
    ...     trial_seed("demo", "cell-a", 4),
    ...     trial_seed("demo", "cell-a", 3, role="adversary"),
    ... })
    5

    >>> trial_seed("ab", "c", 0) == trial_seed("a", "bc", 0)
    False
    """
    check_name(experiment, what="experiment")
    check_name(cell, what="cell")
    check_name(role, what="role")
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError(f"index must be an int, got {type(index).__name__}")
    if index < 0:
        raise ValueError(f"index must be non-negative, got {index}")
    material = b"\x00".join(
        (
            SEED_DOMAIN,
            experiment.encode("utf-8"),
            cell.encode("utf-8"),
            role.encode("utf-8"),
            str(index).encode("ascii"),
        )
    )
    digest = hashlib.blake2b(material, digest_size=8, person=SEED_PERSON).digest()
    return int.from_bytes(digest, "big")


@dataclass(frozen=True)
class TrialSeeds:
    """The two seeds one trial runs under, kept apart on purpose.

    >>> from qsecure.eval.seeds import TrialSeeds, trial_seeds
    >>> pair = trial_seeds("demo", "cell-a", 0)
    >>> pair.experiment, pair.cell, pair.index
    ('demo', 'cell-a', 0)
    >>> sorted(pair.to_dict())
    ['adversary', 'cell', 'experiment', 'index', 'session']

    >>> import numpy as np
    >>> generator = pair.session_rng()
    >>> isinstance(generator, np.random.Generator)
    True
    >>> pair.session_rng() is generator
    False
    """

    experiment: str
    cell: str
    index: int
    session: int
    adversary: int

    def session_rng(self) -> np.random.Generator:
        """Return a fresh generator seeded from :attr:`session`."""
        return seed_to_generator(self.session)

    def adversary_rng(self) -> np.random.Generator:
        """Return a fresh generator seeded from :attr:`adversary`."""
        return seed_to_generator(self.adversary)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "experiment": self.experiment,
            "cell": self.cell,
            "index": self.index,
            "session": self.session,
            "adversary": self.adversary,
        }


def trial_seeds(experiment: str, cell: str, index: int) -> TrialSeeds:
    """Derive the whole seed set for one trial.

    >>> from qsecure.eval.seeds import trial_seeds
    >>> trial_seeds("honest", "l192", 0).session
    15226155242793161573
    """
    return TrialSeeds(
        experiment=experiment,
        cell=cell,
        index=index,
        session=trial_seed(experiment, cell, index, role=SESSION_ROLE),
        adversary=trial_seed(experiment, cell, index, role=ADVERSARY_ROLE),
    )
