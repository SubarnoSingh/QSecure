"""Single point of control for randomness.

Every random function takes a keyword-only ``rng``. :func:`resolve_rng` accepts
a generator or ``None``; :func:`seed_to_generator` turns a scalar seed from a
CLI/config into a generator, once, at the boundary.
"""

from __future__ import annotations

import numpy as np

__all__ = ["resolve_rng", "seed_to_generator"]


def resolve_rng(rng: np.random.Generator | None = None) -> np.random.Generator:
    """Return ``rng`` itself (same object, never a copy), or a fresh generator for ``None``.

    Raises
    ------
    TypeError
        For anything else; integer seeds and ``RandomState`` are rejected because
        re-seeding per call would restart the stream.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.core.rng import resolve_rng
    >>> generator = resolve_rng(np.random.default_rng(0))
    >>> first = generator.integers(100)
    >>> int(first) == int(np.random.default_rng(0).integers(100))
    True
    >>> isinstance(resolve_rng(None), np.random.Generator)
    True
    """
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, (int, np.integer)):
        raise TypeError(
            f"rng must be a numpy.random.Generator or None, got the seed "
            f"{rng!r}. Integer seeds are rejected on purpose: build the "
            f"generator once with numpy.random.default_rng({int(rng)}) and pass "
            f"that object, so repeated calls advance one stream instead of "
            f"restarting it."
        )
    if isinstance(rng, np.random.RandomState):
        raise TypeError(
            "rng must be a numpy.random.Generator or None, got the legacy "
            "numpy.random.RandomState. Replace it with "
            "numpy.random.default_rng(seed); the legacy API is not used "
            "anywhere in this project."
        )
    raise TypeError(
        f"rng must be a numpy.random.Generator or None, got "
        f"{type(rng).__name__}. Pass numpy.random.default_rng(seed) for "
        f"reproducible runs, or None for fresh entropy."
    )


def seed_to_generator(seed: int | None) -> np.random.Generator:
    """Convert a scalar seed from outside (CLI, JSON, config) into a generator.

    Call it once per run, never per call site; generators are rejected so the
    two entry points cannot be confused.

    Raises
    ------
    ValueError
        If ``seed`` is negative.
    TypeError
        If ``seed`` is a bool, a generator, or any non-integer.

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.core.rng import seed_to_generator
    >>> generator = seed_to_generator(20260141)
    >>> reference = np.random.default_rng(20260141)
    >>> bool(np.array_equal(generator.random(4), reference.random(4)))
    True
    >>> isinstance(seed_to_generator(None), np.random.Generator)
    True
    """
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, np.random.Generator):
        raise TypeError(
            "seed_to_generator takes a scalar seed, not a "
            "numpy.random.Generator. Pass the generator to resolve_rng (or "
            "straight to the rng= argument) instead; this function exists only "
            "to convert a scalar from outside the project."
        )
    if isinstance(seed, bool):
        raise TypeError(
            f"seed must be a non-negative integer or None, got the boolean "
            f"{seed!r}. Booleans are integers in Python but never a meaningful "
            f"seed; pass an explicit integer such as 20260141."
        )
    if not isinstance(seed, (int, np.integer)):
        raise TypeError(
            f"seed must be a non-negative integer or None, got "
            f"{type(seed).__name__}. Convert command-line and JSON values with "
            f"int(...) at the parser, so a malformed seed fails there rather "
            f"than seeding something unintended here."
        )
    value = int(seed)
    if value < 0:
        raise ValueError(
            f"seed must be non-negative, got {value}. numpy's SeedSequence "
            f"accepts only non-negative integers; use abs({value}) or a "
            f"different seed."
        )
    return np.random.default_rng(value)
