"""The run manifest: what makes a published number reproducible.

>>> from qsecure.eval.manifest import RunManifest, machine_facts
>>> facts = machine_facts()
>>> sorted(facts)
['blas_env', 'cpu_count_logical', 'machine', 'numpy', 'platform', 'processor', 'python']
>>> manifest = RunManifest.begin(
...     experiment="honest",
...     cells=("l192",),
...     trials=4,
...     workers=1,
...     results_root="/tmp/results",
...     eps=1e-9,
...     command_line="python run.py run honest --trials 4",
... )
>>> manifest.finished_at is None
True
>>> done = manifest.finish(ran=4, skipped=0, failed=0, wall_clock_seconds=1.5)
>>> done.finished_at is None
False
>>> done.to_dict()["trials_ran"], done.to_dict()["trials_skipped"]
(4, 0)
>>> done.to_dict()["seed_derivation"]["domain"]
'qsecure/eval/seed/v1'
"""

from __future__ import annotations

import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Sequence

from .seeds import ADVERSARY_ROLE, NAME_PATTERN, SEED_DOMAIN, SEED_PERSON, SESSION_ROLE

__all__ = [
    "HARNESS_VERSION",
    "MANIFEST_SCHEMA",
    "RunManifest",
    "command_line",
    "git_state",
    "machine_facts",
    "seed_derivation",
    "utc_now",
]

HARNESS_VERSION: Final[str] = "1.0.0"
"""str: Version of the evaluation harness itself, recorded in every manifest."""

MANIFEST_SCHEMA: Final[str] = "qsecure/eval/manifest/v1"
"""str: Schema tag, so a reader can refuse a manifest it does not understand."""

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def git_state(root: Path | None = None) -> dict[str, Any]:
    """Return the commit the code is at and whether the tree was dirty."""
    directory = Path(root) if root is not None else _REPO_ROOT
    state: dict[str, Any] = {"commit": None, "dirty": None, "branch": None}

    def ask(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args],
                cwd=str(directory),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if done.returncode != 0:
            return None
        return done.stdout.strip()

    state["commit"] = ask("rev-parse", "HEAD")
    state["branch"] = ask("rev-parse", "--abbrev-ref", "HEAD")
    status = ask("status", "--porcelain")
    if status is not None:
        state["dirty"] = status != ""
    return state


def machine_facts() -> dict[str, Any]:
    """Return what the run was executed on."""
    import numpy as np

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "blas_env": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }


def seed_derivation() -> dict[str, Any]:
    """Return the seed rule in a form a reviewer can re-implement."""
    return {
        "function": "qsecure.eval.seeds.trial_seed",
        "domain": SEED_DOMAIN.decode("ascii"),
        "person": SEED_PERSON.decode("ascii"),
        "hash": "blake2b",
        "digest_size_bytes": 8,
        "separator": "0x00",
        "material": (
            "domain || 0x00 || utf8(experiment) || 0x00 || utf8(cell) || 0x00 "
            "|| utf8(role) || 0x00 || ascii(decimal(index))"
        ),
        "roles": [SESSION_ROLE, ADVERSARY_ROLE],
        "name_pattern": NAME_PATTERN.pattern,
    }


def command_line(argv: Sequence[str] | None = None) -> str:
    """Return the command that started this process, quoted for a shell.

    >>> from qsecure.eval.manifest import command_line
    >>> command_line(["run.py", "run", "honest", "--trials", "4"])
    'python run.py run honest --trials 4'
    >>> command_line(["a b/sweep.py", "run"])
    "python 'a b/sweep.py' run"
    """
    parts = list(sys.argv if argv is None else argv)
    return "python " + " ".join(shlex.quote(part) for part in parts)


_default_command_line = command_line
"""Alias, so ``RunManifest.begin``'s same-named parameter cannot shadow it."""


@dataclass(frozen=True)
class RunManifest:
    """Everything needed to re-run a sweep, written before and after it."""

    schema: str
    harness_version: str
    experiment: str
    cells: tuple[str, ...]
    trials: int
    workers: int
    results_root: str
    eps: float
    command_line: str
    git: dict[str, Any]
    machine: dict[str, Any]
    seed_derivation: dict[str, Any]
    started_at: str
    finished_at: str | None = None
    trials_ran: int = 0
    trials_skipped: int = 0
    trials_failed: int = 0
    wall_clock_seconds: float | None = None
    notes: str = ""

    @classmethod
    def begin(
        cls,
        *,
        experiment: str,
        cells: Sequence[str],
        trials: int,
        workers: int,
        results_root: Path | str,
        eps: float,
        command_line: str | None = None,
        notes: str = "",
    ) -> RunManifest:
        """Open a manifest at the start of a run."""
        return cls(
            schema=MANIFEST_SCHEMA,
            harness_version=HARNESS_VERSION,
            experiment=experiment,
            cells=tuple(cells),
            trials=int(trials),
            workers=int(workers),
            results_root=str(results_root),
            eps=float(eps),
            command_line=(
                _default_command_line() if command_line is None else command_line
            ),
            git=git_state(),
            machine=machine_facts(),
            seed_derivation=seed_derivation(),
            started_at=utc_now(),
            notes=notes,
        )

    def finish(
        self,
        *,
        ran: int,
        skipped: int,
        failed: int,
        wall_clock_seconds: float,
    ) -> RunManifest:
        """Return a closed copy of this manifest."""
        return replace(
            self,
            finished_at=utc_now(),
            trials_ran=int(ran),
            trials_skipped=int(skipped),
            trials_failed=int(failed),
            wall_clock_seconds=float(wall_clock_seconds),
        )

    def filename(self) -> str:
        """Return the stem this manifest should be stored under.

        >>> from qsecure.eval.manifest import RunManifest
        >>> manifest = RunManifest.begin(
        ...     experiment="honest", cells=("l192",), trials=1, workers=1,
        ...     results_root=".", eps=1e-9, command_line="python x",
        ... )
        >>> manifest.filename().startswith("manifest-")
        True
        >>> ":" in manifest.filename()
        False
        """
        stamp = self.started_at.replace(":", "").replace("-", "")
        return f"manifest-{stamp}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "schema": self.schema,
            "harness_version": self.harness_version,
            "experiment": self.experiment,
            "cells": list(self.cells),
            "trials": self.trials,
            "workers": self.workers,
            "results_root": self.results_root,
            "eps": self.eps,
            "command_line": self.command_line,
            "git": self.git,
            "machine": self.machine,
            "seed_derivation": self.seed_derivation,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "trials_ran": self.trials_ran,
            "trials_skipped": self.trials_skipped,
            "trials_failed": self.trials_failed,
            "wall_clock_seconds": self.wall_clock_seconds,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        """Rebuild from :meth:`to_dict` output."""
        if data.get("schema") != MANIFEST_SCHEMA:
            raise ValueError(
                f"manifest schema {data.get('schema')!r} is not "
                f"{MANIFEST_SCHEMA!r}"
            )
        return cls(
            schema=str(data["schema"]),
            harness_version=str(data["harness_version"]),
            experiment=str(data["experiment"]),
            cells=tuple(data["cells"]),
            trials=int(data["trials"]),
            workers=int(data["workers"]),
            results_root=str(data["results_root"]),
            eps=float(data["eps"]),
            command_line=str(data["command_line"]),
            git=dict(data["git"]),
            machine=dict(data["machine"]),
            seed_derivation=dict(data["seed_derivation"]),
            started_at=str(data["started_at"]),
            finished_at=data.get("finished_at"),
            trials_ran=int(data.get("trials_ran", 0)),
            trials_skipped=int(data.get("trials_skipped", 0)),
            trials_failed=int(data.get("trials_failed", 0)),
            wall_clock_seconds=data.get("wall_clock_seconds"),
            notes=str(data.get("notes", "")),
        )
