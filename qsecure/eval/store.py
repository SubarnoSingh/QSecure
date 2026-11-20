"""One file per trial, on disk, named from the trial's identity.

>>> import tempfile
>>> from pathlib import Path
>>> from qsecure.eval.store import ResultStore
>>> with tempfile.TemporaryDirectory() as tmp:
...     store = ResultStore(Path(tmp) / "results")
...     store.has("honest", "l192", 0)
...     store.path_for("honest", "l192", 42).name
False
'trial-000042.json'
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

from .records import TrialRecord
from .seeds import check_name

__all__ = [
    "DEFAULT_RESULTS_ROOT",
    "MANIFEST_GLOB",
    "ResultStore",
    "TRIAL_GLOB",
]

DEFAULT_RESULTS_ROOT: Final[Path] = Path.home() / ".qsecure" / "results"
"""pathlib.Path: Where a sweep writes when ``--results`` is not given."""

TRIAL_GLOB: Final[str] = "trial-*.json"
"""str: How a cell directory's trial files are found."""

MANIFEST_GLOB: Final[str] = "manifest-*.json"
"""str: How an experiment directory's run manifests are found."""


class ResultStore:
    """A results directory, addressed by trial identity.

    >>> import tempfile
    >>> from pathlib import Path
    >>> from qsecure.eval.store import ResultStore
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     store = ResultStore(Path(tmp) / "absent")
    ...     list(store.experiments()), list(store.cells("nothing"))
    ([], [])
    """

    def __init__(self, root: Path | str) -> None:
        if not isinstance(root, (Path, str)):
            raise TypeError(f"root must be a Path or str, got {type(root).__name__}")
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """pathlib.Path: The results root."""
        return self._root

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return f"ResultStore({str(self._root)!r})"

    def cell_dir(self, experiment: str, cell: str) -> Path:
        """Return the directory holding one cell's trials."""
        check_name(experiment, what="experiment")
        check_name(cell, what="cell")
        return self._root / experiment / cell

    def path_for(self, experiment: str, cell: str, index: int) -> Path:
        """Return the file one trial's record lives at."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"index must be an int, got {type(index).__name__}")
        if index < 0:
            raise ValueError(f"index must be non-negative, got {index}")
        return self.cell_dir(experiment, cell) / f"trial-{index:06d}.json"

    def has(self, experiment: str, cell: str, index: int) -> bool:
        """Return whether this trial is already on disk."""
        return self.path_for(experiment, cell, index).is_file()

    def write(self, record: TrialRecord) -> Path:
        """Write one record atomically and return where it went."""
        if not isinstance(record, TrialRecord):
            raise TypeError(
                f"record must be a TrialRecord, got {type(record).__name__}"
            )
        destination = self.path_for(*record.identity)
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=str(destination.parent), prefix=".trial-", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(record.to_dict(), stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return destination

    def read(self, experiment: str, cell: str, index: int) -> TrialRecord:
        """Read one record back."""
        path = self.path_for(experiment, cell, index)
        with path.open("r", encoding="utf-8") as stream:
            return TrialRecord.from_dict(json.load(stream))

    def experiments(self) -> list[str]:
        """Return the experiment names present, sorted."""
        if not self._root.is_dir():
            return []
        return sorted(p.name for p in self._root.iterdir() if p.is_dir())

    def cells(self, experiment: str) -> list[str]:
        """Return one experiment's cell names, sorted."""
        directory = self._root / experiment
        if not directory.is_dir():
            return []
        return sorted(p.name for p in directory.iterdir() if p.is_dir())

    def indices(self, experiment: str, cell: str) -> list[int]:
        """Return the trial indices present in one cell, sorted numerically."""
        directory = self.cell_dir(experiment, cell)
        if not directory.is_dir():
            return []
        found: list[int] = []
        for path in directory.glob(TRIAL_GLOB):
            stem = path.stem.removeprefix("trial-")
            if stem.isdigit():
                found.append(int(stem))
        return sorted(found)

    def read_cell(self, experiment: str, cell: str) -> list[TrialRecord]:
        """Read one cell's records, in trial-index order."""
        return [
            self.read(experiment, cell, index)
            for index in self.indices(experiment, cell)
        ]

    def read_experiment(self, experiment: str) -> Iterator[TrialRecord]:
        """Yield every record in one experiment, cell by cell, index by index."""
        for cell in self.cells(experiment):
            yield from self.read_cell(experiment, cell)

    def count(self, experiment: str, cell: str | None = None) -> int:
        """Return how many trials are on disk."""
        if cell is not None:
            return len(self.indices(experiment, cell))
        return sum(len(self.indices(experiment, c)) for c in self.cells(experiment))

    def write_manifest(self, experiment: str, name: str, payload: Any) -> Path:
        """Write a run manifest into the experiment's directory."""
        check_name(experiment, what="experiment")
        directory = self._root / experiment
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.json"
        handle, temporary = tempfile.mkstemp(
            dir=str(directory), prefix=".manifest-", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True, default=str)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return path

    def manifests(self, experiment: str) -> list[Path]:
        """Return one experiment's manifest files, oldest name first."""
        directory = self._root / experiment
        if not directory.is_dir():
            return []
        return sorted(directory.glob(MANIFEST_GLOB))
