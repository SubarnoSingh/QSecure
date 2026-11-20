"""The parallel runner: a persistent pool, pinned BLAS, and a resumable sweep.

>>> import tempfile
>>> from pathlib import Path
>>> from qsecure.eval.experiments import experiment
>>> from qsecure.eval.runner import plan_trials, run_experiment
>>> from qsecure.eval.store import ResultStore
>>> with tempfile.TemporaryDirectory() as tmp:
...     store = ResultStore(Path(tmp))
...     smoke = experiment("smoke")
...     first = plan_trials(smoke, store, trials=3)
...     summary = run_experiment(smoke, store, trials=3, workers=1,
...                              in_process=True, quiet=True)
...     second = plan_trials(smoke, store, trials=3)
...     [spec.index for spec in first]
...     (summary["ran"], summary["skipped"], summary["failed"])
...     [spec.index for spec in second]
[0, 1, 2]
(3, 0, 0)
[]

Nothing left to do is not an error -- it is what a resumed sweep looks like:
>>> with tempfile.TemporaryDirectory() as tmp:
...     store = ResultStore(Path(tmp))
...     smoke = experiment("smoke")
...     _ = run_experiment(smoke, store, trials=2, workers=1,
...                        in_process=True, quiet=True)
...     again = run_experiment(smoke, store, trials=2, workers=1,
...                            in_process=True, quiet=True)
...     (again["ran"], again["skipped"])
(0, 2)
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

from .experiments import Experiment, all_cells, experiment, run_trial
from .store import ResultStore

__all__ = [
    "BLAS_ENV_VARS",
    "TrialOutcome",
    "TrialSpec",
    "global_rng_fingerprint",
    "default_results_root",
    "pin_blas_threads",
    "plan_trials",
    "probe_pool",
    "run_experiment",
    "worker_probe",
]

BLAS_ENV_VARS: Final[tuple[str, ...]] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
"""tuple of str: Every thread-limit variable a numpy build might read."""


def pin_blas_threads(threads: int = 1) -> dict[str, Any]:
    """Limit every worker's BLAS pool, and report whether it can have worked.

    >>> from qsecure.eval.runner import pin_blas_threads
    >>> report = pin_blas_threads(1)
    >>> report["threads"], report["values"]["OMP_NUM_THREADS"]
    (1, '1')
    >>> sorted(report)
    ['numpy_already_imported', 'threads', 'values']
    """
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError(f"threads must be a positive int, got {threads!r}")
    for name in BLAS_ENV_VARS:
        os.environ[name] = str(threads)
    return {
        "threads": threads,
        "values": {name: os.environ.get(name) for name in BLAS_ENV_VARS},
        "numpy_already_imported": "numpy" in sys.modules,
    }


def global_rng_fingerprint() -> str:
    """Return a digest of the two global random states nothing may touch."""
    import random as _random

    import numpy as _np

    state = _np.random.get_state()
    keys = _np.asarray(state[1]).tobytes()
    payload = repr((state[0], state[2], state[3], state[4], _random.getstate()))
    digest = hashlib.sha256(keys + payload.encode("utf-8")).hexdigest()
    return digest[:16]


def worker_probe() -> dict[str, Any]:
    """Report what one worker process actually sees. Runs inside the pool."""
    return {
        "pid": os.getpid(),
        "blas_env": {name: os.environ.get(name) for name in BLAS_ENV_VARS},
        "numpy_imported_at_start": "numpy" in sys.modules,
        "global_rng": global_rng_fingerprint(),
    }


@dataclass(frozen=True)
class TrialSpec:
    """One unit of work: which trial, and where to put it."""

    experiment: str
    cell: str
    index: int
    results_root: str
    retain_transcript: bool | None = None


@dataclass(frozen=True)
class TrialOutcome:
    """What a worker reports back once a trial is on disk."""

    experiment: str
    cell: str
    index: int
    fingerprint: str | None
    seconds: float
    pid: int
    touched_global_rng: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "experiment": self.experiment,
            "cell": self.cell,
            "index": self.index,
            "fingerprint": self.fingerprint,
            "seconds": self.seconds,
            "pid": self.pid,
            "touched_global_rng": self.touched_global_rng,
            "error": self.error,
        }


def _execute(spec: TrialSpec) -> TrialOutcome:
    """Run one trial in whatever process picks it up, and write it."""
    started = time.perf_counter()
    before = global_rng_fingerprint()
    try:
        exp = experiment(spec.experiment)
        record = run_trial(
            exp, spec.cell, spec.index, retain_transcript=spec.retain_transcript
        )
        ResultStore(spec.results_root).write(record)
        fingerprint: str | None = record.fingerprint()
        error: str | None = None
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        fingerprint = None
        error = f"{type(exc).__name__}: {exc}"
    return TrialOutcome(
        experiment=spec.experiment,
        cell=spec.cell,
        index=spec.index,
        fingerprint=fingerprint,
        seconds=time.perf_counter() - started,
        pid=os.getpid(),
        touched_global_rng=global_rng_fingerprint() != before,
        error=error,
    )


def plan_trials(
    exp: Experiment,
    store: ResultStore,
    *,
    trials: int | None = None,
    cells: Sequence[str] | None = None,
    retain_transcript: bool | None = None,
) -> list[TrialSpec]:
    """Return the work still to do, skipping what is already on disk.

    >>> import tempfile
    >>> from pathlib import Path
    >>> from qsecure.eval.experiments import experiment
    >>> from qsecure.eval.runner import plan_trials
    >>> from qsecure.eval.store import ResultStore
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     specs = plan_trials(experiment("honest"), ResultStore(Path(tmp)),
    ...                         trials=2, cells=["l192", "l384"])
    ...     [(s.cell, s.index) for s in specs]
    [('l192', 0), ('l192', 1), ('l384', 0), ('l384', 1)]
    """
    count = exp.trials if trials is None else int(trials)
    if count <= 0:
        raise ValueError(f"trials must be positive, got {count}")
    specs: list[TrialSpec] = []
    for cell in all_cells(exp, cells):
        for index in range(count):
            if store.has(exp.name, cell.name, index):
                continue
            specs.append(
                TrialSpec(
                    experiment=exp.name,
                    cell=cell.name,
                    index=index,
                    results_root=str(store.root),
                    retain_transcript=retain_transcript,
                )
            )
    return specs


def _report(message: str, quiet: bool) -> None:
    """Write a progress line to stderr."""
    if not quiet:
        print(message, file=sys.stderr, flush=True)


def run_experiment(
    exp: Experiment,
    store: ResultStore,
    *,
    trials: int | None = None,
    cells: Sequence[str] | None = None,
    workers: int = 1,
    chunk: int = 1,
    in_process: bool = False,
    retain_transcript: bool | None = None,
    quiet: bool = False,
    progress_every: int = 1,
) -> dict[str, Any]:
    """Run every outstanding trial of one experiment and return what happened."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"workers must be a positive int, got {workers!r}")
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError(f"chunk must be a positive int, got {chunk!r}")

    requested = exp.trials if trials is None else int(trials)
    planned = plan_trials(
        exp,
        store,
        trials=requested,
        cells=cells,
        retain_transcript=retain_transcript,
    )
    selected = all_cells(exp, cells)
    skipped = requested * len(selected) - len(planned)

    outcomes: list[TrialOutcome] = []
    started = time.perf_counter()
    total = len(planned)
    if total:
        _report(
            f"[sweep] {exp.name}: {total} trials to run, {skipped} already on "
            f"disk, {workers} worker(s)"
            + (" (in-process)" if in_process else ""),
            quiet,
        )

    def note(done: int, outcome: TrialOutcome) -> None:
        if done % max(1, progress_every) and done != total:
            return
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (total - done) / rate if rate > 0 else float("nan")
        _report(
            f"[sweep] {done}/{total} {outcome.cell}#{outcome.index} "
            f"{outcome.seconds:.2f}s pid={outcome.pid} "
            f"| {rate:.2f} trial/s, ~{remaining / 60:.1f} min left",
            quiet,
        )

    try:
        if in_process or not planned:
            for done, spec in enumerate(planned, start=1):
                outcome = _execute(spec)
                outcomes.append(outcome)
                note(done, outcome)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for done, outcome in enumerate(
                    pool.map(_execute, planned, chunksize=chunk), start=1
                ):
                    outcomes.append(outcome)
                    note(done, outcome)
    except KeyboardInterrupt:
        _report(
            f"[sweep] interrupted after {len(outcomes)}/{total}; every "
            f"finished trial is on disk and the next run resumes from there.",
            quiet,
        )
        raise

    wall = time.perf_counter() - started
    failed = [o for o in outcomes if o.error is not None]
    touched = [o for o in outcomes if o.touched_global_rng]
    if touched:
        _report(
            f"[sweep] RANDOMNESS VIOLATION: {len(touched)} trial(s) moved a global "
            f"random state. Results are not reproducible from their seeds.",
            quiet,
        )
    return {
        "ran": len(outcomes) - len(failed),
        "skipped": skipped,
        "failed": len(failed),
        "wall_clock_seconds": wall,
        "outcomes": [o.to_dict() for o in outcomes],
        "errors": [f"{o.cell}#{o.index}: {o.error}" for o in failed],
        "global_rng_touched": [f"{o.cell}#{o.index}" for o in touched],
    }


def probe_pool(workers: int = 1) -> list[dict[str, Any]]:
    """Ask each of ``workers`` processes what its environment looks like."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"workers must be a positive int, got {workers!r}")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_probe_task, range(workers * 2)))


def _probe_task(_: int) -> dict[str, Any]:
    """Run :func:`worker_probe` in a pool worker."""
    time.sleep(0.02)
    return worker_probe()



def default_results_root() -> Path:
    """Return the results directory a sweep writes to when none is given."""
    from .store import DEFAULT_RESULTS_ROOT

    return DEFAULT_RESULTS_ROOT

