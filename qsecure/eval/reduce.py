"""Reduce stored records to results tables, refusing the ways a table can mislead.

>>> import tempfile
>>> from pathlib import Path
>>> from qsecure.eval.experiments import experiment
>>> from qsecure.eval.reduce import outcome_table, reduce_experiment
>>> from qsecure.eval.runner import run_experiment
>>> from qsecure.eval.store import ResultStore
>>> with tempfile.TemporaryDirectory() as tmp:
...     store = ResultStore(Path(tmp))
...     _ = run_experiment(experiment("smoke"), store, trials=3, workers=1,
...                        in_process=True, quiet=True)
...     records = list(store.read_experiment("smoke"))
...     table = outcome_table(records)
...     len(records)
...     table.columns
...     table.rows[0][:6]
3
('cell', 'count_exchange_timing', 'runs', 'parties', 'accepted', 'rejected', 'refused', 'not_asked')
('tiny', 'before-forwarding', 3, 6, 6, 0)

>>> with tempfile.TemporaryDirectory() as tmp:
...     store = ResultStore(Path(tmp))
...     _ = run_experiment(experiment("smoke"), store, trials=3, workers=1,
...                        in_process=True, quiet=True)
...     tables = reduce_experiment(store, "smoke")
...     [t.slug for t in tables]
['outcomes', 'detection', 'timing']
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Final, Iterable, Mapping, Sequence

from qsecure.detect.statistics import wilson_interval
from qsecure.detect.thresholds_structural import NoVerdictCount, OutcomeTally

from .experiments import EXPERIMENTS
from .records import (
    UNDETECTABLE_ASSUMPTION,
    UNDETECTABLE_BY_CONSTRUCTION,
    TrialRecord,
)
from .store import ResultStore

__all__ = [
    "CONFIDENCE",
    "EXTRA_CHARTS",
    "EXTRA_REDUCTIONS",
    "NOT_EVALUATED",
    "VARIANT_MARKERS",
    "Table",
    "completeness_note",
    "detection_table",
    "escape_xml",
    "expected_trials",
    "group_records",
    "measured_rate",
    "outcome_table",
    "reduce_experiment",
    "refused_run",
    "tally_from_record",
    "timing_table",
]

EXTRA_REDUCTIONS: Final[dict[str, Any]] = {}
"""dict: Per-experiment reductions, appended after the three standard tables."""

EXTRA_CHARTS: Final[dict[str, Any]] = {}
"""dict: Per-experiment chart renderers, keyed by experiment name."""

CONFIDENCE: Final[float] = 0.99
"""float: Confidence level for every measured rate's interval."""

NOT_EVALUATED: Final[str] = "not evaluated"
"""str: What a withheld detector family prints. Never ``passed``, never blank."""


@dataclass(frozen=True)
class Table:
    """One results table, with the command that regenerates it attached.

    >>> from qsecure.eval.reduce import Table
    >>> table = Table(
    ...     slug="demo", title="Demo", columns=("a", "b"),
    ...     rows=((1, 2), (3, 4)),
    ...     command="python run.py reduce demo",
    ...     notes=("n = 2 rows",),
    ... )
    >>> print(table.to_markdown())
    ### Demo
    <BLANKLINE>
    | a | b |
    | --- | --- |
    | 1 | 2 |
    | 3 | 4 |
    <BLANKLINE>
    Regenerate: `python run.py reduce demo`
    <BLANKLINE>
    - n = 2 rows
    """

    slug: str
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    command: str = ""
    notes: tuple[str, ...] = ()

    def to_markdown(self) -> str:
        """Render as a GitHub-flavoured markdown table."""
        lines = [f"### {self.title}", ""]
        lines.append("| " + " | ".join(self.columns) + " |")
        lines.append("| " + " | ".join("---" for _ in self.columns) + " |")
        for row in self.rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        lines.append("")
        if self.command:
            lines.append(f"Regenerate: `{self.command}`")
        else:
            lines.append(
                "Regenerate: **no command recorded -- this table is not "
                "reproducible and must not be published**"
            )
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"- {note}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "slug": self.slug,
            "title": self.title,
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
            "command": self.command,
            "notes": list(self.notes),
        }


VARIANT_MARKERS: Final[tuple[str, ...]] = (
    "count_exchange_timing",
    "symmetrised",
    "counts_exchanged",
    "signer_saw_recipient_logs",
    "security_claim",
)
"""tuple of str: The names of ``Detection.grouping_key``'s five elements."""


def _refuse_mixed_variant_markers(
    grouped: Mapping[tuple[str, str], Sequence[TrialRecord]],
) -> None:
    """Refuse to hand back a group that averages over a variant marker."""
    for (cell, timing), group in grouped.items():
        seen: dict[str, set[Any]] = {}
        for record in group:
            key = record.detection.get("grouping_key")
            if not isinstance(key, (list, tuple)) or len(key) != len(
                VARIANT_MARKERS
            ):
                continue
            for name, value in zip(VARIANT_MARKERS[1:], tuple(key)[1:]):
                seen.setdefault(name, set()).add(value)
        mixed = sorted(name for name, values in seen.items() if len(values) > 1)
        if mixed:
            raise ValueError(
                f"cell {cell!r} ({timing}) mixes two values of "
                + ", ".join(mixed)
                + ": these are grouping_key markers and a table must not "
                "average over them. Split the cell, or widen the group key."
            )


def group_records(
    records: Iterable[TrialRecord],
    *,
    cell_order: Sequence[str] | None = None,
) -> dict[tuple[str, str], list[TrialRecord]]:
    """Group records by ``(cell, count_exchange_timing)``.

    >>> import tempfile
    >>> from pathlib import Path
    >>> from qsecure.eval.experiments import experiment
    >>> from qsecure.eval.reduce import group_records
    >>> from qsecure.eval.runner import run_experiment
    >>> from qsecure.eval.store import ResultStore
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     store = ResultStore(Path(tmp))
    ...     _ = run_experiment(experiment("smoke"), store, trials=2, workers=1,
    ...                        in_process=True, quiet=True)
    ...     sorted(group_records(store.read_experiment("smoke")))
    [('tiny', 'before-forwarding')]
    """
    grouped: dict[tuple[str, str], list[TrialRecord]] = {}
    for record in records:
        key = (record.cell, record.count_exchange_timing)
        grouped.setdefault(key, []).append(record)
    _refuse_mixed_variant_markers(grouped)
    for group in grouped.values():
        group.sort(key=lambda r: r.index)
    rank = {name: i for i, name in enumerate(cell_order or ())}
    unknown = len(rank)
    return dict(
        sorted(
            grouped.items(),
            key=lambda item: (
                rank.get(item[0][0], unknown),
                item[0][0],
                item[0][1],
            ),
        )
    )


def tally_from_record(record: TrialRecord) -> OutcomeTally:
    """Build one run's outcome tally from its transcript summary.

    >>> import tempfile
    >>> from pathlib import Path
    >>> from qsecure.eval.experiments import experiment
    >>> from qsecure.eval.reduce import tally_from_record
    >>> from qsecure.eval.runner import run_experiment
    >>> from qsecure.eval.store import ResultStore
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     store = ResultStore(Path(tmp))
    ...     _ = run_experiment(experiment("smoke"), store, trials=1, workers=1,
    ...                        in_process=True, quiet=True)
    ...     tally = tally_from_record(store.read("smoke", "tiny", 0))
    ...     (tally.accepted, tally.rejected, tally.refused, tally.not_asked)
    (2, 0, 0, 0)
    """
    verdicts: Mapping[str, str] = record.transcript_summary["verdicts"]
    accepted = sum(1 for v in verdicts.values() if v == "accepted")
    rejected = sum(1 for v in verdicts.values() if v == "rejected")
    refused = sum(1 for v in verdicts.values() if v == "refused")
    not_asked = 2 - len(verdicts)
    return OutcomeTally(
        accepted=accepted,
        rejected=rejected,
        refused=NoVerdictCount(refused),
        not_asked=NoVerdictCount(max(0, not_asked)),
        count_exchange_timing=record.count_exchange_timing,
        runs=1,
    )


def outcome_table(
    records: Sequence[TrialRecord],
    *,
    command: str = "",
    cell_order: Sequence[str] | None = None,
) -> Table:
    """Tabulate what actually happened to each run, aborts kept apart.

    >>> import tempfile
    >>> from pathlib import Path
    >>> from qsecure.eval.experiments import experiment
    >>> from qsecure.eval.reduce import outcome_table
    >>> from qsecure.eval.runner import run_experiment
    >>> from qsecure.eval.store import ResultStore
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     store = ResultStore(Path(tmp))
    ...     _ = run_experiment(experiment("smoke"), store, trials=2, workers=1,
    ...                        in_process=True, quiet=True)
    ...     table = outcome_table(list(store.read_experiment("smoke")))
    ...     table.rows
    (('tiny', 'before-forwarding', 2, 4, 4, 0, 0, 0),)
    """
    if not records:
        raise ValueError("outcome_table needs at least one record")
    rows: list[tuple[Any, ...]] = []
    for (cell, timing), group in group_records(
        records, cell_order=cell_order
    ).items():
        tally = tally_from_record(group[0])
        for record in group[1:]:
            tally = tally.merge(tally_from_record(record))
        rows.append(
            (
                cell,
                timing,
                tally.runs,
                tally.verdicts + int(tally.no_verdicts),
                tally.accepted,
                tally.rejected,
                int(tally.refused),
                int(tally.not_asked),
            )
        )
    return Table(
        slug="outcomes",
        title="Run outcomes, by cell and count ordering",
        columns=(
            "cell",
            "count_exchange_timing",
            "runs",
            "parties",
            "accepted",
            "rejected",
            "refused",
            "not_asked",
        ),
        rows=tuple(rows),
        command=command,
        notes=(
            "`parties` is the denominator, and it is two verifiers per run "
            "-- every run has a Bob and a Charlie whatever became of them. "
            "accepted + rejected + refused + not_asked = parties, exactly. "
            "`not_asked` is a party the protocol never put the question to, "
            "which is a third thing again from a refusal.",
            "`refused` is a no-verdict, not a rejection. A run where a "
            "verifier could not score the declaration is neither an "
            "acceptance nor a detection, and it is never folded into either.",
            "`count_exchange_timing` is a column, never averaged over: the "
            "two orderings give different answers to the same attack.",
        ),
    )


def measured_rate(successes: int, trials: int, *, places: int = 3) -> str:
    """Format a measured rate with its interval, or say why there is none.

    >>> from qsecure.eval.reduce import measured_rate
    >>> measured_rate(0, 40)
    '0/40 = 0.000 [0.000, 0.142]'

    An empty denominator is an absence, not a zero:
    >>> measured_rate(0, 0)
    'no trials'

    >>> measured_rate(1, 400, places=4)
    '1/400 = 0.0025 [0.0003, 0.0209]'
    """
    if trials <= 0:
        return "no trials"
    interval = wilson_interval(successes, trials, confidence=CONFIDENCE)
    rate = successes / trials
    return (
        f"{successes}/{trials} = {rate:.{places}f} "
        f"[{interval.low:.{places}f}, {interval.high:.{places}f}]"
    )


def refused_run(record: TrialRecord) -> bool:
    """Say whether any verifier reached no verdict on this run.

    >>> from qsecure.eval.reduce import refused_run
    >>> class Fake:
    ...     def __init__(self, verdicts):
    ...         self.transcript_summary = {"verdicts": verdicts}
    >>> refused_run(Fake({"Bob": "accepted", "Charlie": "refused"}))
    True
    >>> refused_run(Fake({"Bob": "accepted", "Charlie": "rejected"}))
    False

    A rejection is not a refusal, and neither is a party who was never asked:
    >>> refused_run(Fake({"Bob": "rejected", "Charlie": "not_asked"}))
    False
    """
    verdicts = record.transcript_summary.get("verdicts") or {}
    return any(str(v) == "refused" for v in verdicts.values())


def expected_trials(
    store: ResultStore, experiment_name: str
) -> dict[str, int]:
    """Return how many trials per cell the sweep was actually asked for."""
    wanted: dict[str, int] = {}
    for path in store.manifests(experiment_name):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        count = payload.get("trials")
        if not isinstance(count, int) or count <= 0:
            continue
        for cell in payload.get("cells") or ():
            wanted[str(cell)] = max(wanted.get(str(cell), 0), count)
    return wanted


def completeness_note(
    records: Sequence[TrialRecord], expected: Mapping[str, int]
) -> str:
    """Say whether every trial the sweep asked for is on disk.

    >>> from qsecure.eval.reduce import completeness_note
    >>> class Fake:
    ...     def __init__(self, cell, index):
    ...         self.cell, self.index = cell, index
    >>> got = [Fake("a", 0), Fake("a", 1), Fake("b", 0)]
    >>> print(completeness_note(got, {"a": 2, "b": 2}))
    ... # doctest: +NORMALIZE_WHITESPACE
    **INCOMPLETE** -- the denominators above are smaller than the sweep asked
    for, so every rate here is over a shrunken sample: b has 1 of 2 trials.

    With nothing missing it says so, and says what it checked against:
    >>> print(completeness_note(got, {"a": 2, "b": 1}))
    ... # doctest: +NORMALIZE_WHITESPACE
    Complete: every cell has all the trials its manifest asked for, and no
    trial index is missing below the highest present. A trial whose scenario
    raised would leave no file, so this is checked rather than assumed.

    >>> print(completeness_note(got, {"a": 2}))
    ... # doctest: +NORMALIZE_WHITESPACE
    Partly checked: every cell a manifest covers has all the trials it asked
    for, and no trial index is missing below the highest present. No manifest
    recorded a trial count for b, so that cell was checked for interior gaps
    only and may be short of what was asked for.

    With no manifest to check against it does not claim completeness:
    >>> print(completeness_note(got, {}))
    ... # doctest: +NORMALIZE_WHITESPACE
    No manifest recorded a trial count for this store, so completeness could
    not be checked; only interior gaps were, and there are none. The row
    counts above may be short of what was asked for.
    """
    by_cell: dict[str, set[int]] = {}
    for record in records:
        by_cell.setdefault(record.cell, set()).add(record.index)

    short: list[str] = []
    gaps: list[str] = []
    for cell in sorted(by_cell):
        seen = by_cell[cell]
        absent = sorted(set(range(max(seen) + 1)) - seen)
        if absent:
            gaps.append(f"{cell} is missing index {absent}")
        want = expected.get(cell)
        if want is not None and len(seen) < want:
            short.append(f"{cell} has {len(seen)} of {want} trials")
    for cell in sorted(expected):
        if cell not in by_cell:
            short.append(f"{cell} has no records at all")

    # Cells with records but no manifest were never count-checked, so name them.
    unchecked = [cell for cell in sorted(by_cell) if cell not in expected]
    caveat = (
        " No manifest recorded a trial count for "
        + ", ".join(unchecked)
        + ", so "
        + ("that cell was" if len(unchecked) == 1 else "those cells were")
        + " checked for interior gaps only and may be short of what was "
        "asked for."
        if expected and unchecked
        else ""
    )

    if short or gaps:
        return (
            "**INCOMPLETE** -- the denominators above are smaller than the "
            "sweep asked for, so every rate here is over a shrunken sample: "
            + "; ".join(short + gaps)
            + "."
            + caveat
        )
    if caveat:
        return (
            "Partly checked: every cell a manifest covers has all the trials "
            "it asked for, and no trial index is missing below the highest "
            "present." + caveat
        )
    if not expected:
        return (
            "No manifest recorded a trial count for this store, so "
            "completeness could not be checked; only interior gaps were, and "
            "there are none. The row counts above may be short of what was "
            "asked for."
        )
    return (
        "Complete: every cell has all the trials its manifest asked for, and "
        "no trial index is missing below the highest present. A trial whose "
        "scenario raised would leave no file, so this is checked rather than "
        "assumed."
    )


def escape_xml(text: str) -> str:
    """Escape the five XML metacharacters, for a chart renderer.

    >>> from qsecure.eval.reduce import escape_xml
    >>> escape_xml('a & b <c> "d"')
    'a &amp; b &lt;c&gt; &quot;d&quot;'
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def detection_table(
    records: Sequence[TrialRecord],
    *,
    command: str = "",
    cell_order: Sequence[str] | None = None,
) -> Table:
    """Tabulate what the detector flagged, measured apart from proven.

    >>> import tempfile
    >>> from pathlib import Path
    >>> from qsecure.eval.experiments import experiment
    >>> from qsecure.eval.reduce import detection_table
    >>> from qsecure.eval.runner import run_experiment
    >>> from qsecure.eval.store import ResultStore
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     store = ResultStore(Path(tmp))
    ...     _ = run_experiment(experiment("smoke"), store, trials=2, workers=1,
    ...                        in_process=True, quiet=True)
    ...     table = detection_table(list(store.read_experiment("smoke")))
    ...     table.columns[:5]
    ...     row = table.rows[0]
    ...     row[0], row[2], row[3], row[4]
    ...     table.columns[7], row[7]
    ...     row[10].startswith("not evaluated")
    ('cell', 'count_exchange_timing', 'hypothesis', 'runs', 'attacked')
    ('tiny', 'honest', 2, 0)
    ('refusals (any party)', 0)
    True
    """
    if not records:
        raise ValueError("detection_table needs at least one record")
    rows: list[tuple[Any, ...]] = []
    withheld_seen: set[str] = set()
    for (cell, timing), group in group_records(
        records, cell_order=cell_order
    ).items():
        # Per group: a family withheld in one cell says nothing about another.
        withheld_here: set[str] = set()
        runs = len(group)
        attacked = [r for r in group if r.truth.attacked]
        clean = [r for r in group if not r.truth.attacked]
        detectable = all(r.truth.detectable for r in group)
        hypotheses = sorted({r.truth.hypothesis for r in group})
        bounds = [
            float(r.detection["false_positive_bound"])
            for r in group
            if r.detection.get("false_positive_bound") is not None
        ]
        # A refusal is not a rejection: its own column, in neither denominator.
        refusals = sum(1 for r in group if refused_run(r))
        for record in group:
            withheld_here.update(str(w) for w in record.detection.get("withheld", ()))
        withheld_seen.update(withheld_here)

        # Only the attacked column carries the phrase; unattacked runs stay measurable.
        flagged_cell: Any = (
            UNDETECTABLE_BY_CONSTRUCTION
            if not detectable
            else measured_rate(sum(1 for r in attacked if r.flagged), len(attacked))
        )
        clean_cell = measured_rate(sum(1 for r in clean if r.flagged), len(clean))

        nulls = {r.null_is_noiseless for r in group}
        null_cell = (
            "yes" if nulls == {True} else "no" if nulls == {False} else "mixed"
        )
        rows.append(
            (
                cell,
                timing,
                ", ".join(hypotheses),
                runs,
                len(attacked),
                flagged_cell,
                clean_cell,
                refusals,
                f"{max(bounds):.3e}" if bounds else NOT_EVALUATED,
                null_cell,
                f"{NOT_EVALUATED} ({len(withheld_here)})" if withheld_here else "-",
            )
        )
    notes = [
        "`flagged on attacked (measured)` and `flagged on clean (measured)` "
        "are MEASUREMENTS with a sample size, shown as k/n with a "
        f"{CONFIDENCE:.0%} Wilson interval. There is no false-negative bound "
        "and there cannot be one from a transcript.",
        "`max proven FP bound` is PROVEN under the honest null -- the largest "
        "Detection.false_positive_bound in the group. It shares no column "
        "with a measured rate.",
        "`attacked` counts runs whose adversary engaged, read off the "
        "adversary's own log. An untargeted run is byte-identical to an "
        "honest one and is scored as one.",
        "`refusals (any party)` counts runs where AT LEAST ONE verifier "
        "reached no verdict. It is shown beside the rates and added to "
        "neither denominator: a refusal is not a rejection, and it is not a "
        "miss either -- the detector scored every run in both denominators, "
        "refusal or not. A row whose count equals its `runs` did not "
        "necessarily deny transfer, and this is where an experiment's own "
        "per-party table earns its place: at L = 96 the recipient forger "
        "refuses on all 400 runs under either ordering, but before forwarding "
        "it is CHARLIE who reaches no verdict (a denial of transfer) and "
        "after forwarding it is BOB, while Charlie returns a real 120/400 "
        "acceptance. Same count, opposite reading, which is why the column is "
        "named for what it actually counts.",
        "`null_is_noiseless` says whether the detector was given a noiseless "
        "null. On a genuinely noisy link that null is not the truth, honest "
        "runs depart from it, and the mismatch members fire correctly -- so a "
        "`yes` here is a caveat on the row, not a detail.",
        f"`{UNDETECTABLE_BY_CONSTRUCTION}` means an assumption rules "
        f"detection out for that hypothesis -- {UNDETECTABLE_ASSUMPTION}. It "
        "is never a blank, a dash or a zero.",
    ]
    if withheld_seen:
        notes.append(
            "Families withheld on at least one run in this table, reported as "
            f"`{NOT_EVALUATED}` and never as passed: "
            + ", ".join(sorted(withheld_seen))
            + "."
        )
    return Table(
        slug="detection",
        title="Detector signals, measured rates apart from proven bounds",
        columns=(
            "cell",
            "count_exchange_timing",
            "hypothesis",
            "runs",
            "attacked",
            "flagged on attacked (measured)",
            "flagged on clean (measured)",
            "refusals (any party)",
            "max proven FP bound",
            "null_is_noiseless",
            "withheld families",
        ),
        rows=tuple(rows),
        command=command,
        notes=tuple(notes),
    )


def timing_table(
    records: Sequence[TrialRecord],
    *,
    command: str = "",
    cell_order: Sequence[str] | None = None,
) -> Table:
    """Tabulate what the trials cost, so the manifest can be checked against it.

    >>> import tempfile
    >>> from pathlib import Path
    >>> from qsecure.eval.experiments import experiment
    >>> from qsecure.eval.reduce import timing_table
    >>> from qsecure.eval.runner import run_experiment
    >>> from qsecure.eval.store import ResultStore
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     store = ResultStore(Path(tmp))
    ...     _ = run_experiment(experiment("smoke"), store, trials=2, workers=1,
    ...                        in_process=True, quiet=True)
    ...     table = timing_table(list(store.read_experiment("smoke")))
    ...     table.columns
    ...     table.rows[0][0], table.rows[0][1], table.rows[0][2]
    ('cell', 'key_length', 'trials', 'session s (total)', 'session s (mean)', 'detect s (mean)', 'ms/position')
    ('tiny', 96, 2)
    """
    if not records:
        raise ValueError("timing_table needs at least one record")
    by_cell: dict[str, list[TrialRecord]] = {}
    for (cell, _timing), group in group_records(
        records, cell_order=cell_order
    ).items():
        by_cell.setdefault(cell, []).extend(group)
    rows: list[tuple[Any, ...]] = []
    for cell, group in by_cell.items():
        length = int(group[0].params["key_length"])
        session = [float(r.wall_clock.get("session_seconds", 0.0)) for r in group]
        detect_s = [float(r.wall_clock.get("detect_seconds", 0.0)) for r in group]
        mean_session = sum(session) / len(session)
        rows.append(
            (
                cell,
                length,
                len(group),
                f"{sum(session):.2f}",
                f"{mean_session:.3f}",
                f"{sum(detect_s) / len(detect_s):.4f}",
                f"{1000.0 * mean_session / length:.3f}",
            )
        )
    return Table(
        slug="timing",
        title="Per-trial cost, by cell",
        columns=(
            "cell",
            "key_length",
            "trials",
            "session s (total)",
            "session s (mean)",
            "detect s (mean)",
            "ms/position",
        ),
        rows=tuple(rows),
        command=command,
        notes=(
            "Timings are wall clock inside the worker, so at N workers the "
            "sum of `session s (total)` exceeds the sweep's own wall clock by "
            "roughly the achieved speedup. That comparison is the check on "
            "the runner: if it does not, one of the two numbers is wrong.",
            "**`ms/position` HERE IS NOT THE PROTOCOL'S COST.** It carries the "
            "contention of whatever worker count produced these records -- see "
            "`worker counts` in the provenance above. Measured on this machine, "
            "a trial takes 1.61x longer at twenty workers than at one at "
            "L = 384, and 2.53x at L = 768, because the cores share an all-core "
            "turbo budget and an L3. The single-threaded reference is "
            "`qsecure.eval.perf.REFERENCE_MS_PER_POSITION`. Multiplying the figure below by a "
            "key length to project a production run overstates it by that "
            "factor.",
            "Timings are excluded from a record's fingerprint, so they cannot "
            "affect the reproducibility claim -- and every other table here "
            "is byte-identical at one worker and at twenty.",
        ),
    )


def reduce_experiment(
    store: ResultStore,
    experiment_name: str,
    *,
    command: str = "",
    cells: Sequence[str] | None = None,
) -> list[Table]:
    """Read one experiment off disk and return every table it supports.

    >>> import tempfile
    >>> from pathlib import Path
    >>> from qsecure.eval.reduce import reduce_experiment
    >>> from qsecure.eval.store import ResultStore
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     reduce_experiment(ResultStore(Path(tmp)), "smoke")
    Traceback (most recent call last):
        ...
    FileNotFoundError: no records for experiment 'smoke' under ...
    """
    wanted = None if cells is None else set(cells)
    records = [
        record
        for record in store.read_experiment(experiment_name)
        if wanted is None or record.cell in wanted
    ]
    if not records:
        raise FileNotFoundError(
            f"no records for experiment {experiment_name!r} under "
            f"{store.root}. Run the sweep first with "
            f"qsecure.eval.runner.run_experiment."
        )
    order: Sequence[str] | None = None
    registered = EXPERIMENTS.get(experiment_name)
    if registered is not None:
        order = registered.cell_names
    expected = {
        cell: count
        for cell, count in expected_trials(store, experiment_name).items()
        if wanted is None or cell in wanted
    }
    complete = completeness_note(records, expected)
    outcomes = outcome_table(records, command=command, cell_order=order)
    # Prepended: a short sample is the first thing a reader needs.
    outcomes = replace(outcomes, notes=(complete, *outcomes.notes))
    tables = [
        outcomes,
        detection_table(records, command=command, cell_order=order),
        timing_table(records, command=command, cell_order=order),
    ]
    extra = EXTRA_REDUCTIONS.get(experiment_name)
    if extra is not None:
        # Pass `expected` so per-family completeness checks the manifest count too.
        tables.extend(
            extra(
                records,
                command=command,
                cell_order=order,
                expected=expected,
            )
        )
    return tables
