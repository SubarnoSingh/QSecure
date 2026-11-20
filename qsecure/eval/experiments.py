"""What a sweep runs: cells, the scenarios that fill them, and the registry.

>>> from qsecure.eval.experiments import EXPERIMENTS, experiment
>>> set(EXPERIMENTS) >= {'honest', 'noise', 'scaling', 'smoke'}
True
>>> honest = experiment("honest")
>>> [cell.name for cell in honest.cells]
['l192', 'l384', 'l768']
>>> honest.cell("l192").params.key_length
192

>>> [c.name for c in experiment("noise").cells]
['clean', 'p0025', 'p005', 'p010', 'p015', 'p03125']
>>> {c.channel_error_rate for c in experiment("noise").cells}
{0.0}
>>> [c.scenario_options["strength"] for c in experiment("noise").cells]
[0.0, 0.0025, 0.005, 0.01, 0.015, 0.03125]

>>> from qsecure.eval.experiments import run_trial
>>> record = run_trial(experiment("smoke"), "tiny", 0)
>>> record.identity
('smoke', 'tiny', 0)
>>> record.truth.hypothesis, record.truth.engaged
('honest', False)
>>> record.detection["key_length"]
72
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Mapping, Sequence

from qsecure.attacks.channel import DepolarisingChannel
from qsecure.detect.detector import detect
from qsecure.detect.statistics import TranscriptStatistics
from qsecure.protocol.params import ProtocolParams
from qsecure.protocol.session import QDSSession, SessionTranscript

from .records import GroundTruth, TrialRecord
from .seeds import TrialSeeds, check_name, trial_seeds

__all__ = [
    "Cell",
    "DEFAULT_EPS",
    "EXPERIMENTS",
    "Experiment",
    "SCENARIOS",
    "SCENARIO_PROBE_OPTIONS",
    "Scenario",
    "all_cells",
    "claim",
    "experiment",
    "honest_scenario",
    "depolarising_scenario",
    "run_trial",
]

DEFAULT_EPS: Final[float] = 1e-9
"""float: The detector budget a cell inherits when it names none."""


@dataclass(frozen=True)
class Cell:
    """One row of one results table: everything fixed except the trial index.

    >>> from qsecure.eval.experiments import Cell
    >>> from qsecure.protocol.params import ProtocolParams
    >>> cell = Cell(name="l192", params=ProtocolParams(key_length=192))
    >>> cell.scenario, cell.eps, cell.channel_error_rate
    ('honest', 1e-09, 0.0)
    >>> Cell(name="l192", params=ProtocolParams(key_length=192),
    ...      scenario="not-a-scenario")
    Traceback (most recent call last):
        ...
    ValueError: unknown scenario 'not-a-scenario'...
    """

    name: str
    params: ProtocolParams
    scenario: str = "honest"
    scenario_options: Mapping[str, Any] = field(default_factory=dict)
    message_bit: int = 0
    eps: float = DEFAULT_EPS
    channel_error_rate: float = 0.0
    tolerated_depolarising: float = 0.0
    truth_hypothesis: str = "honest"
    detectable: bool = True
    retain_transcript: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate the cell."""
        check_name(self.name, what="cell")
        if not isinstance(self.params, ProtocolParams):
            raise TypeError(
                f"params must be ProtocolParams, got {type(self.params).__name__}"
            )
        if self.scenario not in SCENARIOS:
            raise ValueError(
                f"unknown scenario {self.scenario!r}; registered scenarios are "
                f"{sorted(SCENARIOS)}"
            )
        if self.message_bit not in (0, 1):
            raise ValueError(f"message_bit must be 0 or 1, got {self.message_bit!r}")
        if not 0.0 < float(self.eps) < 1.0:
            raise ValueError(f"eps must lie in (0, 1), got {self.eps!r}")
        for name in ("channel_error_rate", "tolerated_depolarising"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value!r}")

    @property
    def null_is_noiseless(self) -> bool:
        """bool: Whether this cell scores against a noiseless null."""
        return float(self.channel_error_rate) == 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, for the manifest."""
        return {
            "name": self.name,
            "params": self.params.to_dict(),
            "scenario": self.scenario,
            "scenario_options": dict(self.scenario_options),
            "message_bit": self.message_bit,
            "eps": float(self.eps),
            "channel_error_rate": float(self.channel_error_rate),
            "tolerated_depolarising": float(self.tolerated_depolarising),
            "truth_hypothesis": self.truth_hypothesis,
            "detectable": self.detectable,
            "retain_transcript": self.retain_transcript,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Experiment:
    """A named set of cells, and the unit a sweep runs.

    >>> from qsecure.eval.experiments import experiment
    >>> smoke = experiment("smoke")
    >>> smoke.name, [c.name for c in smoke.cells], smoke.trials
    ('smoke', ['tiny'], 4)
    >>> smoke.cell("absent")
    Traceback (most recent call last):
        ...
    KeyError: "experiment 'smoke' has no cell 'absent'; it has ['tiny']"
    """

    name: str
    cells: tuple[Cell, ...]
    trials: int = 30
    description: str = ""

    def __post_init__(self) -> None:
        """Validate the experiment."""
        check_name(self.name, what="experiment")
        if not self.cells:
            raise ValueError(f"experiment {self.name!r} has no cells")
        for cell in self.cells:
            if not isinstance(cell, Cell):
                raise TypeError(f"cells must be Cell, got {type(cell).__name__}")
        names = [cell.name for cell in self.cells]
        if len(set(names)) != len(names):
            raise ValueError(
                f"experiment {self.name!r} has duplicate cell names: {names}. "
                f"Two cells sharing a name share a directory and a seed "
                f"stream, so one would overwrite the other's trials."
            )
        if isinstance(self.trials, bool) or not isinstance(self.trials, int):
            raise TypeError("trials must be an int")
        if self.trials <= 0:
            raise ValueError(f"trials must be positive, got {self.trials}")

    @property
    def cell_names(self) -> tuple[str, ...]:
        """tuple of str: The cell names, in table order."""
        return tuple(cell.name for cell in self.cells)

    def cell(self, name: str) -> Cell:
        """Return one cell by name."""
        for cell in self.cells:
            if cell.name == name:
                return cell
        raise KeyError(
            f"experiment {self.name!r} has no cell {name!r}; it has "
            f"{list(self.cell_names)}"
        )


Scenario = Callable[[Cell, TrialSeeds], "tuple[SessionTranscript, GroundTruth]"]
"""Callable: The signature every registered scenario implements."""


def honest_scenario(
    cell: Cell, seeds: TrialSeeds
) -> tuple[SessionTranscript, GroundTruth]:
    """Run one honest session. No adversary, and the label says so.

    >>> from qsecure.eval.experiments import Cell, honest_scenario
    >>> from qsecure.eval.seeds import trial_seeds
    >>> from qsecure.protocol.params import ProtocolParams
    >>> cell = Cell(name="tiny", params=ProtocolParams(key_length=48))
    >>> run, truth = honest_scenario(cell, trial_seeds("demo", "tiny", 0))
    >>> truth.hypothesis, truth.engaged, truth.engaged_count
    ('honest', False, 0)
    >>> run.is_complete
    True
    """
    session = QDSSession(cell.params, rng=seeds.session_rng())
    transcript = session.run(cell.message_bit)
    return transcript, GroundTruth(
        hypothesis=cell.truth_hypothesis,
        engaged=False,
        engaged_count=0,
        detectable=cell.detectable,
        notes=cell.notes,
    )


def depolarising_scenario(
    cell: Cell, seeds: TrialSeeds
) -> tuple[SessionTranscript, GroundTruth]:
    """Run a session over a depolarising wire, and label it from the wire's log.

    >>> from qsecure.eval.experiments import Cell, depolarising_scenario
    >>> from qsecure.eval.seeds import trial_seeds
    >>> from qsecure.protocol.params import ProtocolParams
    >>> params = ProtocolParams(key_length=96, check_fraction=0.25)
    >>> cell = Cell(
    ...     name="p050", params=params, scenario="depolarising",
    ...     scenario_options={"strength": 0.5, "target": "Bob"},
    ...     truth_hypothesis="channel-manipulation",
    ... )
    >>> run, truth = depolarising_scenario(cell, trial_seeds("demo", "p050", 0))
    >>> truth.hypothesis, truth.targeted_link
    ('channel-manipulation', 'Bob')
    >>> truth.engaged, truth.engaged_count > 0
    (True, True)

    >>> quiet = Cell(
    ...     name="p000", params=params, scenario="depolarising",
    ...     scenario_options={"strength": 0.0},
    ...     truth_hypothesis="channel-manipulation",
    ... )
    >>> _, label = depolarising_scenario(quiet, trial_seeds("demo", "p000", 0))
    >>> label.engaged, label.engaged_count
    (False, 0)
    """
    options = dict(cell.scenario_options)
    if "strength" not in options:
        raise KeyError(
            f"cell {cell.name!r} uses the depolarising scenario but names no "
            f"'strength' in scenario_options. There is no default: a missing "
            f"strength would run an inert arm that looks exactly like a "
            f"detector finding nothing."
        )
    strength = float(options["strength"])
    target = options.get("target")
    attack = DepolarisingChannel(
        strength, rng=seeds.adversary_rng(), target=target
    )
    session = QDSSession(
        cell.params, resource_factory=attack.resource, rng=seeds.session_rng()
    )
    transcript = session.run(cell.message_bit)
    engaged_count = int(attack.engaged_count)
    return transcript, GroundTruth(
        hypothesis=cell.truth_hypothesis,
        engaged=engaged_count > 0,
        engaged_count=engaged_count,
        targeted_link=None if target is None else str(target),
        detectable=cell.detectable,
        notes=cell.notes,
    )


SCENARIOS: Final[dict[str, Scenario]] = {
    "honest": honest_scenario,
    "depolarising": depolarising_scenario,
}
"""dict: Registered scenarios, by name."""

SCENARIO_PROBE_OPTIONS: Final[dict[str, dict[str, Any]]] = {
    "depolarising": {"strength": 0.25},
}
"""dict: The smallest ``scenario_options`` that make each scenario run at all.

>>> from qsecure.eval.experiments import SCENARIO_PROBE_OPTIONS
>>> SCENARIO_PROBE_OPTIONS["depolarising"]
{'strength': 0.25}
"""


def claim(registry: dict[str, Any], name: str, value: Any, what: str) -> None:
    """Put ``value`` in ``registry`` under ``name``, refusing to displace another.

    Claiming a free key, then re-claiming it for the same object:
    >>> from qsecure.eval.experiments import claim
    >>> registry: dict[str, object] = {}
    >>> claim(registry, "mine", len, "scenario")
    >>> claim(registry, "mine", len, "scenario")
    >>> registry
    {'mine': <built-in function len>}

    Both directions of collision raise, and the registry is left alone:
    >>> claim(registry, "mine", sorted, "scenario")
    Traceback (most recent call last):
        ...
    RuntimeError: scenario 'mine' is already registered to something else...
    >>> registry["mine"] is len
    True
    """
    existing = registry.get(name)
    if existing is not None and existing is not value:
        raise RuntimeError(
            f"{what} {name!r} is already registered to something else "
            f"({existing!r}). Two experiment families sharing a registry key "
            f"would give whichever imported last, silently, and the other's "
            f"cells would run under the wrong label or not at all."
        )
    registry[name] = value


def run_trial(
    exp: Experiment,
    cell_name: str,
    index: int,
    *,
    retain_transcript: bool | None = None,
) -> TrialRecord:
    """Execute one trial and return its reduced record.

    >>> from qsecure.eval.experiments import experiment, run_trial
    >>> record = run_trial(experiment("smoke"), "tiny", 1)
    >>> record.index, record.cell, record.flagged
    (1, 'tiny', False)
    >>> record.seeds["session"] > 0
    True
    >>> set(record.wall_clock) == {"session_seconds", "detect_seconds",
    ...                            "total_seconds"}
    True
    """
    cell = exp.cell(cell_name)
    seeds = trial_seeds(exp.name, cell.name, index)
    scenario = SCENARIOS[cell.scenario]

    started = time.perf_counter()
    transcript, truth = scenario(cell, seeds)
    after_session = time.perf_counter()

    # Serialised first, so the detector reads exactly what a saved file holds.
    text = transcript.to_json()
    stats = TranscriptStatistics.from_json(text)
    detection = detect(
        stats,
        eps=cell.eps,
        channel_error_rate=cell.channel_error_rate,
        tolerated_depolarising=cell.tolerated_depolarising,
    )
    finished = time.perf_counter()

    keep = cell.retain_transcript if retain_transcript is None else retain_transcript
    return TrialRecord.build(
        seeds=seeds,
        params=cell.params,
        eps=cell.eps,
        transcript=transcript,
        detection=detection,
        truth=truth,
        wall_clock={
            "session_seconds": after_session - started,
            "detect_seconds": finished - after_session,
            "total_seconds": finished - started,
        },
        retain_transcript=keep,
        stats=stats,
    )


def _honest_experiment() -> Experiment:
    """Build the honest-baseline experiment."""
    return Experiment(
        name="honest",
        description=(
            "Honest runs on a noiseless link at three key lengths: the "
            "false-positive arm, measured against a proven bound."
        ),
        trials=30,
        cells=tuple(
            Cell(
                name=f"l{length}",
                params=ProtocolParams(key_length=length, check_fraction=0.25),
                notes="honest, noiseless link, noiseless null",
            )
            for length in (192, 384, 768)
        ),
    )


def _noise_experiment() -> Experiment:
    """Build the honest-but-noisy experiment."""
    params = ProtocolParams(key_length=384, check_fraction=0.25)
    # 0.03125 is 2*s_a, the design noise level.
    levels = (
        ("clean", 0.0),
        ("p0025", 0.0025),
        ("p005", 0.005),
        ("p010", 0.01),
        ("p015", 0.015),
        ("p03125", 0.03125),
    )
    return Experiment(
        name="noise",
        description=(
            "Honest parties over a depolarising wire, scored against the "
            "noiseless null: the arm where null_is_noiseless is the column "
            "that decides whether the row is a false claim."
        ),
        trials=30,
        cells=tuple(
            Cell(
                name=name,
                params=params,
                scenario="depolarising",
                scenario_options={"strength": strength, "target": None},
                truth_hypothesis="honest" if strength == 0.0 else "channel-noise",
                notes=f"depolarising strength {strength}, noiseless null",
            )
            for name, strength in levels
        ),
    )


def _scaling_experiment() -> Experiment:
    """Build the throughput-against-key-length experiment."""
    return Experiment(
        name="scaling",
        description=(
            "Throughput against key length, measured through the real "
            "runner so it can be checked against the run manifest."
        ),
        trials=20,
        cells=tuple(
            Cell(
                name=f"l{length}",
                params=ProtocolParams(key_length=length, check_fraction=0.25),
                notes="performance measurement; carries no security claim",
            )
            for length in (192, 768, 3072, 12288)
        ),
    )


def _smoke_experiment() -> Experiment:
    """Build the tiny experiment the tests and the CLI's ``--dry-run`` use."""
    return Experiment(
        name="smoke",
        description="A four-trial cell at L=96, for exercising the harness.",
        trials=4,
        cells=(
            Cell(
                name="tiny",
                params=ProtocolParams(key_length=96, check_fraction=0.25),
                notes="harness smoke test; carries no security claim",
            ),
        ),
    )


EXPERIMENTS: Final[dict[str, Experiment]] = {
    exp.name: exp
    for exp in (
        _honest_experiment(),
        _noise_experiment(),
        _scaling_experiment(),
        _smoke_experiment(),
    )
}
"""dict: The registered experiments, by name."""


def experiment(name: str) -> Experiment:
    """Return a registered experiment by name.

    >>> from qsecure.eval.experiments import experiment
    >>> experiment("honest").name
    'honest'
    >>> experiment("nope")
    Traceback (most recent call last):
        ...
    KeyError: "no experiment named 'nope'; registered: [...'scaling', 'smoke']"
    """
    try:
        return EXPERIMENTS[name]
    except KeyError:
        raise KeyError(
            f"no experiment named {name!r}; registered: {sorted(EXPERIMENTS)}"
        ) from None


def all_cells(exp: Experiment, only: Sequence[str] | None = None) -> tuple[Cell, ...]:
    """Return the cells to run, filtered by name.

    >>> from qsecure.eval.experiments import all_cells, experiment
    >>> [c.name for c in all_cells(experiment("honest"), ["l768", "l192"])]
    ['l192', 'l768']
    """
    if only is None:
        return exp.cells
    wanted = set(only)
    for name in wanted:
        exp.cell(name)
    return tuple(cell for cell in exp.cells if cell.name in wanted)

