"""What this costs to run, measured, and every projection derived from it.

The rate itself, and what it says about a full-scale session:
>>> from qsecure.eval.perf import (
...     MEASURED_SESSIONS, REFERENCE_MS_PER_POSITION,
...     projected_session_seconds, projected_sweep_hours,
... )
>>> round(REFERENCE_MS_PER_POSITION, 3)
2.041
>>> round(projected_session_seconds(115200), 1)
235.1
>>> round(projected_session_seconds(115200) / 60, 2)
3.92

Two hundred trials at production parameters, single-threaded:
>>> round(projected_sweep_hours(200, 115200), 1)
13.1

>>> [round(projected_sweep_hours(200, 115200, speedup=s), 1)
...  for s in (7.04, 10.27)]
[1.9, 1.3]
>>> round(projected_sweep_hours(200, 115200, speedup=15.0) * 60, 0)
52.0

>>> lo = min(row["ms_per_position"] for row in MEASURED_SESSIONS)
>>> hi = max(row["ms_per_position"] for row in MEASURED_SESSIONS)
>>> 1.9 < lo and hi < 2.5
True
>>> max(row["key_length"] for row in MEASURED_SESSIONS) // min(
...     row["key_length"] for row in MEASURED_SESSIONS)
1200

>>> from qsecure.eval.perf import MEASURED_TRANSCRIPT_KB, MEASURED_DETECT_SECONDS
>>> MEASURED_TRANSCRIPT_KB[(115200, 0.0)]
0.226
>>> MEASURED_TRANSCRIPT_KB[(115200, 0.25)]
0.33
>>> round(MEASURED_TRANSCRIPT_KB[(115200, 0.25)]
...       / MEASURED_TRANSCRIPT_KB[(115200, 0.0)], 2)
1.46

>>> from qsecure.eval.perf import time_session
>>> for length in (96, 192):
...     for fraction in (0.0, 0.25):
...         got = time_session(
...             length, seed=5 + length, check_fraction=fraction
...         )["kb_per_position"]
...         want = MEASURED_TRANSCRIPT_KB[(length, fraction)]
...         print(length, fraction, round(got, 3) == want)
96 0.0 True
96 0.25 True
192 0.0 True
192 0.25 False

>>> quarter = time_session(96, seed=101, check_fraction=0.25)
>>> heavier = time_session(96, seed=101, check_fraction=0.42)
>>> round(quarter["kb_per_position"], 3), round(heavier["kb_per_position"], 3)
(0.337, 0.407)
>>> spread = [
...     time_session(96, seed=s, check_fraction=0.25)["kb_per_position"]
...     for s in (1, 2, 3, 101)
... ]
>>> round(max(spread) - min(spread), 4) < 0.001
True

>>> MEASURED_DETECT_SECONDS[115200]
1.541
>>> MEASURED_DETECT_SECONDS[115200] > 100 * MEASURED_DETECT_SECONDS[192]
True
"""

from __future__ import annotations

import sys
import time
from typing import Any, Final, Mapping, Sequence

__all__ = [
    "MEASURED_DETECT_SECONDS",
    "DEGENERATE_END",
    "MEASURED_MEMORY",
    "MEASURED_SESSIONS",
    "MEASURED_SPEEDUP",
    "MEASURED_SPEEDUP_KEY_LENGTH",
    "MEASURED_SPEEDUP_TRIALS",
    "MEASURED_TRANSCRIPT_KB",
    "REFERENCE_KEY_LENGTH",
    "REFERENCE_MS_PER_POSITION",
    "SECURITY_CLAIM_MIN_KEY_LENGTH",
    "SPEEDUP_NOTE",
    "detector_latency",
    "peak_rss_bytes",
    "measure_memory",
    "measure_small_end",
    "measured_speedup_at",
    "measured_speedup_curve",
    "measure_scaling",
    "projected_session_seconds",
    "projected_sweep_hours",
    "speedup_table",
    "time_session",
]

REFERENCE_KEY_LENGTH: Final[int] = 115200
"""int: ``DEFAULT_PARAMS.key_length``. The rate is quoted at this scale."""

MEASURED_SESSIONS: Final[tuple[dict[str, Any], ...]] = (
    {
        "key_length": 96,
        "check_fraction": 0.25,
        "seconds": 0.211,
        "ms_per_position": 2.198,
        "source": "harness, single worker, BLAS pinned to 1",
    },
    {
        "key_length": 192,
        "check_fraction": 0.25,
        "seconds": 0.420,
        "ms_per_position": 2.188,
        "source": "harness, single worker, BLAS pinned to 1",
    },
    {
        "key_length": 768,
        "check_fraction": 0.25,
        "seconds": 1.650,
        "ms_per_position": 2.148,
        "source": "earlier measurement, confirmed",
    },
    {
        "key_length": 115200,
        "check_fraction": 0.0,
        "seconds": 235.092,
        "ms_per_position": 2.041,
        "source": "end-to-end, seed 20260905, no profiler attached",
    },
)
"""tuple of dict: Sessions actually timed, end to end, on the target machine."""

MEASURED_TRANSCRIPT_KB: Final[Mapping[tuple[int, float], float]] = {
    (96, 0.25): 0.337,
    (192, 0.25): 0.330,
    (768, 0.25): 0.325,
    (3072, 0.25): 0.326,
    (12288, 0.25): 0.327,
    (115200, 0.25): 0.330,
    (96, 0.0): 0.230,
    (192, 0.0): 0.224,
    (768, 0.0): 0.220,
    (3072, 0.0): 0.221,
    (12288, 0.0): 0.223,
    (115200, 0.0): 0.226,
}
"""Mapping: Transcript JSON size in KiB per position, by ``(L, check_fraction)``."""

MEASURED_DETECT_SECONDS: Final[Mapping[int, float]] = {
    96: 0.0029,
    192: 0.0041,
    768: 0.0113,
    115200: 1.541,
}
"""Mapping: :func:`~qsecure.detect.detector.detect` latency in seconds, by ``L``."""

REFERENCE_MS_PER_POSITION: Final[float] = 2.041
"""float: Milliseconds per position at :data:`REFERENCE_KEY_LENGTH`."""


def projected_session_seconds(
    key_length: int, *, ms_per_position: float = REFERENCE_MS_PER_POSITION
) -> float:
    """Return how long one session at ``key_length`` should take.

    >>> from qsecure.eval.perf import projected_session_seconds
    >>> round(projected_session_seconds(192), 3)
    0.392
    >>> round(projected_session_seconds(1024), 2)
    2.09
    """
    if isinstance(key_length, bool) or not isinstance(key_length, int):
        raise TypeError(f"key_length must be an int, got {type(key_length).__name__}")
    if key_length <= 0:
        raise ValueError(f"key_length must be positive, got {key_length}")
    return key_length * float(ms_per_position) / 1000.0


def projected_sweep_hours(
    trials: int,
    key_length: int,
    *,
    speedup: float = 1.0,
    ms_per_position: float = REFERENCE_MS_PER_POSITION,
) -> float:
    """Return how long a sweep of ``trials`` sessions should take, in hours.

    >>> from qsecure.eval.perf import projected_sweep_hours
    >>> round(projected_sweep_hours(200, 115200), 1)
    13.1
    >>> round(projected_sweep_hours(200, 115200, speedup=20.0) * 60, 1)
    39.2

    >>> projected_sweep_hours(10, 1024) == projected_sweep_hours(
    ...     10, 1024, speedup=1.0)
    True
    """
    if isinstance(trials, bool) or not isinstance(trials, int):
        raise TypeError(f"trials must be an int, got {type(trials).__name__}")
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if float(speedup) <= 0.0:
        raise ValueError(f"speedup must be positive, got {speedup!r}")
    serial = trials * projected_session_seconds(
        key_length, ms_per_position=ms_per_position
    )
    return serial / float(speedup) / 3600.0


def time_session(
    key_length: int,
    *,
    seed: int,
    check_fraction: float = 0.25,
    message_bit: int = 0,
) -> dict[str, Any]:
    """Time one honest session, its serialisation and its detection.

    >>> from qsecure.eval.perf import time_session
    >>> timing = time_session(96, seed=7)
    >>> timing["key_length"], timing["transferable"]
    (96, True)
    >>> timing["ms_per_position"] > 0 and timing["kb_per_position"] > 0
    True
    """
    from qsecure.core.rng import seed_to_generator
    from qsecure.detect.detector import detect
    from qsecure.protocol.params import ProtocolParams
    from qsecure.protocol.session import QDSSession

    params = ProtocolParams(key_length=key_length, check_fraction=check_fraction)
    started = time.perf_counter()
    transcript = QDSSession(params, rng=seed_to_generator(seed)).run(message_bit)
    session_seconds = time.perf_counter() - started

    started = time.perf_counter()
    text = transcript.to_json()
    serialise_seconds = time.perf_counter() - started

    started = time.perf_counter()
    detect(text, eps=1e-9)
    detect_seconds = time.perf_counter() - started

    return {
        "key_length": key_length,
        "check_fraction": check_fraction,
        "seed": seed,
        "seconds": session_seconds,
        "ms_per_position": 1000.0 * session_seconds / key_length,
        "positions_per_second": key_length / session_seconds,
        "json_bytes": len(text),
        "kb_per_position": len(text) / 1024.0 / key_length,
        "serialise_seconds": serialise_seconds,
        "detect_seconds": detect_seconds,
        "transferable": bool(transcript.transferable),
        "aborted": bool(transcript.aborted),
    }


def detector_latency(key_length: int, *, seed: int, repeats: int = 3) -> dict[str, Any]:
    """Time the detector alone, separately from generating the transcript.

    >>> from qsecure.eval.perf import detector_latency
    >>> latency = detector_latency(96, seed=3, repeats=2)
    >>> latency["min_seconds"] <= latency["mean_seconds"]
    True
    >>> latency["json_bytes"] > 0
    True
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError(f"repeats must be a positive int, got {repeats!r}")
    from qsecure.core.rng import seed_to_generator
    from qsecure.detect.detector import detect
    from qsecure.protocol.params import ProtocolParams
    from qsecure.protocol.session import QDSSession

    params = ProtocolParams(key_length=key_length, check_fraction=0.25)
    text = QDSSession(params, rng=seed_to_generator(seed)).run(0).to_json()
    timings: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        detect(text, eps=1e-9)
        timings.append(time.perf_counter() - started)
    mean = sum(timings) / len(timings)
    return {
        "key_length": key_length,
        "json_bytes": len(text),
        "repeats": repeats,
        "mean_seconds": mean,
        "min_seconds": min(timings),
        "mb_per_second": (len(text) / 1_048_576.0) / min(timings),
    }


def measure_scaling(
    key_lengths: Sequence[int], *, seed: int = 5, check_fraction: float = 0.25
) -> list[dict[str, Any]]:
    """Time one session at each key length and return the rows.

    >>> from qsecure.eval.perf import measure_scaling
    >>> rows = measure_scaling([48, 96])
    >>> [row["key_length"] for row in rows]
    [48, 96]
    """
    return [
        time_session(length, seed=seed + length, check_fraction=check_fraction)
        for length in key_lengths
    ]


def speedup_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Turn ``(workers, seconds)`` measurements into a speedup curve.

    >>> from qsecure.eval.perf import speedup_table
    >>> table = speedup_table([
    ...     {"workers": 1, "seconds": 100.0},
    ...     {"workers": 4, "seconds": 27.0},
    ...     {"workers": 20, "seconds": 7.0},
    ... ])
    >>> [(r["workers"], round(r["speedup"], 2)) for r in table]
    [(1, 1.0), (4, 3.7), (20, 14.29)]
    >>> round(table[-1]["efficiency"], 3)
    0.714
    """
    baseline = next(
        (float(row["seconds"]) for row in rows if int(row["workers"]) == 1), None
    )
    if baseline is None:
        raise ValueError(
            "a speedup curve needs a one-worker baseline; none of the rows "
            "has workers == 1, and a curve normalised to its own fastest "
            "point reports an efficiency of 1.0 no matter how bad it is."
        )
    table: list[dict[str, Any]] = []
    for row in rows:
        workers = int(row["workers"])
        seconds = float(row["seconds"])
        speedup = baseline / seconds if seconds > 0 else float("inf")
        table.append(
            {
                "workers": workers,
                "seconds": seconds,
                "speedup": speedup,
                "efficiency": speedup / workers,
            }
        )
    return table


def peak_rss_bytes() -> int | None:
    """Return this process's peak resident set size, without a profiler.

    >>> from qsecure.eval.perf import peak_rss_bytes
    >>> peak = peak_rss_bytes()
    >>> peak is None or peak > 1_000_000
    True
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            """The ``PROCESS_MEMORY_COUNTERS`` struct, fields in order."""

            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # Without argtypes/restype ctypes truncates the -1 pseudo-handle to 32 bits,
        # the call fails and the probe silently reports zero bytes.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_Counters),
            wintypes.DWORD,
        ]
        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return int(counters.PeakWorkingSetSize) if ok else None
    try:
        import resource
    except ImportError:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return int(usage) * (1024 if sys.platform.startswith("linux") else 1)


def measure_memory(key_lengths: Sequence[int], *, seed: int = 5) -> list[dict[str, Any]]:
    """Time and measure peak memory for one session at each key length.

    >>> from qsecure.eval.perf import measure_memory
    >>> rows = measure_memory([48, 96])
    >>> [row["key_length"] for row in rows]
    [48, 96]
    >>> measure_memory([96, 48])
    Traceback (most recent call last):
        ...
    ValueError: key_lengths must be ascending...
    """
    lengths = list(key_lengths)
    if lengths != sorted(lengths):
        raise ValueError(
            f"key_lengths must be ascending, got {lengths}. Peak RSS is a "
            f"high-water mark for the whole process, so a larger run earlier "
            f"would be reported as every later run's peak."
        )
    from qsecure.core.rng import seed_to_generator
    from qsecure.protocol.params import ProtocolParams
    from qsecure.protocol.session import QDSSession

    rows: list[dict[str, Any]] = []
    previous = peak_rss_bytes() or 0
    for length in lengths:
        params = ProtocolParams(key_length=length, check_fraction=0.25)
        transcript = QDSSession(params, rng=seed_to_generator(seed + length)).run(0)
        text = transcript.to_json()
        peak = peak_rss_bytes() or 0
        rows.append(
            {
                "key_length": length,
                "peak_rss_mb": peak / 1_048_576.0,
                "delta_mb": (peak - previous) / 1_048_576.0,
                "json_mb": len(text) / 1_048_576.0,
            }
        )
        previous = peak
        del transcript, text
    return rows


def measure_small_end(
    key_lengths: Sequence[int],
    *,
    trials: int = 5,
    seed: int = 5,
    check_fraction: float = 0.25,
) -> list[dict[str, Any]]:
    """Measure what the protocol does at key lengths too small to claim anything.

    A length whose parameter set cannot exist is a row, not a gap:
    >>> from qsecure.eval.perf import measure_small_end
    >>> rows = measure_small_end([3], trials=1)
    >>> rows[0]["key_length"], rows[0]["refused"]
    (3, True)
    >>> "check" in rows[0]["reason"]
    True

    And a real length reports the floors beside the aborts they cause:
    >>> rows = measure_small_end([24], trials=2)
    >>> row = rows[0]
    >>> row["signing_length"], row["m_min"], row["M_min"]
    (18, 1, 1)
    >>> row["security_claim"], row["aborts"] <= row["trials"]
    (False, True)
    >>> row["withheld_min"] <= row["withheld_max"]
    True
    """
    from qsecure.core.rng import seed_to_generator
    from qsecure.detect.detector import detect
    from qsecure.protocol.params import ProtocolParams
    from qsecure.protocol.session import QDSSession
    from qsecure.protocol.verify import (
        minimum_matched_count,
        minimum_pooled_matched_count,
    )

    rows: list[dict[str, Any]] = []
    for length in key_lengths:
        try:
            params = ProtocolParams(
                key_length=length, check_fraction=check_fraction
            )
        except (ValueError, TypeError) as exc:
            rows.append(
                {
                    "key_length": length,
                    "refused": True,
                    "reason": str(exc).splitlines()[0],
                }
            )
            continue
        aborts = 0
        withheld: list[int] = []
        claims: set[bool] = set()
        for index in range(trials):
            transcript = QDSSession(
                params, rng=seed_to_generator(seed + length + index)
            ).run(0)
            if transcript.aborted:
                aborts += 1
            # Every trial: the evaluable-family count depends on how check rounds split.
            detection = detect(transcript.to_json(), eps=1e-9)
            withheld.append(len(detection.withheld))
            claims.add(bool(detection.security_claim))
        rows.append(
            {
                "key_length": length,
                "refused": False,
                "signing_length": params.signing_length,
                "m_min": minimum_matched_count(params),
                "M_min": minimum_pooled_matched_count(params),
                "aborts": aborts,
                "trials": trials,
                "security_claim": claims == {True},
                "security_claim_varies": len(claims) > 1,
                "withheld_min": min(withheld),
                "withheld_max": max(withheld),
            }
        )
    return rows


MEASURED_SPEEDUP: Final[tuple[dict[str, Any], ...]] = (
    {"workers": 1, "wall_seconds": 204.81, "trial_seconds_total": 203.98},
    {"workers": 4, "wall_seconds": 59.86, "trial_seconds_total": 233.17},
    {"workers": 8, "wall_seconds": 40.37, "trial_seconds_total": 305.56},
    {"workers": 14, "wall_seconds": 34.44, "trial_seconds_total": 436.13},
    {"workers": 20, "wall_seconds": 29.10, "trial_seconds_total": 516.26},
)
"""tuple of dict: The achieved parallel speedup, measured on this machine."""

SPEEDUP_NOTE: Final[str] = (
    "20 workers give 7.04x wall-clock speedup, not the 15x the plan assumed. "
    "The pool is not idling -- 17.7 of 20 workers are busy on average -- but "
    "each worker runs 2.5x slower than a lone one. The slowdown begins at "
    "eight workers, before any E-core is reached, so it is the all-core turbo "
    "budget and shared cache before it is the E-cores. BLAS oversubscription "
    "was ruled out by asking a worker what its own environment says, and a "
    "cold-versus-hot baseline was ruled out by re-measuring the one-worker "
    "point immediately after the twenty-worker run: 1.702 s/trial against "
    "1.700 s, a 0.1% difference."
)
"""str: What the speedup curve means, in one paragraph, for a table caption."""

MEASURED_SPEEDUP_TRIALS: Final[int] = 120
"""int: Sessions per point in :data:`MEASURED_SPEEDUP`."""

MEASURED_SPEEDUP_KEY_LENGTH: Final[int] = 768
"""int: Key length :data:`MEASURED_SPEEDUP` was measured at."""

MEASURED_MEMORY: Final[tuple[dict[str, Any], ...]] = (
    {"key_length": 96, "peak_rss_mb": 78.5, "transcript_mb": 0.03},
    {"key_length": 768, "peak_rss_mb": 80.4, "transcript_mb": 0.24},
    {"key_length": 3072, "peak_rss_mb": 86.2, "transcript_mb": 0.98},
    {"key_length": 12288, "peak_rss_mb": 107.3, "transcript_mb": 3.92},
    {"key_length": 49152, "peak_rss_mb": 197.0, "transcript_mb": 15.82},
    {"key_length": 115200, "peak_rss_mb": 348.7, "transcript_mb": 37.17},
)
"""tuple of dict: Peak resident set against key length, one worker."""

SECURITY_CLAIM_MIN_KEY_LENGTH: Final[int] = 182
"""int: Smallest ``L`` carrying a security claim at ``check_fraction = 0.25``."""

DEGENERATE_END: Final[str] = (
    "Below L=182 at check_fraction=0.25 the pooled floor degenerates to 1 and "
    "no run carries a security claim. Below about L=24 aborts appear on honest "
    "runs -- 1 in 5 at L=6 and L=12 -- because a verifier's matched set falls "
    "under the floor, and those are no-verdicts rather than rejections. At "
    "L=3 the parameter set refuses to be built at all: floor(0.25 * 3) is zero "
    "check rounds, so the run would claim estimation while estimating nothing. "
    "Whether a detector family is evaluable is not a clean function of L "
    "either: below about L=216 the channel family is withheld on some runs and "
    "not others, because it depends on how the check rounds happened to split "
    "between QBER and CHSH cells."
)
"""str: What happens at the small end, measured rather than assumed."""


def measured_speedup_curve() -> list[dict[str, Any]]:
    """Return :data:`MEASURED_SPEEDUP` worked into the numbers a table prints.

    The curve, and the fact that it is nowhere near linear:
    >>> from qsecure.eval.perf import measured_speedup_curve
    >>> curve = measured_speedup_curve()
    >>> [(row["workers"], round(row["speedup"], 2)) for row in curve]
    [(1, 1.0), (4, 3.42), (8, 5.07), (14, 5.95), (20, 7.04)]
    >>> [round(row["efficiency"], 2) for row in curve]
    [1.0, 0.86, 0.63, 0.42, 0.35]

    >>> round(curve[-1]["occupancy"], 1)
    17.7
    >>> [round(row["mean_trial_seconds"], 2) for row in curve]
    [1.7, 1.94, 2.55, 3.63, 4.3]

    And the slowdown starts at eight workers, before an E-core is reached:
    >>> round(curve[2]["mean_trial_seconds"] / curve[0]["mean_trial_seconds"], 2)
    1.5
    """
    rows = [
        {"workers": row["workers"], "seconds": row["wall_seconds"]}
        for row in MEASURED_SPEEDUP
    ]
    curve = speedup_table(rows)
    for entry, measured in zip(curve, MEASURED_SPEEDUP):
        entry["occupancy"] = (
            measured["trial_seconds_total"] / measured["wall_seconds"]
        )
        entry["mean_trial_seconds"] = (
            measured["trial_seconds_total"] / MEASURED_SPEEDUP_TRIALS
        )
    return curve


def measured_speedup_at(workers: int) -> float:
    """Return the measured wall-clock speedup at one worker count.

    The one the run plan uses, and what it does to the plan:
    >>> from qsecure.eval.perf import measured_speedup_at, projected_sweep_hours
    >>> round(measured_speedup_at(20), 2)
    7.04
    >>> round(projected_sweep_hours(200, 115200,
    ...                             speedup=measured_speedup_at(20)), 2)
    1.86

    Against the 51 minutes the plan assumed at an estimated 15x:
    >>> round(projected_sweep_hours(200, 115200, speedup=15.0) * 60, 0)
    52.0

    An unmeasured worker count is refused rather than interpolated:
    >>> measured_speedup_at(12)
    Traceback (most recent call last):
        ...
    KeyError: 'no speedup measured at 12 workers; measured: [1, 4, 8, 14, 20]'
    """
    for row in measured_speedup_curve():
        if row["workers"] == workers:
            return float(row["speedup"])
    raise KeyError(
        f"no speedup measured at {workers} workers; measured: "
        f"{[row['workers'] for row in MEASURED_SPEEDUP]}"
    )
