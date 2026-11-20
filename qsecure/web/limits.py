"""Every bound the dashboard's request surface enforces, in one place.

The caps are data, and a refusal names the one it hit:
>>> from qsecure.web.limits import LIVE_KEY_LENGTH_MAX, RequestRefused, check_key_length
>>> LIVE_KEY_LENGTH_MAX
1024
>>> try:
...     check_key_length(5000)
... except RequestRefused as refusal:
...     print(refusal.field, refusal.cap, str(LIVE_KEY_LENGTH_MAX) in str(refusal))
key_length 1024 True
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Final


__all__ = [
    "CHANNEL_ERROR_RATE_MAX",
    "CHECK_FRACTION_MAX",
    "COST_TABLE",
    "COST_TABLE_NOTE",
    "COUNT_EXCHANGE_TIMINGS",
    "EPS_DEFAULT",
    "EPS_MAX",
    "EPS_MIN",
    "KEY_LENGTH_MIN",
    "LIVE_KEY_LENGTH_MAX",
    "MAX_CONCURRENT_RUNS",
    "MAX_ECHOED_CHARS",
    "MAX_ERROR_BODY_DEPTH",
    "MAX_REQUEST_BYTES",
    "NOISE_MAX",
    "SEED_MAX",
    "RequestRefused",
    "check_check_fraction",
    "check_eps",
    "check_key_length",
    "check_probability",
    "check_seed",
    "check_timing",
    "json_safe",
    "limits_payload",
    "safe_text",
]


#: Shortest run executed; below this every rate is a handful of positions.
KEY_LENGTH_MIN: Final[int] = 24

#: Longest live run; larger requests are refused, never clamped.
LIVE_KEY_LENGTH_MAX: Final[int] = 1024

#: Largest share of a run that may be spent on check rounds.
CHECK_FRACTION_MAX: Final[float] = 0.5

#: Strongest depolarising noise accepted (design noise is 2 s_a = 0.03125).
NOISE_MAX: Final[float] = 0.5

#: Largest accepted seed; bounded so a request cannot carry a huge integer.
SEED_MAX: Final[int] = 2**32 - 1

#: Concurrent session generations; generation is CPU-bound and blocking.
MAX_CONCURRENT_RUNS: Final[int] = 2

#: Max depth json_safe walks: error bodies echo client input, and encoders
#: recurse once per level.
MAX_ERROR_BODY_DEPTH: Final[int] = 32

#: Caller-supplied text a refusal quotes back per value; the rest is truncated
#: with a count.
MAX_ECHOED_CHARS: Final[int] = 200

#: Max request body in bytes; larger bodies get 413 before the app parses them.
MAX_REQUEST_BYTES: Final[int] = 64 * 1024

#: The false-positive budget the dashboard offers by default.
EPS_DEFAULT: Final[float] = 1e-9

#: Tightest accepted budget; far below it every threshold is vacuous.
EPS_MIN: Final[float] = 1e-19

#: The loosest. Above this the "budget" is not a security parameter any more.
EPS_MAX: Final[float] = 0.1

#: Largest null the rate family may be read against.
CHANNEL_ERROR_RATE_MAX: Final[float] = 0.5

#: Measured wall-clock session-generation cost per preset length, so the
#: frontend looks it up rather than computing it.
COST_TABLE: Final[tuple[dict[str, Any], ...]] = (
    {"key_length": 96, "session_ms": 203, "detect_ms": 4, "transcript_kb": 32},
    {"key_length": 192, "session_ms": 407, "detect_ms": 6, "transcript_kb": 63},
    {"key_length": 384, "session_ms": 815, "detect_ms": 14, "transcript_kb": 125},
    {"key_length": 768, "session_ms": 1638, "detect_ms": 18, "transcript_kb": 249},
    {"key_length": 1024, "session_ms": 2509, "detect_ms": 33, "transcript_kb": 332},
)

#: What :data:`COST_TABLE` is and is not, rendered verbatim by the screen.
COST_TABLE_NOTE: Final[str] = (
    "Wall-clock session generation measured on one machine (i7-14700HX, "
    "check_fraction=0.25, no profiler). An estimate for sizing a progress "
    "indicator, not a bound and not a security number. Detection is free at "
    "every length; the session is the whole cost."
)

#: The two Phase C' orderings: separate experiments, never averaged.
COUNT_EXCHANGE_TIMINGS: Final[tuple[str, str]] = (
    "before-forwarding",
    "after-forwarding",
)


def safe_text(value: Any, limit: int = MAX_ECHOED_CHARS) -> str:
    """Return a bounded, printable form of ``value``. Never raises.

    >>> from qsecure.web.limits import safe_text
    >>> safe_text(1024), safe_text("honest"), safe_text(None)
    ('1024', 'honest', 'None')

    >>> len(safe_text(9 * 10 ** 20000)) <= 240
    True
    >>> "20001 digits" in safe_text(9 * 10 ** 20000)
    True

    >>> quoted = safe_text("x" * 1000000)
    >>> len(quoted) < 260, "999800 more characters" in quoted
    (True, True)

    >>> class Hostile:
    ...     def __repr__(self):
    ...         raise RuntimeError("no")
    >>> safe_text(Hostile())
    '<Hostile whose repr() raised>'
    """
    try:
        text = value if isinstance(value, str) else repr(value)
    except ValueError:
        # Huge ints refuse to render; the digit count is what a reader wants.
        try:
            digits = int(math.floor(math.log10(abs(value)))) + 1
            text = f"<integer of about {digits} digits>"
        except Exception:  # noqa: BLE001 - the guard's guard
            text = f"<{type(value).__name__} whose repr() raised>"
    except Exception:  # noqa: BLE001 - deliberately total
        text = f"<{type(value).__name__} whose repr() raised>"
    if len(text) > limit:
        return f"{text[:limit]}... [{len(text) - limit} more characters]"
    return text


def json_safe(item: Any, depth: int = 0) -> Any:
    """Return ``item`` when JSON can carry it, and text where it cannot.

    >>> from qsecure.web.limits import json_safe
    >>> json_safe(1024), json_safe("honest"), json_safe(None)
    (1024, 'honest', None)
    >>> json_safe(float("nan")), json_safe(float("inf")), json_safe(float("-inf"))
    ('nan', 'inf', '-inf')
    >>> json_safe({"nested": [1, 2]})
    {'nested': [1, 2]}
    >>> json_safe({"loc": ["body", "seed"], "input": float("nan")})
    {'loc': ['body', 'seed'], 'input': 'nan'}
    >>> json_safe(object())[:7]
    '<object'

    >>> import json
    >>> deep = []
    >>> nest = deep
    >>> for _ in range(2000):
    ...     inner = []
    ...     nest.append(inner)
    ...     nest = inner
    >>> nest.append(float("nan"))
    >>> quoted = json.dumps(json_safe(deep), allow_nan=False)
    >>> "nested beyond 32 levels" in quoted
    True

    >>> echoed = json_safe({"attack": "x" * 1000000})["attack"]
    >>> len(echoed) < 260, "999800 more characters" in echoed
    (True, True)

    >>> json_safe(9 * 10 ** 20000)
    '<integer of about 20001 digits>'
    """
    if depth >= MAX_ERROR_BODY_DEPTH:
        return f"<nested beyond {MAX_ERROR_BODY_DEPTH} levels>"
    # Always walk containers: a deep list encodes fine, so probing would skip the
    # depth bound.
    if isinstance(item, dict):
        return {
            str(key): json_safe(value, depth + 1)
            for key, value in item.items()
        }
    if isinstance(item, (list, tuple)):
        return [json_safe(value, depth + 1) for value in item]
    try:
        json.dumps(item, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        return safe_text(item)
    # Encodable says nothing about size; only strings and ints can be long.
    if isinstance(item, str) and len(item) > MAX_ECHOED_CHARS:
        return safe_text(item)
    if isinstance(item, int) and not isinstance(item, bool):
        quoted = safe_text(item)
        if not quoted.lstrip("-").isdigit():
            return quoted
    return item


class RequestRefused(Exception):
    """One request parameter was out of bounds, with the bound that refused it.

    >>> from qsecure.web.limits import RequestRefused
    >>> refusal = RequestRefused("key_length", "too long, max 1024", 5000, 1024)
    >>> refusal.to_dict()["cap"]
    1024
    """

    def __init__(
        self,
        field: str,
        message: str,
        value: Any = None,
        cap: Any = None,
    ) -> None:
        super().__init__(message)
        self.field: str = field
        self.value: Any = value
        self.cap: Any = cap

    def to_dict(self) -> dict[str, Any]:
        """Return the refusal as a JSON-serialisable body.

        >>> import json
        >>> from qsecure.web.limits import RequestRefused
        >>> body = RequestRefused("noise", "must be finite", float("nan")).to_dict()
        >>> body["value"]
        'nan'
        >>> json.loads(json.dumps(body, allow_nan=False))["field"]
        'noise'
        """
        return {
            "field": self.field,
            "message": str(self),
            "value": json_safe(self.value),
            "cap": json_safe(self.cap),
        }


def _as_int(value: Any, field: str) -> int:
    """Return ``value`` as an :class:`int`, refusing bools and non-integers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestRefused(
            field,
            f"{field} must be an integer, got {type(value).__name__}.",
            value,
        )
    return int(value)


def _quoted(value: Any) -> str:
    """Return ``value`` as bounded text for a refusal message.

    >>> from qsecure.web.limits import _quoted
    >>> _quoted(1025)
    '1025'
    """
    return safe_text(value)


def _as_float(value: Any, field: str) -> float:
    """Return ``value`` as a finite :class:`float`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestRefused(
            field,
            f"{field} must be a real number, got {type(value).__name__}.",
            value,
        )
    number = float(value)
    if not math.isfinite(number):
        raise RequestRefused(
            field,
            f"{field} must be finite, got {number!r}. A non-finite value "
            f"compares false against every bound and would pass a range check "
            f"unexamined.",
            value,
        )
    return number


def check_key_length(value: Any) -> int:
    """Return ``value`` if it is a runnable live key length, else refuse.

    >>> from qsecure.web.limits import RequestRefused, check_key_length
    >>> check_key_length(1024)
    1024
    >>> try:
    ...     check_key_length(1025)
    ... except RequestRefused as refusal:
    ...     print(refusal.cap, "refused rather than quietly run" in str(refusal))
    1024 True
    >>> try:
    ...     check_key_length(4)
    ... except RequestRefused as refusal:
    ...     print(refusal.cap)
    24

    >>> import json
    >>> try:
    ...     check_key_length(int("9" * 400))
    ... except RequestRefused as refusal:
    ...     body = refusal.to_dict()
    >>> body["field"], body["cap"]
    ('key_length', 1024)
    >>> json.loads(json.dumps(body, allow_nan=False))["value"]
    '999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999... [200 more characters]'
    >>> try:
    ...     check_key_length(9 * 10 ** 20000)
    ... except RequestRefused as refusal:
    ...     print(refusal.to_dict()["value"])
    <integer of about 20001 digits>
    """
    length = _as_int(value, "key_length")
    if length < KEY_LENGTH_MIN:
        raise RequestRefused(
            "key_length",
            f"key_length={_quoted(length)} is below the minimum of "
            f"{KEY_LENGTH_MIN}. Only about a third of positions are matched, "
            f"so a shorter run gives a verifier a handful of records and every "
            f"rate on the screen would be one draw wide.",
            length,
            KEY_LENGTH_MIN,
        )
    if length > LIVE_KEY_LENGTH_MAX:
        # The measured row, not `length * k`, which overflowed for huge lengths.
        ceiling_ms = COST_TABLE[-1]["session_ms"]
        raise RequestRefused(
            "key_length",
            f"key_length={_quoted(length)} is above the live ceiling of "
            f"{LIVE_KEY_LENGTH_MAX}. Session generation is the whole cost of a "
            f"run and it is linear in the key length: the ceiling itself "
            f"measures {ceiling_ms} ms per session, and this asks for more "
            f"positions than that. The request is refused rather than quietly "
            f"run at {LIVE_KEY_LENGTH_MAX}: a screen reporting a clamped run "
            f"under the label of the one that was asked for is the single "
            f"easiest way for this dashboard to lie. The security-grade set "
            f"L=115200 is four minutes a run and is not offered live; its "
            f"bounds are closed forms and are published under /api/defaults.",
            length,
            LIVE_KEY_LENGTH_MAX,
        )
    return length


def check_check_fraction(value: Any, key_length: int) -> float:
    """Return ``value`` if it designates a runnable check-round plan.

    >>> from qsecure.web.limits import RequestRefused, check_check_fraction
    >>> check_check_fraction(0.25, 192)
    0.25
    >>> try:
    ...     check_check_fraction(0.001, 192)
    ... except RequestRefused as refusal:
    ...     print("0 check rounds" in str(refusal))
    True
    >>> try:
    ...     check_check_fraction(0.9, 192)
    ... except RequestRefused as refusal:
    ...     print(refusal.cap)
    0.5
    """
    fraction = _as_float(value, "check_fraction")
    if fraction < 0.0 or fraction > CHECK_FRACTION_MAX:
        raise RequestRefused(
            "check_fraction",
            f"check_fraction={fraction!r} must lie in "
            f"[0.0, {CHECK_FRACTION_MAX}]. Check rounds are excluded from the "
            f"key, so above that ceiling most of the run is instrumentation "
            f"and the signature is scored on what is left.",
            fraction,
            CHECK_FRACTION_MAX,
        )
    if fraction > 0.0 and math.floor(fraction * key_length) < 1:
        raise RequestRefused(
            "check_fraction",
            f"check_fraction={fraction!r} designates "
            f"floor({fraction!r} * {key_length}) = 0 check rounds, so nothing "
            f"would be estimated while the run claims estimation is happening. "
            f"Raise it to at least {1.0 / key_length!r}, lengthen the key, or "
            f"send 0.0 to say plainly that this run publishes no channel "
            f"statistics -- in which case the whole channel family is reported "
            f"as not evaluated, which is not the same as passed.",
            fraction,
            CHECK_FRACTION_MAX,
        )
    return fraction


def check_probability(value: Any, field: str, cap: float = 1.0) -> float:
    """Return ``value`` if it is a probability in ``[0, cap]``.

    >>> from qsecure.web.limits import RequestRefused, check_probability
    >>> check_probability(0.03125, "noise")
    0.03125
    >>> try:
    ...     check_probability(2.0, "channel_error_rate")
    ... except RequestRefused as refusal:
    ...     print(refusal.field, refusal.cap)
    channel_error_rate 1.0
    >>> try:
    ...     check_probability(float("nan"), "noise")
    ... except RequestRefused as refusal:
    ...     print("must be finite" in str(refusal))
    True
    """
    number = _as_float(value, field)
    if number < 0.0 or number > cap:
        raise RequestRefused(
            field,
            f"{field}={number!r} must lie in [0.0, {cap}]. It is a "
            f"probability.",
            number,
            cap,
        )
    return number


def check_eps(value: Any) -> float:
    """Return ``value`` if it is a usable false-positive budget.

    >>> from qsecure.web.limits import RequestRefused, check_eps
    >>> check_eps(1e-9)
    1e-09
    >>> try:
    ...     check_eps(0.0)
    ... except RequestRefused as refusal:
    ...     print(refusal.cap)
    (1e-19, 0.1)
    >>> try:
    ...     check_eps(0.5)
    ... except RequestRefused as refusal:
    ...     print("stopped being a security parameter" in str(refusal))
    True
    """
    number = _as_float(value, "eps")
    if not EPS_MIN <= number <= EPS_MAX:
        raise RequestRefused(
            "eps",
            f"eps={number!r} must lie in [{EPS_MIN}, {EPS_MAX}]. It is the "
            f"false-positive budget for the whole detector. detect() itself "
            f"accepts anything strictly inside (0, 1); the dashboard is "
            f"narrower on purpose. A budget of 0 asks for a detector that "
            f"never fires on an honest run and therefore never fires; far "
            f"below {EPS_MIN} every threshold goes vacuous, which is a "
            f"false-positive probability of exactly zero AND no power at all, "
            f"and a screen on which nothing can light up is worse than a "
            f"refusal. Above {EPS_MAX} the budget has stopped being a security "
            f"parameter.",
            number,
            (EPS_MIN, EPS_MAX),
        )
    return number


def check_seed(value: Any) -> int:
    """Return ``value`` if it is a usable seed.

    >>> from qsecure.web.limits import check_seed
    >>> check_seed(20260141)
    20260141
    """
    seed = _as_int(value, "seed")
    if seed < 0 or seed > SEED_MAX:
        raise RequestRefused(
            "seed",
            f"seed={_quoted(seed)} must lie in [0, {SEED_MAX}]. The seed "
            f"reproduces a "
            f"run exactly -- both the session's stream and the adversary's, "
            f"derived separately -- and is bounded only so a request "
            f"cannot carry an integer with a million digits.",
            seed,
            SEED_MAX,
        )
    return seed


def check_timing(value: Any) -> str:
    """Return ``value`` if it names one of the two Phase C' orderings.

    >>> from qsecure.web.limits import check_timing
    >>> check_timing("after-forwarding")
    'after-forwarding'

    >>> from qsecure.web.limits import RequestRefused
    >>> try:
    ...     check_timing("z" * 1000000)
    ... except RequestRefused as refusal:
    ...     print(len(str(refusal)) < 600, len(refusal.to_dict()["value"]) < 260)
    True True
    """
    if not isinstance(value, str) or value not in COUNT_EXCHANGE_TIMINGS:
        raise RequestRefused(
            "count_exchange_timing",
            f"count_exchange_timing must be one of "
            f"{list(COUNT_EXCHANGE_TIMINGS)}, got {_quoted(value)}. It decides "
            f"which "
            f"declaration each recipient counted against, which is a different "
            f"experiment and not a tuning knob -- one ordering turns a "
            f"substituted declaration into a refusal and the other into a "
            f"scored forgery. Runs under the two are never pooled.",
            value,
            list(COUNT_EXCHANGE_TIMINGS),
        )
    return value


@dataclass(frozen=True)
class _LimitRow:
    """One published cap, as the screen shows it."""

    field: str
    minimum: Any
    maximum: Any
    note: str


def limits_payload() -> dict[str, Any]:
    """Return every cap as JSON, for ``/api/defaults`` to publish.

    >>> from qsecure.web.limits import limits_payload
    >>> payload = limits_payload()
    >>> payload["fields"]["key_length"]["maximum"]
    1024
    >>> payload["max_concurrent_runs"]
    2
    >>> payload["max_request_bytes"]
    65536
    >>> [row["key_length"] for row in payload["cost_table"]]
    [96, 192, 384, 768, 1024]
    """
    rows = (
        _LimitRow(
            "key_length",
            KEY_LENGTH_MIN,
            LIVE_KEY_LENGTH_MAX,
            "L, positions per recipient per message bit. The whole cost of a "
            "run and the only parameter that can make one slow. Above the "
            "ceiling the request is refused, never clamped.",
        ),
        _LimitRow(
            "check_fraction",
            0.0,
            CHECK_FRACTION_MAX,
            "Share of positions spent measuring the channel instead of "
            "carrying key. At 0.0 the run publishes no channel statistics at "
            "all and the whole channel family is reported as not evaluated -- "
            "which is not the same as passed.",
        ),
        _LimitRow(
            "noise",
            0.0,
            NOISE_MAX,
            "Depolarising strength of the physical link, mounted on the "
            "resource line. Applies to the honest arm too, which is the point: "
            "an honest run over a noisy link departs from the detector's "
            "noiseless null.",
        ),
        _LimitRow(
            "eps",
            EPS_MIN,
            EPS_MAX,
            "False-positive budget for the whole detector. What is published "
            "afterwards is the bound the thresholds PROVE -- normally well "
            "inside the budget -- never this number, and never the post hoc "
            "evidence bound either.",
        ),
        _LimitRow(
            "channel_error_rate",
            0.0,
            CHANNEL_ERROR_RATE_MAX,
            "The null the rate family is read against, p_e. Defaults to 0.0, a "
            "claim about a noiseless link. It is set by the operator and never "
            "inferred from the run: a transcript without check rounds carries "
            "no estimate of it, and guessing would be inventing a null.",
        ),
        _LimitRow(
            "tolerated_depolarising",
            0.0,
            NOISE_MAX,
            "The null the CHANNEL family is read against, p0, as a Werner "
            "strength. Also defaults to 0.0, which claims an ideal resource. A "
            "separate number from channel_error_rate on purpose: they are two "
            "parameterisations of the same physics and converting one into the "
            "other silently would state a null the operator did not ask for. "
            "Optional: a request that omits it behaves exactly as the fixed "
            "contract's eight fields say.",
        ),
        _LimitRow(
            "seed",
            0,
            SEED_MAX,
            "Reproduces the run exactly. The session's stream and the "
            "adversary's are derived from it separately and are independent "
            ".",
        ),
        _LimitRow(
            "count_exchange_timing",
            None,
            None,
            "One of "
            + ", ".join(repr(name) for name in COUNT_EXCHANGE_TIMINGS)
            + ". A label on the result and a control, never a thing to pool "
            "over.",
        ),
    )
    return {
        "fields": {
            row.field: {
                "minimum": row.minimum,
                "maximum": row.maximum,
                "note": row.note,
            }
            for row in rows
        },
        "max_concurrent_runs": MAX_CONCURRENT_RUNS,
        "max_request_bytes": MAX_REQUEST_BYTES,
        "max_request_bytes_note": (
            "The largest request body the service will read. A larger one is "
            "answered 413 before the application sees it, so nothing parses "
            "it, echoes it or holds it. The largest legitimate body is nine "
            "scalars and about 300 bytes."
        ),
        "max_echoed_chars": MAX_ECHOED_CHARS,
        "cost_table": [dict(row) for row in COST_TABLE],
        "cost_table_note": COST_TABLE_NOTE,
    }
