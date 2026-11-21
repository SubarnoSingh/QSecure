"""The JSON API and the static host: one process, one command, no network.

>>> from fastapi.testclient import TestClient
>>> from qsecure.web.api import create_app
>>> client = TestClient(create_app())
>>> client.get("/api/health").json()["ok"]
True
>>> defaults = client.get("/api/defaults").json()
>>> defaults["live_key_length_max"]
1024
>>> f'{defaults["bounds"]["headline"]["enforced_repudiation_bound"]:.4e}'
'1.4139e-09'

A run above the ceiling is refused, and the refusal names the ceiling:
>>> refused = client.post("/api/run", json={"key_length": 100000})
>>> refused.status_code, refused.json()["cap"]
(400, 1024)

>>> client.post("/api/run", json={"key_length": 96, "seed": 4}).status_code
200
>>> events = client.get("/api/events").json()
>>> [(event["party"], event["verdict"]) for event in events["events"]]
[('Bob', 'accepted'), ('Charlie', 'accepted')]
>>> events["dropped"], events["persisted"]
(0, False)
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from pathlib import Path
from typing import Any, Final

from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from qsecure import __version__
from qsecure.audit import AuditLog
from qsecure.detect import family_budget
from qsecure.eval.security import FLOOR_CROSSOVER, security_claim_at
from qsecure.protocol.params import (
    DEFAULT_PARAMS,
    DEMO_PARAMS,
    ProtocolParams,
)
from qsecure.protocol.verify import (
    enforced_repudiation_bound,
    minimum_matched_count,
    minimum_pooled_matched_count,
)
from qsecure.web.catalogue import roster_payload
from qsecure.web.driver import MESSAGE_BIT, RunRequest, run_once
from qsecure.web.limits import (
    CHANNEL_ERROR_RATE_MAX,
    CHECK_FRACTION_MAX,
    COST_TABLE,
    COST_TABLE_NOTE,
    COUNT_EXCHANGE_TIMINGS,
    EPS_DEFAULT,
    EPS_MAX,
    EPS_MIN,
    KEY_LENGTH_MIN,
    LIVE_KEY_LENGTH_MAX,
    MAX_CONCURRENT_RUNS,
    MAX_REQUEST_BYTES,
    NOISE_MAX,
    RequestRefused,
    json_safe,
    limits_payload,
)

DEGENERATE_BELOW_KEY_LENGTH: Final[int] = FLOOR_CROSSOVER
"""int: The sifted length below which a run carries no security claim.

>>> from qsecure.web.api import DEGENERATE_BELOW_KEY_LENGTH as boundary
>>> from qsecure.eval.security import security_claim_at
>>> boundary
137
>>> security_claim_at(boundary - 1), security_claim_at(boundary)
(False, True)
"""


__all__ = ["STATIC_DIR", "RunBody", "create_app", "defaults_payload"]



#: Where the frontend lives, served from disk.
STATIC_DIR: Final[Path] = Path(__file__).resolve().parent / "static"

_NO_FRONTEND: Final[str] = """<!doctype html>
<title>QSecure &mdash; API is up, frontend is not installed</title>
<style>
 body{font:15px/1.6 system-ui,sans-serif;margin:3rem auto;max-width:44rem;
      padding:0 1.5rem;color:#1a1a1a;background:#fafaf8}
 code{background:#eeeeea;padding:.1em .35em;border-radius:3px}
 li{margin:.35em 0}
</style>
<h1>QSecure detection API</h1>
<p>The server is running and every endpoint below answers. What is missing is
the dashboard itself: no <code>index.html</code> was found in
<code>qsecure/web/static/</code>.</p>
<ul>
 <li><code>GET /api/health</code></li>
 <li><code>GET /api/attacks</code></li>
 <li><code>GET /api/defaults</code></li>
 <li><code>POST /api/run</code></li>
 <li><code>GET /api/events</code></li>
 <li><code>GET /openapi.json</code> &mdash; generated in-process, fetches nothing</li>
</ul>
<p>This page is served from memory and loads no scripts, fonts or stylesheets
from anywhere. Neither does anything else here.</p>
"""


def _refuse_oversized(declared: int | None) -> JSONResponse:
    """Return the ``413`` body for a request too large to read.

    >>> from qsecure.web.api import _refuse_oversized
    >>> _refuse_oversized(70000).status_code
    413
    """
    return JSONResponse(
        status_code=413,
        content={
            "field": "body",
            "message": (
                f"The request body is larger than the {MAX_REQUEST_BYTES}-byte "
                f"ceiling and was refused without being read. The largest "
                f"legitimate body here is nine scalars and about 300 bytes. "
                f"Nothing parsed it, nothing echoed it, and no run was "
                f"started."
            ),
            "value": (
                f"{declared} bytes declared"
                if declared is not None
                else "more than the ceiling, sent without a Content-Length"
            ),
            "cap": MAX_REQUEST_BYTES,
        },
    )


class _BodyLimit:
    """Refuse an oversized request body **before the application sees it**.

    >>> from fastapi.testclient import TestClient
    >>> from qsecure.web.api import create_app
    >>> client = TestClient(create_app())

    A normal run is untouched:
    >>> client.post("/api/run", json={"key_length": 96, "seed": 4}).status_code
    200

    >>> huge = client.post(
    ...     "/api/run",
    ...     content=b'{"attack": "' + b"x" * 200000 + b'"}',
    ...     headers={"Content-Type": "application/json"},
    ... )
    >>> huge.status_code, huge.json()["cap"]
    (413, 65536)
    >>> len(huge.content) < 600
    True
    """

    def __init__(self, app: Any, limit: int = MAX_REQUEST_BYTES) -> None:
        self.app = app
        self.limit = int(limit)

    @staticmethod
    def _declared(scope: dict[str, Any]) -> int | None:
        """Return the request's ``Content-Length``, or ``None``."""
        for name, value in scope.get("headers", ()):
            if name == b"content-length":
                try:
                    return int(value)
                except ValueError:
                    return None
        return None

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Read at most ``limit`` bytes, then hand the request on or refuse it."""
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        declared = self._declared(scope)
        if declared is not None and declared > self.limit:
            await _refuse_oversized(declared)(scope, receive, send)
            return

        body = bytearray()
        more = True
        while more:
            message = await receive()
            if message["type"] != "http.request":
                # A disconnect mid-body. Nothing to answer and nothing to run.
                return
            body.extend(message.get("body", b""))
            if len(body) > self.limit:
                await _refuse_oversized(declared)(scope, receive, send)
                return
            more = bool(message.get("more_body", False))

        replayed = [bytes(body)]

        async def replay() -> dict[str, Any]:
            """Hand the buffered body to the application, once."""
            if replayed:
                return {
                    "type": "http.request",
                    "body": replayed.pop(),
                    "more_body": False,
                }
            return await receive()

        await self.app(scope, replay, send)


class _NoStaleStatic(StaticFiles):
    """Serve the frontend with ``Cache-Control: no-cache``.

    >>> from fastapi.testclient import TestClient
    >>> from qsecure.web.api import create_app
    >>> client = TestClient(create_app())
    >>> client.get("/static/js/app.js").headers["cache-control"]
    'no-cache'
    >>> client.get("/").headers["cache-control"]
    'no-cache'
    """

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        """Return the file with revalidation forced."""
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


class RunBody(BaseModel):
    """The ``POST /api/run`` body."""

    model_config = ConfigDict(extra="forbid")

    attack: str = "honest"
    key_length: int = DEMO_PARAMS.key_length
    check_fraction: float = 0.25
    noise: float = 0.0
    eps: float = EPS_DEFAULT
    channel_error_rate: float = 0.0
    tolerated_depolarising: float = 0.0
    count_exchange_timing: str = "before-forwarding"
    seed: int = 20260141


def _parameter_set(params: ProtocolParams, note: str) -> dict[str, Any]:
    """Return the derived quantities for one parameter set.

    >>> from qsecure.protocol.params import DEMO_PARAMS
    >>> from qsecure.web.api import _parameter_set
    >>> demo = _parameter_set(DEMO_PARAMS, "the interactive set")
    >>> demo["minimum_matched"], demo["minimum_pooled"]
    (1, 22)
    >>> demo["floors_are_live"]
    True

    >>> from qsecure.eval.security import security_claim_at
    >>> demo["floors_are_live"] == security_claim_at(192)
    True
    """
    floor = minimum_matched_count(params)
    pooled = minimum_pooled_matched_count(params)
    return {
        "key_length": int(params.key_length),
        "signing_length": int(params.signing_length),
        "expected_matched": float(params.expected_matched),
        "minimum_matched": int(floor),
        "minimum_pooled": int(pooled),
        # Aliased for the frontend contract's names; both spellings are in use.
        "matched_minimum": int(floor),
        "pooled_minimum": int(pooled),
        "enforced_repudiation_bound": float(
            enforced_repudiation_bound(params)
        ),
        # Same predicate the transcript is scored by, so /api/run cannot disagree.
        "floors_are_live": security_claim_at(params.signing_length),
        "family_budget": family_budget(params, eps=EPS_DEFAULT).to_dict(),
        "note": note,
    }


def defaults_payload() -> dict[str, Any]:
    """Return the body ``GET /api/defaults`` serves.

    >>> from qsecure.web.api import defaults_payload
    >>> payload = defaults_payload()
    >>> payload["params"]["key_length"], payload["eps_default"]
    (192, 1e-09)
    >>> payload["live_key_length_max"], payload["live_key_length_min"]
    (1024, 24)
    >>> headline = payload["bounds"]["headline"]
    >>> headline["key_length"], headline["minimum_matched"], headline["minimum_pooled"]
    (115200, 36555, 74190)
    >>> bounds = payload["bounds"]
    >>> bounds["matched_minimum"], bounds["pooled_minimum"]
    (36555, 74190)
    >>> f'{headline["enforced_repudiation_bound"]:.4e}', headline["runnable"]
    ('1.4139e-09', False)

    >>> demo = payload["bounds"]["demo"]
    >>> demo["key_length"], demo["floors_are_live"]
    (192, True)
    >>> f'{demo["enforced_repudiation_bound"]:.4f}'
    '0.9940'
    >>> "cannot demonstrate non-repudiation" in demo["note"]
    True
    """
    headline = DEFAULT_PARAMS
    demo = DEMO_PARAMS
    headline_set = _parameter_set(
        headline,
        "The documented security-grade set. NOT RUNNABLE HERE: one session is "
        "about four minutes and roughly 37 MB of transcript, so it is "
        "published as the closed forms it is and never as a run. Anything on "
        "the screen labelled with this set is DERIVED; nothing on the screen "
        "was measured at this length.",
    )
    headline_set["runnable"] = False
    demo_set = _parameter_set(
        demo,
        "The dashboard's starting length. Identical thresholds to the "
        "security-grade set -- the decision rule being demonstrated is THE "
        "SAME decision rule -- with only L cut, so a run answers a click. "
        "Watch the repudiation bound: at this length it is order one, which is "
        "no bound worth the name. A demo-scale run demonstrates the MECHANISM "
        "and cannot demonstrate non-repudiation; a green transferability tick "
        "here would be saying otherwise.",
    )
    demo_set["runnable"] = True
    return {
        # -- the form's starting point ---------------------------------------- #
        "params": {
            "key_length": int(demo.key_length),
            "s_a": float(demo.s_a),
            "s_v": float(demo.s_v),
            "bases": [str(basis) for basis in demo.bases],
            "check_fraction": 0.25,
            "signing_length": int(demo.signing_length),
            "match_probability": float(demo.match_probability),
            "message_bit": MESSAGE_BIT,
            "count_exchange_timing": COUNT_EXCHANGE_TIMINGS[0],
            "attack": "honest",
            "noise": 0.0,
            "eps": EPS_DEFAULT,
            "channel_error_rate": 0.0,
            "tolerated_depolarising": 0.0,
            "seed": 20260141,
            "note": (
                "Where the controls start. Identical thresholds to the "
                "security-grade set; only L is cut, so a run answers a click."
            ),
        },
        "eps_default": EPS_DEFAULT,
        "live_key_length_max": LIVE_KEY_LENGTH_MAX,
        "live_key_length_min": KEY_LENGTH_MIN,
        # -- what is PROVEN, at both sets -------------------------------------- #
        "bounds": {
            "kind": "proven",
            # Headline set at top level and again under `headline`, beside the demo set.
            "enforced_repudiation_bound": headline_set[
                "enforced_repudiation_bound"
            ],
            "matched_minimum": headline_set["matched_minimum"],
            "pooled_minimum": headline_set["pooled_minimum"],
            "family_budget": headline_set["family_budget"],
            "headline": headline_set,
            "demo": demo_set,
            "chsh_classical_bound": 2.0,
            "chsh_tsirelson_bound": 2.0 * math.sqrt(2.0),
            "degenerate_below_key_length": DEGENERATE_BELOW_KEY_LENGTH,
            "note": (
                "Closed forms evaluated at a parameter set: derived, not "
                "measured, and computed without running anything. Nothing here "
                "is a detection rate, and none of it is a false-negative bound "
                "-- there is no such bound in this project and none can be had "
                "from a transcript. What a run publishes afterwards is "
                "Detection.false_positive_bound. It is NOT eps: at L = 384, "
                "eps = 1e-9, false_positive_bound is 3.3964e-10 and the budget "
                "is 2.944 times larger, so quoting the budget overstates the "
                "error rate by that factor. It is also NOT evidence_bound, "
                "which is a post hoc statement about the signals that actually "
                "fired -- a different claim on a different scale entirely, "
                "smaller than false_positive_bound by about 7e9 on a run where "
                "anything fires, and null on a run where nothing does."
            ),
        },
        # -- every cap the request surface enforces ---------------------------- #
        "limits": limits_payload(),
        "check_fraction_max": CHECK_FRACTION_MAX,
        "noise_max": NOISE_MAX,
        "channel_error_rate_max": CHANNEL_ERROR_RATE_MAX,
        "tolerated_depolarising_max": NOISE_MAX,
        "eps_min": EPS_MIN,
        "eps_max": EPS_MAX,
        "count_exchange_timings": list(COUNT_EXCHANGE_TIMINGS),
        "max_concurrent_runs": MAX_CONCURRENT_RUNS,
        # -- MEASURED, and labelled as such ------------------------------------ #
        "noise_null_calibration": {
            "kind": "measured",
            "what": (
                "honest runs over a depolarising wire, scored against the "
                "default noiseless null (channel_error_rate = 0.0). The "
                "denominator is EVERY run at that level, not only the runs "
                "both verifiers accepted: an honest run over a noisy link "
                "can still be rejected on its rate, and at the design "
                "level 16 of the 60 verifier verdicts are rejections. "
                "Filtering to accepting runs would change what the rate "
                "means without changing the number printed beside it."
            ),
            "runs_per_level": 30,
            "key_length": 384,
            "check_fraction": 0.25,
            "source": (
                "Experiment 'noise', 6 cells x 30 trials at "
                "L = 384, check_fraction 0.25. Every trial's seed is a "
                "pure function of (experiment, cell, index)."
            ),
            "regenerate": (
                "the 'noise' experiment in qsecure.eval.experiments, "
                "run and then reduced with qsecure.eval.reduce"
            ),
            "levels": [
                {"noise": 0.0, "detected": 0},
                {"noise": 0.0025, "detected": 14},
                {"noise": 0.005, "detected": 23},
                {"noise": 0.01, "detected": 28},
                {"noise": 0.015, "detected": 30},
                {"noise": 0.03125, "detected": 30},
            ],
            # Must read true alone: detect() has two nulls (see `second_null_note`).
            "with_true_rate_passed": (
                "0/30 at every level when BOTH of detect()'s nulls are stated: "
                "the link's true matched-position error rate as "
                "channel_error_rate, AND the link's Werner strength as "
                "tolerated_depolarising. This table is the RATE family's. "
                "Correcting this null alone does not clear an honest noisy run."
            ),
            "second_null_note": (
                "That calibration is the RATE family. The CHANNEL family has "
                "its own null, tolerated_depolarising, also defaulting to a "
                "perfect resource, and stating only the first leaves an honest "
                "noisy run firing: measured over twelve seeds at L = 192, "
                "check_fraction = 0.25, link strength 0.03125 -- 12/12 "
                "detected with both nulls at zero, 12/12 with only "
                "channel_error_rate corrected, 0/12 with both stated. THE "
                "VERIFIERS DO NOT MOVE WITH THE NULLS AT ALL: their outcomes "
                "are the protocol's and the nulls are the detector's, and over "
                "those twelve seeds the pair of outcomes is identical under "
                "all three settings -- both accept on 7 seeds and Bob rejects "
                "the honest signature on 5, which is what a link at the design "
                "noise level does to a signature and is a separate fact from "
                "anything the detector said."
            ),
        },
        "attacks": roster_payload(),
    }


def _run_digest(request: dict[str, Any]) -> str:
    """Return the identifier the log groups one run's events under.

    >>> from qsecure.web.api import _run_digest
    >>> first = _run_digest({"attack": "honest", "seed": 1})
    >>> len(first), first == _run_digest({"seed": 1, "attack": "honest"})
    (16, True)

    >>> first == _run_digest({"attack": "honest", "seed": 2})
    False
    """
    canonical = json.dumps(request, sort_keys=True, default=str)
    return hashlib.blake2s(
        canonical.encode("utf-8"), digest_size=8
    ).hexdigest()


def _record_verdicts(log: AuditLog, payload: dict[str, Any]) -> None:
    """Append one security event per verification outcome."""
    detection = payload.get("detection")
    if detection is None:
        return
    request = payload["request"]
    by_party = (payload.get("run") or {}).get("aborts", {}).get("by_party", {})
    session_id = _run_digest(request)
    named = tuple(detection.get("named", ()))
    for party, outcome in sorted(detection.get("outcomes", {}).items()):
        log.record(
            party=party,
            verdict=outcome,
            session_id=session_id,
            # Not defaulted: a wrong audit number is worse than a missing one.
            key_length=detection["key_length"],
            check_fraction=request["check_fraction"],
            abort_reason=by_party.get(party),
            hypotheses=named,
        )


def create_app(
    static_dir: Path | None = None, audit_log: AuditLog | None = None
) -> FastAPI:
    """Build the application: the JSON API plus the static frontend.

    >>> from fastapi.testclient import TestClient
    >>> from qsecure.web.api import create_app
    >>> app = create_app()
    >>> app.docs_url is None and app.redoc_url is None
    True
    >>> client = TestClient(app)
    >>> [entry["key"] for entry in client.get("/api/attacks").json()][:2]
    ['honest', 'outside-forgery']

    >>> body = client.post(
    ...     "/api/run", json={"attack": "honest", "key_length": 96, "seed": 2}
    ... ).json()
    >>> sorted(body)
    ['detection', 'error', 'ground_truth', 'request', 'run', 'timings']
    >>> "ground_truth" in body["detection"], body["error"] is None
    (False, True)
    """
    root = STATIC_DIR if static_dir is None else Path(static_dir)
    app = FastAPI(
        title="QSecure - quantum-inspired detection for QDS",
        version=__version__,
        docs_url=None,
        redoc_url=None,
    )
    # Outermost, so oversized bodies are refused before any route sees them.
    app.add_middleware(_BodyLimit)
    # Non-blocking: refuse over-capacity requests rather than queue them.
    gate = threading.BoundedSemaphore(MAX_CONCURRENT_RUNS)
    # Per app, so separate create_app() calls get separate logs.
    log = AuditLog() if audit_log is None else audit_log

    @app.exception_handler(RequestRefused)
    async def _refused(
        request: Request, refusal: RequestRefused
    ) -> JSONResponse:
        """Render a bounds refusal as a body that names the cap."""
        del request
        return JSONResponse(status_code=400, content=refusal.to_dict())

    @app.exception_handler(RequestValidationError)
    async def _invalid(
        request: Request, invalid: RequestValidationError
    ) -> JSONResponse:
        """Render a schema error, including one JSON itself cannot quote."""
        del request
        return JSONResponse(
            status_code=422,
            content={"detail": jsonable_encoder(json_safe(invalid.errors()))},
        )

    @app.get("/api/health")
    def health(response: Response) -> dict[str, Any]:
        """Report that the process is up, and which build it is."""
        response.headers["Cache-Control"] = "no-store"
        return {"ok": True, "version": __version__}

    @app.get("/api/attacks")
    def attacks(response: Response) -> list[dict[str, Any]]:
        """List every arm, with its detectability stated rather than implied."""
        response.headers["Cache-Control"] = "no-store"
        return roster_payload()

    @app.get("/api/defaults")
    def defaults(response: Response) -> dict[str, Any]:
        """Serve the form's starting point, the published caps and the bounds."""
        response.headers["Cache-Control"] = "no-store"
        return defaults_payload()

    @app.post("/api/run")
    def run(body: RunBody, response: Response) -> dict[str, Any]:
        """Run one arm and score it."""
        response.headers["Cache-Control"] = "no-store"
        request = RunRequest.from_mapping(body.model_dump())
        if not gate.acquire(blocking=False):
            response.status_code = 503
            response.headers["Retry-After"] = "3"
            return {
                "request": request.to_dict(),
                "detection": None,
                "run": None,
                "ground_truth": {},
                "timings": {"session_ms": 0.0, "detect_ms": 0.0},
                "error": {
                    "kind": "at-capacity",
                    "message": (
                        f"{MAX_CONCURRENT_RUNS} runs are already generating "
                        f"sessions and this one is refused rather than queued. "
                        f"Session generation is CPU-bound; queueing would "
                        f"leave the screen waiting with nothing to show. Try "
                        f"again in a moment."
                    ),
                    "cap": MAX_CONCURRENT_RUNS,
                },
            }
        try:
            payload = run_once(request).to_dict()
        finally:
            gate.release()
        _record_verdicts(log, payload)
        return payload

    @app.get("/api/events")
    def events(response: Response) -> dict[str, Any]:
        """Serve the security event log, oldest event first."""
        response.headers["Cache-Control"] = "no-store"
        retained, recorded, dropped = log.snapshot()
        return {
            "events": [event.to_dict() for event in retained],
            "count": len(retained),
            "recorded": recorded,
            "dropped": dropped,
            "capacity": log.capacity,
            "persisted": log.path is not None,
            "note": (
                "One event per verification outcome. A request refused for "
                "its size or its range, and a run that failed before a "
                "verdict, appear nowhere here, so a count of these events is "
                "a count of verdicts. The party is SELF-DECLARED: a verifier "
                "using a recipient record obtained outside the protocol "
                "produces an event no field of which differs from the one its "
                "owner would have produced, and nothing in this log detects "
                "that."
            ),
        }

    if root.is_dir():
        app.mount(
            "/static", _NoStaleStatic(directory=str(root)), name="static"
        )

    @app.get("/", response_class=HTMLResponse)
    def index() -> Response:
        """Serve the frontend, or say plainly that it is not installed."""
        page = root / "index.html"
        if page.is_file():
            return FileResponse(
                page, headers={"Cache-Control": "no-cache"}
            )
        return HTMLResponse(
            content=_NO_FRONTEND,
            status_code=200,
            headers={"Cache-Control": "no-cache"},
        )

    return app
