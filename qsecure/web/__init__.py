"""The dashboard's backend -- one process, one command, no network.

>>> from qsecure.web import create_app
>>> from fastapi.testclient import TestClient
>>> client = TestClient(create_app())
>>> client.get("/api/health").json()
{'ok': True, 'version': '0.1.0'}
"""

from qsecure.web.api import STATIC_DIR, RunBody, create_app, defaults_payload
from qsecure.web.catalogue import (
    ATTACK_KEYS,
    ATTACKS,
    AttackSpec,
    attack_spec,
    roster_payload,
)
from qsecure.web.driver import MESSAGE_BIT, RunRequest, RunResult, run_once
from qsecure.web.limits import (
    CHECK_FRACTION_MAX,
    EPS_DEFAULT,
    EPS_MAX,
    EPS_MIN,
    KEY_LENGTH_MIN,
    LIVE_KEY_LENGTH_MAX,
    MAX_CONCURRENT_RUNS,
    RequestRefused,
    limits_payload,
)
from qsecure.web.payload import ground_truth_for, run_facts

__all__ = [
    "ATTACKS",
    "ATTACK_KEYS",
    "CHECK_FRACTION_MAX",
    "EPS_DEFAULT",
    "EPS_MAX",
    "EPS_MIN",
    "KEY_LENGTH_MIN",
    "LIVE_KEY_LENGTH_MAX",
    "MAX_CONCURRENT_RUNS",
    "MESSAGE_BIT",
    "STATIC_DIR",
    "AttackSpec",
    "RequestRefused",
    "RunBody",
    "RunRequest",
    "RunResult",
    "attack_spec",
    "create_app",
    "defaults_payload",
    "ground_truth_for",
    "limits_payload",
    "roster_payload",
    "run_facts",
    "run_once",
]
