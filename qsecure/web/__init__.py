"""The dashboard's backend -- one process, one command, no network.
"""

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
    "AttackSpec",
    "RequestRefused",
    "RunRequest",
    "RunResult",
    "attack_spec",
    "ground_truth_for",
    "limits_payload",
    "roster_payload",
    "run_facts",
    "run_once",
]
