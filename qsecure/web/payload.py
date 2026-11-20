"""The ``run`` and ``ground_truth`` objects, in the shape the screen reads.

>>> import json
>>> import numpy as np
>>> from qsecure.detect import TranscriptStatistics, detect
>>> from qsecure.protocol.params import ProtocolParams
>>> from qsecure.protocol.session import QDSSession
>>> from qsecure.web.payload import run_facts
>>> transcript = QDSSession(
...     ProtocolParams(key_length=96, check_fraction=0.25),
...     rng=np.random.default_rng(7),
... ).run(0)
>>> text = transcript.to_json()
>>> stats = TranscriptStatistics.from_json(text)
>>> facts = run_facts(
...     stats, detect(text, eps=1e-9), json.loads(text),
...     requested_key_length=96, check_fraction=0.25,
... )
>>> facts["nominal_key_length"], facts["sifted_key_length"]
(96, 72)
>>> [row["outcome"] for row in facts["verifiers"]]
['accepted', 'accepted']
>>> facts["channel_evaluable"], len(facts["links"])
(True, 4)

>>> quiet = QDSSession(
...     ProtocolParams(key_length=96), rng=np.random.default_rng(7)
... ).run(0)
>>> text = quiet.to_json()
>>> bare = run_facts(
...     TranscriptStatistics.from_json(text), detect(text, eps=1e-9),
...     json.loads(text), requested_key_length=96, check_fraction=0.0,
... )
>>> bare["links"], bare["channel_evaluable"]
([], False)
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from qsecure.detect.statistics import TranscriptStatistics


__all__ = ["ground_truth_for", "nulls_stated", "run_facts"]


def nulls_stated(
    channel_error_rate: float, tolerated_depolarising: float
) -> dict[str, Any]:
    """Return which of :func:`~qsecure.detect.detect`'s two nulls were stated.

    >>> from qsecure.web.payload import nulls_stated
    >>> default = nulls_stated(0.0, 0.0)
    >>> default["both_are_default"], default["both_are_stated"]
    (True, False)

    >>> half = nulls_stated(0.015625, 0.0)
    >>> half["rate_null_is_default"], half["channel_null_is_default"]
    (False, True)
    >>> half["both_are_default"], half["both_are_stated"]
    (False, False)
    >>> both = nulls_stated(0.015625, 0.03125)
    >>> both["both_are_stated"], both["channel_null_field"]
    (True, 'tolerated_depolarising')
    """
    rate_default = float(channel_error_rate) == 0.0
    channel_default = float(tolerated_depolarising) == 0.0
    return {
        "channel_error_rate": float(channel_error_rate),
        "tolerated_depolarising": float(tolerated_depolarising),
        "rate_null_field": "channel_error_rate",
        "channel_null_field": "tolerated_depolarising",
        "rate_null_is_default": bool(rate_default),
        "channel_null_is_default": bool(channel_default),
        "both_are_default": bool(rate_default and channel_default),
        "both_are_stated": bool(not rate_default and not channel_default),
        "note": (
            "detect() takes TWO nulls and both default to a perfect link: "
            "channel_error_rate for the rate family and tolerated_depolarising "
            "for the channel family. They are two parameterisations of the "
            "same physics and neither is converted into the other, because "
            "that would state a null the operator did not ask for. Stating "
            "only one leaves an honest run over a noisy link DETECTED with an "
            "adversary named -- 12/12 at L = 192, strength 0.03125, over twelve "
            "seeds, against 0/12 with both stated. Neither is ever inferred "
            "from the transcript: at check_fraction = 0 the transcript carries "
            "no estimate of either, and guessing would be inventing a null."
        ),
    }


def _interval(interval: Any) -> dict[str, Any] | None:
    """Return one confidence interval as JSON, or ``None``."""
    if interval is None:
        return None
    return {
        "low": float(interval.low),
        "high": float(interval.high),
        "confidence": float(interval.confidence),
        "method": str(interval.method),
    }


def _count(statistic: Any) -> dict[str, Any]:
    """Return one :class:`~qsecure.detect.statistics.CountStatistic` as JSON."""
    return {
        "name": str(statistic.name),
        "count": int(statistic.count),
        "trials": int(statistic.trials),
        "null_probability": float(statistic.null_probability),
        "null": str(statistic.null),
    }


def _optional_count(statistic: Any) -> int | None:
    """Return one count, or ``None`` where the statistic does not exist."""
    return None if statistic is None else int(statistic.count)


def _published_rate(
    results: Mapping[str, Mapping[str, Any]], party: str
) -> float | None:
    """Return the mismatch rate the transcript published for one verifier."""
    entry = results.get(str(party))
    if entry is None or entry.get("rate") is None:
        return None
    return float(entry["rate"])


def _link_facts(link: Any) -> dict[str, Any]:
    """Return one link's published check-round statistics as JSON."""
    qber = None
    if link.qber is not None:
        qber = {
            "errors": int(link.qber.errors),
            "rounds": int(link.qber.rounds),
            "value": float(link.qber.estimate),
            "interval": _interval(link.qber.interval),
            "bound": _interval(link.qber_bound),
        }
    chsh = None
    if link.chsh is not None:
        chsh = {
            "value": float(link.chsh.statistic),
            "correlators": [float(item) for item in link.chsh.correlators],
            "counts": [int(item) for item in link.chsh.counts],
            "rounds": int(link.chsh.rounds),
            "interval": _interval(link.chsh.interval),
            "bound": _interval(link.chsh_bound),
            "violates_classical_bound": bool(
                link.chsh.violates_classical_bound
            ),
            "consistent_with_ideal": bool(link.chsh.consistent_with_ideal),
        }
    resource = link.resource
    return {
        "party": str(link.party),
        "message_bit": int(link.message_bit),
        "errors": _count(link.errors),
        "qber": qber,
        "chsh": chsh,
        "chsh_unavailable": (
            None
            if link.chsh_unavailable is None
            else str(link.chsh_unavailable)
        ),
        "resource": {
            "samples": int(resource.samples),
            "qber_samples": int(resource.qber_samples),
            "chsh_samples": int(resource.chsh_samples),
            "mean_fidelity": float(resource.mean_fidelity),
            "min_fidelity": float(resource.min_fidelity),
            "mean_purity": float(resource.mean_purity),
            "min_purity": float(resource.min_purity),
            "mean_concurrence": float(resource.mean_concurrence),
            "min_concurrence": float(resource.min_concurrence),
        },
    }


def _verifier_facts(
    stats: TranscriptStatistics,
    outcomes: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return one row per verifier, the four-valued outcome included."""
    parties = sorted(
        {str(party) for party in stats.verifiers}
        | {str(party) for party in outcomes}
    )
    rows: list[dict[str, Any]] = []
    for party in parties:
        verifier = stats.verifiers.get(party)
        outcome = outcomes.get(str(party))
        label = None if outcome is None else str(getattr(outcome, "value", outcome))
        if verifier is None:
            rows.append(
                {
                    "party": str(party),
                    "outcome": label,
                    "accepted": None,
                    "matched": None,
                    "matched_trials": None,
                    "matched_null_probability": None,
                    "mismatches": None,
                    "mismatch_trials": None,
                    "rate": None,
                    "threshold": None,
                    "margin": None,
                    "reported_matched": None,
                    "reported_mismatches": None,
                    "consistent": None,
                    "recomputed": False,
                    "scored": False,
                }
            )
            continue
        rows.append(
            {
                "party": str(party),
                "outcome": label,
                "accepted": (
                    None
                    if verifier.accepted is None
                    else bool(verifier.accepted)
                ),
                "matched": int(verifier.matched.count),
                "matched_trials": int(verifier.matched.trials),
                "matched_null_probability": float(
                    verifier.matched.null_probability
                ),
                "mismatches": int(verifier.mismatch.count),
                "mismatch_trials": int(verifier.mismatch.trials),
                "rate": _published_rate(results, party),
                "threshold": (
                    None
                    if verifier.threshold is None
                    else float(verifier.threshold)
                ),
                "margin": (
                    None if verifier.margin is None else float(verifier.margin)
                ),
                "reported_matched": (
                    None
                    if verifier.reported_matched is None
                    else int(verifier.reported_matched)
                ),
                "reported_mismatches": (
                    None
                    if verifier.reported_mismatches is None
                    else int(verifier.reported_mismatches)
                ),
                "consistent": bool(verifier.consistent),
                "recomputed": bool(verifier.recomputed),
                "scored": True,
            }
        )
    return rows


def run_facts(
    stats: TranscriptStatistics,
    detection: Any,
    transcript_json: Mapping[str, Any],
    *,
    requested_key_length: int,
    check_fraction: float,
) -> dict[str, Any]:
    """Return the ``run`` half of a ``POST /api/run`` response."""
    outcomes = (
        detection["outcomes"]
        if isinstance(detection, Mapping)
        else detection.outcomes
    )
    pooled = stats.pooled
    links = [
        _link_facts(stats.links[key]) for key in sorted(stats.links.keys())
    ]
    results = {
        str(entry["party"]): entry
        for entry in transcript_json.get("results", [])
    }
    return {
        "requested_key_length": int(requested_key_length),
        "nominal_key_length": int(stats.nominal_key_length),
        "sifted_key_length": int(stats.params.key_length),
        "check_fraction": float(check_fraction),
        "message_bit": int(stats.message_bit),
        "s_a": float(stats.params.s_a),
        "s_v": float(stats.params.s_v),
        # -- what the run did and did not do -------------------------------- #
        "check_rounds_present": bool(stats.has_check_rounds),
        "channel_monitored": bool(stats.channel_monitored),
        "channel_evaluable": bool(stats.has_check_rounds and bool(links)),
        "counts_exchanged": bool(stats.counts_exchanged),
        "count_exchange_timing": str(stats.count_exchange_timing),
        "symmetrised": bool(stats.symmetrised),
        "session_coherent": bool(stats.session_coherent),
        "is_complete": bool(stats.is_complete),
        "aborted": bool(stats.aborted),
        "forwarding_altered_signature": bool(
            stats.forwarding_altered_signature
        ),
        "signer_saw_recipient_logs": bool(stats.signer_saw_recipient_logs),
        # -- the transferability question ----------------------------------- #
        "security_claim": bool(stats.security_claim),
        "transferable": bool(stats.transferable),
        "repudiated": bool(stats.repudiated),
        "repudiation_guarantee": (
            None
            if stats.repudiation_guarantee is None
            else float(stats.repudiation_guarantee)
        ),
        "enforced_repudiation_bound": (
            None
            if stats.enforced_repudiation_bound is None
            else float(stats.enforced_repudiation_bound)
        ),
        # -- the floors ------------------------------------------------------ #
        "floors": {
            "matched_minimum": int(pooled.minimum_matched),
            "pooled_minimum": int(pooled.minimum_pooled),
            "matched_floor_bound": float(pooled.matched_floor_bound),
            "meets_pooled_floor": bool(pooled.meets_pooled_floor),
            "meets_every_floor": bool(pooled.meets_every_floor),
            "degenerate": not bool(stats.security_claim),
        },
        "verifiers": _verifier_facts(stats, outcomes, results),
        "pooled": {
            "count": (
                None if pooled.pooled is None else int(pooled.pooled.count)
            ),
            "trials": (
                None if pooled.pooled is None else int(pooled.pooled.trials)
            ),
            "null_probability": (
                None
                if pooled.pooled is None
                else float(pooled.pooled.null_probability)
            ),
            "declared_bob": _optional_count(pooled.declared_bob),
            "declared_charlie": _optional_count(pooled.declared_charlie),
            "declared_pooled": _optional_count(pooled.declared_pooled),
            "declaration_digest_present": bool(
                pooled.declaration_digest_present
            ),
        },
        # EMPTY on an unmonitored run, never zero-filled: an unmonitored link
        # is not a clean one.
        "links": links,
        "replay": {
            "total_refusals": int(stats.replay.total_refusals),
            "refusals_by_party": {
                str(party): int(count)
                for party, count in stats.replay.refusals_by_party.items()
            },
            "spent_rounds": int(stats.replay.spent_rounds),
            "distinct_sessions": int(stats.replay.distinct_sessions),
        },
        "aborts": {
            "total": int(stats.aborts.total),
            "structural": int(stats.aborts.structural),
            "evidence": int(stats.aborts.evidence),
            "by_reason": {
                str(reason): int(count)
                for reason, count in stats.aborts.by_reason.items()
            },
            # party -> the REASON, not a count: one verifier aborts at most
            # once per run and the reason is what the screen has to name.
            "by_party": {
                str(party): str(reason)
                for party, reason in stats.aborts.by_party.items()
            },
            "shortfalls": {
                str(party): int(amount)
                for party, amount in stats.aborts.shortfalls.items()
            },
            "honest_bound": float(stats.aborts.honest_bound),
        },
        # The transcript's own words, incl. the per-run repudiation bound.
        "summary": str(stats.summary()),
    }


def ground_truth_for(
    key: str,
    label: str,
    detectable: str,
    assumption: str | None,
    *,
    acted: bool,
    targeted_links: Sequence[Sequence[Any]] | None = None,
    seams_held: Sequence[str] = (),
    notes: str = "",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the ``ground_truth`` half of a ``POST /api/run`` response.

    >>> from qsecure.web.payload import ground_truth_for
    >>> truth = ground_truth_for(
    ...     "channel-manipulation", "Channel manipulation", "detectable", None,
    ...     acted=False, seams_held=("resource_factory",),
    ... )
    >>> truth["adversary_present"], truth["acted"]
    (True, False)
    >>> truth["identical_to_honest"]
    True

    >>> ground_truth_for(
    ...     "honest", "Honest run", "not-an-attack", None, acted=False
    ... )["identical_to_honest"]
    False
    """
    truth = {
        "attack": str(key),
        "label": str(label),
        "adversary_present": key != "honest",
        "acted": bool(acted),
        "identical_to_honest": key != "honest" and not acted,
        "targeted_links": [list(pair) for pair in (targeted_links or [])],
        "seams_held": list(seams_held),
        "detectable": str(detectable),
        "assumption": assumption,
        "notes": str(notes),
    }
    if extra:
        truth.update(extra)
    return truth
