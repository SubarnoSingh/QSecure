"""Transcript-only attack detection with derived, budgeted false-positive bounds.

Statistics are read from a JSON-round-tripped transcript only; three threshold
families (rate, structural, channel) derive critical values from stated honest
nulls, and :func:`detect` combines them under one family-wise budget.

Examples
--------
>>> import numpy as np
>>> from qsecure.detect import detect
>>> from qsecure.protocol.params import ProtocolParams
>>> from qsecure.protocol.session import QDSSession
>>> result = detect(
...     QDSSession(
...         ProtocolParams(key_length=192), rng=np.random.default_rng(7)
...     ).run(0),
...     eps=1e-9,
... )
>>> result.detected, f"{result.false_positive_bound:.4e}"
(False, '2.6499e-10')

>>> import numpy as np
>>> from qsecure.detect import TranscriptStatistics
>>> from qsecure.protocol.params import ProtocolParams
>>> from qsecure.protocol.session import QDSSession
>>> stats = TranscriptStatistics.from_transcript(
...     QDSSession(
...         ProtocolParams(key_length=192), rng=np.random.default_rng(7)
...     ).run(0)
... )
>>> stats.verifier("Bob").matched.count
69
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


from qsecure.detect.detector import (
    DETECTOR_FAMILIES,
    HYPOTHESES,
    HYPOTHESIS_TABLE,
    LINK_ROSTER,
    SIGNATURE_SUBSTITUTION_GROUP,
    Attribution,
    Detection,
    FamilyBudget,
    Hypothesis,
    HypothesisPredicate,
    Signal,
    SignalKind,
    Support,
    detect,
    family_budget,
)
from qsecure.detect.statistics import (
    EVIDENCE_ABORT_REASONS,
    IDEAL_TOLERANCE,
    STRUCTURAL_ABORT_REASONS,
    AbortStatistics,
    CountStatistic,
    DeclarationStatistics,
    LinkStatistics,
    PooledStatistics,
    RecordStatistics,
    ReplayStatistics,
    ResourceStatistics,
    TranscriptStatistics,
    VerifierStatistics,
    chernoff_deviation_bound,
    pooled_check_qber,
    why_wings_agree_is_absent,
    wilson_interval,
)
from qsecure.detect.thresholds_channel import (
    CHANNEL_STATISTICS,
    CONCURRENCE_RANGE,
    FIDELITY_RANGE,
    PURITY_RANGE,
    ChannelScreen,
    ChannelThreshold,
    CheckRoundShare,
    bounded_mean_threshold,
    chsh_certificate_threshold,
    chsh_threshold,
    concurrence_threshold,
    divide_budget,
    fidelity_threshold,
    link_check_rounds,
    purity_threshold,
    qber_threshold,
    screen_link,
    werner_concurrence,
    werner_fidelity,
    werner_purity,
)
from qsecure.detect.thresholds_rate import (
    EXACT_TRIALS_LIMIT,
    INEQUALITIES,
    RATE_COUNT_FREE_TESTS,
    RATE_COUNT_ROSTER,
    DerivedThreshold,
    RateCountThresholds,
    RateCountVerdict,
    binomial_tail_bound,
    critical_count,
    declared_count_threshold,
    dominance_noise_level,
    floor_comparison,
    forgery_separation_sigma,
    matched_count_threshold,
    mismatch_rate_threshold,
    pooled_count_threshold,
)
from qsecure.detect.thresholds_structural import (
    FLOOR_BUDGET,
    STRUCTURAL_CHECKS,
    AbortAttribution,
    NoVerdictCount,
    OutcomeTally,
    RunOutcome,
    StructuralAlarm,
    StructuralCheck,
    StructuralReport,
    StructuralThreshold,
    abort_reason_bound,
    attribute_aborts,
    chernoff_lower_tail_count,
    evidence_abort_bound,
    evidence_abort_threshold,
    minimum_sifted_length,
    outcome_of,
    outcomes,
    point_mass_threshold,
    replay_refusal_threshold,
    run_shape_threshold,
    run_shape_violations,
    shortfall_threshold,
    structural_abort_threshold,
    structural_report,
    tally_outcomes,
)


@dataclass(frozen=True)
class ThresholdView:
    """One reading of any shipped threshold, whichever family produced it.

    Examples
    --------
    >>> from qsecure.detect import (
    ...     ThresholdView, chsh_threshold, matched_count_threshold,
    ...     qber_threshold, structural_abort_threshold, threshold_view,
    ... )
    >>> rate = threshold_view(
    ...     matched_count_threshold(192, 1 / 3, eps=1e-9, side="lower"))
    >>> rate.family, rate.direction, rate.critical_value, rate.can_fire
    ('rate', 'lower', 26, True)
    >>> channel = threshold_view(qber_threshold(rounds=24, epsilon=1e-9))
    >>> channel.family, channel.direction, channel.critical_value
    ('channel', 'upper', 1.0)
    >>> structural = threshold_view(structural_abort_threshold(1e-9))
    >>> structural.family, structural.direction, structural.can_fire
    ('structural', None, True)

    >>> all(v.false_positive_bound <= v.budget
    ...     for v in (rate, channel, structural))
    True

    >>> from qsecure.detect.thresholds_structural import evidence_abort_threshold
    >>> from qsecure.protocol.params import ProtocolParams
    >>> bad = threshold_view(evidence_abort_threshold(
    ...     ProtocolParams(key_length=384), 1.6e-19))
    >>> bad.can_fire, bad.false_positive_bound > bad.budget
    (False, True)

    >>> thin = threshold_view(chsh_threshold(counts=(2, 2, 2, 2), epsilon=1e-9))
    >>> thin.can_fire, thin.raw.reaches_its_statistic
    (False, False)
    >>> silent = threshold_view(
    ...     matched_count_threshold(8, 1 / 3, eps=1e-30, side="lower"))
    >>> silent.can_fire, silent.raw.vacuous
    (False, True)
    """

    family: str
    statistic: str
    null: str
    inequality: str
    budget: float
    critical_value: float | None
    false_positive_bound: float
    direction: str | None
    can_fire: bool
    is_alarm: bool
    derivation: str
    raw: Any

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, without the carrier it came from.

        Returns
        -------
        dict
        """
        return {
            "family": self.family,
            "statistic": self.statistic,
            "null": self.null,
            "inequality": self.inequality,
            "budget": self.budget,
            "critical_value": self.critical_value,
            "false_positive_bound": self.false_positive_bound,
            "direction": self.direction,
            "can_fire": self.can_fire,
            "is_alarm": self.is_alarm,
            "derivation": self.derivation,
        }


def threshold_view(threshold: Any) -> ThresholdView:
    """Read any family's threshold through one vocabulary.

    Parameters
    ----------
    threshold : DerivedThreshold or ChannelThreshold or StructuralThreshold
        A threshold from any of the three families.

    Returns
    -------
    ThresholdView
        The same threshold, with the conventions reconciled.

    Raises
    ------
    TypeError
        If ``threshold`` is not one of the three carrier types.

    Examples
    --------
    >>> from qsecure.detect import pooled_count_threshold, threshold_view
    >>> view = threshold_view(
    ...     pooled_count_threshold(192, 1 / 3, eps=1e-9, side="upper"))
    >>> view.family, view.statistic.startswith("M ="), view.direction
    ('rate', True, 'upper')
    >>> f"{view.false_positive_bound:.4e}" , view.budget <= 1e-9
    ('6.8419e-10', True)
    """
    if isinstance(threshold, DerivedThreshold):
        return ThresholdView(
            family="rate",
            statistic=threshold.statistic,
            null=threshold.null,
            inequality=threshold.inequality,
            budget=threshold.budget,
            critical_value=threshold.critical_count,
            false_positive_bound=threshold.false_positive_bound,
            direction=threshold.side,
            # Inverted: rate says ``vacuous``, channel says ``reaches_its_statistic``.
            can_fire=not threshold.vacuous,
            is_alarm=True,
            derivation=threshold.derivation,
            raw=threshold,
        )
    if isinstance(threshold, ChannelThreshold):
        return ThresholdView(
            family="channel",
            statistic=threshold.statistic,
            null=threshold.null,
            inequality=threshold.inequality,
            budget=threshold.epsilon,
            critical_value=threshold.critical_value,
            false_positive_bound=threshold.false_positive_bound,
            direction=threshold.direction,
            can_fire=threshold.reaches_its_statistic,
            is_alarm=threshold.is_alarm,
            derivation=threshold.derivation,
            raw=threshold,
        )
    if isinstance(threshold, StructuralThreshold):
        return ThresholdView(
            family="structural",
            statistic=threshold.statistic,
            null=threshold.null,
            inequality=threshold.inequality,
            budget=threshold.budget,
            critical_value=threshold.fires_at,
            false_positive_bound=threshold.false_positive_bound,
            direction=None,
            can_fire=threshold.fires_at is not None,
            is_alarm=True,
            derivation=threshold.derivation,
            raw=threshold,
        )
    raise TypeError(
        f"threshold must be a DerivedThreshold, a ChannelThreshold or a "
        f"StructuralThreshold, got {type(threshold).__name__}. The three "
        f"carriers are not duck-typed on purpose: two of them spell 'can this "
        f"fire' under different names and opposite senses, so a structural "
        f"guess from the attributes present is how a caller reads one as the "
        f"other."
    )

__all__ = [
    "AbortAttribution",
    "AbortStatistics",
    "Attribution",
    "CHANNEL_STATISTICS",
    "CONCURRENCE_RANGE",
    "ChannelScreen",
    "ChannelThreshold",
    "CheckRoundShare",
    "CountStatistic",
    "DETECTOR_FAMILIES",
    "DeclarationStatistics",
    "DerivedThreshold",
    "Detection",
    "EVIDENCE_ABORT_REASONS",
    "EXACT_TRIALS_LIMIT",
    "FIDELITY_RANGE",
    "FLOOR_BUDGET",
    "FamilyBudget",
    "HYPOTHESES",
    "HYPOTHESIS_TABLE",
    "Hypothesis",
    "HypothesisPredicate",
    "IDEAL_TOLERANCE",
    "INEQUALITIES",
    "LINK_ROSTER",
    "LinkStatistics",
    "NoVerdictCount",
    "OutcomeTally",
    "PURITY_RANGE",
    "PooledStatistics",
    "RATE_COUNT_FREE_TESTS",
    "RATE_COUNT_ROSTER",
    "RateCountThresholds",
    "RateCountVerdict",
    "RecordStatistics",
    "ReplayStatistics",
    "ResourceStatistics",
    "RunOutcome",
    "SIGNATURE_SUBSTITUTION_GROUP",
    "STRUCTURAL_ABORT_REASONS",
    "STRUCTURAL_CHECKS",
    "Signal",
    "SignalKind",
    "StructuralAlarm",
    "StructuralCheck",
    "StructuralReport",
    "StructuralThreshold",
    "Support",
    "ThresholdView",
    "TranscriptStatistics",
    "VerifierStatistics",
    "abort_reason_bound",
    "attribute_aborts",
    "binomial_tail_bound",
    "bounded_mean_threshold",
    "chernoff_deviation_bound",
    "chernoff_lower_tail_count",
    "chsh_certificate_threshold",
    "chsh_threshold",
    "concurrence_threshold",
    "critical_count",
    "declared_count_threshold",
    "detect",
    "detector",
    "divide_budget",
    "dominance_noise_level",
    "evidence_abort_bound",
    "evidence_abort_threshold",
    "family_budget",
    "fidelity_threshold",
    "floor_comparison",
    "forgery_separation_sigma",
    "link_check_rounds",
    "matched_count_threshold",
    "minimum_sifted_length",
    "mismatch_rate_threshold",
    "outcome_of",
    "outcomes",
    "point_mass_threshold",
    "pooled_check_qber",
    "pooled_count_threshold",
    "purity_threshold",
    "qber_threshold",
    "replay_refusal_threshold",
    "run_shape_threshold",
    "run_shape_violations",
    "screen_link",
    "shortfall_threshold",
    "statistics",
    "structural_abort_threshold",
    "structural_report",
    "tally_outcomes",
    "threshold_view",
    "thresholds_channel",
    "thresholds_rate",
    "thresholds_structural",
    "werner_concurrence",
    "werner_fidelity",
    "werner_purity",
    "why_wings_agree_is_absent",
    "wilson_interval",
]
