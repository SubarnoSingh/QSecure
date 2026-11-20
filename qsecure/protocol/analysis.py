"""Closed-form probabilities for the teleportation-based QDS scheme.

Analytic only: forgery, repudiation and honest-abort probabilities and bounds,
as finite sums or elementary expressions in ``(L, |B|, s_a, s_v, p_e)``.
Positions reserved for check rounds are sifted out before any count is taken.

Assumption (IND): the declaration is independent of the recipients' logged
bases. Every bound averaged over a binomial matched count needs it; the per-run
:func:`repudiation_bound` and :func:`repudiation_bound_with_abort` do not.

Examples
--------
The shipped matched-count floors:

>>> import math
>>> from qsecure.protocol.analysis import (
...     _log_binomial_pmf_table,
...     _log_factorial_table,
...     _logsumexp,
...     matched_shortfall_probability,
... )
>>> from qsecure.protocol.params import DEFAULT_PARAMS
>>> from qsecure.protocol.verify import (
...     enforced_repudiation_bound,
...     minimum_matched_count,
...     minimum_pooled_matched_count,
... )
>>> m_min = minimum_matched_count(DEFAULT_PARAMS)
>>> M_min = minimum_pooled_matched_count(DEFAULT_PARAMS)
>>> m_min, M_min, M_min - 2 * m_min
(36555, 74190, 1080)

Pricing the coin split a pooled floor alone would leave open:

>>> f"{math.exp(-2 * (M_min / 2 - m_min) ** 2 / M_min):.4e}"
'3.8553e-04'
>>> epsilon = 2.0 ** -64
>>> f"{epsilon ** (3 - 2 * math.sqrt(2)):.4e}"
'4.9487e-04'
>>> log_factorial = _log_factorial_table(M_min)
>>> table = _log_binomial_pmf_table(M_min, 0.5, log_factorial)
>>> exact = math.exp(_logsumexp(table[:m_min]))
>>> f"{exact:.4e}", f"{2 * exact:.4e}"
('3.6116e-05', '7.2233e-05')

The enforced repudiation bound, and the weaker per-verifier figures:

>>> f"{enforced_repudiation_bound(DEFAULT_PARAMS):.4e}"
'1.4139e-09'
>>> f"{math.exp(-max(2 * m_min, M_min) * DEFAULT_PARAMS.gap ** 2 / 8):.4e}"
'1.4139e-09'
>>> f"{math.exp(-2 * m_min * DEFAULT_PARAMS.gap ** 2 / 8):.4e}"
'1.9022e-09'
>>> f"{math.exp(-m_min * DEFAULT_PARAMS.gap ** 2 / 8):.4e}"
'4.3614e-05'

The honest cost of the three floors against ``eps = 2 ** -64``:

>>> own = matched_shortfall_probability(DEFAULT_PARAMS, minimum_matched=m_min)
>>> pooled = matched_shortfall_probability(
...     DEFAULT_PARAMS, minimum_matched=M_min, pooled=True)
>>> f"{own:.1e}", f"{pooled:.1e}", f"{2 * own + pooled:.1e}"
('2.5e-31', '2.9e-31', '8.0e-31')
>>> f"{epsilon:.1e}"
'5.4e-20'

The rate-rule abort term against the floors' shortfall term:

>>> from qsecure.protocol.analysis import (
...     honest_abort_probability, matched_shortfall_probability)
>>> from qsecure.protocol.params import ProtocolParams
>>> from qsecure.protocol.verify import (
...     minimum_matched_count, minimum_pooled_matched_count)
>>> for length in (600, 1200, 4800):
...     small = ProtocolParams(key_length=length)
...     own = matched_shortfall_probability(
...         small, minimum_matched=minimum_matched_count(small))
...     pooled = matched_shortfall_probability(
...         small, minimum_matched=minimum_pooled_matched_count(small),
...         pooled=True)
...     print(length,
...           f"{honest_abort_probability(small):.3e}",
...           f"{2 * own + pooled:.3e}")
600 4.429e-106 1.740e-34
1200 9.807e-212 3.691e-33
4800 0.000e+00 9.972e-32
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

from qsecure.protocol.params import Party, ProtocolParams, _as_party

__all__ = [
    "FORGER_MATCHED_MISMATCH_PROBABILITY",
    "BoundMethod",
    "MatchedStatistics",
    "HonestStatistics",
    "binary_kl_divergence",
    "hoeffding_exponent",
    "max_accepted_mismatches",
    "matched_statistics",
    "matched_count_distribution",
    "matched_shortfall_probability",
    "depolarising_error_rate",
    "honest_statistics",
    "forgery_probability",
    "forgery_bound",
    "recipient_forgery_probability",
    "recipient_forgery_bound",
    "repudiation_probability",
    "repudiation_bound",
    "averaged_repudiation_bound",
    "repudiation_bound_with_abort",
    "symmetric_repudiation_bound",
    "honest_abort_probability",
    "honest_abort_bound",
]

BoundMethod = Literal["hoeffding", "kl"]
"""Which tail inequality a ``*_bound`` function uses: ``"hoeffding"`` or ``"kl"``."""

FORGER_MATCHED_MISMATCH_PROBABILITY: Final[float] = 0.5
"""Per-matched-position mismatch probability of an external forger, ``1/2``."""

_TERNARY_ITERATIONS: Final[int] = 400
"""Ternary-search steps for the KL repudiation exponent (converged well past 1e-16)."""


def _as_params(params: Any, *, name: str = "params") -> ProtocolParams:
    """Validate ``params`` and reduce it to the positions that carry key.

    Returns ``params.sifted()`` when check rounds are reserved, so that diverted
    positions are never counted as evidence.

    Examples
    --------
    >>> from qsecure.protocol.params import DEFAULT_PARAMS
    >>> _as_params(DEFAULT_PARAMS) is DEFAULT_PARAMS
    True
    >>> checked = DEFAULT_PARAMS.with_check_fraction(0.10)
    >>> checked.key_length, _as_params(checked).key_length
    (115200, 103680)
    """
    if not isinstance(params, ProtocolParams):
        raise TypeError(
            f"{name} must be a ProtocolParams, got {type(params).__name__}. "
            f"Build one with ProtocolParams(key_length=...) or "
            f"ProtocolParams.from_dict(...); the closed forms in this module are "
            f"only valid for a parameter set that passed its s_a < s_v < 1/2 "
            f"validation."
        )
    return params.sifted() if params.check_fraction else params


def _as_probability(value: Any, name: str) -> float:
    """Coerce and range-check a probability in ``[0, 1]``."""
    if isinstance(value, bool):
        raise TypeError(
            f"{name} must be a real number in [0, 1], got the boolean {value!r}"
        )
    if not isinstance(value, numbers.Real):
        raise TypeError(
            f"{name} must be a real number in [0, 1], got {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            f"{name} must be a finite probability in [0, 1], got {result!r}"
        )
    if not 0.0 <= result <= 1.0:
        raise ValueError(
            f"{name} must lie in [0, 1], got {result!r}. It is a probability, "
            f"not a count or a rate difference."
        )
    return result


def _as_matched(matched: Any, *, name: str = "matched") -> int:
    """Coerce and check a positive matched-set size."""
    if isinstance(matched, bool):
        raise TypeError(
            f"{name} must be a positive integer, got the boolean {matched!r}"
        )
    if not isinstance(matched, numbers.Integral):
        raise TypeError(
            f"{name} must be a positive integer, got {type(matched).__name__}; "
            f"it counts positions."
        )
    value = int(matched)
    if value < 1:
        raise ValueError(
            f"{name} must be at least 1, got {value}. Conditioning on an empty "
            f"matched set states a bound on a decision that is never taken -- "
            f"verify() raises there. Pass matched=None to average over the "
            f"Binomial(L, 1/|B|) distribution of the matched-set size instead, "
            f"which handles m = 0 correctly."
        )
    return value


def _as_matched_records(matched_records: Any) -> int:
    """Coerce the observed total matched-record count ``M = m_B + m_C``."""
    if matched_records is None:
        raise ValueError(
            "repudiation_bound requires the observed matched-record count "
            "M = m_B + m_C; there is no averaged default. Averaging over "
            "M ~ Binomial(2L, 1/|B|) assumes the declared bases are "
            "independent of the recipients' logged bases (assumption (IND) in "
            "this module's docstring), which a signer reading both raw logs "
            "violates -- that adversary pins M at 13 for every L and repudiates "
            "with probability 1/2, against an averaged bound of 6.9e-10 at the "
            "shipped defaults. Pass matched_records=bob.matched_count + "
            "charlie.matched_count from the run's two VerificationResults for "
            "the per-run guarantee, "
            "repudiation_bound_with_abort(...) for an a-priori one, or "
            "averaged_repudiation_bound(params, signer_sees_recipient_bases="
            "False) if that independence is genuinely enforced by your deployment."
        )
    return _as_matched(matched_records, name="matched_records")


def _as_method(method: Any) -> BoundMethod:
    """Validate the ``method`` selector of a bound function."""
    if not isinstance(method, str):
        raise TypeError(
            f"method must be the string 'hoeffding' or 'kl', got "
            f"{type(method).__name__}"
        )
    cleaned = method.strip().lower()
    if cleaned not in ("hoeffding", "kl"):
        raise ValueError(
            f"method must be 'hoeffding' (exponent 2(a-b)^2, the form quoted in "
            f"qsecure.protocol.params) or 'kl' (exponent D(a||b), never weaker "
            f"and much tighter near 0 or 1), got {method!r}"
        )
    return cleaned  # type: ignore[return-value]


def _as_verifier(party: Party | str) -> Party:
    """Coerce a party argument and refuse Alice."""
    resolved = _as_party(party)
    if resolved is Party.ALICE:
        raise ValueError(
            "Alice is not a verifier: she signs, keeps no measurement record and "
            "has no acceptance threshold, so no acceptance or abort probability "
            "is defined for her. Ask for Party.BOB (threshold s_a) or "
            "Party.CHARLIE (threshold s_v)."
        )
    return resolved


def binary_kl_divergence(a: float, b: float) -> float:
    """Binary relative entropy ``D(a || b)`` in nats.

    Returns ``inf`` when ``b`` is ``0`` or ``1`` and ``a`` differs from it.

    Examples
    --------
    >>> from qsecure.protocol.analysis import binary_kl_divergence
    >>> round(binary_kl_divergence(1 / 6, 1 / 2), 6)
    0.242586
    >>> binary_kl_divergence(0.25, 0.25)
    0.0
    """
    rate = _as_probability(a, "a")
    reference = _as_probability(b, "b")
    if rate == reference:
        return 0.0
    if reference == 0.0 or reference == 1.0:
        return math.inf
    total = 0.0
    if rate > 0.0:
        total += rate * math.log(rate / reference)
    if rate < 1.0:
        total += (1.0 - rate) * math.log((1.0 - rate) / (1.0 - reference))
    return max(total, 0.0)


def hoeffding_exponent(a: float, b: float) -> float:
    """Hoeffding's per-sample exponent ``2 (a - b)**2``, in nats.

    Examples
    --------
    >>> from qsecure.protocol.analysis import hoeffding_exponent
    >>> round(hoeffding_exponent(1 / 6, 1 / 2), 6)
    0.222222
    """
    rate = _as_probability(a, "a")
    reference = _as_probability(b, "b")
    return 2.0 * (rate - reference) ** 2


def _exponent(a: float, b: float, method: BoundMethod) -> float:
    """Return the per-matched-position exponent selected by ``method``."""
    if method == "kl":
        return binary_kl_divergence(a, b)
    return hoeffding_exponent(a, b)


def max_accepted_mismatches(matched: int, threshold: float) -> int:
    """Largest mismatch count ``t(m)`` that still passes ``e / m <= threshold``.

    Evaluated with the same floating-point comparison the verifier uses, not
    exact rational arithmetic (which disagrees at e.g. ``m = 6``, ``s = 1/6``).

    Examples
    --------
    >>> from qsecure.protocol.analysis import max_accepted_mismatches
    >>> max_accepted_mismatches(6, 1 / 6)
    1
    >>> max_accepted_mismatches(60, 1 / 6)
    10
    >>> max_accepted_mismatches(5, 0.0)
    0
    """
    count = _as_matched(matched)
    cut = _as_probability(threshold, "threshold")
    limit = int(cut * count)
    limit = min(max(limit, 0), count)
    while limit < count and (limit + 1) / count <= cut:
        limit += 1
    while limit > 0 and limit / count > cut:
        limit -= 1
    return limit


def depolarising_error_rate(depolarising_parameter: float) -> float:
    """Matched-position error rate ``p / 2`` induced by a depolarising channel.

    Examples
    --------
    >>> from qsecure.protocol.analysis import depolarising_error_rate
    >>> depolarising_error_rate(0.0625)
    0.03125
    """
    return _as_probability(depolarising_parameter, "depolarising_parameter") / 2.0


@dataclass(frozen=True)
class MatchedStatistics:
    """Exact moments of the matched-set size ``|M_R| ~ Binomial(L, 1/|B|)``.

    Examples
    --------
    >>> from qsecure.protocol.analysis import matched_statistics
    >>> from qsecure.protocol.params import ProtocolParams
    >>> stats = matched_statistics(ProtocolParams(key_length=9))
    >>> stats.expected, round(stats.variance, 6)
    (3.0, 2.0)
    """

    key_length: int
    match_probability: float
    expected: float
    variance: float
    empty_probability: float

    @property
    def standard_deviation(self) -> float:
        """float: ``sqrt(Var[|M_R|])``."""
        return math.sqrt(self.variance)

    @property
    def relative_standard_deviation(self) -> float:
        """float: ``sqrt(Var) / E``, the coefficient of variation of ``|M_R|``."""
        return math.sqrt(self.variance) / self.expected


def matched_statistics(params: ProtocolParams) -> MatchedStatistics:
    """Return the exact distributional summary of the matched-set size (under IND).

    Examples
    --------
    >>> from qsecure.protocol.analysis import matched_statistics
    >>> from qsecure.protocol.params import DEFAULT_PARAMS
    >>> matched_statistics(DEFAULT_PARAMS).expected
    38400.0
    """
    checked = _as_params(params)
    probability = checked.match_probability
    length = checked.key_length
    return MatchedStatistics(
        key_length=length,
        match_probability=probability,
        expected=length * probability,
        variance=length * probability * (1.0 - probability),
        empty_probability=math.exp(length * math.log1p(-probability)),
    )


def matched_count_distribution(params: ProtocolParams) -> np.ndarray:
    """Return the pmf of ``|M_R|`` as an array of length ``L + 1`` (under IND).

    Examples
    --------
    >>> import numpy as np
    >>> from qsecure.protocol.analysis import matched_count_distribution
    >>> from qsecure.protocol.params import ProtocolParams
    >>> pmf = matched_count_distribution(ProtocolParams(key_length=3))
    >>> np.round(pmf, 6)
    array([0.296296, 0.444444, 0.222222, 0.037037])
    """
    checked = _as_params(params)
    log_factorial = _log_factorial_table(checked.key_length)
    return np.exp(
        _log_binomial_pmf_table(
            checked.key_length, checked.match_probability, log_factorial
        )
    )


@dataclass(frozen=True)
class HonestStatistics:
    """Exact moments of an honest verifier's mismatch rate ``r_R``.

    Examples
    --------
    >>> from qsecure.protocol.analysis import honest_statistics
    >>> from qsecure.protocol.params import DEMO_PARAMS
    >>> honest_statistics(DEMO_PARAMS).expected_rate
    0.0
    """

    key_length: int
    error_rate: float
    expected_rate: float
    rate_variance: float
    expected_reciprocal_matched: float
    empty_probability: float

    @property
    def rate_standard_deviation(self) -> float:
        """float: ``sqrt(Var[r_R])``, the run-to-run spread of the honest rate."""
        return math.sqrt(self.rate_variance)


def honest_statistics(
    params: ProtocolParams, *, error_rate: float = 0.0
) -> HonestStatistics:
    """Return the exact mean and variance of the honest mismatch rate.

    ``E[r_R] = p_e`` and ``Var[r_R] = p_e (1 - p_e) E[1/m]`` over
    ``m ~ Binomial(L, 1/|B|)`` conditioned on ``m >= 1``.

    Examples
    --------
    >>> from qsecure.protocol.analysis import honest_statistics
    >>> from qsecure.protocol.params import DEMO_PARAMS
    >>> stats = honest_statistics(DEMO_PARAMS, error_rate=0.02)
    >>> stats.expected_rate
    0.02
    >>> round(stats.rate_standard_deviation, 5)
    0.01759
    """
    checked = _as_params(params)
    rate = _as_probability(error_rate, "error_rate")

    pmf = matched_count_distribution(checked)
    empty = float(pmf[0])
    survivors = pmf[1:]
    counts = np.arange(1, checked.key_length + 1, dtype=float)
    non_empty_mass = float(survivors.sum())
    reciprocal = float((survivors / counts).sum()) / non_empty_mass

    return HonestStatistics(
        key_length=checked.key_length,
        error_rate=rate,
        expected_rate=rate,
        rate_variance=rate * (1.0 - rate) * reciprocal,
        expected_reciprocal_matched=reciprocal,
        empty_probability=empty,
    )


def _log_factorial_table(n: int) -> np.ndarray:
    """Return ``[ln 0!, ln 1!, ..., ln n!]`` as a float array."""
    return np.array([math.lgamma(k + 1.0) for k in range(n + 1)], dtype=float)


def _log_binomial_pmf_table(
    trials: int, probability: float, log_factorial: np.ndarray
) -> np.ndarray:
    """Return ``[ln P(X = k)]`` for ``X ~ Binomial(trials, probability)``."""
    counts = np.arange(trials + 1)
    if probability == 0.0:
        table = np.full(trials + 1, -np.inf)
        table[0] = 0.0
        return table
    if probability == 1.0:
        table = np.full(trials + 1, -np.inf)
        table[trials] = 0.0
        return table
    return (
        log_factorial[trials]
        - log_factorial[counts]
        - log_factorial[trials - counts]
        + counts * math.log(probability)
        + (trials - counts) * math.log1p(-probability)
    )


def _logsumexp(values: np.ndarray) -> float:
    """Return ``ln sum exp(values)``, ignoring ``-inf`` entries safely."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -math.inf
    largest = float(finite.max())
    return largest + float(np.log(np.exp(finite - largest).sum()))


def _log_conditional_tail(
    matched: int,
    mismatch_probability: float,
    limit: int,
    *,
    upper: bool,
    log_factorial: np.ndarray,
) -> float:
    """Log of ``P(Bin(matched, q) > limit)`` (``upper``) or ``P(... <= limit)``."""
    if upper:
        low, high = limit + 1, matched
    else:
        low, high = 0, limit
    if low > high:
        return -math.inf
    if mismatch_probability == 0.0:
        return 0.0 if low == 0 else -math.inf
    if mismatch_probability == 1.0:
        return 0.0 if high == matched else -math.inf

    counts = np.arange(low, high + 1)
    log_terms = (
        log_factorial[matched]
        - log_factorial[counts]
        - log_factorial[matched - counts]
        + counts * math.log(mismatch_probability)
        + (matched - counts) * math.log1p(-mismatch_probability)
    )
    return _logsumexp(log_terms)


def _log_decision_probability(
    key_length: int,
    match_probability: float,
    mismatch_probability: float,
    threshold: float,
    *,
    accept: bool,
) -> float:
    """Log-probability that a verifier accepts (or does not accept) a declaration.

    Averages the conditional tail over ``m ~ Binomial(L, p_match)``; ``m = 0``
    counts as not accepted.
    """
    log_factorial = _log_factorial_table(key_length)
    log_pmf = _log_binomial_pmf_table(key_length, match_probability, log_factorial)

    terms = np.full(key_length + 1, -np.inf)
    if not accept:
        terms[0] = log_pmf[0]
    for matched in range(1, key_length + 1):
        weight = float(log_pmf[matched])
        if not math.isfinite(weight):
            continue
        limit = max_accepted_mismatches(matched, threshold)
        terms[matched] = weight + _log_conditional_tail(
            matched,
            mismatch_probability,
            limit,
            upper=not accept,
            log_factorial=log_factorial,
        )
    return _logsumexp(terms)


def _averaged_exponential_bound(
    key_length: int, match_probability: float, exponent: float
) -> float:
    """Return ``E[exp(-m c)] = (1 - p + p e^-c)^L`` for ``m ~ Binomial(L, p)``."""
    if exponent <= 0.0:
        return 1.0
    factor = 1.0 - match_probability + match_probability * math.exp(-exponent)
    if factor <= 0.0:
        return 0.0
    if factor >= 1.0:
        return 1.0
    return min(1.0, math.exp(key_length * math.log(factor)))


def _conditional_or_averaged(
    key_length: int,
    match_probability: float,
    exponent: float,
    matched: int | None,
) -> float:
    """Return ``exp(-matched * exponent)`` or its binomial average over ``m``."""
    if matched is None:
        return _averaged_exponential_bound(key_length, match_probability, exponent)
    if exponent <= 0.0:
        return 1.0
    if math.isinf(exponent):
        return 0.0
    return min(1.0, math.exp(-matched * exponent))


def forgery_probability(
    params: ProtocolParams, *, party: Party | str = Party.CHARLIE
) -> float:
    """Exact probability that an external forger's declaration is accepted.

    Eve holds no key information, so she mismatches each matched position with
    probability ``1/2``; assumes (IND). This is the weakest adversary; quote
    :func:`recipient_forgery_probability` for security claims. Cost is ``O(L^2)``.

    Examples
    --------
    >>> from qsecure.protocol.analysis import forgery_probability
    >>> from qsecure.protocol.params import ProtocolParams
    >>> round(forgery_probability(ProtocolParams(key_length=30)), 6)
    0.004211
    """
    checked = _as_params(params)
    threshold = checked.threshold_for(_as_verifier(party))
    return math.exp(
        _log_decision_probability(
            checked.key_length,
            checked.match_probability,
            FORGER_MATCHED_MISMATCH_PROBABILITY,
            threshold,
            accept=True,
        )
    )


def forgery_bound(
    params: ProtocolParams,
    *,
    party: Party | str = Party.CHARLIE,
    method: BoundMethod = "hoeffding",
    matched: int | None = None,
) -> float:
    """Closed-form upper bound on :func:`forgery_probability`.

    Parameters
    ----------
    matched : int or None, optional
        Condition on this many matched positions (no assumption needed), or
        average over the binomial matched count (requires IND).

    Examples
    --------
    >>> from qsecure.protocol.analysis import forgery_bound, forgery_probability
    >>> from qsecure.protocol.params import ProtocolParams
    >>> params = ProtocolParams(key_length=30)
    >>> forgery_probability(params) <= forgery_bound(params, method="kl")
    True
    >>> forgery_bound(params, method="kl") <= forgery_bound(params)
    True
    """
    checked = _as_params(params)
    selected = _as_method(method)
    threshold = checked.threshold_for(_as_verifier(party))
    limit = None if matched is None else _as_matched(matched)
    if threshold >= FORGER_MATCHED_MISMATCH_PROBABILITY:
        return 1.0
    exponent = _exponent(threshold, FORGER_MATCHED_MISMATCH_PROBABILITY, selected)
    return _conditional_or_averaged(
        checked.key_length, checked.match_probability, exponent, limit
    )


def _recipient_rates(
    params: ProtocolParams, *, symmetrised: bool
) -> tuple[float, float]:
    """Return ``(scored fraction, mismatch rate given scored)`` for a forger."""
    if symmetrised:
        return params.forger_scored_fraction, params.forger_floor
    return params.match_probability, params.unmatched_noise_rate


def recipient_forgery_probability(
    params: ProtocolParams,
    *,
    party: Party | str = Party.CHARLIE,
    symmetrised: bool = True,
) -> float:
    """Exact probability that a *recipient's* forgery is accepted (the binding case).

    The forger is scored on ``p_s = (n + 1)/(2n)`` of positions and mismatches at
    ``rho = (n - 1)/(2n(n + 1))`` among them; ``symmetrised=False`` models the
    protocol without the exchange.

    Examples
    --------
    >>> from qsecure.protocol.analysis import (
    ...     forgery_probability, recipient_forgery_probability
    ... )
    >>> from qsecure.protocol.params import ProtocolParams
    >>> params = ProtocolParams(key_length=30)
    >>> round(recipient_forgery_probability(params), 6)
    0.481427
    >>> recipient_forgery_probability(params) > forgery_probability(params)
    True
    """
    checked = _as_params(params)
    if not isinstance(symmetrised, bool):
        raise TypeError(
            f"symmetrised must be a bool, got {type(symmetrised).__name__}. It "
            f"selects which protocol variant the forger is attacking, not a "
            f"rate."
        )
    threshold = checked.threshold_for(_as_verifier(party))
    scored, rate = _recipient_rates(checked, symmetrised=symmetrised)
    return math.exp(
        _log_decision_probability(
            checked.key_length, scored, rate, threshold, accept=True
        )
    )


def recipient_forgery_bound(
    params: ProtocolParams,
    *,
    party: Party | str = Party.CHARLIE,
    method: BoundMethod = "hoeffding",
    symmetrised: bool = True,
    matched: int | None = None,
) -> float:
    """Closed-form upper bound on :func:`recipient_forgery_probability`.

    Returns ``1.0`` when the threshold is at or above ``rho``.

    Examples
    --------
    >>> from qsecure.protocol.analysis import (
    ...     recipient_forgery_bound, recipient_forgery_probability
    ... )
    >>> from qsecure.protocol.params import DEFAULT_PARAMS, ProtocolParams
    >>> params = ProtocolParams(key_length=300)
    >>> recipient_forgery_probability(params) <= recipient_forgery_bound(
    ...     params, method="kl"
    ... )
    True
    >>> f"{recipient_forgery_bound(DEFAULT_PARAMS, method='kl'):.2e}"
    '1.12e-103'
    """
    checked = _as_params(params)
    selected = _as_method(method)
    if not isinstance(symmetrised, bool):
        raise TypeError(
            f"symmetrised must be a bool, got {type(symmetrised).__name__}"
        )
    threshold = checked.threshold_for(_as_verifier(party))
    limit = None if matched is None else _as_matched(matched)
    scored, rate = _recipient_rates(checked, symmetrised=symmetrised)
    if threshold >= rate:
        return 1.0
    exponent = _exponent(threshold, rate, selected)
    return _conditional_or_averaged(
        checked.key_length, scored, exponent, limit
    )


def repudiation_probability(
    params: ProtocolParams, *, mismatch_probability: float
) -> float:
    """Exact repudiation probability for one symmetric Alice strategy ``q``.

    ``P(r_B <= s_a) * P(r_C > s_v)`` when both copies have mismatch probability
    ``q``. Not a security number: Alice may treat the two recipients differently;
    see :func:`repudiation_bound`.

    Examples
    --------
    >>> from qsecure.protocol.analysis import repudiation_probability
    >>> from qsecure.protocol.params import ProtocolParams
    >>> params = ProtocolParams(key_length=30)
    >>> f"{repudiation_probability(params, mismatch_probability=0.0):.3e}"
    '5.215e-06'
    >>> f"{repudiation_probability(params, mismatch_probability=0.2):.3e}"
    '1.100e-01'
    """
    checked = _as_params(params)
    rate = _as_probability(mismatch_probability, "mismatch_probability")
    log_bob_accepts = _log_decision_probability(
        checked.key_length,
        checked.match_probability,
        rate,
        checked.s_a,
        accept=True,
    )
    log_charlie_rejects = _log_decision_probability(
        checked.key_length,
        checked.match_probability,
        rate,
        checked.s_v,
        accept=False,
    )
    total = log_bob_accepts + log_charlie_rejects
    return 0.0 if total == -math.inf else math.exp(total)


def _kl_repudiation_exponent(s_a: float, s_v: float) -> float:
    """Return ``min_{q in [s_a, s_v]} D(s_a || q) + D(s_v || q)`` by ternary search."""

    def objective(q: float) -> float:
        return binary_kl_divergence(s_a, q) + binary_kl_divergence(s_v, q)

    low, high = s_a, s_v
    for _ in range(_TERNARY_ITERATIONS):
        span = (high - low) / 3.0
        left, right = low + span, high - span
        if objective(left) < objective(right):
            high = right
        else:
            low = left
    return objective(0.5 * (low + high))


def repudiation_bound(params: ProtocolParams, *, matched_records: int) -> float:
    """Per-run repudiation bound ``exp(-M gap^2 / 8)``, valid for every Alice strategy.

    Conditions on the records and the declaration; the only randomness used is
    the recipients' symmetrisation coins. Does not apply without symmetrisation.

    Parameters
    ----------
    matched_records : int
        Observed ``M = m_B + m_C`` from the run, both against the same
        declaration. An expected count would reintroduce (IND).

    Examples
    --------
    >>> from qsecure.protocol.analysis import repudiation_bound
    >>> from qsecure.protocol.params import DEFAULT_PARAMS
    >>> f"{repudiation_bound(DEFAULT_PARAMS, matched_records=76800):.2e}"
    '6.90e-10'
    >>> round(repudiation_bound(DEFAULT_PARAMS, matched_records=13), 4)
    0.9964
    """
    checked = _as_params(params)
    limit = _as_matched_records(matched_records)
    exponent = checked.gap**2 / 8.0
    return _conditional_or_averaged(
        2 * checked.key_length, checked.match_probability, exponent, limit
    )


def averaged_repudiation_bound(
    params: ProtocolParams, *, signer_sees_recipient_bases: bool
) -> float:
    """Repudiation bound averaged over ``M ~ Binomial(2L, 1/n)``; requires (IND).

    A signer who reads both raw logs can pin ``M = 13`` and repudiate with
    probability ``1/2``, so pass ``signer_sees_recipient_bases=False`` only when
    (IND) is enforced; ``True`` raises.

    Examples
    --------
    >>> from qsecure.protocol.analysis import averaged_repudiation_bound
    >>> from qsecure.protocol.params import DEFAULT_PARAMS
    >>> assuming_ind = averaged_repudiation_bound(
    ...     DEFAULT_PARAMS, signer_sees_recipient_bases=False
    ... )
    >>> f"{assuming_ind:.2e}"
    '6.92e-10'
    >>> averaged_repudiation_bound(
    ...     DEFAULT_PARAMS, signer_sees_recipient_bases=True
    ... )
    Traceback (most recent call last):
        ...
    ValueError: no averaged repudiation bound is valid when the signer can see...
    """
    checked = _as_params(params)
    if not isinstance(signer_sees_recipient_bases, bool):
        raise TypeError(
            f"signer_sees_recipient_bases must be a bool, got "
            f"{type(signer_sees_recipient_bases).__name__}. It asserts a fact "
            f"about the deployment -- whether anyone who can choose the "
            f"declaration can also read the recipients' logged bases -- not a "
            f"rate or a count."
        )
    if signer_sees_recipient_bases:
        raise ValueError(
            "no averaged repudiation bound is valid when the signer can see the "
            "recipients' logged bases: the matched set is {i : c_i == d_i}, so "
            "she chooses M rather than drawing it from Binomial(2L, 1/|B|), and "
            "the optimal signer strategy pins M = 13 at every L and repudiates "
            "with probability 1/2. Use repudiation_bound(params, "
            "matched_records=<observed m_B + m_C>) for the per-run guarantee, or "
            "verify.enforced_repudiation_bound(params) for the a-priori one the "
            "shipped floors earn. repudiation_bound_with_abort(params, "
            "minimum_matched_records=...) is the same expression at a floor you "
            "name, and must be read with the event it covers (sections 4b-iii "
            "and 4b-iv) rather than as a blanket number."
        )
    exponent = checked.gap**2 / 8.0
    coin_term = _averaged_exponential_bound(
        2 * checked.key_length, checked.match_probability, exponent
    )
    # Legacy m_C = 0 allowance: slack, but it is the published expression.
    empty = (1.0 - checked.match_probability) ** checked.key_length
    return min(1.0, coin_term + empty)


def repudiation_bound_with_abort(
    params: ProtocolParams, *, minimum_matched_records: int
) -> float:
    """A-priori repudiation bound ``exp(-m_min gap^2 / 8)`` under an abort rule.

    Valid for every Alice strategy when acceptance requires ``M >= m_min``. A
    per-verifier floor at Charlie also needs a joint consequence to cover the
    "no verdict" outcome; :func:`~qsecure.protocol.verify.enforced_repudiation_bound`
    evaluates this at the shipped floors.

    Examples
    --------
    >>> from qsecure.protocol.analysis import (
    ...     matched_shortfall_probability, repudiation_bound_with_abort
    ... )
    >>> from qsecure.protocol.params import DEFAULT_PARAMS
    >>> from qsecure.protocol.verify import (
    ...     minimum_matched_count, minimum_pooled_matched_count
    ... )
    >>> floor = minimum_matched_count(DEFAULT_PARAMS)
    >>> pooled_floor = minimum_pooled_matched_count(DEFAULT_PARAMS)
    >>> floor, pooled_floor
    (36555, 74190)
    >>> local = repudiation_bound_with_abort(
    ...     DEFAULT_PARAMS, minimum_matched_records=floor
    ... )
    >>> both = repudiation_bound_with_abort(
    ...     DEFAULT_PARAMS, minimum_matched_records=2 * floor
    ... )
    >>> pooled = repudiation_bound_with_abort(
    ...     DEFAULT_PARAMS, minimum_matched_records=pooled_floor
    ... )
    >>> f"{local:.1e}", f"{both:.1e}", f"{pooled:.1e}"
    ('4.4e-05', '1.9e-09', '1.4e-09')
    >>> cost = matched_shortfall_probability(
    ...     DEFAULT_PARAMS, minimum_matched=floor
    ... )
    >>> pooled_cost = matched_shortfall_probability(
    ...     DEFAULT_PARAMS, minimum_matched=pooled_floor, pooled=True
    ... )
    >>> f"{cost:.1e}", f"{pooled_cost:.1e}"
    ('2.5e-31', '2.9e-31')
    """
    checked = _as_params(params)
    floor = _as_matched(minimum_matched_records, name="minimum_matched_records")
    exponent = checked.gap**2 / 8.0
    return _conditional_or_averaged(
        2 * checked.key_length, checked.match_probability, exponent, floor
    )


def matched_shortfall_probability(
    params: ProtocolParams, *, minimum_matched: int, pooled: bool = False
) -> float:
    """Probability that an honest run's matched count falls below ``minimum_matched``.

    ``P(Bin(trials, 1/n) < minimum_matched)`` with ``trials = L``, or ``2L`` when
    ``pooled``. Assumes (IND).

    Examples
    --------
    >>> from qsecure.protocol.analysis import (
    ...     matched_shortfall_probability, matched_statistics
    ... )
    >>> from qsecure.protocol.params import DEFAULT_PARAMS, ProtocolParams
    >>> params = ProtocolParams(key_length=30)
    >>> matched_shortfall_probability(params, minimum_matched=1) == (
    ...     matched_statistics(params).empty_probability
    ... )
    True
    >>> cost = matched_shortfall_probability(
    ...     DEFAULT_PARAMS, minimum_matched=73110, pooled=True
    ... )
    >>> f"{cost:.1e}"
    '1.4e-60'

    The three shipped floors and what they cost an honest run:

    >>> from qsecure.protocol.verify import (
    ...     minimum_matched_count, minimum_pooled_matched_count)
    >>> m_min = minimum_matched_count(DEFAULT_PARAMS)
    >>> M_min = minimum_pooled_matched_count(DEFAULT_PARAMS)
    >>> m_min, M_min, M_min - 2 * m_min
    (36555, 74190, 1080)
    >>> own = matched_shortfall_probability(
    ...     DEFAULT_PARAMS, minimum_matched=m_min)
    >>> pooled = matched_shortfall_probability(
    ...     DEFAULT_PARAMS, minimum_matched=M_min, pooled=True)
    >>> f"{own:.1e}", f"{pooled:.1e}", f"{2 * own + pooled:.1e}"
    ('2.5e-31', '2.9e-31', '8.0e-31')
    """
    checked = _as_params(params)
    threshold = _as_matched(minimum_matched, name="minimum_matched")
    if not isinstance(pooled, bool):
        raise TypeError(
            f"pooled must be a bool, got {type(pooled).__name__}. It selects "
            f"which count the abort rule reads -- one verifier's own "
            f"Binomial(L, 1/|B|) or the pair's Binomial(2L, 1/|B|) -- not a "
            f"count itself."
        )
    trials = 2 * checked.key_length if pooled else checked.key_length
    if threshold > trials:
        return 1.0
    log_factorial = _log_factorial_table(trials)
    log_pmf = _log_binomial_pmf_table(
        trials, checked.match_probability, log_factorial
    )
    total = _logsumexp(log_pmf[:threshold])
    return 0.0 if total == -math.inf else min(1.0, math.exp(total))


def symmetric_repudiation_bound(
    params: ProtocolParams,
    *,
    method: BoundMethod = "hoeffding",
    matched: int | None = None,
) -> float:
    """Upper bound on :func:`repudiation_probability`, uniform in ``q`` only.

    Not a security guarantee; :func:`repudiation_bound` is.

    Examples
    --------
    >>> from qsecure.protocol.analysis import (
    ...     repudiation_bound, symmetric_repudiation_bound
    ... )
    >>> from qsecure.protocol.params import DEFAULT_PARAMS
    >>> f"{symmetric_repudiation_bound(DEFAULT_PARAMS, matched=38400):.2e}"
    '4.77e-19'

    The in-model number is far below the guarantee at the same evidence:

    >>> symmetric_repudiation_bound(
    ...     DEFAULT_PARAMS, matched=38400
    ... ) < repudiation_bound(DEFAULT_PARAMS, matched_records=76800)
    True
    """
    checked = _as_params(params)
    selected = _as_method(method)
    limit = None if matched is None else _as_matched(matched)
    if selected == "kl":
        exponent = _kl_repudiation_exponent(checked.s_a, checked.s_v)
    else:
        exponent = checked.gap**2 / 2.0
    return _conditional_or_averaged(
        checked.key_length, checked.match_probability, exponent, limit
    )


def honest_abort_probability(
    params: ProtocolParams,
    *,
    error_rate: float = 0.0,
    party: Party | str | None = None,
) -> float:
    """Exact probability that an honest signature fails the rate rule.

    ``P(m = 0) + sum_m P(m) P(Bin(m, p_e) > t(m))``; ``party=None`` gives the
    probability that at least one verifier aborts. The matched-count floors are
    not modelled: add :func:`matched_shortfall_probability` at each floor for the
    shipped protocol's honest failure probability.

    Examples
    --------
    >>> from qsecure.protocol.analysis import honest_abort_probability
    >>> from qsecure.protocol.params import ProtocolParams
    >>> params = ProtocolParams(key_length=30)
    >>> honest_abort_probability(params, party="Bob") < 1e-4
    True

    The floors this function excludes dominate it at ``p_e = 0``:

    >>> from qsecure.protocol.analysis import matched_shortfall_probability
    >>> from qsecure.protocol.verify import (
    ...     minimum_matched_count, minimum_pooled_matched_count)
    >>> small = ProtocolParams(key_length=600)
    >>> f"{honest_abort_probability(small):.3e}"
    '4.429e-106'
    >>> own = matched_shortfall_probability(
    ...     small, minimum_matched=minimum_matched_count(small))
    >>> pooled = matched_shortfall_probability(
    ...     small, minimum_matched=minimum_pooled_matched_count(small),
    ...     pooled=True)
    >>> f"{2 * own + pooled:.3e}"
    '1.740e-34'
    """
    checked = _as_params(params)
    rate = _as_probability(error_rate, "error_rate")

    if party is None:
        bob = honest_abort_probability(
            checked, error_rate=rate, party=Party.BOB
        )
        charlie = honest_abort_probability(
            checked, error_rate=rate, party=Party.CHARLIE
        )
        return bob + charlie - bob * charlie

    threshold = checked.threshold_for(_as_verifier(party))
    log_probability = _log_decision_probability(
        checked.key_length,
        checked.match_probability,
        rate,
        threshold,
        accept=False,
    )
    return 0.0 if log_probability == -math.inf else math.exp(log_probability)


def honest_abort_bound(
    params: ProtocolParams,
    *,
    error_rate: float = 0.0,
    party: Party | str | None = None,
    method: BoundMethod = "hoeffding",
    matched: int | None = None,
) -> float:
    """Closed-form upper bound on :func:`honest_abort_probability`.

    Prefer ``method="kl"`` at low noise. ``party=None`` uses the union bound.
    Like :func:`honest_abort_probability`, the matched-count floors are excluded.

    Examples
    --------
    >>> from qsecure.protocol.analysis import (
    ...     honest_abort_bound, honest_abort_probability)
    >>> from qsecure.protocol.params import ProtocolParams
    >>> params = ProtocolParams(key_length=60)
    >>> exact = honest_abort_probability(params, error_rate=0.01, party="Bob")
    >>> exact <= honest_abort_bound(params, error_rate=0.01, party="Bob",
    ...                             method="kl")
    True

    It bounds the rate term only; the floors contribute far more:

    >>> from qsecure.protocol.analysis import matched_shortfall_probability
    >>> from qsecure.protocol.verify import (
    ...     minimum_matched_count, minimum_pooled_matched_count)
    >>> small = ProtocolParams(key_length=600)
    >>> f"{honest_abort_bound(small, method='kl'):.3e}"
    '4.429e-106'
    >>> own = matched_shortfall_probability(
    ...     small, minimum_matched=minimum_matched_count(small))
    >>> pooled = matched_shortfall_probability(
    ...     small, minimum_matched=minimum_pooled_matched_count(small),
    ...     pooled=True)
    >>> f"{2 * own + pooled:.3e}"
    '1.740e-34'

    The Hoeffding form is loose here (slack, not coverage):

    >>> honest_abort_bound(small)
    1.0
    """
    checked = _as_params(params)
    selected = _as_method(method)
    rate = _as_probability(error_rate, "error_rate")
    limit = None if matched is None else _as_matched(matched)

    if party is None:
        bob = honest_abort_bound(
            checked,
            error_rate=rate,
            party=Party.BOB,
            method=selected,
            matched=limit,
        )
        charlie = honest_abort_bound(
            checked,
            error_rate=rate,
            party=Party.CHARLIE,
            method=selected,
            matched=limit,
        )
        return min(1.0, bob + charlie)

    threshold = checked.threshold_for(_as_verifier(party))
    if rate >= threshold:
        return 1.0
    exponent = _exponent(threshold, rate, selected)
    return _conditional_or_averaged(
        checked.key_length, checked.match_probability, exponent, limit
    )
