"""Deterministic prototype business-cost policy for dispute recommendations.

Fight Score remains a model assessment signal. This module never treats it as
a win probability and never multiplies it by dispute value or operational cost.
"""

from __future__ import annotations

from math import isfinite
from numbers import Real


# Prototype merchant-policy assumptions for hackathon demonstration. Production
# values should be merchant-specific and calibrated using actual operational
# costs and dispute outcomes.
BASE_HANDLING_COST = 300.0
CRITICAL_EVIDENCE_PREP_COST = 100.0
SUPPORTING_EVIDENCE_PREP_COST = 50.0
MERCHANT_REVIEW_COST = 100.0

# Prototype merchant-policy assumptions. Production bands should be calibrated
# using real merchant economics.
BUSINESS_RISK_RATIO_BANDS = {
    "low_max": 0.10,
    "medium_max": 0.25,
}

# Existing policy thresholds, intentionally unchanged.
COST_SENSITIVITY_THRESHOLDS: dict[str, float] = {
    "low": 0.50,
    "medium": 0.65,
    "high": 0.80,
}

CONTEST_CONSIDERATION_RECOMMENDATIONS = frozenset(
    {"Fight", "Review / Collect Evidence"}
)


def _finite_non_negative(value: float, label: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{label} must be a finite, non-negative amount.")
    return float(value)


def estimate_contest_cost(
    evidence_required: list[str],
    critical_evidence: list[str],
) -> tuple[float, dict[str, float]]:
    """Estimate handling workload from the authoritative evidence configuration."""

    required = list(dict.fromkeys(evidence_required))
    critical = {category for category in critical_evidence if category in required}
    critical_cost = len(critical) * CRITICAL_EVIDENCE_PREP_COST
    supporting_cost = (len(required) - len(critical)) * SUPPORTING_EVIDENCE_PREP_COST
    breakdown = {
        "base_handling": BASE_HANDLING_COST,
        "critical_evidence": critical_cost,
        "supporting_evidence": supporting_cost,
        "merchant_review": MERCHANT_REVIEW_COST,
    }
    return sum(breakdown.values()), breakdown


def derive_business_risk_level(
    estimated_contest_cost: float,
    order_value: float | None,
) -> tuple[str, float | None]:
    """Map transparent cost exposure to a conservative policy level."""

    cost = _finite_non_negative(estimated_contest_cost, "Estimated contest cost")
    exposure = 0.0 if order_value is None else _finite_non_negative(order_value, "Order value")

    # Zero exposure cannot produce a meaningful ratio, so use the most
    # conservative policy rather than divide by zero or fabricate a ratio.
    if exposure <= 0:
        return "high", None

    ratio = cost / exposure
    if ratio < BUSINESS_RISK_RATIO_BANDS["low_max"]:
        return "low", ratio
    if ratio < BUSINESS_RISK_RATIO_BANDS["medium_max"]:
        return "medium", ratio
    return "high", ratio


def calculate_business_cost_context(
    order_value: float | None,
    evidence_required: list[str],
    critical_evidence: list[str],
) -> dict[str, object]:
    """Return workload cost and derived policy metadata without model output."""

    estimated_cost, breakdown = estimate_contest_cost(
        evidence_required, critical_evidence
    )
    level, ratio = derive_business_risk_level(estimated_cost, order_value)
    return {
        "false_positive_sensitivity": level,
        "false_positive_sensitivity_label": level.capitalize(),
        "business_risk_level": level,
        "decision_threshold": COST_SENSITIVITY_THRESHOLDS[level],
        "estimated_contest_cost": estimated_cost,
        "contest_cost_breakdown": breakdown,
        "cost_ratio": ratio,
        "estimated_false_positive_cost": estimated_cost,
    }


def evaluate_business_risk(
    fight_score: float,
    order_value: float | None,
    evidence_required: list[str],
    critical_evidence: list[str],
) -> dict[str, object]:
    """Derive policy level and threshold from contest workload and exposure."""

    if not isinstance(fight_score, Real) or not 0 <= float(fight_score) <= 1:
        raise ValueError("Fight Score must be between 0 and 1.")

    context = calculate_business_cost_context(
        order_value, evidence_required, critical_evidence
    )
    level = context["business_risk_level"]
    threshold = context["decision_threshold"]
    passes = float(fight_score) >= threshold

    if level == "high":
        policy_reason = (
            "The estimated contest cost is significant relative to the amount under dispute, "
            "so a stronger model signal is required."
        )
    else:
        policy_reason = (
            "The required Fight Score reflects estimated contest cost relative "
            "to the amount under dispute."
        )

    if passes:
        status = "Meets threshold"
        reason = f"{policy_reason} The model signal meets the required Fight Score."
    else:
        status = "Below threshold"
        reason = f"{policy_reason} The model signal does not meet the required Fight Score."

    return {
        **context,
        "fight_score": float(fight_score),
        "passes_cost_threshold": passes,
        "business_risk_status": status,
        "business_risk_reason": reason,
    }
