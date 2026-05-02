"""
Insurance Policy Rules Engine
Pure rule-based module — no ML, no dataset dependency.

Pipeline:
    ml_damage_estimate (from ML model — reference only)
        → Rule B: Denial check  (any flag → payout = 0, pipeline stops immediately)
        → Depreciation          (car_price × age → depreciated_car_value)
        → Step 0: Base payout   (depreciated_car_value × severity multiplier;
                                 theft / total-loss → 100%)
        → Rule A: Car price cap  (cap at % of depreciated_car_value)
        → Dashcam trigger        (no_dashcam=True forces overspeeding +
                                  distracted_driving + failure_to_mitigate on)
        → Rule C: Reductions     (multipliers + salvage deduction)
        → Final payout cap       (hard ceiling: final_payout_cap_pct % of original car_price)
        → final_payout

Usage:
    from src.policy_engine.insurance_rules import (
        PolicyConfig, DenialFlags, ReductionFactors, evaluate_claim
    )

    result = evaluate_claim(
        ml_damage_estimate=45000.0,
        car_price=30000.0,
        denial_flags=DenialFlags(lapse_in_coverage=True),
        reduction_factors=ReductionFactors(),
        incident_type="Single Vehicle Collision",
        incident_severity="Major Damage",
    )
    # result["denied"] → True
    # result["final_payout"] → 0.0
"""

import dataclasses
from dataclasses import dataclass
from datetime import date
from typing import Optional


# =============================================================================
# A — CONFIGURATION  (all thresholds in one place, fully overridable)
# =============================================================================

@dataclass
class PolicyConfig:
    """
    Configurable thresholds and penalty rates for the rules engine.
    Override any field to customise behaviour without touching rule logic.
    """
    # Rule A
    car_price_cap_pct: float = 70.0             # max payout as % of car market value

    # Rule C — penalty percentages (0-100); applied multiplicatively to running payout
    overspeeding_penalty_pct:       float = 20.0
    distracted_driving_penalty_pct: float = 15.0
    traffic_violations_penalty_pct: float = 15.0
    no_dashcam_penalty_pct:         float = 15.0
    failure_to_mitigate_penalty_pct: float = 5.0
    poor_maintenance_penalty_pct:   float = 15.0
    unauthorized_repair_penalty_pct: float = 5.0

    # Base payout — severity multipliers (fraction of depreciated car value, 0.0–1.0)
    # Total loss and theft always use 1.0 — handled explicitly in _compute_base_payout
    trivial_damage_multiplier: float = 0.08   # cosmetic / no structural damage
    minor_damage_multiplier: float = 0.25     # single panel / airbags not deployed
    major_damage_multiplier: float = 0.65     # structural damage / airbags deployed

    # Depreciation — derives depreciated_car_value from car_price and vehicle age
    depreciation_rate_per_year: float = 10.0  # % of original value lost per year
    max_depreciation_pct: float = 70.0         # depreciation cannot exceed this cap

    # Final payout cap — hard ceiling applied after ALL reductions
    # References the original car_price, not the depreciated value
    final_payout_cap_pct: float = 70.0


# =============================================================================
# B — DENIAL FLAGS  (one per exclusion condition)
# =============================================================================

@dataclass
class DenialFlags:
    """
    Denial conditions — if ANY field is True the claim is fully denied (payout = 0).
    All default to False; callers set only the flags that apply.

    Fields map 1-to-1 with _DENIAL_LABELS so the denial engine needs no custom logic.
    """
    dui_dwi:                    bool = False   # driving under influence of alcohol/drugs
    no_valid_license:           bool = False   # unlicensed driver OR expired licence
    fraud_or_staged_accident:   bool = False   # fraudulent claim or deliberately staged event
    commercial_use_not_covered: bool = False   # vehicle used commercially without declared coverage
    racing_or_illegal_driving:  bool = False   # racing, stunt driving, or other illegal road use
    geographic_exclusion:       bool = False   # incident occurred outside covered territory
    lapse_in_coverage:          bool = False   # policy was not active at time of incident
    intentional_damage:         bool = False   # owner intentionally caused the damage


# Denial reason labels — keys must match DenialFlags field names exactly
_DENIAL_LABELS: dict[str, str] = {
    "dui_dwi":
        "DUI/DWI — claim denied under policy exclusion.",
    "no_valid_license":
        "No valid licence — driver was unlicensed or licence had expired.",
    "fraud_or_staged_accident":
        "Fraud or staged accident detected — claim denied.",
    "commercial_use_not_covered":
        "Commercial use without declared coverage — policy exclusion applies.",
    "racing_or_illegal_driving":
        "Racing, stunt, or illegal driving — excluded from standard coverage.",
    "geographic_exclusion":
        "Incident occurred in a geographically excluded area.",
    "lapse_in_coverage":
        "Lapse in coverage at time of incident — policy was not active.",
    "intentional_damage":
        "Intentional self-inflicted damage — claim denied.",
}


# =============================================================================
# C — REDUCTION FACTORS
# =============================================================================

@dataclass
class ReductionFactors:
    """
    Inputs for the claim reduction engine.

    comparative_negligence_pct accepts values 0-100.
    salvage_value is a flat dollar amount deducted last.
    Depreciation is NOT here — it is computed from vehicle age in PolicyConfig.
    """
    # Proportional fault — e.g., 30.0 means driver is 30% at fault
    comparative_negligence_pct: float = 0.0

    # Boolean penalty triggers (rates live in PolicyConfig)
    overspeeding:        bool = False
    distracted_driving:  bool = False
    traffic_violations:  bool = False
    no_dashcam:          bool = False
    failure_to_mitigate: bool = False
    poor_maintenance:    bool = False
    unauthorized_repair: bool = False

    salvage_value: float = 0.0       # recovered salvage in $ — deducted from payout


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _pct_reduction(
    current: float,
    pct: float,
    rule: str,
    description: str,
    adjustments: list,
) -> float:
    """Reduce *current* by *pct* percent, append an audit entry, return new value."""
    reduction = current * (pct / 100.0)
    after = current - reduction
    adjustments.append({
        "rule": rule,
        "description": description,
        "payout_before": round(current, 2),
        "payout_after": round(after, 2),
    })
    return after


# =============================================================================
# RULE A — CAR PRICE CAP
# =============================================================================

def _apply_car_price_cap(
    claim: float,
    depreciated_car_value: float,
    cap_pct: float,
    adjustments: list,
) -> float:
    """
    Cap the claim so payout cannot exceed cap_pct% of the depreciated car value.
    Skipped when depreciated_car_value <= 0 (value unknown / not applicable).
    """
    if depreciated_car_value <= 0:
        return claim

    cap_amount = depreciated_car_value * (cap_pct / 100.0)
    if claim > cap_amount:
        adjustments.append({
            "rule": "car_price_cap",
            "description": (
                f"Claim ${claim:,.2f} exceeds {cap_pct:.0f}% of depreciated car value "
                f"(${depreciated_car_value:,.2f} × {cap_pct:.0f}% = ${cap_amount:,.2f}). "
                f"Capped at ${cap_amount:,.2f}."
            ),
            "payout_before": round(claim, 2),
            "payout_after": round(cap_amount, 2),
        })
        return cap_amount

    return claim


# =============================================================================
# DEPRECIATION
# =============================================================================

def _compute_depreciated_car_value(
    car_price: float,
    auto_year: int,
    config: "PolicyConfig",
) -> tuple[float, dict]:
    """
    Compute the depreciated car value from original price and vehicle age.

    Rules:
        - 10% of original value lost per year (config.depreciation_rate_per_year)
        - Depreciation is capped at 70% (config.max_depreciation_pct)
        - Minimum retained value = car_price × (1 − max_depreciation_pct / 100)

    Returns (depreciated_car_value, audit_entry).
    If auto_year is unknown (≤ 0) or car_price is 0, returns car_price unchanged.
    """
    if car_price <= 0 or auto_year <= 0:
        return float(car_price), {
            "rule": "depreciation",
            "description": "Vehicle price or year not provided — depreciation skipped.",
            "payout_before": round(float(car_price), 2),
            "payout_after":  round(float(car_price), 2),
        }

    current_year    = date.today().year
    age_years       = max(0, current_year - auto_year)
    raw_dep_pct     = age_years * config.depreciation_rate_per_year
    dep_pct         = min(raw_dep_pct, config.max_depreciation_pct)
    depreciated_val = round(car_price * (1.0 - dep_pct / 100.0), 2)

    return depreciated_val, {
        "rule": "depreciation",
        "description": (
            f"Vehicle age {age_years} yr × {config.depreciation_rate_per_year:.0f}%/yr "
            f"= {raw_dep_pct:.0f}% (capped at {config.max_depreciation_pct:.0f}%) → "
            f"${car_price:,.2f} × {100.0 - dep_pct:.0f}% = ${depreciated_val:,.2f}."
        ),
        "payout_before": round(float(car_price), 2),
        "payout_after":  depreciated_val,
    }


# =============================================================================
# BASE PAYOUT COMPUTATION
# =============================================================================

def _compute_base_payout(
    incident_type: str,
    incident_severity: str,
    depreciated_car_value: float,
    ml_damage_estimate: float,
    config: "PolicyConfig",
) -> tuple[float, dict]:
    """
    Determine the policy base payout from the pre-computed depreciated car value
    and incident characteristics.
    The ML damage estimate is passed solely for audit contrast — never used in computation.

    Theft / Total Loss  → base = depreciated_car_value (100%)
    Other severities    → base = severity_multiplier × depreciated_car_value
    Fallback (unknown)  → base = ml_damage_estimate (with audit note)
    """
    norm_type     = incident_type.strip().lower()
    norm_severity = incident_severity.strip().lower()

    is_theft      = norm_type == "vehicle theft"
    is_total_loss = norm_severity == "total loss"

    if is_theft or is_total_loss:
        basis = "theft" if is_theft else "total_loss"
        if depreciated_car_value <= 0:
            return float(ml_damage_estimate), {
                "rule": "base_payout_computation",
                "description": (
                    f"{basis.replace('_', ' ').title()}: depreciated vehicle value is zero. "
                    f"Falling back to ML damage estimate ${ml_damage_estimate:,.2f}."
                ),
                "payout_before": round(float(ml_damage_estimate), 2),
                "payout_after":  round(float(ml_damage_estimate), 2),
            }
        return depreciated_car_value, {
            "rule": "base_payout_computation",
            "description": (
                f"{basis.replace('_', ' ').title()}: base = full depreciated vehicle value "
                f"${depreciated_car_value:,.2f}. "
                f"ML estimate ${ml_damage_estimate:,.2f} is reference only."
            ),
            "payout_before": round(float(ml_damage_estimate), 2),
            "payout_after":  round(depreciated_car_value, 2),
        }

    severity_map = {
        "trivial damage": config.trivial_damage_multiplier,
        "minor damage":   config.minor_damage_multiplier,
        "major damage":   config.major_damage_multiplier,
    }
    multiplier = severity_map.get(norm_severity)

    if multiplier is None or depreciated_car_value <= 0:
        return float(ml_damage_estimate), {
            "rule": "base_payout_computation",
            "description": (
                f"Severity '{incident_severity}' (depreciated value unavailable or unrecognised): "
                f"using ML damage estimate ${ml_damage_estimate:,.2f} as base."
            ),
            "payout_before": round(float(ml_damage_estimate), 2),
            "payout_after":  round(float(ml_damage_estimate), 2),
        }

    base = round(max(0.0, depreciated_car_value * multiplier), 2)
    return base, {
        "rule": "base_payout_computation",
        "description": (
            f"{incident_severity}: {multiplier * 100:.0f}% of depreciated value "
            f"(${depreciated_car_value:,.2f} × {multiplier * 100:.0f}% = ${base:,.2f}). "
            f"ML estimate ${ml_damage_estimate:,.2f} is reference only."
        ),
        "payout_before": round(float(ml_damage_estimate), 2),
        "payout_after":  base,
    }


# =============================================================================
# RULE B — DENIAL ENGINE
# =============================================================================

def _check_denial(denial_flags: DenialFlags) -> tuple[bool, Optional[str]]:
    """
    Evaluate every denial condition in declaration order.
    Returns (denied, reason_string).  Stops at the first match.
    """
    for flag_name, label in _DENIAL_LABELS.items():
        if getattr(denial_flags, flag_name, False):
            return True, label
    return False, None


# =============================================================================
# DASHCAM TRIGGER
# =============================================================================

def _apply_dashcam_trigger(
    claim: float,
    factors: ReductionFactors,
    adjustments: list,
) -> ReductionFactors:
    """
    Trigger system — NOT a reduction.

    When dashcam evidence is unavailable (factors.no_dashcam is True), force-activate:
        - overspeeding
        - distracted_driving
        - failure_to_mitigate

    These flags are then applied as normal percentage reductions by Rule C.
    The missing-evidence penalty (no_dashcam_penalty_pct) is also applied by Rule C
    via the existing no_dashcam flag — this function does not touch it.

    Only newly activated flags are listed in the audit entry.
    If all three were already set by the caller, the trigger is still recorded.
    Returns factors unchanged when no_dashcam is False.
    """
    if not factors.no_dashcam:
        return factors

    newly_activated = [
        label
        for flag, label in (
            (factors.overspeeding,        "overspeeding"),
            (factors.distracted_driving,  "distracted driving"),
            (factors.failure_to_mitigate, "failure to mitigate"),
        )
        if not flag
    ]

    description = (
        f"No dashcam evidence — activating: {', '.join(newly_activated)}."
        if newly_activated
        else "No dashcam evidence — all associated penalties already active."
    )

    adjustments.append({
        "rule":          "dashcam_trigger",
        "description":   description,
        "payout_before": round(claim, 2),
        "payout_after":  round(claim, 2),   # trigger only; payout unchanged here
    })

    return dataclasses.replace(
        factors,
        overspeeding=True,
        distracted_driving=True,
        failure_to_mitigate=True,
    )


# =============================================================================
# RULE C — REDUCTION ENGINE
# =============================================================================

def _apply_reductions(
    claim: float,
    factors: ReductionFactors,
    config: PolicyConfig,
) -> tuple[float, list]:
    """
    Apply all reductions multiplicatively to the running payout total.
    Each step operates on the value produced by the previous step.
    Returns (reduced_claim, adjustments_list).

    IMPORTANT: claim must be the post-base-payout value — never the ML estimate.

    Order:
        0. Comparative negligence   (proportional fault — applied first, largest impact)
        1. Overspeeding             20%
        2. Distracted driving       15%
        3. Traffic violations       15%
        4. Missing dashcam          15%
        5. Failure to mitigate       5%
        6. Poor maintenance         15%
        7. Unauthorized repair       5%
        8. Salvage deduction        (flat $ amount — applied last)
    """
    adjustments: list = []

    # 0. Comparative negligence — proportional fault, applied before flat-rate penalties
    if factors.comparative_negligence_pct > 0:
        pct = min(factors.comparative_negligence_pct, 100.0)
        claim = _pct_reduction(
            claim, pct,
            "comparative_negligence",
            f"Driver is {pct:.1f}% at fault — payout reduced by {pct:.1f}%.",
            adjustments,
        )

    # 1. Overspeeding — 20%
    if factors.overspeeding:
        claim = _pct_reduction(
            claim, config.overspeeding_penalty_pct,
            "overspeeding_penalty",
            f"Overspeeding — {config.overspeeding_penalty_pct:.0f}% penalty applied.",
            adjustments,
        )

    # 2. Distracted driving — 15%
    if factors.distracted_driving:
        claim = _pct_reduction(
            claim, config.distracted_driving_penalty_pct,
            "distracted_driving_penalty",
            f"Distracted driving — {config.distracted_driving_penalty_pct:.0f}% penalty applied.",
            adjustments,
        )

    # 3. Traffic violations — 15%
    if factors.traffic_violations:
        claim = _pct_reduction(
            claim, config.traffic_violations_penalty_pct,
            "traffic_violations_penalty",
            f"Traffic violations on record — {config.traffic_violations_penalty_pct:.0f}% penalty applied.",
            adjustments,
        )

    # 4. Missing dashcam evidence — 15%
    if factors.no_dashcam:
        claim = _pct_reduction(
            claim, config.no_dashcam_penalty_pct,
            "no_dashcam_penalty",
            f"No dashcam evidence — {config.no_dashcam_penalty_pct:.0f}% missing-evidence penalty applied.",
            adjustments,
        )

    # 5. Failure to mitigate loss — 5%
    if factors.failure_to_mitigate:
        claim = _pct_reduction(
            claim, config.failure_to_mitigate_penalty_pct,
            "failure_to_mitigate_penalty",
            f"Failure to mitigate loss — {config.failure_to_mitigate_penalty_pct:.0f}% penalty applied.",
            adjustments,
        )

    # 6. Poor vehicle maintenance — 15%
    if factors.poor_maintenance:
        claim = _pct_reduction(
            claim, config.poor_maintenance_penalty_pct,
            "poor_maintenance_penalty",
            f"Poor vehicle maintenance — {config.poor_maintenance_penalty_pct:.0f}% penalty applied.",
            adjustments,
        )

    # 7. Unauthorized repair — 5%
    if factors.unauthorized_repair:
        claim = _pct_reduction(
            claim, config.unauthorized_repair_penalty_pct,
            "unauthorized_repair_penalty",
            f"Unauthorized repair work — {config.unauthorized_repair_penalty_pct:.0f}% penalty applied.",
            adjustments,
        )

    # 8. Salvage value deduction — flat $ amount, applied after all multipliers
    if factors.salvage_value > 0:
        before = claim
        claim = max(0.0, claim - factors.salvage_value)
        adjustments.append({
            "rule":          "salvage_value_deduction",
            "description":   f"Salvage value ${factors.salvage_value:,.2f} deducted.",
            "payout_before": round(before, 2),
            "payout_after":  round(claim, 2),
        })

    return claim, adjustments


# =============================================================================
# FINAL PAYOUT CAP
# =============================================================================

def _apply_final_payout_cap(
    claim: float,
    car_price: float,
    cap_pct: float,
    adjustments: list,
) -> float:
    """
    Hard ceiling on the final payout — applied as the LAST step in the pipeline,
    after all reductions have been applied.

    Cap = car_price × cap_pct / 100  (references original car price, not depreciated).
    Skipped when car_price <= 0.
    """
    if car_price <= 0:
        return claim

    ceiling = round(car_price * cap_pct / 100.0, 2)
    if claim > ceiling:
        adjustments.append({
            "rule":          "final_payout_cap",
            "description":   (
                f"Final payout ${claim:,.2f} exceeds {cap_pct:.0f}% of original car price "
                f"(${car_price:,.2f} × {cap_pct:.0f}% = ${ceiling:,.2f}). "
                f"Capped at ${ceiling:,.2f}."
            ),
            "payout_before": round(claim, 2),
            "payout_after":  ceiling,
        })
        return ceiling

    return claim


# =============================================================================
# PUBLIC API — single entry point
# =============================================================================

def evaluate_claim(
    ml_damage_estimate: float,
    car_price: float,
    denial_flags: DenialFlags,
    reduction_factors: ReductionFactors,
    incident_type: str = "",
    incident_severity: str = "",
    auto_year: int = 0,
    config: Optional[PolicyConfig] = None,
) -> dict:
    """
    Run the full insurance policy rules engine.

    ml_damage_estimate is the raw ML model output — stored in the result for
    reference only.  All downstream calculations use depreciated_car_value,
    which is computed from car_price and auto_year before any other rule runs.

    Execution order:
        B. Denial check    → ANY flag True → payout = 0, pipeline stops immediately
        D. Depreciation    → depreciated_car_value = car_price × (1 − age×rate),
                             capped at config.max_depreciation_pct
        0. Base payout     → depreciated_car_value × severity multiplier
                             (theft / total-loss → 100% of depreciated value)
        A. Car price cap   → payout ≤ config.car_price_cap_pct % of depreciated_car_value
        C. Reductions      → sequential multipliers + salvage deduction

    Args:
        ml_damage_estimate: Raw ML model output in dollars — reference only.
        car_price:          Original market value of the insured vehicle in dollars.
        denial_flags:       DenialFlags instance — any True field voids the claim.
        reduction_factors:  ReductionFactors instance — penalty/deduction inputs.
        incident_type:      Incident type string (e.g. "Vehicle Theft").
        incident_severity:  Severity string (e.g. "Total Loss", "Major Damage").
        auto_year:          Vehicle model year — used to compute age-based depreciation.
                            Pass 0 to skip depreciation (depreciated_car_value = car_price).
        config:             Optional PolicyConfig for custom thresholds.

    Returns:
        dict with keys:
            "ml_damage_estimate"   (float)      — raw ML output, for reference
            "depreciated_car_value"(float)      — car value after age-based depreciation
            "final_payout"         (float)      — adjusted payout in dollars
            "denied"               (bool)       — True if claim was voided
            "denial_reason"        (str | None) — human-readable denial reason
            "adjustments"          (list[dict]) — ordered audit trail
    """
    if config is None:
        config = PolicyConfig()

    # ---- RULE B: Denial — checked first, stops pipeline immediately ----------
    denied, denial_reason = _check_denial(denial_flags)
    if denied:
        return {
            "ml_damage_estimate":    round(float(ml_damage_estimate), 2),
            "depreciated_car_value": 0.0,
            "final_payout":          0.0,
            "denied":                True,
            "denial_reason":         denial_reason,
            "adjustments": [
                {
                    "rule":          "claim_denial",
                    "description":   denial_reason,
                    "payout_before": round(float(ml_damage_estimate), 2),
                    "payout_after":  0.0,
                }
            ],
        }

    adjustments: list = []

    # ---- DEPRECIATION: compute once, used by every downstream rule -----------
    depreciated_car_value, dep_audit = _compute_depreciated_car_value(
        car_price=float(car_price),
        auto_year=int(auto_year),
        config=config,
    )
    adjustments.append(dep_audit)

    # ---- STEP 0: Base payout — derived from depreciated_car_value only ------
    current, base_audit = _compute_base_payout(
        incident_type=incident_type,
        incident_severity=incident_severity,
        depreciated_car_value=depreciated_car_value,
        ml_damage_estimate=float(ml_damage_estimate),
        config=config,
    )
    adjustments.append(base_audit)

    # ---- RULE A: Car price cap — uses depreciated_car_value as the ceiling ---
    current = _apply_car_price_cap(
        current, depreciated_car_value, config.car_price_cap_pct, adjustments
    )

    # ---- DASHCAM TRIGGER: force-activate penalties before reductions ----------
    reduction_factors = _apply_dashcam_trigger(current, reduction_factors, adjustments)

    # ---- RULE C: Reductions --------------------------------------------------
    current, reduction_adjustments = _apply_reductions(current, reduction_factors, config)
    adjustments.extend(reduction_adjustments)

    # ---- FINAL PAYOUT CAP: hard ceiling after all reductions -----------------
    current = _apply_final_payout_cap(
        current, float(car_price), config.final_payout_cap_pct, adjustments
    )

    return {
        "ml_damage_estimate":    round(float(ml_damage_estimate), 2),
        "depreciated_car_value": round(depreciated_car_value, 2),
        "final_payout":          round(max(0.0, current), 2),
        "denied":                False,
        "denial_reason":         None,
        "adjustments":           adjustments,
    }
