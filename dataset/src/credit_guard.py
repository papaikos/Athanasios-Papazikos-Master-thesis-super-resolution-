"""Local credit allowance calculator; no cloud, billing, or payment access.

Run with ``python src/credit_guard.py < request.json``. A positive allowance
depends on the accuracy of caller-provided EUR costs and is not an account
spending limit or a runtime shutdown mechanism.
"""

import json
import math
import sys
from datetime import datetime, timezone
from fractions import Fraction


NUMERIC_FIELDS = (
    "available_credit_eur", "reserved_eur", "committed_eur", "unbilled_eur",
    "other_anticipated_eur", "whole_vm_upper_eur_per_hour",
    "stage_cap_seconds", "user_time_cap_seconds",
)


def _denied(reason):
    return {"allowed_seconds": 0, "spend_allowance_eur": 0, "reasons": [reason]}


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO 8601 string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _seconds(delta):
    return Fraction((delta.days * 86400 + delta.seconds) * 1000000
                    + delta.microseconds, 1000000)


def calculate_allowance(request, *, now):
    """Return whole allowed seconds, capped spend in EUR, and limiting reasons.

    ``now``, ``snapshot_at`` and ``credit_expires_at`` are timezone-aware ISO
    8601 strings. All monetary values and caps must be finite, nonnegative
    JSON numbers. The reserve must be at least max(EUR 30, 25% of available
    credit). Anticipated costs must cover the full planned stage, including
    costs outside the supplied whole-VM hourly upper bound.

    Snapshots older than six hours, future snapshots, expired credit, invalid
    inputs, and less than one affordable whole second return zero. This pure
    function reads neither the clock nor external state. Callers must enforce
    the returned duration and account for shutdown time in their inputs.
    """
    if not isinstance(request, dict):
        return _denied("invalid_request")
    amounts = {}
    for field in NUMERIC_FIELDS:
        value = request.get(field)
        try:
            valid = (type(value) in (int, float) and math.isfinite(value)
                     and value >= 0)
        except OverflowError:
            valid = False
        if not valid:
            return _denied("invalid_" + field)
        amounts[field] = Fraction(str(value))
    times = {}
    for field, value in (("now", now), ("snapshot_at", request.get("snapshot_at")),
                         ("credit_expires_at", request.get("credit_expires_at"))):
        try:
            times[field] = _timestamp(value)
        except (ValueError, OverflowError):
            return _denied("invalid_" + field)

    age = _seconds(times["now"] - times["snapshot_at"])
    if age < 0:
        return _denied("future_snapshot")
    if age > 6 * 3600:
        return _denied("stale_snapshot")
    remaining = _seconds(times["credit_expires_at"] - times["now"])
    if remaining <= 0:
        return _denied("credit_expired")
    if amounts["reserved_eur"] < max(30, amounts["available_credit_eur"] / 4):
        return _denied("insufficient_reserve")
    rate = amounts["whole_vm_upper_eur_per_hour"]
    if rate <= 0:
        return _denied("invalid_whole_vm_upper_eur_per_hour")
    budget = amounts["available_credit_eur"] - sum(
        amounts[field] for field in (
            "reserved_eur", "committed_eur", "unbilled_eur", "other_anticipated_eur"
        )
    )
    if budget <= 0:
        return _denied("credit_budget_exhausted")
    caps = {
        "credit_budget": budget * 3600 / rate,
        "stage_cap": amounts["stage_cap_seconds"],
        "user_time_cap": amounts["user_time_cap_seconds"],
        "credit_expiry": remaining,
    }
    limit = min(caps.values())
    allowed = math.floor(limit)
    reasons = [name for name, cap in caps.items() if cap == limit]
    if allowed <= 0:
        return {"allowed_seconds": 0, "spend_allowance_eur": 0,
                "reasons": reasons + ["less_than_one_second"]}
    exact_spend = allowed * rate / 3600
    spend = float(exact_spend)
    # Avoid reporting a floating-point allowance above the exact calculation.
    if Fraction(spend) > exact_spend:
        spend = math.nextafter(spend, 0)
    return {"allowed_seconds": allowed, "spend_allowance_eur": spend,
            "reasons": reasons}


def main():
    """Read one JSON request from stdin; emit JSON, exiting 2 on denial."""
    try:
        request = json.load(sys.stdin)
    except (ValueError, UnicodeError):
        result = _denied("invalid_json")
    else:
        result = calculate_allowance(request, now=datetime.now(timezone.utc).isoformat())
    print(json.dumps(result, allow_nan=False))
    return 0 if result["allowed_seconds"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
