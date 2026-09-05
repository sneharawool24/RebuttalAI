"""Razorpay webhook verification and conservative dispute normalization."""

from __future__ import annotations

import hashlib
import hmac
from typing import Any


ACTIVE_DISPUTE_EVENT = "payment.dispute.created"


def signature_is_valid(raw_body: bytes, signature: str, secret: str) -> bool:
    """Verify the signature over the exact received request bytes."""

    expected = hmac.new(
        secret.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _number(value: Any) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def normalize_internal_reason(
    reason: str | None,
    reason_code: str | None,
    payment_method: str | None,
) -> str | None:
    """Map only clearly supported Razorpay reason/method combinations."""

    text = " ".join(
        item.lower() for item in (reason, reason_code) if isinstance(item, str)
    )
    method = (payment_method or "").lower()

    unauthorized = any(
        marker in text
        for marker in ("unauthor", "not_recogn", "transaction_not_recogn")
    )
    non_delivery = any(
        marker in text
        for marker in ("non_delivery", "not_delivered", "goods_not_received", "service_not_received")
    )

    if unauthorized and method == "upi":
        return "upi_unauthorized"
    if unauthorized and method in {"netbanking", "net_banking"}:
        return "netbanking_unauthorized"
    if non_delivery:
        return "non_delivery"
    return None


def extract_dispute_metadata(event_payload: dict[str, Any]) -> dict[str, Any] | None:
    """Extract only safe, documented payment/dispute metadata when present."""

    payload = event_payload.get("payload")
    if not isinstance(payload, dict):
        return None
    dispute_container = payload.get("dispute")
    payment_container = payload.get("payment")
    dispute = dispute_container.get("entity") if isinstance(dispute_container, dict) else None
    payment = payment_container.get("entity") if isinstance(payment_container, dict) else None
    if not isinstance(dispute, dict):
        return None
    if not isinstance(payment, dict):
        payment = {}

    razorpay_dispute_id = _text(dispute.get("id"))
    if not razorpay_dispute_id:
        return None

    reason = _text(dispute.get("reason"))
    reason_code = _text(dispute.get("reason_code"))
    payment_method = _text(payment.get("method"))
    payment_amount = _number(payment.get("amount"))
    dispute_amount = _number(dispute.get("amount"))
    currency = _text(dispute.get("currency")) or _text(payment.get("currency"))
    payment_created_at = _number(payment.get("created_at"))

    # Razorpay amounts are in currency subunits. This prefill is used only for
    # INR, where the existing merchant form expects rupees.
    order_value = (
        round(payment_amount / 100, 2)
        if isinstance(payment_amount, (int, float)) and currency == "INR"
        else None
    )
    internal_reason = normalize_internal_reason(reason, reason_code, payment_method)
    return {
        "source_label": (
            "Razorpay Webhook Test"
            if event_payload.get("rebuttalai_simulated_webhook") is True
            else "Razorpay Webhook"
        ),
        "is_simulated": event_payload.get("rebuttalai_simulated_webhook") is True,
        "event": _text(event_payload.get("event")),
        "razorpay_dispute_id": razorpay_dispute_id,
        "payment_id": _text(dispute.get("payment_id")) or _text(payment.get("id")),
        "amount": dispute_amount,
        "currency": currency,
        "reason": reason,
        "reason_code": reason_code,
        "status": _text(dispute.get("status")),
        "phase": _text(dispute.get("phase")),
        "created_at": _number(dispute.get("created_at")) or _number(event_payload.get("created_at")),
        "payment_method": payment_method,
        "payment_amount": payment_amount,
        "payment_currency": _text(payment.get("currency")),
        "order_id": _text(payment.get("order_id")),
        "internal_dispute_type": internal_reason,
        "form_prefill": {
            "dispute_type": internal_reason,
            "order_value": order_value,
        },
    }


def public_incoming_dispute(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return compact safe metadata for the merchant's incoming-disputes UI."""

    metadata = workflow["webhook_metadata"]
    return {
        "workflow_id": workflow["dispute_id"],
        "source_label": metadata["source_label"],
        "is_simulated": metadata["is_simulated"],
        "razorpay_dispute_id": metadata["razorpay_dispute_id"],
        "payment_id": metadata.get("payment_id"),
        "amount": metadata.get("amount"),
        "currency": metadata.get("currency"),
        "reason": metadata.get("reason"),
        "reason_code": metadata.get("reason_code"),
        "status": metadata.get("status"),
        "phase": metadata.get("phase"),
        "created_at": metadata.get("created_at"),
        "payment_method": metadata.get("payment_method"),
        "internal_dispute_type": metadata.get("internal_dispute_type"),
        "analysis_ready": bool(workflow.get("analysis_ready")),
        "additional_merchant_input_required": not bool(workflow.get("analysis_ready")),
        "form_prefill": metadata.get("form_prefill", {}),
    }
