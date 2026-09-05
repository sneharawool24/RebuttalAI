"""Backend-only Razorpay handoff helpers.

This module deliberately contains no FastAPI routes or frontend-facing secrets.
It maps RebuttalAI categories to Razorpay evidence fields and keeps the small
HTTP adapter isolated from workflow and recommendation logic.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4


RAZORPAY_API_BASE = "https://api.razorpay.com/v1"
RAZORPAY_TIMEOUT_SECONDS = 12

# `others` is intentionally explicit: Razorpay's contest API accepts a custom
# type paired with document IDs for this bucket. We preserve the RebuttalAI
# category in a conservative slug rather than claiming a more specific field.
EVIDENCE_BUCKETS: dict[str, tuple[str, str | None]] = {
    "Customer communication": ("customer_communication", None),
    "Merchant transaction/order records": ("billing_proof", None),
    "Order/invoice details": ("billing_proof", None),
    "Shipping proof": ("shipping_proof", None),
    "Shipment tracking information": ("shipping_proof", None),
    "Delivery confirmation/proof of delivery": ("proof_of_service", None),
    "Proof of service": ("proof_of_service", None),
    "Refund/cancellation records": ("refund_confirmation", None),
    "Refund/cancellation policy": ("refund_cancellation_policy", None),
    "Explanation of transaction authorization": ("explanation_letter", None),
    "UPI transaction details": ("others", "upi_transaction_details"),
    "UPI authentication/authorization records": (
        "others",
        "upi_authentication_authorization_records",
    ),
    "Netbanking transaction details": ("others", "netbanking_transaction_details"),
    "Authentication/authorization records": (
        "others",
        "authentication_authorization_records",
    ),
    "Transaction timestamp": ("others", "transaction_timestamp"),
}


class RazorpayAdapterError(Exception):
    """Safe provider-facing failure; never include upstream response bodies."""


def evidence_mapping(category: str) -> dict[str, str | None]:
    """Return the centrally defined Razorpay field for an allowed category."""

    try:
        bucket, other_type = EVIDENCE_BUCKETS[category]
    except KeyError as error:
        raise RazorpayAdapterError(
            "This verified evidence category does not have a Razorpay mapping."
        ) from error

    return {
        "razorpay_bucket": bucket,
        "razorpay_other_type": other_type,
    }


def demo_document_reference(evidence_id: str) -> str:
    """Create a deterministic, visibly non-production document reference."""

    return f"demo_doc_{evidence_id.replace('-', '')[:16]}"


def demo_dispute_reference(workflow_id: str) -> str:
    """Create a deterministic, visibly non-production dispute reference."""

    return f"demo_disp_{workflow_id.replace('-', '')[:16]}"


def contest_summary(dispute_type: str, verified_categories: list[str]) -> str:
    """Make a concise, category-grounded summary without ML/policy claims."""

    readable_type = {
        "upi_unauthorized": "unauthorized UPI payment",
        "netbanking_unauthorized": "unauthorized netbanking payment",
        "non_delivery": "non-delivery",
    }.get(dispute_type, "payment")

    if verified_categories:
        categories = ", ".join(verified_categories)
        summary = (
            f"The merchant is responding to this {readable_type} dispute. "
            f"Verified {categories} have been provided in support of the "
            "merchant's position. Please review the attached evidence when "
            "assessing the dispute."
        )
    else:
        summary = (
            f"The merchant is responding to this {readable_type} dispute. "
            "No verified evidence files have been prepared."
        )

    return summary[:1000]


def build_contest_payload(
    summary: str,
    mapped_documents: list[dict[str, str | None]],
) -> dict[str, Any]:
    """Build only a Razorpay *draft* payload from already-synced documents."""

    payload: dict[str, Any] = {"summary": summary[:1000], "action": "draft"}
    others: dict[str, list[str]] = {}

    for document in mapped_documents:
        document_id = document.get("razorpay_document_id")
        bucket = document.get("razorpay_bucket")
        if not isinstance(document_id, str) or not document_id.startswith("doc_"):
            raise RazorpayAdapterError("A synced Razorpay document reference is required.")
        if not isinstance(bucket, str):
            raise RazorpayAdapterError("A Razorpay evidence mapping is required.")

        if bucket == "others":
            other_type = document.get("razorpay_other_type")
            if not isinstance(other_type, str) or not re.fullmatch(r"[a-z0-9_]{1,80}", other_type):
                raise RazorpayAdapterError("A Razorpay custom evidence type is required.")
            others.setdefault(other_type, []).append(document_id)
        else:
            payload.setdefault(bucket, []).append(document_id)

    if others:
        payload["others"] = [
            {"type": other_type, "document_ids": document_ids}
            for other_type, document_ids in others.items()
        ]

    # This guard is intentionally close to the only provider write payload.
    if payload["action"] != "draft" or "submit" in json.dumps(payload).lower():
        raise RazorpayAdapterError("Only non-submitting Razorpay drafts are supported.")

    return payload


class RazorpayClient:
    """Minimal, timeout-bound server-side adapter using HTTP Basic auth."""

    def __init__(self, key_id: str, key_secret: str):
        self._key_id = key_id
        self._key_secret = key_secret

    def _headers(self) -> dict[str, str]:
        credentials = f"{self._key_id}:{self._key_secret}".encode("utf-8")
        return {
            "Authorization": "Basic " + base64.b64encode(credentials).decode("ascii"),
            "Accept": "application/json",
        }

    def _request_json(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        headers = self._headers()
        if content_type:
            headers["Content-Type"] = content_type
        request = Request(
            f"{RAZORPAY_API_BASE}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=RAZORPAY_TIMEOUT_SECONDS) as response:
                raw_response = response.read()
        except HTTPError as error:
            raise RazorpayAdapterError(
                "Razorpay could not complete this request. Review the dispute in the Razorpay Dashboard and try again."
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise RazorpayAdapterError(
                "Razorpay could not be reached right now. Please try again later."
            ) from error

        try:
            result = json.loads(raw_response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RazorpayAdapterError(
                "Razorpay returned an unexpected response. Please try again later."
            ) from error
        if not isinstance(result, dict):
            raise RazorpayAdapterError("Razorpay returned an unexpected response.")
        return result

    def fetch_dispute(self, razorpay_dispute_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/disputes/{quote(razorpay_dispute_id, safe='')}")

    def upload_document(
        self,
        file_path: Path,
        original_filename: str,
        content_type: str,
    ) -> str:
        boundary = f"----rebuttalai-{uuid4().hex}"
        safe_filename = original_filename.replace("\r", "").replace("\n", "").replace('"', "")
        try:
            file_content = file_path.read_bytes()
        except OSError as error:
            raise RazorpayAdapterError(
                "The verified evidence file could not be read for Razorpay handoff."
            ) from error
        body = b"".join(
            (
                f"--{boundary}\r\n".encode(),
                b'Content-Disposition: form-data; name="purpose"\r\n\r\n',
                b"dispute_evidence\r\n",
                f"--{boundary}\r\n".encode(),
                (
                    'Content-Disposition: form-data; name="file"; '
                    f'filename="{safe_filename}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode("ascii"),
                file_content,
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            )
        )
        response = self._request_json(
            "POST",
            "/documents",
            body=body,
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        document_id = response.get("id")
        if not isinstance(document_id, str) or not document_id.startswith("doc_"):
            raise RazorpayAdapterError("Razorpay did not return a valid document reference.")
        return document_id

    def prepare_draft(
        self,
        razorpay_dispute_id: str,
        summary: str,
        mapped_documents: list[dict[str, str | None]],
    ) -> dict[str, Any]:
        payload = build_contest_payload(summary, mapped_documents)
        return self._request_json(
            "PATCH",
            f"/disputes/{quote(razorpay_dispute_id, safe='')}/contest",
            body=json.dumps(payload).encode("utf-8"),
            content_type="application/json",
        )
