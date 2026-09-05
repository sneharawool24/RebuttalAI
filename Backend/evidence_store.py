"""Small local persistence layer for the evidence-management MVP.

Uploaded documents are intentionally kept outside FastAPI's static files.  The
metadata index is deliberately simple for the hackathon workflow and is ignored
by Git so merchant documents never enter source control.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4


BACKEND_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BACKEND_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "evidence_uploads"
INDEX_PATH = STORAGE_DIR / "evidence_index.json"


class EvidenceStoreError(Exception):
    """Raised when an evidence workflow record cannot be safely handled."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_index() -> dict[str, dict[str, Any]]:
    return {"disputes": {}, "evidence": {}}


def _ensure_storage() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _load_index() -> dict[str, dict[str, Any]]:
    if not INDEX_PATH.exists():
        return _empty_index()

    try:
        data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceStoreError(
            "Evidence storage is unavailable. Please try again later."
        ) from error

    if not isinstance(data, dict) or not isinstance(data.get("disputes"), dict) or not isinstance(data.get("evidence"), dict):
        raise EvidenceStoreError(
            "Evidence storage is unavailable. Please try again later."
        )

    return data


def _save_index(index: dict[str, dict[str, Any]]) -> None:
    _ensure_storage()
    temporary_path = INDEX_PATH.with_suffix(".tmp")

    try:
        temporary_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary_path, INDEX_PATH)
    except OSError as error:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)
        raise EvidenceStoreError(
            "Evidence storage is unavailable. Please try again later."
        ) from error


def create_dispute_snapshot(snapshot: dict[str, Any]) -> str:
    """Create a backend-owned workflow ID and store the immutable ML result."""

    index = _load_index()
    dispute_id = str(uuid4())

    index["disputes"][dispute_id] = {
        "dispute_id": dispute_id,
        "created_at": _timestamp(),
        "prediction_snapshot": snapshot,
        "razorpay_handoff": {
            "razorpay_dispute_id": None,
            "razorpay_dispute_metadata": None,
            "handoff_status": "not_prepared",
            "contest_summary": None,
            "prepared_at": None,
            "razorpay_draft_status": "not_prepared",
            "rebuttal_ready": False,
            "demo_dispute_reference": None,
        },
    }
    _save_index(index)
    return dispute_id


def get_dispute_snapshot(dispute_id: str) -> dict[str, Any]:
    index = _load_index()
    dispute = index["disputes"].get(dispute_id)
    if not dispute:
        raise EvidenceStoreError("Dispute workflow was not found.")
    return dispute


def add_evidence(
    dispute_id: str,
    original_filename: str,
    evidence_category: str,
    content_type: str,
    file_bytes: bytes,
    extension: str,
) -> dict[str, Any]:
    """Store a validated upload under a non-user-controlled filename."""

    index = _load_index()
    if dispute_id not in index["disputes"]:
        raise EvidenceStoreError("Dispute workflow was not found.")

    _ensure_storage()
    evidence_id = str(uuid4())
    stored_filename = f"{uuid4().hex}{extension}"
    storage_path = UPLOAD_DIR / stored_filename

    try:
        storage_path.write_bytes(file_bytes)
        metadata = {
            "evidence_id": evidence_id,
            "dispute_id": dispute_id,
            "original_filename": original_filename,
            "evidence_category": evidence_category,
            "uploaded_at": _timestamp(),
            "status": "Pending Review",
            "content_type": content_type,
            "file_size": len(file_bytes),
            "stored_filename": stored_filename,
            "razorpay_sync_status": "not_synced",
            "razorpay_document_id": None,
            "razorpay_synced_at": None,
        }
        index["evidence"][evidence_id] = metadata
        _save_index(index)
    except OSError as error:
        storage_path.unlink(missing_ok=True)
        raise EvidenceStoreError(
            "Unable to store this evidence file. Please try again."
        ) from error

    return metadata


def list_evidence(dispute_id: str) -> list[dict[str, Any]]:
    get_dispute_snapshot(dispute_id)
    index = _load_index()
    return [
        evidence
        for evidence in index["evidence"].values()
        if evidence.get("dispute_id") == dispute_id
    ]


def get_evidence_file(dispute_id: str, evidence_id: str) -> tuple[dict[str, Any], Path]:
    """Resolve only a file that is registered for the requested dispute."""

    index = _load_index()
    if dispute_id not in index["disputes"]:
        raise EvidenceStoreError("Dispute workflow was not found.")

    evidence = index["evidence"].get(evidence_id)
    if not evidence or evidence.get("dispute_id") != dispute_id:
        raise EvidenceStoreError("Evidence record was not found for this dispute.")

    # The persisted name is server-generated. Taking only its basename and
    # requiring it to be a direct child of UPLOAD_DIR prevents path traversal.
    upload_root = UPLOAD_DIR.resolve()
    file_path = (upload_root / Path(str(evidence.get("stored_filename", ""))).name).resolve()
    if file_path.parent != upload_root or not file_path.is_file():
        raise EvidenceStoreError("Evidence file was not found.")

    return evidence, file_path


def verify_evidence(dispute_id: str, evidence_id: str) -> dict[str, Any]:
    index = _load_index()
    if dispute_id not in index["disputes"]:
        raise EvidenceStoreError("Dispute workflow was not found.")

    evidence = index["evidence"].get(evidence_id)
    if not evidence or evidence.get("dispute_id") != dispute_id:
        raise EvidenceStoreError("Evidence record was not found for this dispute.")

    if evidence["status"] != "Verified":
        evidence["status"] = "Verified"
        evidence["verified_at"] = _timestamp()
        _save_index(index)

    return evidence


def update_evidence_razorpay_sync(
    dispute_id: str,
    evidence_id: str,
    sync_status: str,
    razorpay_document_id: str | None = None,
) -> dict[str, Any]:
    """Persist only real Razorpay document-sync outcomes for verified files."""

    if sync_status not in {"not_synced", "synced", "failed"}:
        raise EvidenceStoreError("Evidence sync status is invalid.")

    index = _load_index()
    if dispute_id not in index["disputes"]:
        raise EvidenceStoreError("Dispute workflow was not found.")
    evidence = index["evidence"].get(evidence_id)
    if not evidence or evidence.get("dispute_id") != dispute_id:
        raise EvidenceStoreError("Evidence record was not found for this dispute.")
    if evidence.get("status") != "Verified":
        raise EvidenceStoreError("Only verified evidence can be synced to Razorpay.")

    evidence["razorpay_sync_status"] = sync_status
    evidence["razorpay_document_id"] = (
        razorpay_document_id if sync_status == "synced" else None
    )
    evidence["razorpay_synced_at"] = _timestamp() if sync_status == "synced" else None
    _save_index(index)
    return evidence


def update_razorpay_handoff(dispute_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Persist safe handoff state alongside the backend-owned workflow."""

    index = _load_index()
    dispute = index["disputes"].get(dispute_id)
    if not dispute:
        raise EvidenceStoreError("Dispute workflow was not found.")

    handoff = dispute.setdefault("razorpay_handoff", {})
    handoff.update(updates)
    _save_index(index)
    return handoff


def mark_rebuttal_ready(dispute_id: str) -> None:
    """Record that a merchant-facing draft was generated for this workflow."""

    update_razorpay_handoff(dispute_id, {"rebuttal_ready": True})


def delete_evidence(dispute_id: str, evidence_id: str) -> None:
    index = _load_index()
    if dispute_id not in index["disputes"]:
        raise EvidenceStoreError("Dispute workflow was not found.")

    evidence = index["evidence"].get(evidence_id)
    if not evidence or evidence.get("dispute_id") != dispute_id:
        raise EvidenceStoreError("Evidence record was not found for this dispute.")

    stored_filename = Path(str(evidence.get("stored_filename", ""))).name
    del index["evidence"][evidence_id]
    _save_index(index)

    if stored_filename:
        (UPLOAD_DIR / stored_filename).unlink(missing_ok=True)


def public_metadata(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return safe metadata only; never leak server-side storage paths."""

    return {
        "evidence_id": evidence["evidence_id"],
        "dispute_id": evidence["dispute_id"],
        "original_filename": evidence["original_filename"],
        "evidence_category": evidence["evidence_category"],
        "uploaded_at": evidence["uploaded_at"],
        "status": evidence["status"],
        "file_size": evidence["file_size"],
        "content_type": evidence["content_type"],
        "verified_at": evidence.get("verified_at"),
        "razorpay_sync_status": evidence.get("razorpay_sync_status", "not_synced"),
        "razorpay_document_id": evidence.get("razorpay_document_id"),
        "razorpay_synced_at": evidence.get("razorpay_synced_at"),
    }
