import hashlib
import hmac
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import evidence_store
import main
import razorpay_handoff


class EvidenceManagementApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.temp_directory = tempfile.TemporaryDirectory()
        storage_directory = Path(self.temp_directory.name) / "storage"

        self.original_store_paths = (
            evidence_store.STORAGE_DIR,
            evidence_store.UPLOAD_DIR,
            evidence_store.INDEX_PATH,
        )
        evidence_store.STORAGE_DIR = storage_directory
        evidence_store.UPLOAD_DIR = storage_directory / "evidence_uploads"
        evidence_store.INDEX_PATH = storage_directory / "evidence_index.json"

    def tearDown(self):
        (
            evidence_store.STORAGE_DIR,
            evidence_store.UPLOAD_DIR,
            evidence_store.INDEX_PATH,
        ) = self.original_store_paths
        self.temp_directory.cleanup()

    @staticmethod
    def prediction_payload(start_evidence_workflow=False):
        return {
            "dispute_type": "upi_unauthorized",
            "order_value": 3137.73,
            "days_since_transaction": 4,
            "device_ip_match_history": 1.0,
            "auth_flow_type": "pin_entry",
            "customer_account_age_days": 1051,
            "customer_past_order_count": 4,
            "customer_past_dispute_count": 0,
            "delivery_confirmed": None,
            "tracking_available": None,
            "merchant_comm_log_exists": 1,
            "refund_already_issued": 1,
            "start_evidence_workflow": start_evidence_workflow,
        }

    def create_workflow(self):
        response = self.client.post(
            "/predict",
            json=self.prediction_payload(start_evidence_workflow=True),
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertIn("dispute_id", payload)
        return payload

    @staticmethod
    def webhook_payload(
        dispute_id="disp_test_webhook_001",
        reason="unauthorized_transaction",
        reason_code="transaction_not_recognized",
        method="upi",
        simulated=False,
    ):
        payload = {
            "entity": "event",
            "event": "payment.dispute.created",
            "payload": {
                "payment": {
                    "entity": {
                        "id": "pay_test_webhook_001",
                        "amount": 429900,
                        "currency": "INR",
                        "order_id": "order_test_webhook_001",
                        "method": method,
                    }
                },
                "dispute": {
                    "entity": {
                        "id": dispute_id,
                        "payment_id": "pay_test_webhook_001",
                        "amount": 429900,
                        "currency": "INR",
                        "reason": reason,
                        "reason_code": reason_code,
                        "status": "open",
                        "phase": "chargeback",
                        "created_at": 1735689700,
                    }
                },
            },
            "created_at": 1735689700,
        }
        if simulated:
            payload["rebuttalai_simulated_webhook"] = True
        return payload

    def send_signed_webhook(self, payload, event_id="evt_test_webhook_001"):
        raw_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        secret = "webhook_test_secret"
        signature = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
        with patch.dict(
            os.environ, {"RAZORPAY_WEBHOOK_SECRET": secret}, clear=False
        ):
            return self.client.post(
                "/webhooks/razorpay",
                content=raw_body,
                headers={
                    "Content-Type": "application/json",
                    "X-Razorpay-Signature": signature,
                    "x-razorpay-event-id": event_id,
                },
            )

    def upload_pdf(self, dispute_id, category="UPI transaction details"):
        return self.client.post(
            "/evidence/upload",
            data={
                "dispute_id": dispute_id,
                "evidence_category": category,
            },
            files={
                "file": (
                    "merchant-record.pdf",
                    b"%PDF-1.4\nmerchant evidence",
                    "application/pdf",
                )
            },
        )

    def upload_and_verify(self, dispute_id, category="UPI transaction details"):
        upload = self.upload_pdf(dispute_id, category)
        self.assertEqual(upload.status_code, 201, upload.text)
        verify = self.client.post(
            f"/evidence/{dispute_id}/{upload.json()['evidence']['evidence_id']}/verify"
        )
        self.assertEqual(verify.status_code, 200, verify.text)
        return upload.json()["evidence"]

    def test_existing_models_endpoint(self):
        response = self.client.get("/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["upi"], "Random Forest loaded")

    def test_legacy_predict_retains_expected_fields(self):
        response = self.client.post("/predict", json=self.prediction_payload())
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertNotIn("dispute_id", payload)
        self.assertTrue(
            {
                "base_decision",
                "fight_score",
                "evidence_required",
                "evidence_available",
                "evidence_missing",
                "critical_evidence_missing",
                "evidence_coverage",
                "final_recommendation",
                "recommendation_reason",
            }.issubset(payload)
        )

    def test_existing_generate_rebuttal_retains_legacy_values(self):
        request_data = {
            "dispute_type": "upi_unauthorized",
            "order_value": 10.0,
            "days_since_transaction": 1,
            "base_decision": "Don't Fight",
            "fight_score": 0.1,
            "final_recommendation": "Don't Fight",
            "recommendation_reason": "Legacy reason",
            "evidence_available": [],
            "evidence_missing": ["UPI transaction details"],
        }

        with patch.object(main, "generate_rebuttal", return_value="Legacy draft"):
            response = self.client.post("/generate-rebuttal", json=request_data)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["base_decision"], "Don't Fight")
        self.assertEqual(response.json()["rebuttal"], "Legacy draft")
        self.assertNotIn("dispute_id", response.json())

    def test_workflow_creation_is_backend_owned(self):
        workflow = self.create_workflow()
        self.assertEqual(workflow["evidence_available"], [])
        self.assertEqual(workflow["evidence_coverage"], 0.0)

    def test_valid_upload_is_pending_and_does_not_count(self):
        workflow = self.create_workflow()
        response = self.upload_pdf(workflow["dispute_id"])
        self.assertEqual(response.status_code, 201, response.text)
        payload = response.json()
        self.assertEqual(payload["evidence"]["status"], "Pending Review")
        self.assertEqual(payload["recommendation"]["evidence_available"], [])
        self.assertEqual(payload["recommendation"]["evidence_coverage"], 0.0)

        # The refresh endpoint is what the frontend uses after upload. It
        # must return the newly persisted Pending Review record immediately.
        refreshed = self.client.get(f"/evidence/{workflow['dispute_id']}")
        self.assertEqual(refreshed.status_code, 200, refreshed.text)
        self.assertEqual(len(refreshed.json()["evidence"]), 1)
        self.assertEqual(
            refreshed.json()["evidence"][0]["status"],
            "Pending Review",
        )
        self.assertEqual(refreshed.json()["recommendation"]["evidence_coverage"], 0.0)

    def test_preview_serves_registered_file_without_changing_status(self):
        workflow = self.create_workflow()
        upload = self.upload_pdf(workflow["dispute_id"])
        evidence_id = upload.json()["evidence"]["evidence_id"]

        preview = self.client.get(
            f"/evidence/{workflow['dispute_id']}/{evidence_id}/preview"
        )
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertTrue(preview.headers["content-type"].startswith("application/pdf"))
        self.assertEqual(preview.content, b"%PDF-1.4\nmerchant evidence")
        self.assertIn("inline", preview.headers["content-disposition"])

        refreshed = self.client.get(f"/evidence/{workflow['dispute_id']}")
        self.assertEqual(refreshed.status_code, 200)
        self.assertEqual(refreshed.json()["evidence"][0]["status"], "Pending Review")
        self.assertEqual(refreshed.json()["recommendation"]["evidence_coverage"], 0.0)

    def test_preview_rejects_evidence_owned_by_another_dispute(self):
        first_workflow = self.create_workflow()
        second_workflow = self.create_workflow()
        upload = self.upload_pdf(first_workflow["dispute_id"])
        evidence_id = upload.json()["evidence"]["evidence_id"]

        response = self.client.get(
            f"/evidence/{second_workflow['dispute_id']}/{evidence_id}/preview"
        )
        self.assertEqual(response.status_code, 404)

    def test_invalid_file_type_is_rejected(self):
        workflow = self.create_workflow()
        response = self.client.post(
            "/evidence/upload",
            data={
                "dispute_id": workflow["dispute_id"],
                "evidence_category": "UPI transaction details",
            },
            files={"file": ("unsafe.exe", b"MZ", "application/octet-stream")},
        )
        self.assertEqual(response.status_code, 400)

    def test_malformed_file_signature_is_rejected(self):
        workflow = self.create_workflow()
        response = self.client.post(
            "/evidence/upload",
            data={
                "dispute_id": workflow["dispute_id"],
                "evidence_category": "UPI transaction details",
            },
            files={"file": ("not-a-pdf.pdf", b"MZ", "application/pdf")},
        )
        self.assertEqual(response.status_code, 400)

    def test_invalid_evidence_category_is_rejected(self):
        workflow = self.create_workflow()
        response = self.upload_pdf(
            workflow["dispute_id"],
            category="Shipping proof",
        )
        self.assertEqual(response.status_code, 422)

    def test_verified_evidence_counts_once_even_with_duplicate_uploads(self):
        workflow = self.create_workflow()
        first_upload = self.upload_pdf(workflow["dispute_id"])
        second_upload = self.upload_pdf(workflow["dispute_id"])
        self.assertEqual(first_upload.status_code, 201)
        self.assertEqual(second_upload.status_code, 201)

        first_verify = self.client.post(
            "/evidence/"
            f"{workflow['dispute_id']}/"
            f"{first_upload.json()['evidence']['evidence_id']}/verify"
        )
        second_verify = self.client.post(
            "/evidence/"
            f"{workflow['dispute_id']}/"
            f"{second_upload.json()['evidence']['evidence_id']}/verify"
        )

        self.assertEqual(first_verify.status_code, 200, first_verify.text)
        self.assertEqual(second_verify.status_code, 200, second_verify.text)
        recommendation = second_verify.json()["recommendation"]
        self.assertEqual(
            recommendation["evidence_available"],
            ["UPI transaction details"],
        )
        self.assertEqual(recommendation["evidence_coverage"], 0.17)

    def test_deterministic_critical_and_dont_fight_logic(self):
        requirements = main.get_evidence_requirements("upi_unauthorized")
        critical = main.get_critical_evidence("upi_unauthorized")

        missing_critical = main.determine_final_recommendation(
            "Fight", 0.9, requirements, [], critical
        )
        ready_to_fight = main.determine_final_recommendation(
            "Fight", 0.9, requirements, critical, critical
        )
        dont_fight = main.determine_final_recommendation(
            "Don't Fight", 0.9, requirements, critical, critical
        )

        self.assertEqual(
            missing_critical["final_recommendation"],
            "Review / Collect Evidence",
        )
        self.assertEqual(ready_to_fight["final_recommendation"], "Fight")
        self.assertEqual(dont_fight["final_recommendation"], "Don't Fight")

    def test_verified_critical_categories_update_workflow_to_fight(self):
        workflow = self.create_workflow()
        dispute_id = workflow["dispute_id"]

        # This endpoint test isolates the evidence workflow from ML model
        # selection; deterministic recommendation behavior is under test.
        index = evidence_store._load_index()
        index["disputes"][dispute_id]["prediction_snapshot"]["base_decision"] = "Fight"
        evidence_store._save_index(index)

        before = self.client.post(f"/evidence/{dispute_id}/recalculate")
        self.assertEqual(
            before.json()["final_recommendation"],
            "Review / Collect Evidence",
        )

        for category in main.get_critical_evidence("upi_unauthorized"):
            upload = self.upload_pdf(dispute_id, category)
            self.assertEqual(upload.status_code, 201, upload.text)
            verify = self.client.post(
                f"/evidence/{dispute_id}/{upload.json()['evidence']['evidence_id']}/verify"
            )
            self.assertEqual(verify.status_code, 200, verify.text)

        after = self.client.post(f"/evidence/{dispute_id}/recalculate")
        self.assertEqual(after.status_code, 200, after.text)
        self.assertEqual(after.json()["final_recommendation"], "Fight")

    def test_dispute_id_makes_rebuttal_state_authoritative(self):
        workflow = self.create_workflow()
        upload = self.upload_pdf(workflow["dispute_id"])
        self.assertEqual(upload.status_code, 201, upload.text)

        captured = {}

        def capture_generation(generation_data):
            captured["data"] = generation_data
            return "Server-authoritative draft"

        client_claims = {
            "dispute_id": workflow["dispute_id"],
            "dispute_type": "non_delivery",
            "order_value": 1.0,
            "days_since_transaction": 999,
            "base_decision": "Client value",
            "fight_score": 0.9999,
            "final_recommendation": "Fight",
            "recommendation_reason": "Client value",
            "evidence_available": ["Shipping proof"],
            "evidence_missing": [],
            "merchant_context": "The customer authenticated it.",
        }

        with patch.object(main, "generate_rebuttal", side_effect=capture_generation):
            response = self.client.post("/generate-rebuttal", json=client_claims)

        self.assertEqual(response.status_code, 200, response.text)
        authoritative = captured["data"]
        self.assertEqual(authoritative.dispute_type, "upi_unauthorized")
        self.assertEqual(authoritative.order_value, 3137.73)
        self.assertNotEqual(authoritative.base_decision, "Client value")
        self.assertEqual(authoritative.evidence_available, [])
        self.assertEqual(authoritative.verified_evidence_metadata, [])
        self.assertEqual(
            authoritative.merchant_context,
            "The customer authenticated it.",
        )

        verify = self.client.post(
            "/evidence/"
            f"{workflow['dispute_id']}/"
            f"{upload.json()['evidence']['evidence_id']}/verify"
        )
        self.assertEqual(verify.status_code, 200, verify.text)

        with patch.object(main, "generate_rebuttal", side_effect=capture_generation):
            verified_response = self.client.post("/generate-rebuttal", json=client_claims)

        self.assertEqual(verified_response.status_code, 200, verified_response.text)
        self.assertEqual(
            captured["data"].evidence_available,
            ["UPI transaction details"],
        )
        self.assertEqual(
            captured["data"].verified_evidence_metadata[0].original_filename,
            "merchant-record.pdf",
        )

    def test_gemini_prompt_forbids_unsupported_policy_claims(self):
        captured = {}

        def fake_generation(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text="Safe draft")

        input_data = main.RebuttalInput(
            dispute_type="upi_unauthorized",
            order_value=3137.73,
            days_since_transaction=4,
            base_decision="Fight",
            fight_score=0.82,
            final_recommendation="Review / Collect Evidence",
            recommendation_reason="Critical evidence is missing.",
            evidence_available=[],
            evidence_missing=["UPI transaction details"],
        )

        with patch.object(
            main.client.models,
            "generate_content",
            side_effect=fake_generation,
        ):
            self.assertEqual(main.generate_rebuttal(input_data), "Safe draft")

        self.assertIn(
            "Do not cite UPI guidelines, payment-network rules, merchant",
            captured["contents"],
        )
        self.assertIn(
            "The model result, Fight Score, final recommendation, evidence engine",
            captured["contents"],
        )
        self.assertIn("Use EXACTLY this structure and no other sections", captured["contents"])

    def test_internal_decision_language_is_not_returned_in_draft(self):
        for draft_with_internal_language in (
            "Recommendation:\nFight Score: 82%\nThe evidence engine recommends Fight.",
            "Subject: Case response\nRebuttal: Review / Collect Evidence before responding.",
            "Subject: Case response\nRebuttal: The model recommends contesting this case.",
        ):
            with self.subTest(draft=draft_with_internal_language):
                draft = main.merchant_facing_draft_or_fallback(
                    draft_with_internal_language
                )
                self.assertNotIn("Fight", draft)
                self.assertNotIn("evidence engine", draft.lower())
                self.assertNotIn("Review / Collect Evidence", draft)
                self.assertNotIn("model", draft.lower())

    def test_demo_handoff_prepares_only_verified_evidence_without_provider_call(self):
        workflow = self.create_workflow()
        verified = self.upload_and_verify(workflow["dispute_id"])
        pending = self.upload_pdf(
            workflow["dispute_id"], "UPI authentication/authorization records"
        )
        self.assertEqual(pending.status_code, 201)

        with patch.dict(os.environ, {"RAZORPAY_MODE": "demo"}, clear=False), patch.object(
            main, "RazorpayClient"
        ) as provider_client:
            response = self.client.post(
                f"/razorpay/{workflow['dispute_id']}/prepare-draft"
            )

        self.assertEqual(response.status_code, 200, response.text)
        provider_client.assert_not_called()
        handoff = response.json()
        self.assertEqual(handoff["razorpay_mode"], "demo")
        self.assertEqual(handoff["handoff_status"], "prepared_demo")
        self.assertEqual(handoff["submission_status"], "not_submitted")
        self.assertTrue(handoff["demo_dispute_reference"].startswith("demo_disp_"))
        self.assertLessEqual(len(handoff["contest_summary"]), 1000)
        self.assertEqual(handoff["verified_evidence_count"], 1)
        self.assertEqual(len(handoff["evidence"]), 1)
        self.assertEqual(handoff["evidence"][0]["evidence_id"], verified["evidence_id"])
        self.assertEqual(handoff["evidence"][0]["razorpay_bucket"], "others")
        self.assertTrue(
            handoff["evidence"][0]["demo_document_reference"].startswith("demo_doc_")
        )
        self.assertEqual(handoff["evidence"][0]["razorpay_sync_status"], "not_synced")

    def test_connected_handoff_requires_credentials_and_imported_dispute(self):
        workflow = self.create_workflow()
        with patch.dict(
            os.environ,
            {
                "RAZORPAY_MODE": "connected",
                "RAZORPAY_KEY_ID": "",
                "RAZORPAY_KEY_SECRET": "",
            },
            clear=False,
        ):
            missing_credentials = self.client.post(
                f"/razorpay/{workflow['dispute_id']}/prepare-draft"
            )
        self.assertEqual(missing_credentials.status_code, 503)
        self.assertIn("backend credentials", missing_credentials.json()["detail"])

        with patch.dict(
            os.environ,
            {
                "RAZORPAY_MODE": "connected",
                "RAZORPAY_KEY_ID": "key_test",
                "RAZORPAY_KEY_SECRET": "secret_test",
            },
            clear=False,
        ):
            missing_dispute = self.client.post(
                f"/razorpay/{workflow['dispute_id']}/prepare-draft"
            )
        self.assertEqual(missing_dispute.status_code, 400)
        self.assertIn("dispute ID", missing_dispute.json()["detail"])

    def test_connected_failure_never_reports_a_prepared_draft(self):
        workflow = self.create_workflow()
        self.upload_and_verify(workflow["dispute_id"])
        evidence_store.update_razorpay_handoff(
            workflow["dispute_id"],
            {
                "razorpay_dispute_id": "disp_AbCdEf12345678",
                "razorpay_dispute_imported": True,
            },
        )

        fake_provider = SimpleNamespace(
            upload_document=lambda *_: (_ for _ in ()).throw(
                razorpay_handoff.RazorpayAdapterError("Razorpay could not be reached right now.")
            )
        )
        with patch.dict(
            os.environ,
            {
                "RAZORPAY_MODE": "connected",
                "RAZORPAY_KEY_ID": "key_test",
                "RAZORPAY_KEY_SECRET": "secret_test",
            },
            clear=False,
        ), patch.object(main, "RazorpayClient", return_value=fake_provider):
            response = self.client.post(
                f"/razorpay/{workflow['dispute_id']}/prepare-draft"
            )

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("prepared", response.json()["detail"].lower())
        with patch.dict(os.environ, {"RAZORPAY_MODE": "connected"}, clear=False):
            handoff = self.client.get(
                f"/razorpay/{workflow['dispute_id']}/handoff"
            ).json()
        self.assertEqual(handoff["handoff_status"], "failed")
        self.assertEqual(handoff["razorpay_draft_status"], "not_prepared")
        evidence = self.client.get(f"/evidence/{workflow['dispute_id']}").json()["evidence"]
        self.assertEqual(evidence[0]["razorpay_sync_status"], "failed")

    def test_connected_sync_uses_verified_files_once_and_only_drafts(self):
        workflow = self.create_workflow()
        verified = self.upload_and_verify(workflow["dispute_id"])
        self.upload_pdf(workflow["dispute_id"], "UPI authentication/authorization records")
        evidence_store.update_razorpay_handoff(
            workflow["dispute_id"],
            {
                "razorpay_dispute_id": "disp_AbCdEf12345678",
                "razorpay_dispute_imported": True,
            },
        )

        class FakeProvider:
            def __init__(self):
                self.uploads = []
                self.drafts = []

            def upload_document(self, _path, filename, _content_type):
                self.uploads.append(filename)
                return "doc_TestEvidence123"

            def prepare_draft(self, dispute_id, summary, mapped_documents):
                self.drafts.append((dispute_id, summary, mapped_documents))
                return {"id": dispute_id, "status": "open"}

        fake_provider = FakeProvider()
        environment = {
            "RAZORPAY_MODE": "connected",
            "RAZORPAY_KEY_ID": "key_test",
            "RAZORPAY_KEY_SECRET": "secret_test",
        }
        with patch.dict(os.environ, environment, clear=False), patch.object(
            main, "RazorpayClient", return_value=fake_provider
        ):
            first = self.client.post(f"/razorpay/{workflow['dispute_id']}/prepare-draft")
            second = self.client.post(f"/razorpay/{workflow['dispute_id']}/prepare-draft")

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(fake_provider.uploads, ["merchant-record.pdf"])
        self.assertEqual(len(fake_provider.drafts), 2)
        self.assertEqual(fake_provider.drafts[0][2][0]["razorpay_document_id"], "doc_TestEvidence123")
        self.assertEqual(first.json()["evidence"][0]["evidence_id"], verified["evidence_id"])
        self.assertEqual(first.json()["evidence"][0]["razorpay_sync_status"], "synced")

        payload = razorpay_handoff.build_contest_payload(
            "A concise summary.",
            [
                {
                    "razorpay_bucket": "billing_proof",
                    "razorpay_other_type": None,
                    "razorpay_document_id": "doc_TestEvidence123",
                }
            ],
        )
        self.assertEqual(payload["action"], "draft")
        self.assertNotIn("submit", str(payload).lower())
        self.assertNotIn("accept", str(payload).lower())

    def test_connected_fetch_rejects_invalid_ids_and_provider_errors_safely(self):
        workflow = self.create_workflow()
        environment = {
            "RAZORPAY_MODE": "connected",
            "RAZORPAY_KEY_ID": "key_test",
            "RAZORPAY_KEY_SECRET": "secret_test",
        }
        with patch.dict(os.environ, environment, clear=False), patch.object(
            main, "RazorpayClient"
        ) as client_factory:
            invalid = self.client.post(
                "/razorpay/dispute/fetch",
                json={"workflow_id": workflow["dispute_id"], "razorpay_dispute_id": "bad"},
            )
        self.assertEqual(invalid.status_code, 422)
        client_factory.assert_not_called()

        with patch.dict(os.environ, environment, clear=False), patch.object(
            main, "RazorpayClient"
        ) as client_factory:
            local_fixture = self.client.post(
                "/razorpay/dispute/fetch",
                json={
                    "workflow_id": workflow["dispute_id"],
                    "razorpay_dispute_id": "disp_test_rebuttalai_001",
                },
            )
        self.assertEqual(local_fixture.status_code, 422)
        client_factory.assert_not_called()

        fake_provider = SimpleNamespace(
            fetch_dispute=lambda *_: (_ for _ in ()).throw(
                razorpay_handoff.RazorpayAdapterError("provider failure")
            )
        )
        with patch.dict(os.environ, environment, clear=False), patch.object(
            main, "RazorpayClient", return_value=fake_provider
        ):
            failed = self.client.post(
                "/razorpay/dispute/fetch",
                json={
                    "workflow_id": workflow["dispute_id"],
                    "razorpay_dispute_id": "disp_AbCdEf12345678",
                },
            )
        self.assertEqual(failed.status_code, 502)
        self.assertNotIn("key_test", failed.json()["detail"])
        handoff = self.client.get(f"/razorpay/{workflow['dispute_id']}/handoff").json()
        self.assertIsNone(handoff["razorpay_dispute_id"])
        self.assertFalse(handoff["razorpay_dispute_imported"])

    def test_connected_fetch_stores_limited_provider_metadata(self):
        workflow = self.create_workflow()
        environment = {
            "RAZORPAY_MODE": "connected",
            "RAZORPAY_KEY_ID": "key_test",
            "RAZORPAY_KEY_SECRET": "secret_test",
        }
        fake_provider = SimpleNamespace(
            fetch_dispute=lambda *_: {
                "id": "disp_AbCdEf12345678",
                "payment_id": "pay_Example123",
                "amount": 313773,
                "currency": "INR",
                "status": "open",
                "evidence": {"not": "stored"},
                "private_provider_field": "not stored",
            }
        )
        with patch.dict(os.environ, environment, clear=False), patch.object(
            main, "RazorpayClient", return_value=fake_provider
        ):
            response = self.client.post(
                "/razorpay/dispute/fetch",
                json={
                    "workflow_id": workflow["dispute_id"],
                    "razorpay_dispute_id": "disp_AbCdEf12345678",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["razorpay_dispute_imported"])
        metadata = response.json()["razorpay_dispute_metadata"]
        self.assertEqual(metadata["id"], "disp_AbCdEf12345678")
        self.assertNotIn("evidence", metadata)
        self.assertNotIn("private_provider_field", metadata)

    def test_connected_draft_requires_a_provider_confirmed_dispute_import(self):
        workflow = self.create_workflow()
        self.upload_and_verify(workflow["dispute_id"])
        # A webhook or stale local value may carry an ID. It is not an import.
        evidence_store.update_razorpay_handoff(
            workflow["dispute_id"],
            {"razorpay_dispute_id": "disp_AbCdEf12345678"},
        )
        environment = {
            "RAZORPAY_MODE": "connected",
            "RAZORPAY_KEY_ID": "key_test",
            "RAZORPAY_KEY_SECRET": "secret_test",
        }
        with patch.dict(os.environ, environment, clear=False), patch.object(
            main, "RazorpayClient"
        ) as client_factory:
            response = self.client.post(
                f"/razorpay/{workflow['dispute_id']}/prepare-draft"
            )

        self.assertEqual(response.status_code, 400)
        client_factory.assert_called_once()
        handoff = self.client.get(f"/razorpay/{workflow['dispute_id']}/handoff").json()
        self.assertFalse(handoff["razorpay_dispute_imported"])
        self.assertEqual(handoff["razorpay_draft_status"], "not_prepared")

    def test_sync_evidence_connected_needs_no_razorpay_dispute_id(self):
        workflow = self.create_workflow()
        verified = self.upload_and_verify(workflow["dispute_id"])
        pending = self.upload_pdf(
            workflow["dispute_id"], "UPI authentication/authorization records"
        )
        self.assertEqual(pending.status_code, 201)

        class FakeProvider:
            def __init__(self):
                self.uploads = []

            def upload_document(self, _path, filename, content_type):
                self.uploads.append((filename, content_type))
                return "doc_RealTestDocument123"

        fake_provider = FakeProvider()
        environment = {
            "RAZORPAY_MODE": "connected",
            "RAZORPAY_KEY_ID": "key_test",
            "RAZORPAY_KEY_SECRET": "secret_test",
        }
        with patch.dict(os.environ, environment, clear=False), patch.object(
            main, "RazorpayClient", return_value=fake_provider
        ):
            response = self.client.post(
                f"/razorpay/{workflow['dispute_id']}/sync-evidence"
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["razorpay_mode"], "connected")
        self.assertEqual(payload["evidence_sync_status"], "synced")
        self.assertEqual(payload["results"][0]["status"], "synced")
        self.assertEqual(
            payload["results"][0]["razorpay_document_id"], "doc_RealTestDocument123"
        )
        self.assertEqual(fake_provider.uploads, [("merchant-record.pdf", "application/pdf")])
        self.assertIsNone(payload["handoff"]["razorpay_dispute_id"])
        self.assertEqual(payload["handoff"]["prepared_evidence_count"], 1)
        self.assertEqual(payload["handoff"]["waiting_evidence_count"], 0)
        stored = self.client.get(f"/evidence/{workflow['dispute_id']}").json()["evidence"]
        self.assertEqual(stored[0]["evidence_id"], verified["evidence_id"])
        self.assertEqual(stored[0]["razorpay_sync_status"], "synced")
        self.assertEqual(stored[0]["razorpay_document_id"], "doc_RealTestDocument123")

    def test_sync_evidence_skips_already_synced_files(self):
        workflow = self.create_workflow()
        self.upload_and_verify(workflow["dispute_id"])

        class FakeProvider:
            def __init__(self):
                self.calls = 0

            def upload_document(self, *_):
                self.calls += 1
                return "doc_SyncOnly123"

        fake_provider = FakeProvider()
        environment = {
            "RAZORPAY_MODE": "connected",
            "RAZORPAY_KEY_ID": "key_test",
            "RAZORPAY_KEY_SECRET": "secret_test",
        }
        with patch.dict(os.environ, environment, clear=False), patch.object(
            main, "RazorpayClient", return_value=fake_provider
        ):
            first = self.client.post(f"/razorpay/{workflow['dispute_id']}/sync-evidence")
            second = self.client.post(f"/razorpay/{workflow['dispute_id']}/sync-evidence")

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(fake_provider.calls, 1)
        self.assertEqual(second.json()["results"][0]["status"], "already_synced")

    def test_sync_evidence_partial_failure_keeps_successful_document_id(self):
        workflow = self.create_workflow()
        first = self.upload_and_verify(workflow["dispute_id"])
        second = self.upload_and_verify(
            workflow["dispute_id"], "UPI authentication/authorization records"
        )

        class FakeProvider:
            def __init__(self):
                self.calls = 0

            def upload_document(self, *_):
                self.calls += 1
                if self.calls == 1:
                    return "doc_FirstSuccess123"
                raise razorpay_handoff.RazorpayAdapterError("Safe provider failure.")

        environment = {
            "RAZORPAY_MODE": "connected",
            "RAZORPAY_KEY_ID": "key_test",
            "RAZORPAY_KEY_SECRET": "secret_test",
        }
        with patch.dict(os.environ, environment, clear=False), patch.object(
            main, "RazorpayClient", return_value=FakeProvider()
        ):
            response = self.client.post(f"/razorpay/{workflow['dispute_id']}/sync-evidence")

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["evidence_sync_status"], "partial_failed")
        self.assertEqual(payload["results"][0]["status"], "synced")
        self.assertEqual(payload["results"][1]["status"], "failed")
        self.assertNotIn("doc_", payload["results"][1].get("error", ""))
        stored = {
            item["evidence_id"]: item
            for item in self.client.get(f"/evidence/{workflow['dispute_id']}").json()["evidence"]
        }
        self.assertEqual(
            stored[first["evidence_id"]]["razorpay_document_id"], "doc_FirstSuccess123"
        )
        self.assertEqual(stored[second["evidence_id"]]["razorpay_sync_status"], "failed")
        self.assertIsNone(stored[second["evidence_id"]]["razorpay_document_id"])

    def test_sync_evidence_connected_missing_credentials_fails_safely(self):
        workflow = self.create_workflow()
        self.upload_and_verify(workflow["dispute_id"])
        with patch.dict(
            os.environ,
            {
                "RAZORPAY_MODE": "connected",
                "RAZORPAY_KEY_ID": "",
                "RAZORPAY_KEY_SECRET": "",
            },
            clear=False,
        ):
            response = self.client.post(
                f"/razorpay/{workflow['dispute_id']}/sync-evidence"
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("backend credentials", response.json()["detail"])

    def test_sync_evidence_demo_makes_zero_provider_calls(self):
        workflow = self.create_workflow()
        self.upload_and_verify(workflow["dispute_id"])
        with patch.dict(os.environ, {"RAZORPAY_MODE": "demo"}, clear=False), patch.object(
            main, "RazorpayClient"
        ) as provider_client:
            response = self.client.post(
                f"/razorpay/{workflow['dispute_id']}/sync-evidence"
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        provider_client.assert_not_called()
        self.assertEqual(payload["evidence_sync_status"], "simulated_demo")
        self.assertTrue(payload["results"][0]["demo_document_reference"].startswith("demo_doc_"))
        self.assertIsNone(payload["results"][0]["razorpay_document_id"])
        self.assertIn("No data was sent to Razorpay", payload["message"])

    def test_document_adapter_uses_dispute_evidence_purpose(self):
        source_file = Path(self.temp_directory.name) / "document.jpg"
        source_file.write_bytes(b"image bytes")
        adapter = razorpay_handoff.RazorpayClient("key_test", "secret_test")
        captured = {}

        def capture_request(method, path, body=None, content_type=None):
            captured.update(
                method=method,
                path=path,
                body=body,
                content_type=content_type,
            )
            return {"id": "doc_AdapterTest123"}

        with patch.object(adapter, "_request_json", side_effect=capture_request):
            document_id = adapter.upload_document(
                source_file, "merchant-photo.jpg", "image/jpeg"
            )

        self.assertEqual(document_id, "doc_AdapterTest123")
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["path"], "/documents")
        self.assertIn("multipart/form-data", captured["content_type"])
        self.assertIn(
            b'Content-Disposition: form-data; name="purpose"\r\n\r\ndispute_evidence',
            captured["body"],
        )

    def test_valid_signed_webhook_creates_safe_preanalysis_workflow(self):
        response = self.send_signed_webhook(self.webhook_payload())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "created")
        workflow_id = response.json()["workflow_id"]

        incoming = self.client.get("/razorpay/incoming-disputes")
        self.assertEqual(incoming.status_code, 200, incoming.text)
        dispute = incoming.json()["disputes"][0]
        self.assertEqual(dispute["workflow_id"], workflow_id)
        self.assertEqual(dispute["razorpay_dispute_id"], "disp_test_webhook_001")
        self.assertEqual(dispute["internal_dispute_type"], "upi_unauthorized")
        self.assertTrue(dispute["additional_merchant_input_required"])
        self.assertEqual(dispute["form_prefill"]["order_value"], 4299.0)
        self.assertEqual(dispute["form_prefill"]["dispute_type"], "upi_unauthorized")

        stored = evidence_store.get_webhook_workflow(workflow_id)
        self.assertIsNone(stored["prediction_snapshot"])
        self.assertFalse(stored["analysis_ready"])
        self.assertEqual(
            stored["razorpay_handoff"]["razorpay_dispute_id"],
            "disp_test_webhook_001",
        )

    def test_invalid_or_missing_webhook_signature_creates_no_workflow(self):
        payload = self.webhook_payload()
        raw_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        with patch.dict(
            os.environ, {"RAZORPAY_WEBHOOK_SECRET": "webhook_test_secret"}, clear=False
        ):
            missing = self.client.post("/webhooks/razorpay", content=raw_body)
            invalid = self.client.post(
                "/webhooks/razorpay",
                content=raw_body,
                headers={"X-Razorpay-Signature": "not-a-valid-signature"},
            )

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(invalid.status_code, 401)
        self.assertEqual(self.client.get("/razorpay/incoming-disputes").json()["disputes"], [])

    def test_webhook_missing_secret_and_raw_body_mismatch_fail_safely(self):
        payload = self.webhook_payload()
        raw_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        signature = hmac.new(
            b"webhook_test_secret", raw_body, hashlib.sha256
        ).hexdigest()
        with patch.dict(os.environ, {"RAZORPAY_WEBHOOK_SECRET": ""}, clear=False):
            missing_secret = self.client.post(
                "/webhooks/razorpay",
                content=raw_body,
                headers={"X-Razorpay-Signature": signature},
            )
        with patch.dict(
            os.environ, {"RAZORPAY_WEBHOOK_SECRET": "webhook_test_secret"}, clear=False
        ):
            altered_body = self.client.post(
                "/webhooks/razorpay",
                content=raw_body + b" ",
                headers={"X-Razorpay-Signature": signature},
            )

        self.assertEqual(missing_secret.status_code, 503)
        self.assertEqual(altered_body.status_code, 401)
        self.assertEqual(self.client.get("/razorpay/incoming-disputes").json()["disputes"], [])

    def test_duplicate_event_and_dispute_id_reuse_existing_webhook_workflow(self):
        payload = self.webhook_payload()
        first = self.send_signed_webhook(payload, event_id="evt_duplicate_001")
        duplicate_event = self.send_signed_webhook(payload, event_id="evt_duplicate_001")
        same_dispute_new_event = self.send_signed_webhook(
            payload, event_id="evt_duplicate_002"
        )

        self.assertEqual(first.json()["status"], "created")
        self.assertEqual(duplicate_event.json()["status"], "duplicate_ignored")
        self.assertEqual(same_dispute_new_event.json()["status"], "updated")
        self.assertEqual(
            first.json()["workflow_id"], same_dispute_new_event.json()["workflow_id"]
        )
        self.assertEqual(len(self.client.get("/razorpay/incoming-disputes").json()["disputes"]), 1)

    def test_signed_unsupported_and_incomplete_webhooks_are_safely_ignored(self):
        unsupported = self.webhook_payload()
        unsupported["event"] = "payment.dispute.won"
        ignored = self.send_signed_webhook(unsupported, event_id="evt_ignored_001")

        incomplete = {
            "entity": "event",
            "event": "payment.dispute.created",
            "payload": {"dispute": {"entity": {"id": "disp_test_incomplete_001"}}},
        }
        accepted_incomplete = self.send_signed_webhook(incomplete, event_id="evt_ignored_002")

        self.assertEqual(ignored.status_code, 200)
        self.assertEqual(ignored.json()["status"], "ignored")
        self.assertEqual(accepted_incomplete.status_code, 200)
        self.assertEqual(accepted_incomplete.json()["status"], "created")
        incoming = self.client.get("/razorpay/incoming-disputes").json()["disputes"]
        self.assertEqual(len(incoming), 1)
        self.assertIsNone(incoming[0]["internal_dispute_type"])
        self.assertTrue(incoming[0]["additional_merchant_input_required"])

    def test_unknown_reason_is_not_force_mapped_and_test_source_is_labelled(self):
        response = self.send_signed_webhook(
            self.webhook_payload(
                dispute_id="disp_test_unknown_001",
                reason="excessive_fee",
                reason_code="processing_charge",
                method="upi",
                simulated=True,
            ),
            event_id="evt_test_source_001",
        )
        workflow_id = response.json()["workflow_id"]
        incoming = self.client.get(
            f"/razorpay/incoming-disputes/{workflow_id}"
        )
        self.assertEqual(incoming.status_code, 200, incoming.text)
        self.assertIsNone(incoming.json()["internal_dispute_type"])
        self.assertEqual(incoming.json()["source_label"], "Razorpay Webhook Test")
        self.assertTrue(incoming.json()["is_simulated"])

    def test_webhook_case_uses_the_existing_prediction_workflow_after_input(self):
        webhook = self.send_signed_webhook(
            self.webhook_payload(dispute_id="disp_test_converge_001"),
            event_id="evt_converge_001",
        )
        workflow_id = webhook.json()["workflow_id"]
        prediction = self.prediction_payload(start_evidence_workflow=True)
        prediction["existing_workflow_id"] = workflow_id
        response = self.client.post("/predict", json=prediction)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["dispute_id"], workflow_id)
        stored = evidence_store.get_webhook_workflow(workflow_id)
        self.assertTrue(stored["analysis_ready"])
        self.assertIsNotNone(stored["prediction_snapshot"])
        self.assertEqual(
            stored["razorpay_handoff"]["razorpay_dispute_id"],
            "disp_test_converge_001",
        )
