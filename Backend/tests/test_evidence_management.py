import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import evidence_store
import main


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
