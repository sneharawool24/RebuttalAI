from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, FiniteFloat
from datetime import datetime, timezone
from typing import Literal
import joblib
import json
import os
import pandas as pd
import re
from dotenv import load_dotenv
from google import genai

try:
    from .evidence_store import (
        EvidenceStoreError,
        add_evidence,
        attach_prediction_snapshot_to_webhook_workflow,
        create_dispute_snapshot,
        create_or_update_webhook_workflow,
        delete_evidence,
        get_evidence_file,
        get_webhook_workflow,
        get_dispute_snapshot,
        list_evidence,
        list_webhook_workflows,
        mark_rebuttal_ready,
        public_metadata,
        update_evidence_razorpay_sync,
        update_business_risk_settings,
        update_razorpay_handoff,
        verify_evidence,
    )
    from .razorpay_handoff import (
        RazorpayAdapterError,
        RazorpayClient,
        contest_summary,
        demo_dispute_reference,
        demo_document_reference,
        evidence_mapping,
    )
    from .razorpay_webhooks import (
        ACTIVE_DISPUTE_EVENT,
        extract_dispute_metadata,
        public_incoming_dispute,
        signature_is_valid,
    )
    from .business_risk import (
        CONTEST_CONSIDERATION_RECOMMENDATIONS,
        calculate_business_cost_context,
        evaluate_business_risk,
    )
except ImportError:
    from evidence_store import (
        EvidenceStoreError,
        add_evidence,
        attach_prediction_snapshot_to_webhook_workflow,
        create_dispute_snapshot,
        create_or_update_webhook_workflow,
        delete_evidence,
        get_evidence_file,
        get_webhook_workflow,
        get_dispute_snapshot,
        list_evidence,
        list_webhook_workflows,
        mark_rebuttal_ready,
        public_metadata,
        update_evidence_razorpay_sync,
        update_business_risk_settings,
        update_razorpay_handoff,
        verify_evidence,
    )
    from razorpay_handoff import (
        RazorpayAdapterError,
        RazorpayClient,
        contest_summary,
        demo_dispute_reference,
        demo_document_reference,
        evidence_mapping,
    )
    from razorpay_webhooks import (
        ACTIVE_DISPUTE_EVENT,
        extract_dispute_metadata,
        public_incoming_dispute,
        signature_is_valid,
    )
    from business_risk import (
        CONTEST_CONSIDERATION_RECOMMENDATIONS,
        calculate_business_cost_context,
        evaluate_business_risk,
    )


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(
    title="RebuttalAI API",
    version="0.1.0"
)

# Allows the Vite development server to call the API without broadening access
# beyond the local development origins used by this MVP.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ML",
    "models"
)

upi_model = joblib.load(
    os.path.join(MODEL_DIR, "upi_rf.pkl")
)

netbanking_model = joblib.load(
    os.path.join(MODEL_DIR, "netbanking_rf.pkl")
)

non_delivery_model = joblib.load(
    os.path.join(MODEL_DIR, "non_delivery_lr.pkl")
)


# ============================================================
# INPUT SCHEMA - DISPUTE PREDICTION
# ============================================================

class DisputeInput(BaseModel):

    # --------------------------------------------------------
    # Dispute information
    # --------------------------------------------------------

    dispute_type: str

    # --------------------------------------------------------
    # Transaction information
    # --------------------------------------------------------

    order_value: float
    days_since_transaction: int

    # Merchant policy setting. This does not alter the trained ML model.
    false_positive_sensitivity: Literal["low", "medium", "high"] = "medium"

    # Merchant-entered business estimates only. Never included in ML inputs.
    contest_handling_cost: FiniteFloat = Field(default=0, ge=0)
    staff_operational_cost: FiniteFloat = Field(default=0, ge=0)

    # --------------------------------------------------------
    # Authentication / fraud-related features
    # --------------------------------------------------------

    device_ip_match_history: float | None = None
    auth_flow_type: str | None = None

    # --------------------------------------------------------
    # Customer history
    # --------------------------------------------------------

    customer_account_age_days: int
    customer_past_order_count: int
    customer_past_dispute_count: int

    # --------------------------------------------------------
    # Delivery-related features
    # --------------------------------------------------------

    delivery_confirmed: float | None = None
    tracking_available: float | None = None

    # --------------------------------------------------------
    # Merchant evidence indicators
    # --------------------------------------------------------

    merchant_comm_log_exists: int
    refund_already_issued: int

    # --------------------------------------------------------
    # Evidence actually available to the merchant
    # --------------------------------------------------------

    evidence_available: list[str] = []

    # When true, the backend creates an evidence-management workflow ID.
    # Omitted by legacy callers, preserving their existing contract.
    start_evidence_workflow: bool = False

    # Used only when a merchant opens a pre-analysis Razorpay webhook case.
    # It lets the existing prediction result attach to that same workflow.
    existing_workflow_id: str | None = None


class BusinessRiskPolicyInput(BaseModel):
    """Policy-only update for an existing workflow; does not invoke ML."""

    false_positive_sensitivity: Literal["low", "medium", "high"] | None = None
    contest_handling_cost: FiniteFloat | None = Field(default=None, ge=0)
    staff_operational_cost: FiniteFloat | None = Field(default=None, ge=0)


# ============================================================
# INPUT SCHEMA - REBUTTAL GENERATION
# ============================================================

class RebuttalInput(BaseModel):

    # --------------------------------------------------------
    # Dispute information
    # --------------------------------------------------------

    dispute_type: str
    order_value: float
    days_since_transaction: int

    # --------------------------------------------------------
    # ML recommendation
    # --------------------------------------------------------

    base_decision: str
    fight_score: float

    # --------------------------------------------------------
    # FINAL EVIDENCE-BASED RECOMMENDATION
    # --------------------------------------------------------

    final_recommendation: str
    recommendation_reason: str

    # --------------------------------------------------------
    # Evidence information
    # --------------------------------------------------------

    evidence_available: list[str] = []
    evidence_missing: list[str] = []

    # --------------------------------------------------------
    # Optional merchant-provided context
    # --------------------------------------------------------

    merchant_context: str | None = None

    # When supplied, stored server-side state replaces caller-supplied
    # recommendation and evidence fields before Gemini is called.
    dispute_id: str | None = None


class VerifiedEvidenceReference(BaseModel):

    evidence_category: str
    original_filename: str
    uploaded_at: str
    verified_at: str | None = None


class AuthoritativeRebuttalInput(RebuttalInput):

    verified_evidence_metadata: list[VerifiedEvidenceReference] = Field(
        default_factory=list
    )


class RazorpayDisputeFetchInput(BaseModel):
    """Import an external Razorpay dispute without replacing ML workflow data."""

    workflow_id: str
    razorpay_dispute_id: str


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "RebuttalAI API is running!"
    }


# ============================================================
# RAZORPAY WEBHOOKS
# ============================================================


@app.post("/webhooks/razorpay")
async def receive_razorpay_webhook(request: Request):
    """Verify Razorpay's raw-body signature before trusting webhook JSON."""

    webhook_secret = os.getenv("RAZORPAY_WEBHOOK_SECRET")
    if not webhook_secret:
        raise HTTPException(
            status_code=503,
            detail="Razorpay webhook verification is not configured.",
        )

    raw_body = await request.body()
    signature = request.headers.get("x-razorpay-signature")
    if not signature:
        raise HTTPException(status_code=401, detail="Razorpay webhook signature is required.")
    if not signature_is_valid(raw_body, signature, webhook_secret):
        raise HTTPException(status_code=401, detail="Razorpay webhook signature is invalid.")

    try:
        event_payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="Razorpay webhook payload is invalid.")
    if not isinstance(event_payload, dict):
        raise HTTPException(status_code=400, detail="Razorpay webhook payload is invalid.")

    event_name = event_payload.get("event")
    if event_name != ACTIVE_DISPUTE_EVENT:
        return {"status": "ignored", "event": event_name or "unknown"}

    metadata = extract_dispute_metadata(event_payload)
    if not metadata:
        return {
            "status": "ignored",
            "event": event_name,
            "reason": "missing_dispute_metadata",
        }

    event_id = request.headers.get("x-razorpay-event-id")
    try:
        workflow_id, status = create_or_update_webhook_workflow(event_id, metadata)
    except EvidenceStoreError as error:
        raise evidence_error_to_http(error)

    return {
        "status": status,
        "event": event_name,
        "workflow_id": workflow_id,
    }


@app.get("/razorpay/incoming-disputes")
def list_incoming_razorpay_disputes():
    """List only normalized, non-sensitive webhook-created cases."""

    try:
        return {
            "disputes": [
                public_incoming_dispute(workflow)
                for workflow in list_webhook_workflows()
            ]
        }
    except EvidenceStoreError as error:
        raise evidence_error_to_http(error)


@app.get("/razorpay/incoming-disputes/{workflow_id}")
def get_incoming_razorpay_dispute(workflow_id: str):
    """Load one safe pre-analysis webhook case into the existing form."""

    try:
        return public_incoming_dispute(get_webhook_workflow(workflow_id))
    except EvidenceStoreError as error:
        raise evidence_error_to_http(error)


# ============================================================
# MODEL STATUS ENDPOINT
# ============================================================

@app.get("/models")
def models():

    return {
        "upi": "Random Forest loaded",
        "netbanking": "Random Forest loaded",
        "non_delivery": "Logistic Regression loaded"
    }


# ============================================================
# EVIDENCE REQUIREMENTS
# ============================================================

def get_evidence_requirements(dispute_type):

    # --------------------------------------------------------
    # UPI UNAUTHORIZED
    # --------------------------------------------------------

    if dispute_type == "upi_unauthorized":

        return [
            "UPI transaction details",
            "UPI authentication/authorization records",
            "Transaction timestamp",
            "Customer communication",
            "Merchant transaction/order records",
            "Explanation of transaction authorization"
        ]

    # --------------------------------------------------------
    # NETBANKING UNAUTHORIZED
    # --------------------------------------------------------

    elif dispute_type == "netbanking_unauthorized":

        return [
            "Netbanking transaction details",
            "Authentication/authorization records",
            "Transaction timestamp",
            "Customer communication",
            "Merchant transaction/order records",
            "Explanation of transaction authorization"
        ]

    # --------------------------------------------------------
    # NON-DELIVERY
    # --------------------------------------------------------

    elif dispute_type == "non_delivery":

        return [
            "Order/invoice details",
            "Shipping proof",
            "Shipment tracking information",
            "Delivery confirmation/proof of delivery",
            "Customer communication",
            "Proof of service",
            "Refund/cancellation records",
            "Refund/cancellation policy"
        ]

    return []


# ============================================================
# CRITICAL EVIDENCE
# ============================================================

def get_critical_evidence(dispute_type):

    # --------------------------------------------------------
    # UPI UNAUTHORIZED
    # --------------------------------------------------------

    if dispute_type == "upi_unauthorized":

        return [
            "UPI transaction details",
            "UPI authentication/authorization records"
        ]

    # --------------------------------------------------------
    # NETBANKING UNAUTHORIZED
    # --------------------------------------------------------

    elif dispute_type == "netbanking_unauthorized":

        return [
            "Netbanking transaction details",
            "Authentication/authorization records"
        ]

    # --------------------------------------------------------
    # NON-DELIVERY
    # --------------------------------------------------------

    elif dispute_type == "non_delivery":

        return [
            "Shipping proof",
            "Shipment tracking information",
            "Delivery confirmation/proof of delivery"
        ]

    return []


# ============================================================
# FINAL RECOMMENDATION ENGINE
# ============================================================

def determine_final_recommendation(
    base_decision,
    fight_score,
    evidence_required,
    evidence_available,
    critical_evidence,
    order_value=None,
):

    # --------------------------------------------------------
    # FIND MISSING EVIDENCE
    # --------------------------------------------------------

    # Only known, unique requirements may affect completeness. This prevents
    # duplicate or unrelated values from inflating evidence coverage.
    available_categories = {
        evidence
        for evidence in evidence_available
        if evidence in evidence_required
    }

    evidence_missing = [
        evidence
        for evidence in evidence_required
        if evidence not in available_categories
    ]

    critical_evidence_missing = [
        evidence
        for evidence in critical_evidence
        if evidence not in available_categories
    ]

    # --------------------------------------------------------
    # CALCULATE EVIDENCE COVERAGE
    # --------------------------------------------------------

    if len(evidence_required) > 0:

        evidence_coverage = (
            len(available_categories)
            / len(evidence_required)
        )

    else:

        evidence_coverage = 0.0

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    # The model class is retained as a transparent ML output, but it must not
    # bypass the automatically derived business-cost policy. The Fight Score
    # is only an assessment signal, not a win-probability estimate.
    business_risk = evaluate_business_risk(
        fight_score=fight_score,
        order_value=order_value,
        evidence_required=evidence_required,
        critical_evidence=critical_evidence,
    )

    # CASE 1: the Fight Score does not meet the business-risk threshold.
    if not business_risk["passes_cost_threshold"]:

        final_recommendation = "Don't Fight"

        recommendation_reason = (
            f"{business_risk['business_risk_reason']} "
            "RebuttalAI does not recommend a dispute fight under the current policy."
        )

    # CASE 2: the policy threshold is met but critical evidence is missing.

    elif len(critical_evidence_missing) > 0:

        final_recommendation = "Review / Collect Evidence"

        recommendation_reason = (
            f"{business_risk['business_risk_reason']} Critical supporting evidence is missing. "
            "Collect the missing evidence before deciding "
            "whether to contest."
        )

    # CASE 3: the policy threshold and critical evidence requirements are met.

    else:

        final_recommendation = "Fight"

        recommendation_reason = (
            f"{business_risk['business_risk_reason']} "
            "All required critical evidence is verified."
        )

    # Costs are relevant only while a contest is still under consideration.
    # This is derived from canonical backend outcomes, not UI presentation text.
    business_risk["show_false_positive_cost"] = (
        final_recommendation in CONTEST_CONSIDERATION_RECOMMENDATIONS
    )

    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {

        "evidence_missing": evidence_missing,

        "critical_evidence_missing": (
            critical_evidence_missing
        ),

        "evidence_coverage": round(
            evidence_coverage,
            2
        ),

        "final_recommendation": (
            final_recommendation
        ),

        "recommendation_reason": recommendation_reason,

        # Deterministic prototype business policy metadata.
        **business_risk,
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.get("/business-risk/estimate")
def preview_business_risk_estimate(dispute_type: str, order_value: FiniteFloat):
    """Return a read-only workload estimate before the merchant analyzes."""

    if dispute_type not in {
        "upi_unauthorized",
        "netbanking_unauthorized",
        "non_delivery",
    }:
        raise HTTPException(status_code=422, detail="Unsupported dispute type.")
    if order_value < 0:
        raise HTTPException(status_code=422, detail="Order value must be non-negative.")

    evidence_required = get_evidence_requirements(dispute_type)
    critical_evidence = get_critical_evidence(dispute_type)
    return {
        "dispute_type": dispute_type,
        "order_value": order_value,
        **calculate_business_cost_context(
            order_value,
            evidence_required,
            critical_evidence,
        ),
    }


@app.post("/predict")
def predict_dispute(data: DisputeInput):

    # --------------------------------------------------------
    # PREPARE MODEL INPUT
    # --------------------------------------------------------

    input_data = {

        "order_value": data.order_value,

        "days_since_transaction":
            data.days_since_transaction,

        "device_ip_match_history":
            data.device_ip_match_history,

        "auth_flow_type":
            data.auth_flow_type,

        "customer_account_age_days":
            data.customer_account_age_days,

        "customer_past_order_count":
            data.customer_past_order_count,

        "customer_past_dispute_count":
            data.customer_past_dispute_count,

        "delivery_confirmed":
            data.delivery_confirmed,

        "tracking_available":
            data.tracking_available,

        "merchant_comm_log_exists":
            data.merchant_comm_log_exists,

        "refund_already_issued":
            data.refund_already_issued
    }

    # --------------------------------------------------------
    # CONVERT INPUT INTO DATAFRAME
    # --------------------------------------------------------

    df = pd.DataFrame([input_data])

    # --------------------------------------------------------
    # SELECT MODEL
    # --------------------------------------------------------

    if data.dispute_type == "upi_unauthorized":

        model = upi_model

    elif data.dispute_type == "netbanking_unauthorized":

        model = netbanking_model

    elif data.dispute_type == "non_delivery":

        model = non_delivery_model

    else:

        return {
            "error": "Unsupported dispute type"
        }

    # --------------------------------------------------------
    # ML PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(df)[0]

    fight_score = model.predict_proba(df)[0][1]

    # --------------------------------------------------------
    # CONVERT PREDICTION TO HUMAN-READABLE DECISION
    # --------------------------------------------------------

    base_decision = (
        "Fight"
        if prediction == 1
        else "Don't Fight"
    )

    # --------------------------------------------------------
    # GET EVIDENCE REQUIREMENTS
    # --------------------------------------------------------

    evidence_required = get_evidence_requirements(
        data.dispute_type
    )

    critical_evidence = get_critical_evidence(
        data.dispute_type
    )

    # A new evidence workflow starts with no verified evidence. Legacy calls
    # retain their existing caller-provided evidence behavior.
    evidence_available = (
        []
        if data.start_evidence_workflow
        else data.evidence_available
    )

    # --------------------------------------------------------
    # DETERMINE FINAL RECOMMENDATION
    # --------------------------------------------------------

    recommendation = determine_final_recommendation(

        base_decision=base_decision,

        fight_score=fight_score,

        evidence_required=evidence_required,

        evidence_available=evidence_available,

        critical_evidence=critical_evidence,

        order_value=data.order_value,
    )

    # --------------------------------------------------------
    # RETURN COMPLETE RESPONSE
    # --------------------------------------------------------

    result = {

        "dispute_type":
            data.dispute_type,

        "order_value": data.order_value,

        # ----------------------------------------------------
        # ML result
        # ----------------------------------------------------

        "base_decision":
            base_decision,

        "fight_score":
            round(
                float(fight_score),
                4
            ),

        "false_positive_sensitivity": recommendation["false_positive_sensitivity"],

        "decision_threshold": recommendation["decision_threshold"],

        "passes_cost_threshold": recommendation["passes_cost_threshold"],

        "business_risk_status": recommendation["business_risk_status"],

        "business_risk_reason": recommendation["business_risk_reason"],

        "business_risk_level": recommendation["business_risk_level"],

        "estimated_contest_cost": recommendation["estimated_contest_cost"],

        "contest_cost_breakdown": recommendation["contest_cost_breakdown"],

        "cost_ratio": recommendation["cost_ratio"],

        "estimated_false_positive_cost": recommendation["estimated_false_positive_cost"],

        "show_false_positive_cost": recommendation["show_false_positive_cost"],

        # ----------------------------------------------------
        # Evidence information
        # ----------------------------------------------------

        "evidence_required":
            evidence_required,

        "critical_evidence":
            critical_evidence,

        "evidence_available":
            evidence_available,

        "evidence_missing":
            recommendation[
                "evidence_missing"
            ],

        "critical_evidence_missing":
            recommendation[
                "critical_evidence_missing"
            ],

        "evidence_coverage":
            recommendation[
                "evidence_coverage"
            ],

        # ----------------------------------------------------
        # Final recommendation
        # ----------------------------------------------------

        "final_recommendation":
            recommendation[
                "final_recommendation"
            ],

        "recommendation_reason":
            recommendation[
                "recommendation_reason"
            ]
    }

    if data.start_evidence_workflow:
        snapshot = {
            "dispute_type": data.dispute_type,
            "order_value": data.order_value,
            "days_since_transaction": data.days_since_transaction,
            "model_input": input_data,
            "base_decision": base_decision,
            "fight_score": round(float(fight_score), 4),
            "false_positive_sensitivity": recommendation["false_positive_sensitivity"],
            "decision_threshold": recommendation["decision_threshold"],
            "passes_cost_threshold": recommendation["passes_cost_threshold"],
            "business_risk_reason": recommendation["business_risk_reason"],
            "business_risk_level": recommendation["business_risk_level"],
            "estimated_contest_cost": recommendation["estimated_contest_cost"],
            "contest_cost_breakdown": recommendation["contest_cost_breakdown"],
            "cost_ratio": recommendation["cost_ratio"],
            "estimated_false_positive_cost": recommendation["estimated_false_positive_cost"],
            "evidence_required": evidence_required,
            "critical_evidence": critical_evidence,
        }
        try:
            if data.existing_workflow_id:
                attach_prediction_snapshot_to_webhook_workflow(
                    data.existing_workflow_id, snapshot
                )
                result["dispute_id"] = data.existing_workflow_id
            else:
                result["dispute_id"] = create_dispute_snapshot(snapshot)
        except EvidenceStoreError as error:
            raise evidence_error_to_http(error)

    return result


# ============================================================
# LLM REBUTTAL GENERATION
# ============================================================

class RebuttalGenerationError(Exception):
    """A safe, merchant-facing error for a Gemini generation failure."""


def merchant_facing_draft_or_fallback(draft: str) -> str:
    """Prevent internal decision-support language from reaching the draft UI."""

    forbidden_patterns = (
        r"\bfight\b",
        r"\bdon't fight\b",
        r"\bfight score\b",
        r"\bmodel\b",
        r"\bevidence engine\b",
        r"\bsystem flagged\b",
        r"\bfinal recommendation\b",
        r"\breview / collect evidence\b",
        r"(?m)^recommendation:\s*$",
        r"(?m)^evidence still needed:\s*$",
        r"(?m)^merchant review note:\s*$",
    )

    if any(re.search(pattern, draft, flags=re.IGNORECASE) for pattern in forbidden_patterns):
        return (
            "Unable to prepare a suitable draft. Please try again."
        )

    return draft


def generate_rebuttal(data: RebuttalInput):

    # --------------------------------------------------------
    # FORMAT AVAILABLE EVIDENCE
    # --------------------------------------------------------

    verified_evidence_metadata = getattr(
        data,
        "verified_evidence_metadata",
        [],
    )

    if verified_evidence_metadata:

        evidence_text = "\n".join(
            f"- {evidence.evidence_category} - {evidence.original_filename}"
            for evidence in verified_evidence_metadata
        )

    elif data.evidence_available:

        evidence_text = "\n".join(
            f"- {evidence}"
            for evidence in data.evidence_available
        )

    else:

        evidence_text = "- No evidence provided."

    # --------------------------------------------------------
    # FORMAT MISSING EVIDENCE
    # --------------------------------------------------------

    if data.evidence_missing:

        missing_evidence_text = "\n".join(
            f"- {evidence}"
            for evidence in data.evidence_missing
        )

    else:

        missing_evidence_text = "- None."

    # --------------------------------------------------------
    # MERCHANT CONTEXT
    # --------------------------------------------------------

    merchant_context = (
        data.merchant_context
        if data.merchant_context
        else "No additional merchant context provided."
    )

    # --------------------------------------------------------
    # LLM PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are RebuttalAI, an AI assistant helping merchants
respond to payment disputes.

Your task is to draft a professional, factual dispute
rebuttal for HUMAN MERCHANT REVIEW.

============================================================
STRICT SAFETY AND ACCURACY RULES
============================================================

1. NEVER invent evidence.

2. NEVER invent:
   - transaction IDs
   - tracking numbers
   - dates
   - authentication events
   - OTP events
   - customer statements
   - delivery events
   - refund events
   - order details
   - payment details
   - names
   - amounts other than the supplied order value
   - any other unsupported fact

3. ONLY use facts explicitly supplied in this request.

4. Only evidence listed under "VERIFIED EVIDENCE ACTUALLY AVAILABLE"
   is merchant-verified evidence and can be referenced as available.

5. Evidence listed under "EVIDENCE CURRENTLY MISSING"
   MUST NOT be described as if it exists.

6. Do not claim that the merchant will definitely win.

7. Do not make legal conclusions.

8. Do not pretend that the ML Fight Score is a probability
   of winning the dispute.

9. The rebuttal is a DRAFT only.

10. The merchant must review and verify the draft before
    submitting it.

11. If there is insufficient evidence to support a claim,
    clearly state that additional evidence is needed.

12. Do not automatically recommend submission merely because
    the ML model recommends fighting.

13. The FINAL RECOMMENDATION FROM THE EVIDENCE ENGINE
    MUST BE RESPECTED.

14. The final recommendation is more important than the
    base ML decision when generating the merchant-facing
    response.

15. If the final recommendation is:
    "Review / Collect Evidence"

    DO NOT write the rebuttal as if the merchant is ready
    to contest the dispute.

    Instead, explain that the model identified the case as
    potentially suitable for contesting, but the merchant
    should first collect and verify the missing evidence.

16. If the final recommendation is:
    "Don't Fight"

    Do not encourage the merchant to contest the dispute.

17. If the final recommendation is:
    "Fight"

    The response may present the merchant's position using
    the available evidence, but it must still remain factual
    and must not guarantee success.

18. Merchant-provided context is an unverified merchant statement,
    not evidence. Do not describe it as verified or convert it into
    a factual claim without supporting verified evidence.

19. Do not cite UPI guidelines, payment-network rules, merchant
    policies, legal standards, or any authority unless that exact
    source is supplied as verified evidence. Use neutral factual
    wording instead of unsupported authoritative claims.

20. The model result, Fight Score, final recommendation, evidence engine,
    recommendation labels, and missing-evidence workflow are INTERNAL
    merchant decision support. NEVER mention them in the external rebuttal.

21. Merchant verification means the merchant confirmed that an uploaded
    file belongs to its selected evidence category. It does NOT independently
    verify every fact inside that file. Do not claim independent verification.

22. Do not infer a document's contents, successful processing, customer
    authorization, delivery, identity, or provider confirmation from an
    evidence category or filename alone.

23. Merchant-provided context remains an unverified statement. It may be
    used only when clearly framed as a merchant statement, never as verified
    evidence or an established fact.

24. Keep the rebuttal concise and non-repetitive. Do not restate the same
    authorization, transaction, or evidence claim in multiple paragraphs.

============================================================
DISPUTE INFORMATION
============================================================

Dispute type:
{data.dispute_type}

Order value:
₹{data.order_value}

Days since transaction:
{data.days_since_transaction}

============================================================
MODEL INFORMATION
============================================================

Base model decision:
{data.base_decision}

Model Fight Score:
{data.fight_score}

IMPORTANT:
The Fight Score represents the model's confidence in the
"Fight" class based on the available dispute signals.

It is NOT:
- a probability of winning
- a probability of recovering the money
- a guarantee of success

============================================================
FINAL EVIDENCE ENGINE DECISION
============================================================

Final recommendation:
{data.final_recommendation}

Recommendation reason:
{data.recommendation_reason}

IMPORTANT:
You MUST respect this final recommendation.

The evidence engine combines the ML recommendation with
evidence availability.

============================================================
VERIFIED EVIDENCE ACTUALLY AVAILABLE
============================================================

{evidence_text}

============================================================
EVIDENCE CURRENTLY MISSING
============================================================

{missing_evidence_text}

============================================================
MERCHANT-PROVIDED CONTEXT (UNVERIFIED)
============================================================

{merchant_context}

============================================================
TASK
============================================================

Write a concise, professional merchant-facing dispute response draft.

The external draft must contain NO internal decision-support language,
including: Fight, Don't Fight, Fight Score, model, evidence engine,
recommendation labels, internal rules, or missing-evidence workflow.

Use EXACTLY this structure and no other sections:

Subject:
A short professional subject line.

Rebuttal:
Write up to four short paragraphs:
1. Identify the dispute and supplied transaction information.
2. State the merchant's position only where supplied trusted case facts or
   verified evidence supports it.
3. Reference the strongest verified evidence without inferring unseen file
   contents from its category or filename.
4. Request review or resolution without guaranteeing an outcome.

Evidence Referenced:
Provide a bullet list containing ONLY verified evidence in this format:
- Evidence category — original filename

If no evidence is available, write:
- None provided

FINAL REMINDER:
Never fabricate evidence.
Never convert missing evidence into existing evidence.
Never guarantee a successful dispute.
Never treat Fight Score as a win probability.
Never infer facts from evidence categories or filenames.
Keep internal merchant review information out of the external draft.
"""


    # --------------------------------------------------------
    # CALL GEMINI
    # --------------------------------------------------------

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
    except Exception as error:
        raise RebuttalGenerationError(
            "Unable to generate a rebuttal draft right now. Please try again."
        ) from error

    # --------------------------------------------------------
    # SAFETY CHECK FOR EMPTY RESPONSE
    # --------------------------------------------------------

    if not response.text:

        return (
            "Unable to generate a rebuttal draft. "
            "Please review the dispute information and "
            "try again."
        )

    # --------------------------------------------------------
    # RETURN GENERATED TEXT
    # --------------------------------------------------------

    return merchant_facing_draft_or_fallback(response.text)


# ============================================================
# REBUTTAL GENERATION ENDPOINT
# ============================================================

@app.post("/generate-rebuttal")
def generate_rebuttal_endpoint(
    data: RebuttalInput
):

    generation_data: RebuttalInput = data

    if data.dispute_id:
        try:
            generation_data = authoritative_rebuttal_input(data)
        except EvidenceStoreError as error:
            raise evidence_error_to_http(error)

    try:
        rebuttal = generate_rebuttal(generation_data)
    except RebuttalGenerationError as error:
        raise HTTPException(status_code=502, detail=str(error))

    result = {

        "dispute_type":
            generation_data.dispute_type,

        "base_decision":
            generation_data.base_decision,

        "fight_score":
            generation_data.fight_score,

        "final_recommendation":
            generation_data.final_recommendation,

        "recommendation_reason":
            generation_data.recommendation_reason,

        "rebuttal":
            rebuttal
    }

    if generation_data.dispute_id:
        result["dispute_id"] = generation_data.dispute_id
        try:
            mark_rebuttal_ready(generation_data.dispute_id)
        except EvidenceStoreError:
            # A generated draft remains available even if the optional handoff
            # readiness marker cannot be persisted at that moment.
            pass

    return result

# ============================================================
# EVIDENCE WORKFLOW HELPERS
# ============================================================


MAX_EVIDENCE_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_UPLOAD_TYPES = {
    ".pdf": {
        "content_types": {"application/pdf"},
        "signature": b"%PDF-",
    },
    ".png": {
        "content_types": {"image/png"},
        "signature": b"\x89PNG\r\n\x1a\n",
    },
    ".jpg": {
        "content_types": {"image/jpeg", "image/pjpeg"},
        "signature": b"\xff\xd8\xff",
    },
    ".jpeg": {
        "content_types": {"image/jpeg", "image/pjpeg"},
        "signature": b"\xff\xd8\xff",
    },
}


def workflow_state(dispute_id: str):
    """Recalculate evidence state from the stored prediction and verified files."""

    dispute = get_dispute_snapshot(dispute_id)
    snapshot = dispute["prediction_snapshot"]
    evidence_records = list_evidence(dispute_id)

    verified_categories = {
        evidence["evidence_category"]
        for evidence in evidence_records
        if evidence.get("status") == "Verified"
    }

    # Return categories in requirement order, rather than storage order.
    evidence_available = [
        category
        for category in snapshot["evidence_required"]
        if category in verified_categories
    ]

    recommendation = determine_final_recommendation(
        base_decision=snapshot["base_decision"],
        fight_score=snapshot["fight_score"],
        evidence_required=snapshot["evidence_required"],
        evidence_available=evidence_available,
        critical_evidence=snapshot["critical_evidence"],
        order_value=snapshot.get("order_value"),
    )

    result = {
        "dispute_id": dispute_id,
        "dispute_type": snapshot["dispute_type"],
        "order_value": snapshot.get("order_value", 0),
        "base_decision": snapshot["base_decision"],
        "fight_score": snapshot["fight_score"],
        "false_positive_sensitivity": recommendation["false_positive_sensitivity"],
        "decision_threshold": recommendation["decision_threshold"],
        "passes_cost_threshold": recommendation["passes_cost_threshold"],
        "business_risk_status": recommendation["business_risk_status"],
        "business_risk_reason": recommendation["business_risk_reason"],
        "business_risk_level": recommendation["business_risk_level"],
        "estimated_contest_cost": recommendation["estimated_contest_cost"],
        "contest_cost_breakdown": recommendation["contest_cost_breakdown"],
        "cost_ratio": recommendation["cost_ratio"],
        "estimated_false_positive_cost": recommendation["estimated_false_positive_cost"],
        "show_false_positive_cost": recommendation["show_false_positive_cost"],
        "evidence_required": snapshot["evidence_required"],
        "critical_evidence": snapshot["critical_evidence"],
        "evidence_available": evidence_available,
        "evidence_missing": recommendation["evidence_missing"],
        "critical_evidence_missing": recommendation[
            "critical_evidence_missing"
        ],
        "evidence_coverage": recommendation["evidence_coverage"],
        "final_recommendation": recommendation["final_recommendation"],
        "recommendation_reason": recommendation["recommendation_reason"],
    }

    return result


def evidence_error_to_http(error: EvidenceStoreError) -> HTTPException:
    message = str(error)
    status_code = 404 if "not found" in message.lower() else 503
    return HTTPException(status_code=status_code, detail=message)


def normalise_original_filename(filename: str | None) -> str:
    """Keep a display name, never a path supplied by the browser."""

    if not filename:
        return ""
    return filename.replace("\\", "/").split("/")[-1]


def validate_upload(
    filename: str,
    content_type: str | None,
    file_bytes: bytes,
) -> str:
    """Validate extension, declared MIME type, and magic bytes before storage."""

    extension = os.path.splitext(filename)[1].lower()
    upload_type = ALLOWED_UPLOAD_TYPES.get(extension)
    if not upload_type:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, PNG, JPG, and JPEG evidence files are supported.",
        )

    if (content_type or "").lower() not in upload_type["content_types"]:
        raise HTTPException(
            status_code=400,
            detail="The file content type does not match its allowed extension.",
        )

    if not file_bytes.startswith(upload_type["signature"]):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file does not match the selected file type.",
        )

    return extension


def authoritative_rebuttal_input(data: RebuttalInput) -> AuthoritativeRebuttalInput:
    """Build Gemini input from persisted state, not browser-supplied decisions."""

    if not data.dispute_id:
        raise ValueError("A dispute ID is required for an authoritative workflow.")

    dispute = get_dispute_snapshot(data.dispute_id)
    snapshot = dispute["prediction_snapshot"]
    state = workflow_state(data.dispute_id)
    verified_metadata = [
        VerifiedEvidenceReference(
            evidence_category=evidence["evidence_category"],
            original_filename=evidence["original_filename"],
            uploaded_at=evidence["uploaded_at"],
            verified_at=evidence.get("verified_at"),
        )
        for evidence in list_evidence(data.dispute_id)
        if evidence.get("status") == "Verified"
    ]

    return AuthoritativeRebuttalInput(
        dispute_id=data.dispute_id,
        dispute_type=snapshot["dispute_type"],
        order_value=snapshot["order_value"],
        days_since_transaction=snapshot["days_since_transaction"],
        base_decision=state["base_decision"],
        fight_score=state["fight_score"],
        final_recommendation=state["final_recommendation"],
        recommendation_reason=state["recommendation_reason"],
        evidence_available=state["evidence_available"],
        evidence_missing=state["evidence_missing"],
        merchant_context=data.merchant_context,
        verified_evidence_metadata=verified_metadata,
    )


# ============================================================
# RAZORPAY HANDOFF HELPERS
# ============================================================


RAZORPAY_DISPUTE_ID_PATTERN = re.compile(r"disp_[A-Za-z0-9]{8,64}$")


def razorpay_mode() -> str:
    """Read only a non-secret mode flag from backend environment variables."""

    configured_mode = os.getenv("RAZORPAY_MODE", "demo").strip().lower()
    if configured_mode not in {"demo", "connected"}:
        raise HTTPException(
            status_code=503,
            detail="Razorpay mode is not configured correctly. Use demo or connected.",
        )
    return configured_mode


def connected_razorpay_client() -> RazorpayClient:
    """Construct the provider adapter only when backend credentials are present."""

    key_id = os.getenv("RAZORPAY_KEY_ID")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        raise HTTPException(
            status_code=503,
            detail="Razorpay connected mode is not configured with backend credentials.",
        )
    return RazorpayClient(key_id, key_secret)


def handoff_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_razorpay_dispute_metadata(provider_dispute: dict) -> dict:
    """Persist only small, workflow-relevant provider metadata, never raw data."""

    allowed_fields = (
        "id",
        "payment_id",
        "amount",
        "currency",
        "reason_code",
        "respond_by",
        "status",
        "phase",
    )
    return {
        field: provider_dispute[field]
        for field in allowed_fields
        if field in provider_dispute
        and isinstance(provider_dispute[field], (str, int, float, type(None)))
    }


def handoff_view(workflow_id: str) -> dict:
    """Return safe, current handoff state calculated from verified files only."""

    dispute = get_dispute_snapshot(workflow_id)
    state = workflow_state(workflow_id)
    mode = razorpay_mode()
    stored = dispute.get("razorpay_handoff", {})
    demo_evidence_prepared = (
        stored.get("handoff_status") == "prepared_demo"
        or stored.get("evidence_sync_status") == "simulated_demo"
    )
    evidence_items = []

    for evidence in list_evidence(workflow_id):
        if evidence.get("status") != "Verified":
            continue
        mapping = evidence_mapping(evidence["evidence_category"])
        item = {
            "evidence_id": evidence["evidence_id"],
            "evidence_category": evidence["evidence_category"],
            "original_filename": evidence["original_filename"],
            **mapping,
            "razorpay_sync_status": evidence.get(
                "razorpay_sync_status", "not_synced"
            ),
            "razorpay_document_id": evidence.get("razorpay_document_id"),
        }
        if demo_evidence_prepared:
            item["preparation_status"] = "Prepared for demo"
            item["demo_document_reference"] = demo_document_reference(
                evidence["evidence_id"]
            )
        elif item["razorpay_sync_status"] == "synced":
            item["preparation_status"] = "Synced to Razorpay"
        elif item["razorpay_sync_status"] == "failed":
            item["preparation_status"] = "Sync failed"
        else:
            item["preparation_status"] = "Waiting to sync"
        evidence_items.append(item)

    prepared_evidence_count = (
        len(evidence_items)
        if demo_evidence_prepared
        else sum(
            item["razorpay_sync_status"] == "synced"
            for item in evidence_items
        )
    )

    return {
        "workflow_id": workflow_id,
        "razorpay_mode": mode,
        "razorpay_dispute_id": stored.get("razorpay_dispute_id"),
        "razorpay_dispute_imported": bool(stored.get("razorpay_dispute_imported")),
        "razorpay_dispute_metadata": stored.get("razorpay_dispute_metadata"),
        "handoff_status": stored.get("handoff_status", "not_prepared"),
        "contest_summary": stored.get("contest_summary"),
        "prepared_at": stored.get("prepared_at"),
        "razorpay_draft_status": stored.get("razorpay_draft_status", "not_prepared"),
        "rebuttal_ready": bool(stored.get("rebuttal_ready")),
        "verified_evidence_count": len(evidence_items),
        "prepared_evidence_count": prepared_evidence_count,
        "waiting_evidence_count": len(evidence_items) - prepared_evidence_count,
        "critical_evidence_missing": state["critical_evidence_missing"],
        "submission_status": "not_submitted",
        "demo_dispute_reference": stored.get("demo_dispute_reference"),
        "evidence_sync_status": stored.get("evidence_sync_status", "not_synced"),
        "evidence": evidence_items,
    }


def verified_evidence_for_handoff(workflow_id: str) -> list[dict]:
    """The single verified-only source for both demo and connected handoff."""

    return [
        evidence
        for evidence in list_evidence(workflow_id)
        if evidence.get("status") == "Verified"
    ]


def prepare_demo_handoff(workflow_id: str) -> dict:
    """Persist a deterministic no-network handoff result for the MVP demo."""

    dispute = get_dispute_snapshot(workflow_id)
    verified_evidence = verified_evidence_for_handoff(workflow_id)
    summary = contest_summary(
        dispute["prediction_snapshot"]["dispute_type"],
        [evidence["evidence_category"] for evidence in verified_evidence],
    )
    update_razorpay_handoff(
        workflow_id,
        {
            "razorpay_mode": "demo",
            "handoff_status": "prepared_demo",
            "contest_summary": summary,
            "prepared_at": handoff_timestamp(),
            "razorpay_draft_status": "prepared",
            "demo_dispute_reference": demo_dispute_reference(workflow_id),
            "evidence_sync_status": "simulated_demo",
        },
    )
    return handoff_view(workflow_id)


def evidence_sync_result(evidence: dict, status: str, error: str | None = None) -> dict:
    """Return one safe, frontend-ready document sync outcome."""

    result = {
        "evidence_id": evidence["evidence_id"],
        "evidence_category": evidence["evidence_category"],
        "original_filename": evidence["original_filename"],
        "status": status,
        "razorpay_document_id": evidence.get("razorpay_document_id"),
    }
    if error:
        result["error"] = error
    return result


def persist_connected_sync_status(workflow_id: str, results: list[dict]) -> None:
    """Record aggregate sync state without changing draft/submission state."""

    statuses = {result["status"] for result in results}
    if not results:
        aggregate_status = "not_synced"
    elif "failed" in statuses and statuses <= {"failed"}:
        aggregate_status = "failed"
    elif "failed" in statuses:
        aggregate_status = "partial_failed"
    else:
        aggregate_status = "synced"

    update_razorpay_handoff(
        workflow_id,
        {
            "razorpay_mode": "connected",
            "evidence_sync_status": aggregate_status,
            "evidence_synced_at": handoff_timestamp(),
        },
    )


def sync_connected_evidence(
    workflow_id: str,
    provider: RazorpayClient | None = None,
) -> dict:
    """Upload only verified, previously-unsynced files; no external dispute ID."""

    get_dispute_snapshot(workflow_id)
    provider = provider or connected_razorpay_client()
    results = []

    for evidence in verified_evidence_for_handoff(workflow_id):
        document_id = evidence.get("razorpay_document_id")
        if evidence.get("razorpay_sync_status") == "synced" and isinstance(
            document_id, str
        ) and document_id.startswith("doc_"):
            results.append(evidence_sync_result(evidence, "already_synced"))
            continue

        try:
            _, file_path = get_evidence_file(workflow_id, evidence["evidence_id"])
            document_id = provider.upload_document(
                file_path,
                evidence["original_filename"],
                evidence["content_type"],
            )
            evidence = update_evidence_razorpay_sync(
                workflow_id,
                evidence["evidence_id"],
                "synced",
                document_id,
            )
            results.append(evidence_sync_result(evidence, "synced"))
        except (EvidenceStoreError, RazorpayAdapterError) as error:
            # Preserve previously-synced files and record this file's failure
            # without leaking raw provider responses or credentials.
            try:
                evidence = update_evidence_razorpay_sync(
                    workflow_id,
                    evidence["evidence_id"],
                    "failed",
                )
            except EvidenceStoreError:
                pass
            results.append(evidence_sync_result(evidence, "failed", str(error)))

    persist_connected_sync_status(workflow_id, results)
    return {
        "workflow_id": workflow_id,
        "razorpay_mode": "connected",
        "evidence_sync_status": (
            "not_synced"
            if not results
            else "partial_failed"
            if any(result["status"] == "failed" for result in results)
            and any(result["status"] != "failed" for result in results)
            else "failed"
            if any(result["status"] == "failed" for result in results)
            else "synced"
        ),
        "results": results,
        "handoff": handoff_view(workflow_id),
    }


def sync_demo_evidence(workflow_id: str) -> dict:
    """Deterministically simulate evidence sync without calling Razorpay."""

    get_dispute_snapshot(workflow_id)
    verified_evidence = verified_evidence_for_handoff(workflow_id)
    update_razorpay_handoff(
        workflow_id,
        {
            "razorpay_mode": "demo",
            "evidence_sync_status": "simulated_demo",
            "evidence_synced_at": handoff_timestamp(),
        },
    )
    return {
        "workflow_id": workflow_id,
        "razorpay_mode": "demo",
        "evidence_sync_status": "simulated_demo",
        "results": [
            {
                **evidence_sync_result(evidence, "simulated"),
                "demo_document_reference": demo_document_reference(
                    evidence["evidence_id"]
                ),
            }
            for evidence in verified_evidence
        ],
        "handoff": handoff_view(workflow_id),
        "message": "DEMO MODE — No data was sent to Razorpay.",
    }


def prepare_connected_handoff(workflow_id: str) -> dict:
    """Sync verified files and save a Razorpay *draft*, never a submission."""

    dispute = get_dispute_snapshot(workflow_id)
    provider = connected_razorpay_client()
    stored_handoff = dispute.get("razorpay_handoff", {})
    razorpay_dispute_id = stored_handoff.get("razorpay_dispute_id")
    if (
        not stored_handoff.get("razorpay_dispute_imported")
        or not isinstance(razorpay_dispute_id, str)
        or not RAZORPAY_DISPUTE_ID_PATTERN.fullmatch(razorpay_dispute_id)
    ):
        raise HTTPException(
            status_code=400,
            detail="A valid Razorpay dispute ID must be imported before preparing a connected draft.",
        )

    verified_evidence = verified_evidence_for_handoff(workflow_id)
    if not verified_evidence:
        raise HTTPException(
            status_code=422,
            detail="At least one verified evidence file is required for a connected Razorpay draft.",
        )

    sync_result = sync_connected_evidence(workflow_id, provider)
    if any(result["status"] == "failed" for result in sync_result["results"]):
        update_razorpay_handoff(
            workflow_id,
            {
                "razorpay_mode": "connected",
                "handoff_status": "failed",
                "razorpay_draft_status": "not_prepared",
            },
        )
        raise HTTPException(
            status_code=502,
            detail="Some verified evidence files could not be synced to Razorpay. Review the file statuses and try again.",
        )

    mapped_documents: list[dict[str, str | None]] = []
    for evidence in verified_evidence_for_handoff(workflow_id):
        mapping = evidence_mapping(evidence["evidence_category"])
        mapped_documents.append(
            {
                **mapping,
                "razorpay_document_id": evidence.get("razorpay_document_id"),
            }
        )

    try:

        summary = contest_summary(
            dispute["prediction_snapshot"]["dispute_type"],
            [evidence["evidence_category"] for evidence in verified_evidence],
        )
        provider.prepare_draft(razorpay_dispute_id, summary, mapped_documents)
    except (EvidenceStoreError, RazorpayAdapterError) as error:
        # A failure never produces a prepared result. Synced document IDs are
        # retained so a retry cannot upload those same files again.
        update_razorpay_handoff(
            workflow_id,
            {
                "razorpay_mode": "connected",
                "handoff_status": "failed",
                "razorpay_draft_status": "not_prepared",
            },
        )
        raise HTTPException(status_code=502, detail=str(error))

    update_razorpay_handoff(
        workflow_id,
        {
            "razorpay_mode": "connected",
            "handoff_status": "prepared_connected",
            "contest_summary": summary,
            "prepared_at": handoff_timestamp(),
            "razorpay_draft_status": "prepared",
            "demo_dispute_reference": None,
        },
    )
    return handoff_view(workflow_id)


# ============================================================
# EVIDENCE MANAGEMENT ENDPOINTS
# ============================================================


@app.post("/evidence/upload", status_code=201)
async def upload_evidence(
    dispute_id: str = Form(...),
    evidence_category: str = Form(...),
    file: UploadFile = File(...),
):
    """Store a merchant-selected document as Pending Review."""

    try:
        dispute = get_dispute_snapshot(dispute_id)
    except EvidenceStoreError as error:
        raise evidence_error_to_http(error)

    category = evidence_category.strip()
    allowed_categories = dispute["prediction_snapshot"]["evidence_required"]
    if category not in allowed_categories:
        raise HTTPException(
            status_code=422,
            detail="This evidence category is not valid for the dispute type.",
        )

    original_filename = normalise_original_filename(file.filename)
    if not original_filename:
        raise HTTPException(status_code=400, detail="An evidence filename is required.")

    declared_content_type = file.content_type

    try:
        file_bytes = await file.read(MAX_EVIDENCE_FILE_SIZE + 1)
    finally:
        await file.close()

    if len(file_bytes) > MAX_EVIDENCE_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="Evidence files must be 10 MB or smaller.",
        )

    extension = validate_upload(
        filename=original_filename,
        content_type=declared_content_type,
        file_bytes=file_bytes,
    )

    try:
        evidence = add_evidence(
            dispute_id=dispute_id,
            original_filename=original_filename,
            evidence_category=category,
            content_type=(declared_content_type or "").lower(),
            file_bytes=file_bytes,
            extension=extension,
        )
        recommendation = workflow_state(dispute_id)
    except EvidenceStoreError as error:
        raise evidence_error_to_http(error)

    return {
        "evidence": public_metadata(evidence),
        "recommendation": recommendation,
    }


@app.get("/evidence/{dispute_id}")
def get_evidence(dispute_id: str):
    """List metadata only; uploaded files are never exposed as static assets."""

    try:
        evidence = [
            public_metadata(record)
            for record in list_evidence(dispute_id)
        ]
        recommendation = workflow_state(dispute_id)
    except EvidenceStoreError as error:
        raise evidence_error_to_http(error)

    return {
        "dispute_id": dispute_id,
        "evidence": evidence,
        "recommendation": recommendation,
    }


@app.get("/evidence/{dispute_id}/{evidence_id}/preview")
def preview_evidence(dispute_id: str, evidence_id: str):
    """Serve a registered upload inline without exposing the storage directory."""

    try:
        evidence, file_path = get_evidence_file(dispute_id, evidence_id)
    except EvidenceStoreError as error:
        raise evidence_error_to_http(error)

    return FileResponse(
        path=file_path,
        media_type=evidence["content_type"],
        filename=evidence["original_filename"],
        content_disposition_type="inline",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.post("/evidence/{dispute_id}/{evidence_id}/verify")
def verify_uploaded_evidence(dispute_id: str, evidence_id: str):
    """Merchant review transition: Pending Review -> Verified."""

    try:
        evidence = verify_evidence(dispute_id, evidence_id)
        recommendation = workflow_state(dispute_id)
    except EvidenceStoreError as error:
        raise evidence_error_to_http(error)

    return {
        "evidence": public_metadata(evidence),
        "recommendation": recommendation,
    }


@app.delete("/evidence/{dispute_id}/{evidence_id}")
def remove_evidence(dispute_id: str, evidence_id: str):
    """Remove an uploaded document and re-evaluate the server-side workflow."""

    try:
        delete_evidence(dispute_id, evidence_id)
        recommendation = workflow_state(dispute_id)
    except EvidenceStoreError as error:
        raise evidence_error_to_http(error)

    return {
        "message": "Evidence removed.",
        "recommendation": recommendation,
    }


@app.post("/evidence/{dispute_id}/recalculate")
def recalculate_recommendation(dispute_id: str):
    """Return deterministic evidence state without re-running the ML model."""

    try:
        return workflow_state(dispute_id)
    except EvidenceStoreError as error:
        raise evidence_error_to_http(error)


@app.post("/workflows/{workflow_id}/business-risk")
def update_business_risk_policy(
    workflow_id: str,
    data: BusinessRiskPolicyInput,
):
    """Compatibility recalculation endpoint; policy is now automatic."""

    try:
        # Keep the existing route callable for older clients while ignoring
        # obsolete manual policy fields. The derived policy is authoritative.
        del data
        return workflow_state(workflow_id)
    except EvidenceStoreError as error:
        raise evidence_error_to_http(error)


# ============================================================
# RAZORPAY HANDOFF ENDPOINTS
# ============================================================


@app.get("/razorpay/{workflow_id}/handoff")
def get_razorpay_handoff(workflow_id: str):
    """Return safe handoff state; credentials and raw provider data stay server-side."""

    try:
        return handoff_view(workflow_id)
    except EvidenceStoreError as error:
        raise evidence_error_to_http(error)
    except RazorpayAdapterError as error:
        raise HTTPException(status_code=422, detail=str(error))


@app.post("/razorpay/dispute/fetch")
def fetch_razorpay_dispute(data: RazorpayDisputeFetchInput):
    """Import limited Razorpay metadata for a connected workflow only."""

    if razorpay_mode() != "connected":
        raise HTTPException(
            status_code=400,
            detail="Razorpay dispute import is available only in connected mode.",
        )
    external_id = data.razorpay_dispute_id.strip()
    if not RAZORPAY_DISPUTE_ID_PATTERN.fullmatch(external_id):
        raise HTTPException(status_code=422, detail="The Razorpay dispute ID is not valid.")

    try:
        get_dispute_snapshot(data.workflow_id)
        provider_dispute = connected_razorpay_client().fetch_dispute(external_id)
    except EvidenceStoreError as error:
        raise evidence_error_to_http(error)
    except RazorpayAdapterError as error:
        raise HTTPException(status_code=502, detail=str(error))

    if provider_dispute.get("id") != external_id:
        raise HTTPException(
            status_code=502,
            detail="Razorpay returned an unexpected dispute record.",
        )

    try:
        update_razorpay_handoff(
            data.workflow_id,
            {
                "razorpay_mode": "connected",
                "razorpay_dispute_id": external_id,
                "razorpay_dispute_imported": True,
                "razorpay_dispute_metadata": safe_razorpay_dispute_metadata(
                    provider_dispute
                ),
            },
        )
        return handoff_view(data.workflow_id)
    except EvidenceStoreError as error:
        raise evidence_error_to_http(error)


@app.post("/razorpay/{workflow_id}/prepare-draft")
def prepare_razorpay_draft(workflow_id: str):
    """Prepare a non-submitting Razorpay handoff from verified evidence only."""

    try:
        if razorpay_mode() == "demo":
            return prepare_demo_handoff(workflow_id)
        return prepare_connected_handoff(workflow_id)
    except EvidenceStoreError as error:
        raise evidence_error_to_http(error)


@app.post("/razorpay/{workflow_id}/sync-evidence")
def sync_razorpay_evidence(workflow_id: str):
    """Sync verified documents without requiring a Razorpay dispute ID."""

    try:
        if razorpay_mode() == "demo":
            return sync_demo_evidence(workflow_id)
        return sync_connected_evidence(workflow_id)
    except EvidenceStoreError as error:
        raise evidence_error_to_http(error)
