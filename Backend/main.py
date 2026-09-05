from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd
from dotenv import load_dotenv
from google import genai


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


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "RebuttalAI API is running!"
    }


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
    critical_evidence
):

    # --------------------------------------------------------
    # FIND MISSING EVIDENCE
    # --------------------------------------------------------

    evidence_missing = [
        evidence
        for evidence in evidence_required
        if evidence not in evidence_available
    ]

    critical_evidence_missing = [
        evidence
        for evidence in critical_evidence
        if evidence not in evidence_available
    ]

    # --------------------------------------------------------
    # CALCULATE EVIDENCE COVERAGE
    # --------------------------------------------------------

    if len(evidence_required) > 0:

        evidence_coverage = (
            len(evidence_available)
            / len(evidence_required)
        )

    else:

        evidence_coverage = 0.0

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    # CASE 1:
    # ML model says DON'T FIGHT

    if base_decision == "Don't Fight":

        final_recommendation = "Don't Fight"

        recommendation_reason = (
            "The dispute model does not recommend contesting "
            "this dispute based on the available dispute signals."
        )

    # CASE 2:
    # ML says FIGHT but critical evidence is missing

    elif len(critical_evidence_missing) > 0:

        final_recommendation = "Review / Collect Evidence"

        recommendation_reason = (
            "The dispute model recommends fighting, but "
            "critical supporting evidence is missing. "
            "Collect the missing evidence before deciding "
            "whether to contest."
        )

    # CASE 3:
    # ML says FIGHT and critical evidence is available

    else:

        final_recommendation = "Fight"

        recommendation_reason = (
            "The dispute model recommends fighting and "
            "the required critical evidence is available."
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

        "recommendation_reason": (
            recommendation_reason
        )
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

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

    evidence_available = data.evidence_available

    # --------------------------------------------------------
    # DETERMINE FINAL RECOMMENDATION
    # --------------------------------------------------------

    recommendation = determine_final_recommendation(

        base_decision=base_decision,

        fight_score=fight_score,

        evidence_required=evidence_required,

        evidence_available=evidence_available,

        critical_evidence=critical_evidence
    )

    # --------------------------------------------------------
    # RETURN COMPLETE RESPONSE
    # --------------------------------------------------------

    return {

        "dispute_type":
            data.dispute_type,

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

        # ----------------------------------------------------
        # Evidence information
        # ----------------------------------------------------

        "evidence_required":
            evidence_required,

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


# ============================================================
# LLM REBUTTAL GENERATION
# ============================================================

def generate_rebuttal(data: RebuttalInput):

    # --------------------------------------------------------
    # FORMAT AVAILABLE EVIDENCE
    # --------------------------------------------------------

    if data.evidence_available:

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

4. Evidence listed under "EVIDENCE ACTUALLY AVAILABLE"
   can be referenced.

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
EVIDENCE ACTUALLY AVAILABLE
============================================================

{evidence_text}

============================================================
EVIDENCE CURRENTLY MISSING
============================================================

{missing_evidence_text}

============================================================
ADDITIONAL MERCHANT CONTEXT
============================================================

{merchant_context}

============================================================
TASK
============================================================

Write a concise, professional dispute response draft.

Use EXACTLY this structure:

Subject:
A short professional subject line.

Recommendation:
State the final recommendation from the evidence engine
and briefly explain what the merchant should do next.

Rebuttal:
Write 2-4 concise paragraphs explaining the merchant's
position using ONLY the supplied facts and available evidence.

If the final recommendation is "Review / Collect Evidence",
make it clear that additional evidence must be collected
and verified before contesting.

If the final recommendation is "Don't Fight", do not
encourage contesting.

If the final recommendation is "Fight", explain the
merchant's position using only the evidence actually
available.

Evidence Referenced:
Provide a bullet list containing ONLY evidence that is
actually available.

If no evidence is available, write:
- None provided

Evidence Still Needed:
Provide a bullet list of missing evidence that could
strengthen or support the dispute response.

If no evidence is missing, write:
- None

Merchant Review Note:
Write one short sentence reminding the merchant to verify
all factual statements and attach the appropriate evidence
before submitting.

FINAL REMINDER:
Never fabricate evidence.
Never convert missing evidence into existing evidence.
Never guarantee a successful dispute.
Never treat Fight Score as a win probability.
The merchant remains the final decision-maker.
"""


    # --------------------------------------------------------
    # CALL GEMINI
    # --------------------------------------------------------

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

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

    return response.text


# ============================================================
# REBUTTAL GENERATION ENDPOINT
# ============================================================

@app.post("/generate-rebuttal")
def generate_rebuttal_endpoint(
    data: RebuttalInput
):

    rebuttal = generate_rebuttal(data)

    return {

        "dispute_type":
            data.dispute_type,

        "base_decision":
            data.base_decision,

        "fight_score":
            data.fight_score,

        "final_recommendation":
            data.final_recommendation,

        "recommendation_reason":
            data.recommendation_reason,

        "rebuttal":
            rebuttal
    }