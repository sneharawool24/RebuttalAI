from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd


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
# INPUT SCHEMA
# ============================================================

class DisputeInput(BaseModel):

    # Dispute information
    dispute_type: str

    # Transaction information
    order_value: float
    days_since_transaction: int

    # Authentication / fraud-related features
    device_ip_match_history: float | None = None
    auth_flow_type: str | None = None

    # Customer history
    customer_account_age_days: int
    customer_past_order_count: int
    customer_past_dispute_count: int

    # Delivery-related features
    delivery_confirmed: float | None = None
    tracking_available: float | None = None

    # Merchant evidence indicators
    merchant_comm_log_exists: int
    refund_already_issued: int

    # Evidence actually available to the merchant
    evidence_available: list[str] = []


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


    # Convert input into DataFrame
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


    # Convert prediction to human-readable decision

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

        # ML result
        "base_decision":
            base_decision,

        "fight_score":
            round(
                float(fight_score),
                4
            ),

        # Evidence information
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

        # Final recommendation
        "final_recommendation":
            recommendation[
                "final_recommendation"
            ],

        "recommendation_reason":
            recommendation[
                "recommendation_reason"
            ]
    }