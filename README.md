# RebuttalAI

### AI Intelligence Layer for Razorpay Chargeback Management

**RebuttalAI is not a replacement for Razorpay's existing dispute-management system.**

It is an additional intelligence and decision-support layer that helps merchants decide **whether a dispute is worth contesting, whether the required evidence is ready, and how to prepare an evidence-grounded rebuttal** before handing the case back to Razorpay.

```text
Razorpay Dispute
      ↓
Reason-Specific ML Model
      ↓
Fight Score
      ↓
Cost-Aware Decision Policy
      ↓
Evidence Readiness
      ↓
Final Recommendation
      ↓
Gemini-Assisted Rebuttal
      ↓
Merchant Review
      ↓
Razorpay Handoff
```

The merchant always remains the final decision-maker.

---

## Problem

Chargeback management requires merchants to quickly determine:

- Is the dispute worth contesting?
- Is the operational effort justified by the disputed amount?
- Is critical supporting evidence missing?
- How should the rebuttal be structured?

RebuttalAI combines **machine learning, cost-aware decisioning, verified evidence management and generative AI** to assist with these decisions.

---

## Core Features

### 1. Reason-Specific ML Models

Instead of one generic classifier, RebuttalAI uses separate models for different dispute types:

| Dispute Type | Selected Model |
|---|---|
| UPI Unauthorized | Random Forest |
| Netbanking Unauthorized | Random Forest |
| Non-Delivery | Logistic Regression |

Each model generates a **Fight Score**, which acts as an assessment signal for the decision engine.

> Fight Score is not a guaranteed or calibrated probability of winning a dispute.

### 2. Model Evaluation & Visualisation

For each dispute category, we compared:

- Logistic Regression
- Random Forest
- XGBoost

Models were evaluated using **Accuracy, Precision, Recall, F1 Score and ROC-AUC**.

The ML experimentation also includes visualisations for:

- Confusion matrices
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

| Selected Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| UPI — Random Forest | 91.67% | 94.22% | 94.22% | 94.22% | 0.9306 |
| Netbanking — Random Forest | 86.00% | 88.24% | 88.24% | 88.24% | 0.9109 |
| Non-Delivery — Logistic Regression | 84.29% | 96.69% | 82.16% | 88.83% | 0.9047 |

> The prototype models were trained on synthetic dispute datasets. These metrics demonstrate the modelling pipeline and should not be interpreted as expected production performance.

### 3. Cost-Aware Decision Engine

RebuttalAI does not blindly follow the ML classification.

It considers the estimated operational effort of contesting relative to the amount under dispute:

```text
Estimated Contest Cost
        ÷
Dispute Exposure
        ↓
Required Fight Score
```

This allows the system to require a stronger model signal when contesting would be relatively expensive compared with the disputed amount.

The current costs are prototype business assumptions and should be calibrated using merchant-specific operational data in production.

### 4. Verified Evidence Management

Evidence moves through:

```text
Missing → Pending Review → Verified
```

Only **Verified evidence** contributes to evidence readiness.

The system identifies required evidence, critical evidence, evidence completeness and missing critical documents.

Final recommendations can therefore be:

**Fight • Review / Collect Evidence • Don't Fight**

### 5. Gemini-Assisted Rebuttal

Gemini assists in generating a structured rebuttal grounded in verified evidence.

Merchant-provided additional context is kept separate from verified evidence, while internal ML scores and cost-policy information are excluded from the external rebuttal.

The generated response remains a **draft for merchant review**.

---

## Razorpay Integration

RebuttalAI integrates with Razorpay's existing dispute infrastructure rather than replacing it.

### Documents API

In Connected Mode, verified evidence can be uploaded through Razorpay's Documents API.

This integration was tested using **Razorpay Test Mode** and successfully returned real Razorpay `doc_...` document IDs.

### Disputes API

For an existing Razorpay dispute, RebuttalAI supports the workflow for preparing a contest draft using:

```text
PATCH /v1/disputes/:id/contest
action: "draft"
```

RebuttalAI intentionally does **not** automatically submit or accept disputes.

A real Razorpay `disp_...` ID is required for the contest-draft workflow.

### Test-Mode Limitation

Our Razorpay Test account did not contain an actual test dispute. Razorpay's documented merchant Disputes API operates on existing disputes but does not provide a documented endpoint for creating an arbitrary test dispute.

Therefore, the real **Documents API integration was tested end-to-end**, while the final contest-draft call could not be demonstrated without fabricating a `disp_...` ID.

RebuttalAI handles this explicitly by waiting for a real Razorpay dispute ID before enabling contest-draft preparation.

---

## Razorpay Webhooks

RebuttalAI implements support for Razorpay's:

```text
payment.dispute.created
```

webhook.

The implementation includes:

- HMAC-SHA256 signature verification
- Exact raw-body validation
- Webhook secret protection
- Duplicate-event/idempotency handling
- Dispute workflow creation
- Dispute-reason normalization

### Webhook Demo Limitation

Because the Test account contained no real dispute event, a live Razorpay-generated dispute webhook could not be triggered.

The webhook flow was instead validated locally using a **correctly signed Razorpay-style test fixture**.

This is explicitly labelled **Razorpay Webhook Test** in the application and is not represented as a live Razorpay dispute.

---

## Safety by Design

- No automatic dispute submission
- No automatic dispute acceptance
- No fabricated evidence
- Only verified evidence contributes to evidence readiness
- API credentials remain backend-side
- Webhook signatures are validated
- Gemini assists with drafting; it does not make the final decision
- Merchant remains the final reviewer

---

## Tech Stack

**ML:** Python, pandas, scikit-learn, XGBoost, joblib  
**Backend:** FastAPI, Python, Uvicorn  
**Frontend:** React, Vite  
**AI:** Google Gemini  
**Integration:** Razorpay Documents API, Disputes API & Dispute Webhooks

---

## Prototype Scope & Future Work

The current prototype supports:

- UPI Unauthorized
- Netbanking Unauthorized
- Non-Delivery

Future development can include training and calibration on authorized real merchant dispute histories, additional dispute categories, merchant-specific operational cost policies, production-grade document storage and deeper order/fulfilment integrations.

---

## Run Locally

### Prerequisites

- Python 3.10+
- Node.js & npm
- Git

### 1. Clone

```bash
git clone git@github.com:sneharawool24/RebuttalAI.git
cd RebuttalAI
```

### 2. Backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r Backend/requirements.txt
```

Create `Backend/.env`:

```env
GEMINI_API_KEY=your_gemini_api_key
RAZORPAY_MODE=demo
RAZORPAY_KEY_ID=your_razorpay_test_key_id
RAZORPAY_KEY_SECRET=your_razorpay_test_key_secret
RAZORPAY_WEBHOOK_SECRET=your_webhook_secret
```

Start the backend:

```powershell
cd Backend
uvicorn main:app --reload
```

### 3. Frontend

Open another terminal:

```powershell
cd RebuttalAI/Frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, typically:

```text
http://localhost:5173/
```

### 4. Optional — Test Webhook

With the backend running:

```powershell
cd Backend
python scripts/send_test_webhook.py
```

This sends a locally signed `payment.dispute.created` test event. It validates the webhook workflow but is **not a live Razorpay-generated dispute**.

---

## RebuttalAI in One Line

> **Razorpay provides the dispute-management rails; RebuttalAI adds the intelligence that helps merchants decide how and when to use them.**
