# RebuttalAI

### An AI Intelligence Layer for Razorpay Chargeback Management

**RebuttalAI is not a replacement for Razorpay's existing dispute-management system.**

It is an **additional intelligence and decision-support layer** that sits before the merchant's existing Razorpay dispute workflow. Steps to run the project at the end of readme file.

Razorpay already provides the infrastructure to view, contest and manage disputes. RebuttalAI focuses on the decisions that come before submission:

> **Should this dispute be fought? Is contesting economically sensible? Is the required evidence ready? And how can the merchant prepare a stronger rebuttal?**

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
AI-Assisted Rebuttal
      ↓
Merchant Review
      ↓
Razorpay Handoff
```

The merchant always remains the final decision-maker.

---

## The Problem

Chargeback management is not simply an evidence-upload problem.

For every dispute, a merchant has to decide:

- Is it worth contesting?
- How strong is the case?
- Is critical evidence missing?
- Is the operational effort justified by the disputed amount?
- How should the rebuttal be structured?

RebuttalAI combines **machine learning, cost-aware decisioning, verified evidence management and generative AI** to support these decisions before handing the case back to Razorpay.

---

## Core Features

### 1. Reason-Specific ML Models

Instead of using one generic classifier, RebuttalAI currently supports three dispute categories:

| Dispute Type | Selected Model |
|---|---|
| UPI Unauthorized | Random Forest |
| Netbanking Unauthorized | Random Forest |
| Non-Delivery | Logistic Regression |

Each model produces a **Fight Score** — an assessment signal used by the decision engine.

> Fight Score is not presented as a guaranteed or calibrated probability of winning.

---

### 2. Model Evaluation & Visualisation

For every dispute category, we compared:

- Logistic Regression
- Random Forest
- XGBoost

Models were evaluated using:

**Accuracy • Precision • Recall • F1 Score • ROC-AUC**

The ML experimentation also includes visualisations such as:

- Confusion matrices
- Accuracy comparison
- Precision comparison
- Recall comparison
- F1 comparison
- ROC-AUC comparison

Final selected models:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| UPI — Random Forest | **91.67%** | 94.22% | 94.22% | 94.22% | 0.9306 |
| Netbanking — Random Forest | **86.00%** | 88.24% | 88.24% | 88.24% | 0.9109 |
| Non-Delivery — Logistic Regression | **84.29%** | 96.69% | 82.16% | 88.83% | 0.9047 |

The prototype models were trained using **synthetic dispute datasets**, so these metrics demonstrate the modelling pipeline rather than expected production performance.

---

### 3. Cost-Aware Decision Engine

RebuttalAI does not blindly use the ML classification.

It estimates the operational effort required to contest the dispute and compares it with the **Dispute Exposure**.

```text
Estimated Contest Cost
        ÷
Dispute Exposure
        ↓
Required Fight Score
```

This means an ₹800 handling effort has a very different significance for a ₹3,000 dispute than for a ₹50,000 dispute.

The cost values are centralized **prototype business assumptions** and should be calibrated using merchant-specific operational data in production.

---

### 4. Verified Evidence Management

Evidence moves through:

```text
Missing → Pending Review → Verified
```

Only **Verified** documents contribute to evidence readiness.

The engine identifies:

- required evidence
- critical evidence
- missing critical evidence
- evidence completeness

This allows the final recommendation to distinguish between:

**Fight • Review / Collect Evidence • Don't Fight**

---

### 5. Gemini-Assisted Rebuttal

Gemini generates a structured rebuttal using verified evidence.

Merchant-provided additional context is kept separate from verified evidence, and internal information such as ML scores and cost thresholds is not inserted into the external rebuttal.

The rebuttal remains a **draft for merchant review**, not an automatically submitted response.

---

## Razorpay Integration

RebuttalAI integrates with the existing Razorpay ecosystem rather than replacing it.

### Documents API

In **Connected Mode**, verified evidence can be uploaded through Razorpay's Documents API.

The integration was tested using Razorpay Test Mode and successfully returned real Razorpay:

```text
doc_...
```

document IDs.

### Dispute API

For a real Razorpay dispute, RebuttalAI is designed to fetch the dispute and prepare a contest using:

```text
PATCH /v1/disputes/:id/contest
```

with:

```text
action: "draft"
```

RebuttalAI intentionally does **not** call `action: "submit"` automatically.

A real Razorpay:

```text
disp_...
```

ID is required for this workflow.

### Why the Full Dispute API Flow Could Not Be Demonstrated

Our Razorpay Test account did not contain a real test dispute.

Razorpay's documented merchant Disputes API provides operations for existing disputes, such as fetching, accepting and contesting disputes, but does not provide a documented endpoint for creating an arbitrary test dispute.

Therefore we could test real **document synchronization**, but could not truthfully fabricate a `disp_...` ID to demonstrate the final contest-draft call.

The application handles this explicitly:

```text
No dispute linked
→ Evidence can still be synchronized
→ Contest Draft waits for a real Razorpay dispute ID
```

---

## Razorpay Webhooks

RebuttalAI also implements the Razorpay:

```text
payment.dispute.created
```

webhook flow.

The webhook implementation includes:

- HMAC-SHA256 signature verification
- exact raw-body validation
- webhook secret protection
- duplicate-event/idempotency handling
- dispute workflow creation
- dispute-reason normalization

### Why a Live Razorpay Webhook Was Not Used in the Demo

The webhook integration was tested locally using a correctly signed Razorpay-style `payment.dispute.created` fixture.

A live Razorpay-generated dispute webhook could not be demonstrated because the Test account had no real dispute event to generate.

We deliberately did **not fabricate a live Razorpay event or represent the local fixture as one**.

The local webhook test is therefore explicitly identified as:

**Razorpay Webhook Test**

rather than a live dispute.

---

## Safety by Design

RebuttalAI deliberately avoids irreversible automated financial actions.

- No automatic dispute submission
- No automatic dispute acceptance
- No fabricated evidence
- Only verified evidence contributes to evidence readiness
- API secrets remain backend-side
- Webhook signatures are validated
- Gemini assists with drafting but does not make the final decision
- Merchant remains the final reviewer

---

## Tech Stack

**ML:** Python, pandas, scikit-learn, XGBoost, joblib  
**Backend:** FastAPI, Python, Uvicorn  
**Frontend:** React, Vite  
**AI:** Google Gemini  
**Integration:** Razorpay Documents API, Disputes API & Dispute Webhooks

---

## Run Locally

### Backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r Backend/requirements.txt
cd Backend
uvicorn main:app --reload
```

Create `Backend/.env` with the required local credentials. Never commit secrets.

### Frontend

```powershell
cd Frontend
npm install
npm run dev
```

Then open the local Vite URL, typically:

```text
http://localhost:5173
```

---

## Prototype Limitations & Future Scope

The current prototype uses synthetic training data and supports:

- UPI Unauthorized
- Netbanking Unauthorized
- Non-Delivery

Production development would include real authorized historical dispute data, model calibration, merchant-specific operational costs, additional dispute categories, production-grade document storage and deeper merchant order/fulfilment integrations.

---

## The Idea Behind RebuttalAI

Razorpay already provides the **rails for dispute management**.

RebuttalAI adds the **intelligence before the merchant acts**.

> **ML assesses the dispute.  
> Cost-aware policy evaluates whether fighting makes business sense.  
> Evidence management checks readiness.  
> Gemini assists with the rebuttal.  
> Razorpay handles the dispute workflow.  
> The merchant makes the final decision.**
> ##  Run Locally

### Prerequisites
- Python 3.10+
- Node.js & npm
- Git

### 1. Clone the Repository

```bash
git clone git@github.com:sneharawool24/RebuttalAI.git
cd RebuttalAI
```

### 2. Set Up the Backend

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

Then start the backend:

```powershell
cd Backend
uvicorn main:app --reload
```

Backend runs at `http://127.0.0.1:8000`.

### 3. Start the Frontend

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

### 4. Optional — Test the Webhook

With the backend running:

```powershell
cd Backend
python scripts/send_test_webhook.py
```

This sends a locally signed `payment.dispute.created` test event. It simulates the webhook flow and is **not a live Razorpay-generated dispute**.
