import { useMemo, useState } from "react";
import {
  analyzeDispute,
  fetchEvidence,
  generateRebuttal,
  recalculateRecommendation,
  removeEvidence,
  uploadEvidence,
  verifyEvidence,
} from "./api";

const INITIAL_FORM = {
  dispute_type: "upi_unauthorized",
  order_value: "3137.73",
  days_since_transaction: "4",
  device_ip_match_history: "1",
  auth_flow_type: "pin_entry",
  customer_account_age_days: "1051",
  customer_past_order_count: "4",
  customer_past_dispute_count: "0",
  delivery_confirmed: "1",
  tracking_available: "1",
  merchant_comm_log_exists: "1",
  refund_already_issued: "0",
};

const DISPUTE_LABELS = {
  upi_unauthorized: "UPI unauthorized transaction",
  netbanking_unauthorized: "Netbanking unauthorized transaction",
  non_delivery: "Non-delivery",
};

function Field({ label, children, helper }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {helper && <small>{helper}</small>}
    </label>
  );
}

function SectionTitle({ step, title, detail }) {
  return (
    <div className="section-title">
      <span className="step">{step}</span>
      <div>
        <h2>{title}</h2>
        {detail && <p>{detail}</p>}
      </div>
    </div>
  );
}

function DecisionCard({ analysis }) {
  if (!analysis) {
    return (
      <section className="card muted-card">
        <SectionTitle
          step="2"
          title="Analyze Dispute"
          detail="Complete dispute details to receive a model score and workflow recommendation."
        />
      </section>
    );
  }

  const fightScore = `${(Number(analysis.fight_score) * 100).toFixed(2)}%`;
  const recommendationClass = {
    "Don't Fight": "dont-fight",
    "Review / Collect Evidence": "review-collect-evidence",
    Fight: "fight",
  }[analysis.final_recommendation] || "";

  return (
    <section className="card decision-card">
      <SectionTitle
        step="2"
        title="Analysis Result"
        detail={DISPUTE_LABELS[analysis.dispute_type] || analysis.dispute_type}
      />
      <div className="decision-grid">
        <div>
          <p className="metric-label">Model Fight Score</p>
          <strong className="score">{fightScore}</strong>
          <small>Model score — not a guaranteed outcome.</small>
        </div>
        <div>
          <p className="metric-label">Base Decision</p>
          <strong>{analysis.base_decision}</strong>
          <small>ML output before evidence review.</small>
        </div>
        <div className={`recommendation ${recommendationClass}`}>
          <p className="metric-label">Final Recommendation</p>
          <strong>{analysis.final_recommendation}</strong>
          <small>{analysis.recommendation_reason}</small>
        </div>
      </div>
    </section>
  );
}

function App() {
  const [form, setForm] = useState(INITIAL_FORM);
  const [analysis, setAnalysis] = useState(null);
  const [evidence, setEvidence] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [merchantContext, setMerchantContext] = useState("");
  const [draft, setDraft] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState("");

  const isUnauthorized = form.dispute_type !== "non_delivery";
  const pendingByCategory = useMemo(() => {
    const categories = new Set();
    evidence.forEach((item) => {
      if (item.status === "Pending Review") categories.add(item.evidence_category);
    });
    return categories;
  }, [evidence]);

  function updateForm(event) {
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: value }));
  }

  function requestPayload() {
    return {
      dispute_type: form.dispute_type,
      order_value: Number(form.order_value),
      days_since_transaction: Number(form.days_since_transaction),
      device_ip_match_history: isUnauthorized
        ? Number(form.device_ip_match_history)
        : null,
      auth_flow_type: isUnauthorized ? form.auth_flow_type : null,
      customer_account_age_days: Number(form.customer_account_age_days),
      customer_past_order_count: Number(form.customer_past_order_count),
      customer_past_dispute_count: Number(form.customer_past_dispute_count),
      delivery_confirmed: isUnauthorized ? null : Number(form.delivery_confirmed),
      tracking_available: isUnauthorized ? null : Number(form.tracking_available),
      merchant_comm_log_exists: Number(form.merchant_comm_log_exists),
      refund_already_issued: Number(form.refund_already_issued),
      start_evidence_workflow: true,
    };
  }

  async function synchronizeEvidence(disputeId = analysis?.dispute_id) {
    if (!disputeId) return;
    const payload = await fetchEvidence(disputeId);
    setEvidence(payload.evidence);
    setAnalysis(payload.recommendation);
  }

  async function handleAnalyze(event) {
    event.preventDefault();
    setNotice("");
    setDraft("");
    setBusy("analyze");

    try {
      const result = await analyzeDispute(requestPayload());
      setAnalysis(result);
      setEvidence([]);
      setSelectedCategory(result.evidence_required?.[0] || "");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy("");
    }
  }

  async function handleUpload(event) {
    event.preventDefault();
    if (!analysis?.dispute_id || !selectedFile || !selectedCategory) return;

    setNotice("");
    setBusy("upload");
    try {
      await uploadEvidence({
        disputeId: analysis.dispute_id,
        evidenceCategory: selectedCategory,
        file: selectedFile,
      });
      setSelectedFile(null);
      event.currentTarget.reset();
      await synchronizeEvidence(analysis.dispute_id);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy("");
    }
  }

  async function handleVerify(evidenceId) {
    if (!analysis?.dispute_id) return;
    setNotice("");
    setBusy(`verify-${evidenceId}`);
    try {
      await verifyEvidence(analysis.dispute_id, evidenceId);
      await synchronizeEvidence(analysis.dispute_id);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy("");
    }
  }

  async function handleRemove(evidenceId) {
    if (!analysis?.dispute_id) return;
    setNotice("");
    setBusy(`remove-${evidenceId}`);
    try {
      await removeEvidence(analysis.dispute_id, evidenceId);
      await synchronizeEvidence(analysis.dispute_id);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy("");
    }
  }

  async function handleRecalculate() {
    if (!analysis?.dispute_id) return;
    setNotice("");
    setBusy("recalculate");
    try {
      const result = await recalculateRecommendation(analysis.dispute_id);
      setAnalysis(result);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy("");
    }
  }

  async function handleGenerate() {
    if (!analysis?.dispute_id) return;
    setNotice("");
    setBusy("generate");
    try {
      const response = await generateRebuttal({
        dispute_id: analysis.dispute_id,
        dispute_type: analysis.dispute_type,
        order_value: Number(form.order_value),
        days_since_transaction: Number(form.days_since_transaction),
        base_decision: analysis.base_decision,
        fight_score: analysis.fight_score,
        final_recommendation: analysis.final_recommendation,
        recommendation_reason: analysis.recommendation_reason,
        evidence_available: analysis.evidence_available,
        evidence_missing: analysis.evidence_missing,
        merchant_context: merchantContext || null,
      });
      setDraft(response.rebuttal);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy("");
    }
  }

  async function copyDraft() {
    try {
      await navigator.clipboard.writeText(draft);
      setNotice("Draft copied to the clipboard.");
    } catch {
      setNotice("Unable to copy the draft. Select the text and copy it manually.");
    }
  }

  function exportDraft() {
    const file = new Blob([draft], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(file);
    const link = document.createElement("a");
    link.href = url;
    link.download = "rebuttalai-draft.txt";
    link.click();
    URL.revokeObjectURL(url);
  }

  const requiredEvidence = analysis?.evidence_required || [];
  const availableEvidence = new Set(analysis?.evidence_available || []);
  const criticalEvidence = new Set(analysis?.critical_evidence || []);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Merchant dispute workspace</p>
          <h1>RebuttalAI</h1>
        </div>
        <p className="review-reminder">Defense support only · Merchant review required</p>
      </header>

      {notice && <div className="notice" role="alert">{notice}</div>}

      <section className="card">
        <SectionTitle
          step="1"
          title="Dispute Details"
          detail="Enter the available case information to begin a new evidence workflow."
        />
        <form className="form-grid" onSubmit={handleAnalyze}>
          <Field label="Dispute type">
            <select name="dispute_type" value={form.dispute_type} onChange={updateForm}>
              {Object.entries(DISPUTE_LABELS).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </Field>
          <Field label="Order value (₹)">
            <input name="order_value" type="number" min="0" step="0.01" required value={form.order_value} onChange={updateForm} />
          </Field>
          <Field label="Days since transaction">
            <input name="days_since_transaction" type="number" min="0" required value={form.days_since_transaction} onChange={updateForm} />
          </Field>
          <Field label="Customer account age (days)">
            <input name="customer_account_age_days" type="number" min="0" required value={form.customer_account_age_days} onChange={updateForm} />
          </Field>
          <Field label="Past order count">
            <input name="customer_past_order_count" type="number" min="0" required value={form.customer_past_order_count} onChange={updateForm} />
          </Field>
          <Field label="Past dispute count">
            <input name="customer_past_dispute_count" type="number" min="0" required value={form.customer_past_dispute_count} onChange={updateForm} />
          </Field>

          {isUnauthorized ? (
            <>
              <Field label="Device and IP match history" helper="Use a value from 0 to 1.">
                <input name="device_ip_match_history" type="number" min="0" max="1" step="0.001" required value={form.device_ip_match_history} onChange={updateForm} />
              </Field>
              <Field label="Authentication flow">
                <select name="auth_flow_type" value={form.auth_flow_type} onChange={updateForm}>
                  <option value="pin_entry">PIN entry</option>
                  <option value="collect_request">Collect request</option>
                  <option value="otp">OTP</option>
                </select>
              </Field>
            </>
          ) : (
            <>
              <Field label="Delivery confirmed">
                <select name="delivery_confirmed" value={form.delivery_confirmed} onChange={updateForm}>
                  <option value="1">Yes</option><option value="0">No</option>
                </select>
              </Field>
              <Field label="Tracking available">
                <select name="tracking_available" value={form.tracking_available} onChange={updateForm}>
                  <option value="1">Yes</option><option value="0">No</option>
                </select>
              </Field>
            </>
          )}

          <Field label="Customer communication log exists">
            <select name="merchant_comm_log_exists" value={form.merchant_comm_log_exists} onChange={updateForm}>
              <option value="1">Yes</option><option value="0">No</option>
            </select>
          </Field>
          <Field label="Refund already issued">
            <select name="refund_already_issued" value={form.refund_already_issued} onChange={updateForm}>
              <option value="1">Yes</option><option value="0">No</option>
            </select>
          </Field>
          <div className="form-action">
            <button className="primary" disabled={busy === "analyze"} type="submit">
              {busy === "analyze" ? "Analyzing…" : "Analyze Dispute"}
            </button>
          </div>
        </form>
      </section>

      <DecisionCard analysis={analysis} />

      {analysis && (
        <>
          <section className="card">
            <SectionTitle
              step="3"
              title="Evidence Checklist"
              detail="Only merchant-verified uploads count toward evidence completeness."
            />
            <div className="checklist">
              {requiredEvidence.map((category) => {
                const isVerified = availableEvidence.has(category);
                const isPending = pendingByCategory.has(category);
                const isCritical = criticalEvidence.has(category);
                const status = isVerified ? "Verified" : isPending ? "Pending Review" : isCritical ? "Critical Missing" : "Missing";
                return (
                  <div className={`check-row ${isCritical ? "critical" : ""}`} key={category}>
                    <div>
                      <strong>{category}</strong>
                      {isCritical && <span className="critical-tag">Critical evidence</span>}
                    </div>
                    <div className="check-actions">
                      <span className={`status ${status.toLowerCase().replaceAll(" ", "-")}`}>{status}</span>
                      {!isVerified && (
                        <button type="button" className="text-button" onClick={() => setSelectedCategory(category)}>
                          Upload
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          <section className="card" id="evidence-upload">
            <SectionTitle
              step="4"
              title="Evidence Upload"
              detail="Files remain Pending Review until you explicitly verify the category and document."
            />
            <form className="upload-form" onSubmit={handleUpload}>
              <Field label="Evidence category">
                <select value={selectedCategory} onChange={(event) => setSelectedCategory(event.target.value)} required>
                  {requiredEvidence.map((category) => <option key={category} value={category}>{category}</option>)}
                </select>
              </Field>
              <Field label="Document" helper="PDF, PNG, JPG, or JPEG · maximum 10 MB">
                <input type="file" accept=".pdf,.png,.jpg,.jpeg,application/pdf,image/png,image/jpeg" required onChange={(event) => setSelectedFile(event.target.files?.[0] || null)} />
              </Field>
              <div className="form-action">
                <button className="secondary" disabled={busy === "upload"} type="submit">
                  {busy === "upload" ? "Uploading…" : "Upload Evidence"}
                </button>
              </div>
            </form>
          </section>

          <section className="card">
            <SectionTitle
              step="5"
              title="Evidence Review & Verification"
              detail="Review each upload before marking it Verified. Uploading alone does not establish a fact."
            />
            {evidence.length === 0 ? (
              <p className="empty-state">No files uploaded for this dispute workflow.</p>
            ) : (
              <div className="evidence-table" role="table">
                {evidence.map((item) => (
                  <div className="evidence-row" role="row" key={item.evidence_id}>
                    <div><strong>{item.original_filename}</strong><small>{item.evidence_category}</small></div>
                    <span className={`status ${item.status.toLowerCase().replaceAll(" ", "-")}`}>{item.status}</span>
                    <div className="row-actions">
                      {item.status !== "Verified" && (
                        <button className="secondary small" disabled={busy === `verify-${item.evidence_id}`} onClick={() => handleVerify(item.evidence_id)} type="button">
                          {busy === `verify-${item.evidence_id}` ? "Verifying…" : "Mark Verified"}
                        </button>
                      )}
                      <button className="text-button danger" disabled={busy === `remove-${item.evidence_id}`} onClick={() => handleRemove(item.evidence_id)} type="button">Remove</button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="card recommendation-update">
            <SectionTitle
              step="6"
              title="Recommendation Update"
              detail="The backend evidence engine recalculates this workflow; the ML model score does not change after upload."
            />
            <div className="coverage-row">
              <div><span>Evidence completeness</span><strong>{(Number(analysis.evidence_coverage) * 100).toFixed(0)}%</strong></div>
              <div><span>Critical evidence missing</span><strong>{analysis.critical_evidence_missing.length || "None"}</strong></div>
              <button className="secondary" disabled={busy === "recalculate"} onClick={handleRecalculate} type="button">
                {busy === "recalculate" ? "Updating…" : "Update Recommendation"}
              </button>
            </div>
          </section>

          <section className="card">
            <SectionTitle
              step="7"
              title="Generate Rebuttal"
              detail="Creates an AI-generated draft using only verified evidence and the backend recommendation."
            />
            <Field label="Merchant review context" helper="Optional merchant statement. It is not verified evidence.">
              <textarea value={merchantContext} onChange={(event) => setMerchantContext(event.target.value)} placeholder="Add context for merchant review. Do not enter secrets." rows="4" />
            </Field>
            <div className="generate-row">
              <p>AI-generated draft — merchant review required. No action in RebuttalAI submits a dispute.</p>
              <button className="primary" disabled={busy === "generate"} onClick={handleGenerate} type="button">
                {busy === "generate" ? "Generating…" : "Generate Rebuttal"}
              </button>
            </div>
          </section>

          <section className="card">
            <SectionTitle
              step="8"
              title="Rebuttal Review"
              detail="Review and edit the draft before independently deciding whether to contest."
            />
            {draft ? (
              <>
                <textarea className="draft" value={draft} onChange={(event) => setDraft(event.target.value)} rows="16" />
                <div className="draft-actions">
                  <button className="secondary" onClick={copyDraft} type="button">Copy Draft</button>
                  <button className="secondary" onClick={exportDraft} type="button">Export Draft</button>
                </div>
              </>
            ) : <p className="empty-state">A generated draft will appear here for merchant review.</p>}
          </section>
        </>
      )}
    </main>
  );
}

export default App;
