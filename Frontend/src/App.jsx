import { useEffect, useMemo, useState } from "react";
import {
  analyzeDispute,
  fetchBusinessRiskEstimate,
  fetchEvidence,
  fetchIncomingRazorpayDispute,
  fetchIncomingRazorpayDisputes,
  fetchRazorpayDispute,
  fetchRazorpayHandoff,
  generateRebuttal,
  prepareRazorpayDraft,
  previewEvidenceUrl,
  removeEvidence,
  uploadEvidence,
  syncRazorpayEvidence,
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

const WEBHOOK_FORM = {
  dispute_type: "",
  order_value: "",
  days_since_transaction: "",
  device_ip_match_history: "",
  auth_flow_type: "",
  customer_account_age_days: "",
  customer_past_order_count: "",
  customer_past_dispute_count: "",
  delivery_confirmed: "",
  tracking_available: "",
  merchant_comm_log_exists: "",
  refund_already_issued: "",
};

const DISPUTE_LABELS = {
  upi_unauthorized: "UPI unauthorized transaction",
  netbanking_unauthorized: "Netbanking unauthorized transaction",
  non_delivery: "Non-delivery",
};

const WORKFLOW_STEPS = [
  { id: "dispute", label: "Dispute", number: "1" },
  { id: "analysis", label: "Analysis", number: "2" },
  { id: "evidence", label: "Evidence", number: "3" },
  { id: "recommendation", label: "Recommendation", number: "4" },
  { id: "rebuttal", label: "Rebuttal", number: "5" },
  { id: "handoff", label: "Razorpay Handoff", number: "6" },
];

const IMPORTANT_EVIDENCE = new Set([
  "Transaction timestamp",
  "Customer communication",
  "Merchant transaction/order records",
  "Order/invoice details",
  "Refund/cancellation records",
]);

function evidencePriority(category, criticalEvidence) {
  if (criticalEvidence.has(category)) return "Critical";
  if (IMPORTANT_EVIDENCE.has(category)) return "Important";
  return "Supporting";
}

function evidenceSyncLabel(status) {
  return {
    not_synced: "Not started",
    synced: "Synced",
    partial_failed: "Partially synced",
    failed: "Sync failed",
    simulated_demo: "Simulated",
  }[status] || "Not started";
}

function incomingAmount(amount, currency) {
  if (typeof amount !== "number") return "Amount unavailable";
  if (currency === "INR") return `₹${(amount / 100).toLocaleString("en-IN")}`;
  return `${amount} ${currency || ""}`.trim();
}

function formatInr(amount) {
  const numericAmount = Number(amount);
  const safeAmount = Number.isFinite(numericAmount) ? numericAmount : 0;
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    minimumFractionDigits: 0,
    maximumFractionDigits: 2,
  }).format(safeAmount);
}

function draftSections(draft) {
  const headings = [
    "Subject",
    "Rebuttal",
    "Evidence Referenced",
  ];
  const matches = [...draft.matchAll(
    /^(Subject|Rebuttal|Evidence Referenced):\s*$/gm,
  )];

  if (!matches.length) return [{ heading: "Draft", content: draft }];

  return matches.map((match, index) => ({
    heading: match[1],
    content: draft
      .slice(match.index + match[0].length, matches[index + 1]?.index)
      .trim() || "—",
  })).filter((section) => headings.includes(section.heading));
}

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

function DecisionCard({ analysis, onBack, onContinue }) {
  if (!analysis) {
    return (
      <section className="card muted-card">
        <SectionTitle step="2" title="Analysis" detail="Analyze a dispute to view its ML assessment." />
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
        title="Analysis"
        detail={DISPUTE_LABELS[analysis.dispute_type] || analysis.dispute_type}
      />
      <div className="decision-grid">
        <div>
          <p className="metric-label">Model Fight Score</p>
          <strong className="score">{fightScore}</strong>
          <small>Assessment signal only — not a probability of winning or a guaranteed outcome.</small>
        </div>
        <div>
          <p className="metric-label">Dispute Exposure</p>
          <strong>{formatInr(analysis.order_value)}</strong>
          <small>Amount currently under dispute.</small>
        </div>
        <div>
          <p className="metric-label">Base Decision</p>
          <strong>{analysis.base_decision}</strong>
          <small>ML output before evidence review.</small>
        </div>
        <div>
          <p className="metric-label">Required Fight Score</p>
          <strong>{(Number(analysis.decision_threshold) * 100).toFixed(0)}%</strong>
          <small>Minimum model signal required after considering estimated contest cost relative to the amount under dispute.</small>
          <small>{analysis.passes_cost_threshold ? "Meets threshold" : "Below threshold"}</small>
        </div>
        <div>
          <p className="metric-label">Estimated Contest Cost</p>
          <strong>{formatInr(analysis.estimated_contest_cost)}</strong>
          <small>Prototype estimate based on expected dispute-handling workload.</small>
        </div>
        <div className={`recommendation ${recommendationClass}`}>
          <p className="metric-label">Final Recommendation</p>
          <strong>{analysis.final_recommendation}</strong>
          <small>{analysis.recommendation_reason}</small>
        </div>
      </div>
      <div className="workflow-actions analysis-summary">
        <span><strong>{analysis.evidence_available.length}</strong> verified evidence categories</span>
        <span><strong>{analysis.critical_evidence_missing.length}</strong> critical evidence categories missing</span>
        <div className="workflow-action-buttons">
          <button className="secondary" type="button" onClick={onBack}>Back to Dispute</button>
          <button className="primary" type="button" onClick={onContinue}>Continue to Evidence</button>
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
  const [fileInputKey, setFileInputKey] = useState(0);
  const [merchantContext, setMerchantContext] = useState("");
  const [draft, setDraft] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState("");
  const [activeStep, setActiveStep] = useState("dispute");
  const [preview, setPreview] = useState(null);
  const [handoff, setHandoff] = useState(null);
  const [razorpayDisputeId, setRazorpayDisputeId] = useState("");
  const [entryPath, setEntryPath] = useState("manual");
  const [incomingDisputes, setIncomingDisputes] = useState([]);
  const [webhookCase, setWebhookCase] = useState(null);
  const [businessCostPreview, setBusinessCostPreview] = useState(null);

  const isUnauthorized = form.dispute_type !== "non_delivery";
  const authenticationOptions = form.dispute_type === "upi_unauthorized"
    ? [
        { value: "pin_entry", label: "Customer entered a UPI PIN" },
        { value: "collect_request", label: "Customer approved a collect request" },
        { value: "autopay", label: "Recurring or mandate payment" },
      ]
    : [
        { value: "netbanking_otp", label: "One-time password (OTP) flow" },
        { value: "saved_credentials", label: "Saved bank credentials" },
      ];
  const pendingByCategory = useMemo(() => {
    const categories = new Set();
    evidence.forEach((item) => {
      if (item.status === "Pending Review") categories.add(item.evidence_category);
    });
    return categories;
  }, [evidence]);
  const renderedDraftSections = useMemo(() => draftSections(draft), [draft]);

  useEffect(() => {
    const orderValue = Number(form.order_value);
    if (!form.dispute_type || !Number.isFinite(orderValue) || orderValue < 0) {
      setBusinessCostPreview(null);
      return undefined;
    }

    let cancelled = false;
    fetchBusinessRiskEstimate(form.dispute_type, orderValue)
      .then((estimate) => {
        if (!cancelled) setBusinessCostPreview(estimate);
      })
      .catch(() => {
        if (!cancelled) setBusinessCostPreview(null);
      });
    return () => {
      cancelled = true;
    };
  }, [form.dispute_type, form.order_value]);

  function updateForm(event) {
    const { name, value } = event.target;
    setForm((current) => {
      if (name === "dispute_type") {
        return {
          ...current,
          dispute_type: value,
          auth_flow_type: value === "netbanking_unauthorized"
            ? "netbanking_otp"
            : "pin_entry",
        };
      }
      return { ...current, [name]: value };
    });
  }

  function requestPayload() {
    return {
      dispute_type: form.dispute_type,
      order_value: Number(form.order_value),
      days_since_transaction: Number(form.days_since_transaction),
      device_ip_match_history: isUnauthorized
        ? form.device_ip_match_history === "not_recorded"
          ? null
          : Number(form.device_ip_match_history)
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
      existing_workflow_id: webhookCase?.workflow_id || null,
    };
  }

  async function showIncomingDisputes() {
    setEntryPath("razorpay");
    setNotice("");
    setBusy("incoming-disputes");
    try {
      const payload = await fetchIncomingRazorpayDisputes();
      setIncomingDisputes(payload.disputes);
    } catch (error) {
      setNotice(`Unable to load Razorpay disputes: ${error.message}`);
    } finally {
      setBusy("");
    }
  }

  function showManualEntry() {
    setEntryPath("manual");
    setWebhookCase(null);
  }

  async function openIncomingDispute(workflowId) {
    setNotice("");
    setBusy(`open-incoming-${workflowId}`);
    try {
      const incoming = await fetchIncomingRazorpayDispute(workflowId);
      const prefill = incoming.form_prefill || {};
      setForm({
        ...WEBHOOK_FORM,
        dispute_type: prefill.dispute_type || "",
        order_value: prefill.order_value == null ? "" : String(prefill.order_value),
      });
      setWebhookCase(incoming);
      setAnalysis(null);
      setEvidence([]);
      setHandoff(null);
      setDraft("");
      setEntryPath("manual");
    } catch (error) {
      setNotice(`Unable to open the Razorpay dispute: ${error.message}`);
    } finally {
      setBusy("");
    }
  }

  async function synchronizeEvidence(disputeId = analysis?.dispute_id) {
    if (!disputeId) return;
    const payload = await fetchEvidence(disputeId);
    setEvidence(payload.evidence);
    setAnalysis(payload.recommendation);
    return payload;
  }

  async function synchronizeHandoff(disputeId = analysis?.dispute_id) {
    if (!disputeId) return;
    const payload = await fetchRazorpayHandoff(disputeId);
    setHandoff(payload);
    return payload;
  }

  async function navigateTo(step) {
    if (step === "recommendation" && analysis?.dispute_id) {
      try {
        await synchronizeEvidence(analysis.dispute_id);
      } catch (error) {
        setNotice(`Unable to refresh the latest recommendation: ${error.message}`);
      }
    }
    if (step === "handoff" && analysis?.dispute_id) {
      try {
        await synchronizeHandoff(analysis.dispute_id);
      } catch (error) {
        setNotice(`Unable to load Razorpay handoff status: ${error.message}`);
      }
    }
    setActiveStep(step);
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
      setHandoff(null);
      setRazorpayDisputeId("");
      setSelectedCategory(result.evidence_required?.[0] || "");
      setActiveStep("analysis");
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
      const uploadResult = await uploadEvidence({
        disputeId: analysis.dispute_id,
        evidenceCategory: selectedCategory,
        file: selectedFile,
      });

      // Update immediately from the authoritative upload response, then
      // reconcile from the server. No synthetic event property is accessed
      // after awaiting the upload request.
      setEvidence((current) => [...current, uploadResult.evidence]);
      setAnalysis(uploadResult.recommendation);
      setSelectedFile(null);
      setFileInputKey((current) => current + 1);
      try {
        await synchronizeEvidence(analysis.dispute_id);
      } catch {
        setNotice(
          "Evidence was uploaded as Pending Review, but the latest document list could not refresh. The document is saved; please refresh this step.",
        );
      }
    } catch (error) {
      setNotice(`Unable to upload evidence: ${error.message}`);
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

  function handlePreview(item) {
    if (!analysis?.dispute_id) return;
    setPreview({
      filename: item.original_filename,
      contentType: item.content_type,
      url: previewEvidenceUrl(analysis.dispute_id, item.evidence_id),
    });
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
      try {
        await synchronizeHandoff(analysis.dispute_id);
      } catch {
        // The generated draft remains available even if the status card does
        // not refresh immediately.
      }
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy("");
    }
  }

  async function handlePrepareRazorpayDraft() {
    if (!analysis?.dispute_id) return;
    setNotice("");
    setBusy("prepare-handoff");
    try {
      const payload = await prepareRazorpayDraft(analysis.dispute_id);
      setHandoff(payload);
    } catch (error) {
      setNotice(`Unable to prepare the Razorpay draft: ${error.message}`);
    } finally {
      setBusy("");
    }
  }

  async function handleSyncRazorpayEvidence() {
    if (!analysis?.dispute_id) return;
    setNotice("");
    setBusy("sync-razorpay-evidence");
    try {
      const payload = await syncRazorpayEvidence(analysis.dispute_id);
      setHandoff(payload.handoff);
      if (payload.evidence_sync_status === "partial_failed") {
        setNotice("Some verified evidence files could not be synced. Review the per-file statuses and try again.");
      }
    } catch (error) {
      setNotice(`Unable to sync verified evidence to Razorpay: ${error.message}`);
    } finally {
      setBusy("");
    }
  }

  async function handleImportRazorpayDispute(event) {
    event.preventDefault();
    const externalDisputeId = razorpayDisputeId.trim();
    if (!externalDisputeId) {
      setNotice("Enter a Razorpay dispute ID.");
      return;
    }
    if (!/^disp_[A-Za-z0-9]+$/.test(externalDisputeId) || externalDisputeId.startsWith("disp_test_")) {
      setNotice("Enter a valid Razorpay dispute ID beginning with disp_.");
      return;
    }
    if (!analysis?.dispute_id) return;
    setNotice("");
    setBusy("import-razorpay-dispute");
    try {
      const payload = await fetchRazorpayDispute({
        workflowId: analysis.dispute_id,
        razorpayDisputeId: externalDisputeId,
      });
      setHandoff(payload);
    } catch (error) {
      setNotice("Could not import this Razorpay dispute. Check the dispute ID and your Test Mode account.");
    } finally {
      setBusy("");
    }
  }

  function openRazorpayDashboard() {
    window.open("https://dashboard.razorpay.com/", "_blank", "noopener,noreferrer");
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
  const hasConfirmedRazorpayDispute = handoff?.razorpay_mode === "connected"
    && handoff?.razorpay_dispute_imported === true
    && typeof handoff?.razorpay_dispute_id === "string";
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

      <nav className="workflow-nav" aria-label="Dispute workflow steps">
        {WORKFLOW_STEPS.map((step) => {
          const unavailable = step.id !== "dispute" && !analysis;
          return (
            <button
              key={step.id}
              type="button"
              disabled={unavailable}
              className={activeStep === step.id ? "workflow-tab active" : "workflow-tab"}
              onClick={() => navigateTo(step.id)}
            >
              <span>{step.number}</span>{step.label}
            </button>
          );
        })}
      </nav>

      {notice && <div className="notice" role="alert">{notice}</div>}

      {activeStep === "dispute" && <section className="card">
        <SectionTitle
          step="1"
          title="Dispute"
          detail="Choose how to start this case. Both paths use the same analysis and evidence workflow."
        />
        <div className="entry-paths" aria-label="Dispute entry path">
          <div>
            <strong>How would you like to start?</strong>
            <small>Manual entry remains available even when Razorpay is disconnected.</small>
          </div>
          <div className="entry-path-buttons">
            <button className={entryPath === "manual" ? "primary" : "secondary"} type="button" onClick={showManualEntry}>Enter Dispute Manually</button>
            <button className={entryPath === "razorpay" ? "primary" : "secondary"} type="button" onClick={showIncomingDisputes} disabled={busy === "incoming-disputes"}>
              {busy === "incoming-disputes" ? "Loading…" : "Razorpay Disputes"}
            </button>
          </div>
        </div>

        {entryPath === "manual" && <>
        {webhookCase && (
          <div className="webhook-case-banner">
            <strong>Source: {webhookCase.source_label}</strong>
            <span>Razorpay Dispute ID: {webhookCase.razorpay_dispute_id}</span>
            <small>Additional merchant input is required before analysis. Only compatible webhook values were prefilled.</small>
          </div>
        )}
        <form className="form-grid" onSubmit={handleAnalyze}>
          <Field label="Dispute type">
            <select name="dispute_type" value={form.dispute_type} onChange={updateForm} required>
              <option value="">Select a dispute type</option>
              {Object.entries(DISPUTE_LABELS).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </Field>
          <Field label="Order value (₹)">
            <input name="order_value" type="number" min="0" step="0.01" required value={form.order_value} onChange={updateForm} />
          </Field>
          {businessCostPreview && (
            <section className="business-cost-preview" aria-label="Estimated contest cost">
              <div>
                <p>Estimated contest cost</p>
                <strong>{formatInr(businessCostPreview.estimated_contest_cost)}</strong>
                <small>Prototype estimate based on expected dispute-handling workload.</small>
              </div>
              <ul>
                <li>Base handling: {formatInr(businessCostPreview.contest_cost_breakdown.base_handling)}</li>
                <li>Critical evidence preparation: {formatInr(businessCostPreview.contest_cost_breakdown.critical_evidence)}</li>
                <li>Supporting evidence preparation: {formatInr(businessCostPreview.contest_cost_breakdown.supporting_evidence)}</li>
                <li>Merchant review: {formatInr(businessCostPreview.contest_cost_breakdown.merchant_review)}</li>
              </ul>
            </section>
          )}
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
              <Field label="Device/IP familiarity" helper="Select how the current device and IP compare with prior customer activity.">
                <select name="device_ip_match_history" value={form.device_ip_match_history} onChange={updateForm} required>
                  <option value="">Select device/IP familiarity</option>
                  <option value="1">Matches known customer activity</option>
                  <option value="0">Does not match known customer activity</option>
                  <option value="not_recorded">Not recorded in merchant systems</option>
                </select>
              </Field>
              <Field label="How was the payment authorized?" helper="Choose the flow recorded for this transaction. Only supported recorded flows are shown.">
                <select name="auth_flow_type" value={form.auth_flow_type} onChange={updateForm} required>
                  <option value="">Select the recorded authorization flow</option>
                  {authenticationOptions.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
              </Field>
            </>
          ) : (
            <>
              <Field label="Delivery confirmed">
                <select name="delivery_confirmed" value={form.delivery_confirmed} onChange={updateForm} required>
                  <option value="">Select delivery status</option>
                  <option value="1">Yes</option><option value="0">No</option>
                </select>
              </Field>
              <Field label="Tracking available">
                <select name="tracking_available" value={form.tracking_available} onChange={updateForm} required>
                  <option value="">Select tracking availability</option>
                  <option value="1">Yes</option><option value="0">No</option>
                </select>
              </Field>
            </>
          )}

          <Field label="Customer communication log exists">
            <select name="merchant_comm_log_exists" value={form.merchant_comm_log_exists} onChange={updateForm} required>
              <option value="">Select an option</option>
              <option value="1">Yes</option><option value="0">No</option>
            </select>
          </Field>
          <Field label="Refund already issued">
            <select name="refund_already_issued" value={form.refund_already_issued} onChange={updateForm} required>
              <option value="">Select an option</option>
              <option value="1">Yes</option><option value="0">No</option>
            </select>
          </Field>
          <div className="form-action">
            <button className="primary" disabled={busy === "analyze"} type="submit">
              {busy === "analyze" ? "Analyzing…" : "Analyze & Continue"}
            </button>
          </div>
        </form>
        </>}

        {entryPath === "razorpay" && (
          <section className="incoming-disputes" aria-label="Incoming Razorpay disputes">
            <h3>Razorpay Disputes</h3>
            {incomingDisputes.length === 0 ? (
              <div className="empty-incoming">
                <strong>No Razorpay disputes received yet.</strong>
                <p>New Razorpay disputes received through the configured webhook will appear here automatically.</p>
              </div>
            ) : (
              <div className="incoming-dispute-list">
                {incomingDisputes.map((incoming) => (
                  <article className="incoming-dispute-card" key={incoming.workflow_id}>
                    <div>
                      <p className="eyebrow">{incoming.source_label}</p>
                      <h3>{DISPUTE_LABELS[incoming.internal_dispute_type] || incoming.reason || "Razorpay dispute"}</h3>
                      <p>{incoming.razorpay_dispute_id}</p>
                    </div>
                    <div className="incoming-dispute-meta">
                      <span>{incomingAmount(incoming.amount, incoming.currency)}</span>
                      <span>{incoming.payment_method || "Payment method unavailable"}</span>
                      <span>{incoming.status || "Status unavailable"}</span>
                    </div>
                    <div className="incoming-dispute-action">
                      {incoming.is_simulated && <span className="status pending-review">Webhook Test</span>}
                      <small>{incoming.additional_merchant_input_required ? "Additional merchant input required" : "Analysis available"}</small>
                      <button className="secondary" type="button" disabled={busy === `open-incoming-${incoming.workflow_id}`} onClick={() => openIncomingDispute(incoming.workflow_id)}>
                        {busy === `open-incoming-${incoming.workflow_id}` ? "Opening…" : "Open Case"}
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>
        )}
      </section>}

      {activeStep === "analysis" && (
        <DecisionCard
          analysis={analysis}
          onBack={() => navigateTo("dispute")}
          onContinue={() => navigateTo("evidence")}
        />
      )}

      {analysis && activeStep === "evidence" && (
        <>
          <section className="card">
            <SectionTitle
              step="3"
              title="Evidence"
              detail="Only verified documents count toward evidence completeness and the recommendation."
            />
            <div className="checklist">
              {requiredEvidence.map((category) => {
                const isVerified = availableEvidence.has(category);
                const isPending = pendingByCategory.has(category);
                const status = isVerified ? "Verified" : isPending ? "Pending Review" : "Missing";
                const priority = evidencePriority(category, criticalEvidence);
                return (
                  <div className={`check-row ${priority.toLowerCase()}`} key={category}>
                    <div>
                      <strong>{category}</strong>
                      <span className={`priority ${priority.toLowerCase()}`}>{priority}</span>
                    </div>
                    <div className="check-actions">
                      <span className={`status ${status.toLowerCase().replaceAll(" ", "-")}`}>{status}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          <section className="card" id="evidence-upload">
            <SectionTitle
              step="3"
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
                <input key={fileInputKey} type="file" accept=".pdf,.png,.jpg,.jpeg,application/pdf,image/png,image/jpeg" required onChange={(event) => setSelectedFile(event.target.files?.[0] || null)} />
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
              step="3"
              title="Uploaded Documents"
              detail="Uploading alone does not establish a fact."
            />
            <p className="document-review-help">Preview each uploaded document and confirm that it matches the selected evidence category before marking it Verified.</p>
            {evidence.length === 0 ? (
              <p className="empty-state">No files uploaded for this dispute workflow.</p>
            ) : (
              <div className="evidence-table" role="table">
                {evidence.map((item) => (
                  <div className="evidence-row" role="row" key={item.evidence_id}>
                    <div><strong>{item.original_filename}</strong><small>{item.evidence_category}</small></div>
                    <span className={`status ${item.status.toLowerCase().replaceAll(" ", "-")}`}>{item.status}</span>
                    <div className="row-actions">
                      <button className="secondary small" onClick={() => handlePreview(item)} type="button">Preview</button>
                      {item.status !== "Verified" && (
                        <button className="secondary small" disabled={busy === `verify-${item.evidence_id}`} onClick={() => handleVerify(item.evidence_id)} type="button">
                          {busy === `verify-${item.evidence_id}` ? "Verifying…" : "Verify"}
                        </button>
                      )}
                      <button className="text-button danger" disabled={busy === `remove-${item.evidence_id}`} onClick={() => handleRemove(item.evidence_id)} type="button">Remove</button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
          <div className="workflow-actions page-actions">
            <button className="secondary" type="button" onClick={() => navigateTo("analysis")}>Back to Analysis</button>
            <button className="primary" type="button" onClick={() => navigateTo("recommendation")}>Continue to Recommendation</button>
          </div>
        </>
      )}

      {analysis && activeStep === "recommendation" && (
          <section className="card recommendation-update">
            <SectionTitle
              step="4"
              title="Recommendation"
              detail="The deterministic evidence engine is backend-authoritative; the Model Fight Score remains an assessment signal."
            />
            <div className="recommendation-grid">
              <div className="recommendation-highlight">
                <span>Final recommendation</span>
                <strong>{analysis.final_recommendation}</strong>
              </div>
              <div><span>Model Fight Score</span><strong>{(Number(analysis.fight_score) * 100).toFixed(2)}%</strong><small>Not a win probability.</small></div>
              <div><span>Required Fight Score</span><strong>{(Number(analysis.decision_threshold) * 100).toFixed(0)}%</strong><small>{analysis.passes_cost_threshold ? "Meets threshold" : "Below threshold"}. Minimum model signal required after considering estimated contest cost relative to the amount under dispute.</small></div>
              <div><span>Dispute Exposure</span><strong>{formatInr(analysis.order_value)}</strong><small>Amount currently under dispute.</small></div>
              <div><span>Estimated Contest Cost</span><strong>{formatInr(analysis.estimated_contest_cost)}</strong><small>Prototype estimate based on expected dispute-handling workload.</small></div>
              <div><span>Evidence completeness</span><strong>{(Number(analysis.evidence_coverage) * 100).toFixed(0)}%</strong></div>
              <div><span>Critical evidence missing</span><strong>{analysis.critical_evidence_missing.length || "None"}</strong></div>
            </div>
            <p className="recommendation-reason"><strong>Why this is recommended:</strong> {analysis.recommendation_reason}</p>
            <div className="workflow-actions page-actions">
              <button className="secondary" type="button" onClick={() => navigateTo("evidence")}>Back to Evidence</button>
              <button className="primary" type="button" onClick={() => navigateTo("rebuttal")}>Continue to Rebuttal</button>
            </div>
          </section>
      )}

      {analysis && activeStep === "rebuttal" && (
        <>
          <section className="card">
            <SectionTitle
              step="5"
              title="Generate Rebuttal"
              detail="Creates an AI-generated draft using only verified evidence and the backend recommendation."
            />
            <Field label="Additional case context (optional)" helper="Add relevant information that may help draft the response but is not contained in the uploaded evidence.">
              <textarea value={merchantContext} onChange={(event) => setMerchantContext(event.target.value)} placeholder="Example: Customer contacted support and said they did not recognize the transaction. Customer previously purchased using the same account. Merchant attempted to contact the customer but received no response." rows="5" />
            </Field>
            <p className="context-note">This is treated as a merchant statement, not verified evidence.</p>
            <div className="generate-row">
              <p>AI-generated draft — merchant review required. No action in RebuttalAI submits a dispute.</p>
              <button className="primary" disabled={busy === "generate"} onClick={handleGenerate} type="button">
                {busy === "generate" ? "Generating…" : "Generate Rebuttal"}
              </button>
            </div>
          </section>

          <section className="card">
            <SectionTitle
              step="5"
              title="Rebuttal Review"
              detail="Review and edit the draft before independently deciding whether to contest."
            />
            {draft ? (
              <>
                <article className="draft-document" aria-label="Generated rebuttal document">
                  {renderedDraftSections.map((section) => (
                    <section key={section.heading} className="draft-section">
                      <h3>{section.heading}</h3>
                      <p>{section.content}</p>
                    </section>
                  ))}
                </article>
                <label className="field draft-edit-label">
                  <span>Edit draft before using it</span>
                <textarea className="draft" value={draft} onChange={(event) => setDraft(event.target.value)} rows="16" />
                </label>
                <div className="draft-actions">
                  <button className="secondary" onClick={copyDraft} type="button">Copy Draft</button>
                  <button className="secondary" onClick={exportDraft} type="button">Export Draft</button>
                </div>
                <aside className="merchant-review-info">
                  <h3>Merchant Review Information</h3>
                  <div>
                    <h4>Additional Supporting Evidence Not Yet Provided</h4>
                    {analysis.evidence_missing.length ? (
                      <ul>{analysis.evidence_missing.map((item) => <li key={item}>{item}</li>)}</ul>
                    ) : <p>None identified by the current evidence checklist.</p>}
                  </div>
                  <div>
                    <h4>Merchant Review Note</h4>
                    <p>Review all factual statements and ensure the referenced evidence files are attached before using this draft.</p>
                  </div>
                </aside>
              </>
            ) : <p className="empty-state">A generated draft will appear here for merchant review.</p>}
          </section>
          <div className="workflow-actions page-actions">
            <button className="secondary" type="button" onClick={() => navigateTo("recommendation")}>Back to Recommendation</button>
            <button className="primary" type="button" onClick={() => navigateTo("handoff")}>Continue to Razorpay Handoff</button>
          </div>
        </>
      )}

      {analysis && activeStep === "handoff" && (
        <>
          <section className="card handoff-card">
            <SectionTitle
              step="6"
              title="Razorpay Handoff"
              detail="Sync merchant-verified evidence to Razorpay. Final submission remains merchant-controlled."
            />
            <p className="handoff-safety">RebuttalAI prepares the dispute response only. Final submission remains a merchant-controlled action.</p>

            <section className="evidence-sync-panel" aria-label="Evidence Sync">
              <h3>Evidence Sync</h3>
              <div className="handoff-result-grid">
                <div><span>Razorpay connection</span><strong>{handoff ? (handoff.razorpay_mode === "demo" ? "Demo" : "Connected") : "Loading…"}</strong></div>
                <div><span>Verified evidence</span><strong>{handoff ? handoff.verified_evidence_count : "Loading…"}</strong></div>
                <div><span>Synced to Razorpay</span><strong>{handoff ? handoff.prepared_evidence_count : "Loading…"}</strong></div>
                <div><span>Waiting to sync</span><strong>{handoff ? handoff.waiting_evidence_count : "Loading…"}</strong></div>
                <div><span>Evidence sync status</span><strong>{handoff ? evidenceSyncLabel(handoff.evidence_sync_status) : "Loading…"}</strong></div>
              </div>
              {handoff?.razorpay_mode === "demo" && (
                <p className="demo-mode">DEMO MODE — No data was sent to Razorpay.</p>
              )}
              {handoff?.razorpay_mode === "connected" && handoff.evidence_sync_status === "synced" && (
                <p className="sync-success">Evidence successfully synced to Razorpay.</p>
              )}
              <button className="primary" type="button" disabled={busy === "sync-razorpay-evidence" || !handoff} onClick={handleSyncRazorpayEvidence}>
                {busy === "sync-razorpay-evidence" ? "Syncing…" : "Sync Verified Evidence to Razorpay"}
              </button>

              {handoff?.evidence.length ? (
                <div className="handoff-evidence-table evidence-sync-records" role="table">
                  {handoff.evidence.map((item) => (
                    <div className="handoff-evidence-row" role="row" key={item.evidence_id}>
                      <div><strong>{item.evidence_category}</strong><small>{item.original_filename}</small></div>
                      <div><span>Status</span><strong>{item.preparation_status}</strong></div>
                      <div><span>{item.razorpay_document_id ? "Razorpay Document ID" : "Simulated document reference"}</span><strong>{item.razorpay_document_id || item.demo_document_reference || "Waiting to sync"}</strong></div>
                    </div>
                  ))}
                </div>
              ) : <p className="empty-state">No verified evidence files are currently available for sync.</p>}
            </section>

            {handoff?.razorpay_mode === "connected" && (
              <form className="handoff-import" onSubmit={handleImportRazorpayDispute}>
                <Field label="Razorpay Dispute" helper="A real Razorpay dispute ID is required to prepare a contest draft.">
                  {hasConfirmedRazorpayDispute ? (
                    <strong className="imported-dispute">✓ {handoff.razorpay_dispute_id} imported</strong>
                  ) : (
                    <p className="empty-state">No Razorpay dispute is linked to this case.</p>
                  )}
                  <input value={razorpayDisputeId} onChange={(event) => setRazorpayDisputeId(event.target.value)} placeholder="disp_..." />
                </Field>
                <button className="secondary" type="submit" disabled={busy === "import-razorpay-dispute" || !razorpayDisputeId.trim()}>
                  {busy === "import-razorpay-dispute" ? "Importing…" : "Import Razorpay Dispute"}
                </button>
              </form>
            )}

            {handoff?.razorpay_mode === "connected" && (
              <section className="contest-draft-panel" aria-label="Contest Draft status">
                <h3>Contest Draft</h3>
                {hasConfirmedRazorpayDispute && handoff.razorpay_draft_status === "prepared" ? (
                  <>
                    <p className="sync-success">✓ Razorpay contest draft prepared for merchant review.</p>
                  </>
                ) : hasConfirmedRazorpayDispute ? (
                  <>
                    <p>Ready to prepare. This creates a draft for merchant review and does not submit the dispute.</p>
                    <button className="primary" type="button" disabled={busy === "prepare-handoff"} onClick={handlePrepareRazorpayDraft}>
                      {busy === "prepare-handoff" ? "Preparing…" : "Prepare Razorpay Draft"}
                    </button>
                  </>
                ) : (
                  <>
                    <strong>Waiting for Razorpay dispute ID</strong>
                    <p>Your verified evidence has been uploaded to Razorpay. A Razorpay dispute ID is required before RebuttalAI can prepare a contest draft.</p>
                    <button className="primary" type="button" disabled>Prepare Razorpay Draft</button>
                  </>
                )}
              </section>
            )}

            {handoff && (
              <div className="handoff-actions">
                <button className="secondary" type="button" onClick={openRazorpayDashboard}>Open Razorpay Dashboard</button>
              </div>
            )}
          </section>

          <div className="workflow-actions page-actions">
            <button className="secondary" type="button" onClick={() => navigateTo("rebuttal")}>Back to Rebuttal</button>
          </div>
        </>
      )}

      {preview && (
        <div className="preview-backdrop" role="presentation" onMouseDown={() => setPreview(null)}>
          <section className="preview-modal" role="dialog" aria-modal="true" aria-label={`Preview ${preview.filename}`} onMouseDown={(event) => event.stopPropagation()}>
            <header>
              <div>
                <p className="eyebrow">Evidence preview</p>
                <h2>{preview.filename}</h2>
              </div>
              <button className="secondary small" type="button" onClick={() => setPreview(null)}>Close</button>
            </header>
            {preview.contentType === "application/pdf" ? (
              <iframe className="pdf-preview" src={preview.url} title={`Preview of ${preview.filename}`} />
            ) : (
              <img className="image-preview" src={preview.url} alt={`Preview of ${preview.filename}`} />
            )}
          </section>
        </div>
      )}
    </main>
  );
}

export default App;
