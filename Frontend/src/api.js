const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

async function request(path, options = {}) {
  let response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, options);
  } catch {
    throw new Error(
      "RebuttalAI could not reach the backend. Confirm the API is running and try again.",
    );
  }

  const isJson = response.headers.get("content-type")?.includes("application/json");
  const payload = isJson ? await response.json() : null;

  if (!response.ok) {
    throw new Error(payload?.detail || "The request could not be completed. Please try again.");
  }

  return payload;
}

export function analyzeDispute(payload) {
  return request("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function fetchEvidence(disputeId) {
  return request(`/evidence/${encodeURIComponent(disputeId)}`);
}

export function uploadEvidence({ disputeId, evidenceCategory, file }) {
  const body = new FormData();
  body.append("dispute_id", disputeId);
  body.append("evidence_category", evidenceCategory);
  body.append("file", file);

  return request("/evidence/upload", { method: "POST", body });
}

export function verifyEvidence(disputeId, evidenceId) {
  return request(
    `/evidence/${encodeURIComponent(disputeId)}/${encodeURIComponent(evidenceId)}/verify`,
    { method: "POST" },
  );
}

export function previewEvidenceUrl(disputeId, evidenceId) {
  return `${API_BASE_URL}/evidence/${encodeURIComponent(disputeId)}/${encodeURIComponent(evidenceId)}/preview`;
}

export function removeEvidence(disputeId, evidenceId) {
  return request(
    `/evidence/${encodeURIComponent(disputeId)}/${encodeURIComponent(evidenceId)}`,
    { method: "DELETE" },
  );
}

export function generateRebuttal(payload) {
  return request("/generate-rebuttal", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function fetchRazorpayHandoff(workflowId) {
  return request(`/razorpay/${encodeURIComponent(workflowId)}/handoff`);
}

export function prepareRazorpayDraft(workflowId) {
  return request(`/razorpay/${encodeURIComponent(workflowId)}/prepare-draft`, {
    method: "POST",
  });
}

export function fetchRazorpayDispute({ workflowId, razorpayDisputeId }) {
  return request("/razorpay/dispute/fetch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      workflow_id: workflowId,
      razorpay_dispute_id: razorpayDisputeId,
    }),
  });
}
