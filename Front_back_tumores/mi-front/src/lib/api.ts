export type ModelKind = "ml" | "dl";

export type PredictResponse = {
  model: "ml";
  top_class: "glioma" | "meningioma" | "notumor" | "pituitary";
  probabilities: Record<"glioma"|"meningioma"|"notumor"|"pituitary", number>;
};

export type Metrics = {
  precision: number | null;
  recall: number | null;
  f1: number | null;
  inference_ms: number | null;
};

const BASE = import.meta.env.VITE_API_BASE;

export async function predictML(file: File): Promise<PredictResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${BASE}/predict/ml`, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.detail || `HTTP ${res.status}`);
  }
  const data = await res.json();
  return { model: "ml", ...data };
}

export async function fetchMetricsML(): Promise<Metrics> {
  const res = await fetch(`${BASE}/metrics/ml`);
  if (!res.ok) return { precision: null, recall: null, f1: null, inference_ms: null };
  return res.json();
}
