// src/lib/api.ts
export type ModelKind = "ml" | "rf";

export type PredictResponse = {
  model: ModelKind;
  top_class: "glioma" | "meningioma" | "notumor" | "pituitary";
  probabilities: Record<"glioma"|"meningioma"|"notumor"|"pituitary", number>;
  inference_ms?: number | null;
};

export type Metrics = {
  precision: number | null;
  recall: number | null;
  f1: number | null;
  inference_ms: number | null;
};

// Base de la API (si no hay vars, usa proxy /api)
const RAW_BASE =
  (import.meta.env as any).VITE_API_BASE ??
  (import.meta.env as any).VITE_API_URL ??
  "";
const BASE = typeof RAW_BASE === "string" ? RAW_BASE.replace(/\/+$/, "") : "";

function apiUrl(path: string) {
  const clean = path.startsWith("/") ? path : `/${path}`;
  return BASE ? `${BASE}${clean}` : `/api${clean}`;
}

async function getJsonOrNull(url: string) {
  const r = await fetch(url);
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text().catch(() => r.statusText)}`);
  return r.json();
}

async function predictGeneric(model: ModelKind, file: File): Promise<PredictResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(apiUrl(`/predict/${model}`), { method: "POST", body: fd });
  if (!res.ok) {
    const msg = await res.text().catch(() => res.statusText);
    throw new Error(`HTTP ${res.status}: ${msg}`);
  }
  const data = await res.json();
  return { model, ...data };
}

// ML
export async function predictML(file: File) { return predictGeneric("ml", file); }
export async function fetchMetricsML(): Promise<Metrics> {
  const data = await getJsonOrNull(apiUrl(`/metrics/ml`));
  return data ?? { precision: null, recall: null, f1: null, inference_ms: null };
}

// RF
export async function predictRF(file: File) { return predictGeneric("rf", file); }
export async function fetchMetricsRF(): Promise<Metrics> {
  const data = await getJsonOrNull(apiUrl(`/metrics/rf`));
  return data ?? { precision: null, recall: null, f1: null, inference_ms: null };
}
