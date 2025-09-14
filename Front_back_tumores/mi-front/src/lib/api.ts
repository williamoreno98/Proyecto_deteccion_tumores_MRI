// src/lib/api.ts
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

// Lee base si existe; si no, usa proxy (/api)
const RAW_BASE =
  (import.meta.env as any).VITE_API_BASE ??
  (import.meta.env as any).VITE_API_URL ??
  "";
const BASE = typeof RAW_BASE === "string" ? RAW_BASE.replace(/\/+$/, "") : "";

function apiUrl(path: string) {
  const clean = path.startsWith("/") ? path : `/${path}`;
  return BASE ? `${BASE}${clean}` : `/api${clean}`;
}

export async function predictML(file: File): Promise<PredictResponse> {
  const fd = new FormData();
  fd.append("file", file); // el backend espera "file"

  const res = await fetch(apiUrl("/predict/ml"), {
    method: "POST",
    body: fd, // no pongas Content-Type manual
  });

  if (!res.ok) {
    const msg = await res.text().catch(() => res.statusText);
    throw new Error(`HTTP ${res.status}: ${msg}`);
  }
  const data = await res.json();
  return { model: "ml", ...data };
}

export async function fetchMetricsML(): Promise<Metrics> {
  const res = await fetch(apiUrl("/metrics/ml"));
  if (!res.ok) return { precision: null, recall: null, f1: null, inference_ms: null };
  return res.json();
}
