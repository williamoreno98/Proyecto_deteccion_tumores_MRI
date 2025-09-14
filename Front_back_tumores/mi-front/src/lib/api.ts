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

// Soporta cualquiera de estas vars y cae al proxy si no hay ninguna
const RAW_BASE =
  (import.meta.env as any).VITE_API_BASE ??
  (import.meta.env as any).VITE_API_URL ??
  "";
const BASE = typeof RAW_BASE === "string"
  ? RAW_BASE.replace(/\/+$/, "") // sin slash final
  : "";

// helper para construir la URL (usa proxy /api cuando BASE está vacío)
function apiUrl(path: string) {
  const clean = path.startsWith("/") ? path : `/${path}`;
  return BASE ? `${BASE}${clean}` : `/api${clean}`;
}

export async function predictML(file: File): Promise<PredictResponse> {
  const fd = new FormData();
  fd.append("file", file);                 // el backend espera "file"

  const res = await fetch(apiUrl("/predict/ml"), {
    method: "POST",
    body: fd,
    // ¡no fijar Content-Type! el navegador pone el boundary
  });

  if (!res.ok) {
    // intenta leer texto por si el backend no responde JSON en error
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
