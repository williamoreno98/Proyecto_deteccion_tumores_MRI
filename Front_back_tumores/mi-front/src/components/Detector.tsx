import { useEffect, useMemo, useState } from "react";
import type { PredictResponse } from "../lib/api";
import { predictML, predictRF, fetchMetricsML } from "../lib/api";
import ProbabilityBar from "./ProbabilityBar";
import MetricsCard from "./MetricsCard";

type HistoryItem = { filename: string; top: string; pct: number; model: string };

export default function Detector() {
  // CAMBIO: ahora el modelo puede ser "ml" o "rf"
  const [model, setModel] = useState<"ml"|"rf">("rf");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [metricsML, setMetricsML] = useState<any>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);

  useEffect(() => {
    fetchMetricsML().then(setMetricsML).catch(() => {});
  }, []);

  useEffect(() => {
    if (!file) { setPreview(null); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const probs = useMemo(() => result?.probabilities ?? null, [result]);

  async function onProcess() {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    try {
      let r: PredictResponse;
      if (model === "ml") {
        r = await predictML(file);
      } else {
        // NUEVO: cuando el modelo es RF usamos /predict/rf
        r = await predictRF(file);
      }
      setResult(r);
      const pct = Math.round((r.probabilities[r.top_class] ?? 0) * 100);
      setHistory((h) => [{ filename: file.name, top: r.top_class, pct, model: r.model }, ...h].slice(0,6));
    } catch (e:any) {
      setError(e.message || "Error");
    } finally {
      setLoading(false);
    }
  }

  function onClear() {
    setFile(null); setPreview(null); setResult(null); setError(null);
  }

  return (
    <div className="mx-auto max-w-6xl p-6 space-y-6">
      <header className="flex items-center justify-between">
        <h1 className="text-xl sm:text-2xl font-semibold">
          MRI Tumor Detector — <span className="text-indigo-600">ML vs RF</span>
        </h1>
        <a className="text-sm text-indigo-600 hover:underline" href="#">Ver historial</a>
      </header>

      {/* Selector de modelo */}
      <section className="rounded-2xl border p-4">
        <h3 className="font-medium mb-3">Seleccionar modelo</h3>
        <div className="flex items-center gap-4">
          <label className="inline-flex items-center gap-2 px-3 py-2 rounded-2xl border cursor-pointer">
            <input type="radio" name="model" checked={model==="ml"} onChange={()=>setModel("ml")} />
            ML
          </label>
          <label className="inline-flex items-center gap-2 px-3 py-2 rounded-2xl border cursor-pointer">
            <input type="radio" name="model" checked={model==="rf"} onChange={()=>setModel("rf")} />
            RF
          </label>
        </div>
      </section>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Columna izquierda: upload + resultado */}
        <div className="lg:col-span-2 space-y-6">
          <section className="rounded-2xl border p-4 space-y-4">
            <h3 className="font-medium">Seleccionar archivo</h3>
            <div className="flex flex-wrap items-center gap-3">
              <label className="inline-flex">
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e)=>setFile(e.target.files?.[0] ?? null)}
                />
              </label>
              <button
                onClick={onProcess}
                disabled={!file || loading}
                className="px-4 py-2 rounded-2xl shadow font-medium disabled:opacity-50"
              >
                {loading ? "Procesando..." : "Procesar"}
              </button>
              <button onClick={onClear} className="px-4 py-2 rounded-2xl border">Limpiar</button>
            </div>

            {preview && (
              <div>
                <div className="text-sm text-gray-500 mb-2">Vista previa:</div>
                <img src={preview} alt="preview" className="h-44 w-auto rounded-xl border object-contain bg-gray-50" />
              </div>
            )}
          </section>

          <section className="rounded-2xl border p-4 space-y-4">
            <h3 className="font-medium">Resultado</h3>

            {error && <div className="text-red-600 text-sm">{error}</div>}

            {!result && !error && (
              <div className="text-sm text-gray-500">Sube una imagen y presiona “Procesar”.</div>
            )}

            {result && (
              <>
                <div className="rounded-xl border p-3">
                  <div className="text-sm text-gray-500">Predicción</div>
                  <div className="text-lg font-semibold">
                    {result.top_class}{" "}
                    <span className="text-gray-500 text-sm">
                      ({Math.round((result.probabilities[result.top_class] ?? 0)*100)}%)
                    </span>
                  </div>
                </div>

                <div className="space-y-3">
                  <ProbabilityBar label="glioma"     value={probs!.glioma} />
                  <ProbabilityBar label="meningioma" value={probs!.meningioma} />
                  <ProbabilityBar label="notumor"    value={probs!.notumor} />
                  <ProbabilityBar label="pituitary"  value={probs!.pituitary} />
                </div>

                <div className="text-xs text-gray-500">
                  Basado en la maqueta (DL/ML, resultados y comparación).
                </div>
              </>
            )}
          </section>
        </div>

        {/* Columna derecha: métricas + historial */}
        <div className="space-y-6">
          <MetricsCard ml={metricsML} />

          <div className="rounded-2xl border p-4 space-y-3">
            <h3 className="font-semibold">Historial de análisis</h3>
            <div className="rounded-xl border p-3 h-[120px] overflow-auto text-sm space-y-2">
              {history.length === 0 && <div className="text-gray-500">Sin items aún.</div>}
              {history.map((h, i) => (
                <div key={i} className="flex items-center justify-between">
                  <span className="truncate">{h.filename}</span>
                  <span className="text-gray-600">
                    {h.model.toUpperCase()} · {h.top} ({h.pct}%)
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
