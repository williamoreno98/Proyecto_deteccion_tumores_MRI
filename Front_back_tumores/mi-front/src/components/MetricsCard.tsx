import type { Metrics } from "../lib/api";

export default function MetricsCard({ ml }: { ml: Metrics | null }) {
  const dash = "—";
  const fmt = (x: number | null) =>
    x == null ? dash : (Math.round(x * 100) / 100).toFixed(2);
  const fmtMs = (x: number | null) => x == null ? dash : (Math.round(x * 100) / 100).toFixed(2);


  return (
    <div className="rounded-2xl border p-4 space-y-3">
      <h3 className="font-semibold">Comparación ML vs DL</h3>
      <p className="text-xs text-gray-500">Métricas reales desde el backend.</p>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-xl border p-3">
          <div className="text-gray-500">Precisión</div>
          <div>ML: <b>{fmt(ml?.precision)}</b>  |  DL: {dash}</div>
        </div>
        <div className="rounded-xl border p-3">
          <div className="text-gray-500">Recall</div>
          <div>ML: <b>{fmt(ml?.recall)}</b>  |  DL: {dash}</div>
        </div>
        <div className="rounded-xl border p-3">
          <div className="text-gray-500">F1-score</div>
          <div>ML: <b>{fmt(ml?.f1)}</b>  |  DL: {dash}</div>
        </div>
        <div className="rounded-xl border p-3">
          <div className="text-gray-500">Inferencia (ms)</div>
          <div>ML: <b>{fmtMs(ml?.inference_ms)}</b>  |  DL: {dash}</div>
        </div>
      </div>
    </div>
  );
}
