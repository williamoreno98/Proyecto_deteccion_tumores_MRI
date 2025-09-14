type Metrics = {
  precision: number | null;
  recall: number | null;
  f1: number | null;
  inference_ms: number | null;
};

function fmt(v: number | null) {
  if (v == null || Number.isNaN(v)) return "—";
  // muestra 2 decimales; si viene en 0-1 lo deja con 2, si son ms también
  return v.toFixed(2);
}

export default function MetricsCard({
  ml,
  rf,
  loading = false,
}: {
  ml: Metrics | null;
  rf: Metrics | null;
  loading?: boolean;
}) {
  return (
    <section className="rounded-2xl border p-4 space-y-3" aria-busy={loading}>
      <h3 className="font-semibold">Comparación ML vs RF</h3>
      <p className="text-sm text-gray-500 -mt-1">Métricas reales desde el backend.</p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div className="rounded-xl border p-3">
          <div className="text-sm text-gray-500">Precisión</div>
          <div>
            ML: <strong>{fmt(ml?.precision ?? null)}</strong> | RF: <strong>{fmt(rf?.precision ?? null)}</strong>
          </div>
        </div>

        <div className="rounded-xl border p-3">
          <div className="text-sm text-gray-500">Recall</div>
          <div>
            ML: <strong>{fmt(ml?.recall ?? null)}</strong> | RF: <strong>{fmt(rf?.recall ?? null)}</strong>
          </div>
        </div>

        <div className="rounded-xl border p-3">
          <div className="text-sm text-gray-500">F1-score</div>
          <div>
            ML: <strong>{fmt(ml?.f1 ?? null)}</strong> | RF: <strong>{fmt(rf?.f1 ?? null)}</strong>
          </div>
        </div>

        <div className="rounded-xl border p-3">
          <div className="text-sm text-gray-500">Inferencia (ms)</div>
          <div>
            ML: <strong>{fmt(ml?.inference_ms ?? null)}</strong> | RF: <strong>{fmt(rf?.inference_ms ?? null)}</strong>
          </div>
        </div>
      </div>
    </section>
  );
}
