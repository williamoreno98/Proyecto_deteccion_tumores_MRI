type Props = { label: string; value: number };
export default function ProbabilityBar({ label, value }: Props) {
  const pct = Math.round(value * 100);
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span className="text-gray-700">{label}</span>
        <span className="text-gray-700">{pct}%</span>
      </div>
      <div className="h-2 w-full rounded-full bg-gray-200">
        <div
          className="h-2 rounded-full bg-black/80"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
