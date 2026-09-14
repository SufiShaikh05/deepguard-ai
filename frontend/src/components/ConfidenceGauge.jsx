const RADIUS = 54
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

export default function ConfidenceGauge({ label, confidence, probabilities }) {
  const toneClass = label === 'fake' ? 'text-alert' : 'text-scan'
  const strokeDashoffset = CIRCUMFERENCE * (1 - confidence)

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="relative h-32 w-32">
        <svg viewBox="0 0 120 120" className="h-full w-full -rotate-90">
          <circle cx="60" cy="60" r={RADIUS} fill="none" strokeWidth="8" className="stroke-line" />
          <circle
            cx="60" cy="60" r={RADIUS} fill="none" strokeWidth="8" strokeLinecap="round"
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={strokeDashoffset}
            className={`${toneClass} stroke-current transition-[stroke-dashoffset] duration-700 ease-out`}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`font-mono text-2xl font-semibold ${toneClass}`}>{(confidence * 100).toFixed(0)}%</span>
          <span className="text-xs uppercase tracking-wide text-ink-muted">{label}</span>
        </div>
      </div>

      <div className="w-full max-w-[240px] space-y-2">
        <ProbabilityRow label="real" value={probabilities.real} toneClass="bg-scan" />
        <ProbabilityRow label="fake" value={probabilities.fake} toneClass="bg-alert" />
      </div>
    </div>
  )
}

function ProbabilityRow({ label, value, toneClass }) {
  return (
    <div className="flex items-center gap-3 font-mono text-xs">
      <span className="w-10 uppercase text-ink-muted">{label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-line">
        <div className={`h-full rounded-full ${toneClass}`} style={{ width: `${value * 100}%` }} />
      </div>
      <span className="w-10 text-right text-ink">{(value * 100).toFixed(0)}%</span>
    </div>
  )
}
