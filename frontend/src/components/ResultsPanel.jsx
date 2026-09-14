import ScanFrame from './ScanFrame'
import ConfidenceGauge from './ConfidenceGauge'

export default function ResultsPanel({ previewUrl, result, onReset }) {
  const isFake = result.label === 'fake'

  return (
    <div className="flex flex-col gap-8">
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
        <div>
          <p className="mb-2 font-mono text-xs uppercase tracking-wide text-ink-muted">Original</p>
          <ScanFrame tone="neutral">
            <img src={previewUrl} alt="Original upload" className="h-64 w-full rounded-lg bg-surface object-contain" />
          </ScanFrame>
        </div>
        <div>
          <p className="mb-2 font-mono text-xs uppercase tracking-wide text-ink-muted">Heatmap</p>
          <ScanFrame tone={isFake ? 'alert' : 'scan'}>
            <img
              src={result.heatmapImage}
              alt="Grad-CAM heatmap overlay"
              className="h-64 w-full rounded-lg bg-surface object-contain"
            />
          </ScanFrame>
        </div>
      </div>

      <div className="flex flex-col items-center gap-6 border-t border-line pt-8 sm:flex-row sm:justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-wide text-ink-muted">Verdict</p>
          <p className={`font-display text-4xl font-semibold uppercase ${isFake ? 'text-alert' : 'text-scan'}`}>
            {result.label}
          </p>
        </div>
        <ConfidenceGauge label={result.label} confidence={result.confidence} probabilities={result.probabilities} />
      </div>

      <button
        type="button"
        onClick={onReset}
        className="self-center rounded-full border border-line px-6 py-2 font-mono text-sm text-ink-muted transition-colors hover:border-scan hover:text-ink"
      >
        Scan another image
      </button>
    </div>
  )
}
