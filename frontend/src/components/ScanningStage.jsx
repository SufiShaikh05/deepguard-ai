import ScanFrame from './ScanFrame'

export default function ScanningStage({ previewUrl }) {
  return (
    <div>
      <ScanFrame tone="scan">
        <div className="relative h-80 w-full overflow-hidden rounded-lg bg-surface">
          <img src={previewUrl} alt="Image being analyzed" className="h-full w-full object-contain" />
          <div className="animate-scan-sweep absolute inset-x-0 h-12 bg-gradient-to-b from-transparent via-scan/50 to-transparent" />
        </div>
      </ScanFrame>
      <p className="mt-4 text-center font-mono text-sm text-ink-muted">Analyzing pixel and frequency patterns…</p>
    </div>
  )
}
