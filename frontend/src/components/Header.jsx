import { ShieldCheck } from 'lucide-react'

export default function Header() {
  return (
    <header className="flex items-center justify-between border-b border-line px-6 py-4 sm:px-10">
      <div className="flex items-center gap-2">
        <ShieldCheck className="h-5 w-5 text-scan" strokeWidth={1.5} />
        <span className="font-display text-lg font-semibold tracking-tight text-ink">DeepGuard</span>
      </div>
      <span className="font-mono text-xs uppercase tracking-wide text-ink-muted">Explainable synthetic image detection</span>
    </header>
  )
}
