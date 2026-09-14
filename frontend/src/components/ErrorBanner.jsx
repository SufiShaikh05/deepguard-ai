import { AlertTriangle } from 'lucide-react'

export default function ErrorBanner({ message }) {
  if (!message) return null
  return (
    <div className="mb-6 flex items-start gap-3 rounded-lg border border-alert/40 bg-alert/10 px-4 py-3">
      <AlertTriangle className="mt-0.5 h-5 w-5 flex-shrink-0 text-alert" strokeWidth={1.5} />
      <p className="text-sm text-ink">{message}</p>
    </div>
  )
}
