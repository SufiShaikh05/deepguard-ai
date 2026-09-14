const CORNER_POSITIONS = [
  'top-0 left-0 border-t-2 border-l-2 rounded-tl-lg',
  'top-0 right-0 border-t-2 border-r-2 rounded-tr-lg',
  'bottom-0 left-0 border-b-2 border-l-2 rounded-bl-lg',
  'bottom-0 right-0 border-b-2 border-r-2 rounded-br-lg',
]

const COLOR_CLASSES = {
  neutral: 'border-ink-muted/40',
  scan: 'border-scan',
  alert: 'border-alert',
}

export default function ScanFrame({ children, tone = 'neutral', className = '' }) {
  const colorClass = COLOR_CLASSES[tone] || COLOR_CLASSES.neutral

  return (
    <div className={`relative ${className}`}>
      {children}
      {CORNER_POSITIONS.map((position) => (
        <span
          key={position}
          className={`pointer-events-none absolute h-6 w-6 transition-colors duration-500 ${position} ${colorClass}`}
        />
      ))}
    </div>
  )
}
