interface EquityBarProps {
  equity: number    // 0 to 1
  winRate: number   // 0 to 1
  tieRate: number   // 0 to 1
  label?: string
  showDetails?: boolean
}

export function EquityBar({
  equity,
  winRate,
  tieRate,
  label,
  showDetails = true,
}: EquityBarProps) {
  const winWidth = winRate * 100
  const tieWidth = tieRate * 100

  return (
    <div className="space-y-2">
      {label && (
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium">{label}</span>
          <span className="text-lg font-semibold">{(equity * 100).toFixed(1)}%</span>
        </div>
      )}

      {/* Stacked bar */}
      <div className="h-4 bg-[var(--muted)] rounded-full overflow-hidden flex">
        <div
          className="h-full bg-[var(--equity-win)] transition-all duration-300"
          style={{ width: `${winWidth}%` }}
        />
        <div
          className="h-full bg-[var(--equity-tie)] transition-all duration-300"
          style={{ width: `${tieWidth}%` }}
        />
      </div>

      {showDetails && (
        <div className="flex gap-4 text-xs text-[var(--muted-foreground)]">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-[var(--equity-win)] rounded" />
            <span>Win: {(winRate * 100).toFixed(1)}%</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-[var(--equity-tie)] rounded" />
            <span>Tie: {(tieRate * 100).toFixed(1)}%</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default EquityBar
