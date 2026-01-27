import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { CanonicalHandInfo } from '../../api/types'
import preflopData from '../../data/preflop-equity.json'

interface PreflopMatrixProps {
  hands: CanonicalHandInfo[]
}

// Calculate background color based on equity (blue gradient matching HandMatrix)
const getEquityColor = (equity: number): string => {
  const normalized = Math.max(0, Math.min(1, (equity - 5) / 80))

  // 使用蓝色系渐变，与 HandMatrix 统一
  // 低胜率: 浅灰 (#F3F4F6), 高胜率: 蓝色 (#DBEAFE → 更深)
  const hue = 214                            // 蓝色色相
  const saturation = 10 + normalized * 70    // 10% → 80%
  const lightness = 96 - normalized * 18     // 96% → 78%

  return `hsl(${hue}, ${saturation}%, ${lightness}%)`
}

// Get text color based on background brightness
const getTextColor = (equity: number): string => {
  const normalized = (equity - 5) / 80
  // Dark text for all since we're using light backgrounds
  return normalized > 0.6 ? '#1a1a1a' : '#374151'
}

export function PreflopMatrix({ hands }: PreflopMatrixProps) {
  const { t } = useTranslation()
  const [numPlayers, setNumPlayers] = useState(2)

  // Get equity data for current player count
  const equityData = (preflopData as Record<string, Record<string, number>>)[String(numPlayers)] || {}

  // Build matrix grid
  const matrix = useMemo(() => {
    const grid: (CanonicalHandInfo | null)[][] = Array(13)
      .fill(null)
      .map(() => Array(13).fill(null))

    for (const hand of hands) {
      grid[hand.matrix_row][hand.matrix_col] = hand
    }
    return grid
  }, [hands])

  return (
    <div className="space-y-6">
      {/* Header and player selector */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">{t('preflop.title')}</h2>
        <div className="flex items-center gap-2">
          <span className="text-sm text-[var(--muted-foreground)]">{t('preflop.players')}</span>
          <div className="flex gap-1">
            {[2, 3, 4, 5, 6, 7, 8, 9, 10].map((n) => (
              <button
                key={n}
                onClick={() => setNumPlayers(n)}
                className={`px-2 py-1 text-xs rounded-[var(--radius-sm)] transition-colors ${
                  numPlayers === n
                    ? 'bg-[var(--primary)] text-white'
                    : 'bg-[var(--muted)] text-[var(--muted-foreground)] hover:bg-[var(--border)]'
                }`}
              >
                {n}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Description */}
      <p className="text-sm text-[var(--muted-foreground)]">
        {t('preflop.description')}
      </p>

      {/* Matrix Grid */}
      <div className="inline-grid gap-1.5" style={{ gridTemplateColumns: 'repeat(13, 42px)' }}>
        {matrix.map((row, rowIdx) =>
          row.map((hand, colIdx) => {
            const equity = hand ? equityData[hand.notation] : null
            return (
              <div
                key={`${rowIdx}-${colIdx}`}
                className="relative group"
              >
                <div
                  className="w-[42px] h-[42px] flex flex-col items-center justify-center rounded-[var(--radius-sm)] text-xs cursor-default"
                  style={{
                    backgroundColor: equity != null ? getEquityColor(equity) : 'var(--muted)',
                    color: equity != null ? getTextColor(equity) : 'var(--muted-foreground)',
                  }}
                >
                  <span className="font-medium leading-none">{hand?.notation || ''}</span>
                  {equity != null && (
                    <span className="text-[10px] leading-none opacity-90">{equity}%</span>
                  )}
                </div>
                {/* Tooltip */}
                {hand && (
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-[var(--foreground)] text-white text-xs rounded opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap z-20">
                    {hand.notation} · {equity ?? '-'}% · {hand.num_combos} {t('results.combos', { count: hand.num_combos })}
                  </div>
                )}
              </div>
            )
          })
        )}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 text-xs">
        <span className="text-[var(--muted-foreground)]">{t('preflop.legend')}:</span>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded-[var(--radius-sm)]" style={{ backgroundColor: getEquityColor(80) }} />
          <span>{t('preflop.high')}</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded-[var(--radius-sm)]" style={{ backgroundColor: getEquityColor(45) }} />
          <span>{t('preflop.medium')}</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded-[var(--radius-sm)]" style={{ backgroundColor: getEquityColor(10) }} />
          <span>{t('preflop.low')}</span>
        </div>
      </div>
    </div>
  )
}

export default PreflopMatrix
