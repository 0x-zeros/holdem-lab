import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { CanonicalHandInfo } from '../../api/types'
import preflopData from '../../data/preflop-equity.json'

interface PreflopMatrixProps {
  hands: CanonicalHandInfo[]
}

// Calculate background color based on equity (soft blue-green gradient)
const getEquityColor = (equity: number): string => {
  // Normalize to 0-1 range (assume equity ranges from ~5% to ~85%)
  const normalized = Math.max(0, Math.min(1, (equity - 5) / 80))

  // Use soft blue-to-green gradient
  // Low equity: soft blue-gray, High equity: soft teal-green
  const hue = 200 - normalized * 50        // 200 (blue) -> 150 (teal-green)
  const saturation = 25 + normalized * 30  // 25% -> 55%
  const lightness = 82 - normalized * 22   // 82% -> 60%

  return `hsl(${Math.round(hue)}, ${Math.round(saturation)}%, ${Math.round(lightness)}%)`
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
  const [hoveredHand, setHoveredHand] = useState<string | null>(null)

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

  // Get hovered hand info
  const hoveredInfo = useMemo(() => {
    if (!hoveredHand) return null
    const hand = hands.find((h) => h.notation === hoveredHand)
    const equity = equityData[hoveredHand]
    return { hand, equity }
  }, [hoveredHand, hands, equityData])

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
            const isHovered = hand?.notation === hoveredHand
            return (
              <div
                key={`${rowIdx}-${colIdx}`}
                className={`w-[42px] h-[42px] flex flex-col items-center justify-center rounded-[var(--radius-sm)] text-xs cursor-pointer transition-all ${
                  isHovered ? 'ring-2 ring-[var(--primary)] z-10' : ''
                }`}
                style={{
                  backgroundColor: equity != null ? getEquityColor(equity) : 'var(--muted)',
                  color: equity != null ? getTextColor(equity) : 'var(--muted-foreground)',
                }}
                onMouseEnter={() => hand && setHoveredHand(hand.notation)}
                onMouseLeave={() => setHoveredHand(null)}
                title={hand ? `${hand.notation}: ${equity ?? '-'}%` : ''}
              >
                <span className="font-medium leading-none">{hand?.notation || ''}</span>
                {equity != null && (
                  <span className="text-[10px] leading-none opacity-90">{equity}%</span>
                )}
              </div>
            )
          })
        )}
      </div>

      {/* Hovered hand details */}
      {hoveredInfo && hoveredInfo.hand && (
        <div className="p-3 bg-[var(--muted)] rounded-[var(--radius-md)]">
          <div className="flex items-center gap-4">
            <div className="text-2xl font-bold text-[var(--primary)]">
              {hoveredInfo.hand.notation}
            </div>
            <div className="text-lg">
              {hoveredInfo.equity != null ? `${hoveredInfo.equity}%` : '-'}
            </div>
            <div className="text-sm text-[var(--muted-foreground)]">
              {hoveredInfo.hand.num_combos} {t('results.combos', { count: hoveredInfo.hand.num_combos })}
            </div>
          </div>
        </div>
      )}

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
