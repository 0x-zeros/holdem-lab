import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import type { CanonicalHandInfo } from '../../api/types'

interface HandMatrixProps {
  hands: CanonicalHandInfo[]
  selectedHands: Set<string>
  onToggleHand: (notation: string) => void
  onSelectAll?: () => void
  onClearAll?: () => void
}

export function HandMatrix({
  hands,
  selectedHands,
  onToggleHand,
  onSelectAll,
  onClearAll,
}: HandMatrixProps) {
  const { t } = useTranslation()

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

  const getCellStyle = (hand: CanonicalHandInfo | null, isSelected: boolean) => {
    if (isSelected) {
      return 'bg-[var(--matrix-selected)] text-[var(--matrix-selected-text)]'
    }
    if (!hand) return 'bg-[var(--muted)]'
    if (hand.is_pair) return 'bg-[var(--matrix-pair)]'
    if (hand.suited) return 'bg-[var(--matrix-suited)]'
    return 'bg-[var(--matrix-offsuit)]'
  }

  return (
    <div className="space-y-3">
      {/* Controls */}
      <div className="flex gap-2">
        {onSelectAll && (
          <button
            onClick={onSelectAll}
            className="px-3 py-1 text-sm bg-[var(--muted)] rounded-[var(--radius-sm)] hover:bg-[var(--border)] transition-colors"
          >
            {t('actions.selectAll')}
          </button>
        )}
        {onClearAll && (
          <button
            onClick={onClearAll}
            className="px-3 py-1 text-sm bg-[var(--muted)] rounded-[var(--radius-sm)] hover:bg-[var(--border)] transition-colors"
          >
            {t('actions.clear')}
          </button>
        )}
        <span className="ml-auto text-sm text-[var(--muted-foreground)]">
          {t('matrix.handsSelected', { count: selectedHands.size })}
        </span>
      </div>

      {/* Matrix Grid */}
      <div className="inline-grid gap-[2px]" style={{ gridTemplateColumns: 'repeat(13, minmax(0, 1fr))' }}>
        {matrix.map((row, rowIdx) =>
          row.map((hand, colIdx) => {
            const isSelected = hand ? selectedHands.has(hand.notation) : false
            return (
              <button
                key={`${rowIdx}-${colIdx}`}
                className={`matrix-cell w-9 h-9 text-xs font-medium rounded-[var(--radius-sm)] ${getCellStyle(
                  hand,
                  isSelected
                )} flex items-center justify-center`}
                onClick={() => hand && onToggleHand(hand.notation)}
                disabled={!hand}
              >
                {hand?.notation || ''}
              </button>
            )
          })
        )}
      </div>

      {/* Legend */}
      <div className="flex gap-4 text-xs text-[var(--muted-foreground)]">
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 bg-[var(--matrix-pair)] rounded-[var(--radius-sm)]" />
          <span>{t('matrix.pairs')}</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 bg-[var(--matrix-suited)] rounded-[var(--radius-sm)]" />
          <span>{t('matrix.suited')}</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 bg-[var(--matrix-offsuit)] rounded-[var(--radius-sm)]" />
          <span>{t('matrix.offsuit')}</span>
        </div>
      </div>
    </div>
  )
}

export default HandMatrix
