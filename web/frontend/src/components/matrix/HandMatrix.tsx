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
      <div className="flex flex-wrap gap-2">
        {onSelectAll && (
          <button
            onClick={onSelectAll}
            className="px-2 sm:px-3 py-1 text-xs sm:text-sm bg-[var(--muted)] rounded-[var(--radius-sm)] hover:bg-[var(--border)] transition-colors touch-manipulation"
          >
            {t('actions.selectAll')}
          </button>
        )}
        {onClearAll && (
          <button
            onClick={onClearAll}
            className="px-2 sm:px-3 py-1 text-xs sm:text-sm bg-[var(--muted)] rounded-[var(--radius-sm)] hover:bg-[var(--border)] transition-colors touch-manipulation"
          >
            {t('actions.clear')}
          </button>
        )}
        <span className="ml-auto text-xs sm:text-sm text-[var(--muted-foreground)]">
          {t('matrix.handsSelected', { count: selectedHands.size })}
        </span>
      </div>

      {/* Matrix Grid - responsive with horizontal scroll on mobile */}
      <div className="overflow-x-auto -mx-4 px-4 sm:mx-0 sm:px-0 pb-2">
        <div className="inline-grid gap-1 sm:gap-2" style={{ gridTemplateColumns: 'repeat(13, minmax(26px, 36px))' }}>
          {matrix.map((row, rowIdx) =>
            row.map((hand, colIdx) => {
              const isSelected = hand ? selectedHands.has(hand.notation) : false
              return (
                <button
                  key={`${rowIdx}-${colIdx}`}
                  className={`matrix-cell w-[26px] h-[26px] sm:w-9 sm:h-9 text-[10px] sm:text-xs font-medium rounded-[var(--radius-sm)] ${getCellStyle(
                    hand,
                    isSelected
                  )} flex items-center justify-center touch-manipulation`}
                  onClick={() => hand && onToggleHand(hand.notation)}
                  disabled={!hand}
                >
                  {hand?.notation || ''}
                </button>
              )
            })
          )}
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-2 sm:gap-4 text-[10px] sm:text-xs text-[var(--muted-foreground)]">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 sm:w-4 sm:h-4 bg-[var(--matrix-pair)] rounded-[var(--radius-sm)]" />
          <span>{t('matrix.pairs')}</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 sm:w-4 sm:h-4 bg-[var(--matrix-suited)] rounded-[var(--radius-sm)]" />
          <span>{t('matrix.suited')}</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 sm:w-4 sm:h-4 bg-[var(--matrix-offsuit)] rounded-[var(--radius-sm)]" />
          <span>{t('matrix.offsuit')}</span>
        </div>
      </div>
    </div>
  )
}

export default HandMatrix
