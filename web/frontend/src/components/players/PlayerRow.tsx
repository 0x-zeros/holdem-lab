import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Card, CardPickerDialog } from '../cards'

interface PlayerRowProps {
  index: number
  cards: string[]
  range: string[]
  useRange: boolean
  onCardsChange: (cards: string[]) => void
  onRangeClick: () => void
  onToggleMode: () => void
  onRemove?: () => void
  canRemove?: boolean
  usedCards?: string[]
}

export function PlayerRow({
  index,
  cards,
  range,
  useRange,
  onCardsChange,
  onRangeClick,
  onToggleMode,
  onRemove,
  canRemove = true,
  usedCards = [],
}: PlayerRowProps) {
  const { t } = useTranslation()
  const [showPicker, setShowPicker] = useState(false)

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    const parsed = value
      .split(/[\s,]+/)
      .map((s) => s.trim())
      .filter(Boolean)
    onCardsChange(parsed)
  }

  const handleCardClick = () => {
    setShowPicker(true)
  }

  const handlePickerConfirm = (selectedCards: string[]) => {
    onCardsChange(selectedCards)
  }

  return (
    <div className="p-4 bg-white border border-[var(--border)] rounded-[var(--radius-lg)]">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <span className="font-medium">{t('player.title', { index: index + 1 })}</span>
        <div className="flex items-center gap-2">
          <button
            onClick={onToggleMode}
            className={`px-2 py-1 text-xs rounded-[var(--radius-sm)] ${
              !useRange
                ? 'bg-[var(--primary)] text-white'
                : 'bg-[var(--muted)] text-[var(--muted-foreground)]'
            }`}
          >
            {t('player.cards')}
          </button>
          <button
            onClick={onToggleMode}
            className={`px-2 py-1 text-xs rounded-[var(--radius-sm)] ${
              useRange
                ? 'bg-[var(--primary)] text-white'
                : 'bg-[var(--muted)] text-[var(--muted-foreground)]'
            }`}
          >
            {t('player.range')}
          </button>
          {canRemove && onRemove && (
            <button
              onClick={onRemove}
              className="ml-2 text-[var(--muted-foreground)] hover:text-red-500"
            >
              ×
            </button>
          )}
        </div>
      </div>

      {/* Input based on mode */}
      {useRange ? (
        <div>
          <button
            onClick={onRangeClick}
            className="w-full px-3 py-2 text-left border border-[var(--border)] rounded-[var(--radius-md)] text-sm hover:border-[var(--primary)] transition-colors"
          >
            {range.length > 0 ? (
              <span>{range.join(', ')}</span>
            ) : (
              <span className="text-[var(--muted-foreground)]">{t('player.selectRange')}</span>
            )}
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="flex gap-2">
            <Card notation={cards[0]} empty={!cards[0]} size="sm" onClick={handleCardClick} />
            <Card notation={cards[1]} empty={!cards[1]} size="sm" onClick={handleCardClick} />
          </div>
          <input
            type="text"
            placeholder={t('player.cardPlaceholder')}
            value={cards.join(' ')}
            onChange={handleInputChange}
            className="w-full px-3 py-2 border border-[var(--border)] rounded-[var(--radius-md)] text-sm"
          />
        </div>
      )}

      {/* Card Picker Dialog */}
      <CardPickerDialog
        isOpen={showPicker}
        onClose={() => setShowPicker(false)}
        title={t('cardPicker.selectHoleCards')}
        maxCards={2}
        initialCards={cards}
        usedCards={usedCards}
        onConfirm={handlePickerConfirm}
      />
    </div>
  )
}

export default PlayerRow
