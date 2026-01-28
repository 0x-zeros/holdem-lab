import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Card, CardPickerDialog } from '../cards'

type PlayerMode = 'cards' | 'range' | 'random'

interface PlayerRowProps {
  index: number
  cards: string[]
  range: string[]
  mode: PlayerMode
  onCardsChange: (cards: string[]) => void
  onRangeClick: () => void
  onSetMode: (mode: PlayerMode) => void
  onRemove?: () => void
  canRemove?: boolean
  usedCards?: string[]
}

export function PlayerRow({
  index,
  cards,
  range,
  mode,
  onCardsChange,
  onRangeClick,
  onSetMode,
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

  const modeButtonClass = (m: PlayerMode) =>
    `px-1.5 sm:px-2 py-1 text-[10px] sm:text-xs rounded-[var(--radius-sm)] touch-manipulation ${
      mode === m
        ? 'bg-[var(--primary)] text-white'
        : 'bg-[var(--muted)] text-[var(--muted-foreground)]'
    }`

  return (
    <div className="p-3 sm:p-4 bg-white border border-[var(--border)] rounded-[var(--radius-lg)]">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <span className="text-sm sm:text-base font-medium">{t('player.title', { index: index + 1 })}</span>
        <div className="flex items-center gap-1 sm:gap-2">
          <button onClick={() => onSetMode('cards')} className={modeButtonClass('cards')}>
            {t('player.cards')}
          </button>
          <button onClick={() => onSetMode('range')} className={modeButtonClass('range')}>
            {t('player.range')}
          </button>
          <button onClick={() => onSetMode('random')} className={modeButtonClass('random')}>
            {t('player.random')}
          </button>
          {canRemove && onRemove && (
            <button
              onClick={onRemove}
              className="ml-1 sm:ml-2 w-6 h-6 flex items-center justify-center text-[var(--muted-foreground)] hover:text-red-500 hover:bg-red-50 rounded-full touch-manipulation"
            >
              ×
            </button>
          )}
        </div>
      </div>

      {/* Input based on mode */}
      {mode === 'range' ? (
        <div>
          <button
            onClick={onRangeClick}
            className="w-full px-3 py-2.5 text-left border border-[var(--border)] rounded-[var(--radius-md)] text-xs sm:text-sm hover:border-[var(--primary)] transition-colors touch-manipulation"
          >
            {range.length > 0 ? (
              <span className="line-clamp-2">{range.join(', ')}</span>
            ) : (
              <span className="text-[var(--muted-foreground)]">{t('player.selectRange')}</span>
            )}
          </button>
        </div>
      ) : mode === 'random' ? (
        <div className="py-2 px-3 text-center text-xs sm:text-sm text-[var(--muted-foreground)] border border-dashed border-[var(--border)] rounded-[var(--radius-md)]">
          {t('player.randomHand')}
        </div>
      ) : (
        <div className="space-y-2 sm:space-y-3">
          <div className="flex gap-1 sm:gap-2">
            <Card notation={cards[0]} empty={!cards[0]} size="sm" onClick={handleCardClick} />
            <Card notation={cards[1]} empty={!cards[1]} size="sm" onClick={handleCardClick} />
          </div>
          <input
            type="text"
            placeholder={t('player.cardPlaceholder')}
            value={cards.join(' ')}
            onChange={handleInputChange}
            className="w-full px-3 py-2.5 border border-[var(--border)] rounded-[var(--radius-md)] text-sm"
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
