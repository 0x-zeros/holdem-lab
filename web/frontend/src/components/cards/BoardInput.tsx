import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Card } from './Card'
import { CardPickerDialog } from './CardPickerDialog'
import { CameraButton } from '../camera'
import { useCardRecognition } from '../../hooks/useCardRecognition'

interface BoardInputProps {
  cards: string[]
  onCardsChange: (cards: string[]) => void
  onClear: () => void
  usedCards?: string[]
}

export function BoardInput({ cards, onCardsChange, onClear, usedCards = [] }: BoardInputProps) {
  const { t } = useTranslation()
  const [showPicker, setShowPicker] = useState(false)
  const { recognize, isRecognizing } = useCardRecognition()

  const handleCameraCapture = async (blob: Blob) => {
    await recognize(blob)
    // Cards will be automatically applied to store via useCardRecognition
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    const parsed = value
      .split(/[\s,]+/)
      .map((s) => s.trim())
      .filter(Boolean)
      .slice(0, 5)
    onCardsChange(parsed)
  }

  const handleCardClick = () => {
    setShowPicker(true)
  }

  const handlePickerConfirm = (selectedCards: string[]) => {
    onCardsChange(selectedCards)
  }

  return (
    <div className="space-y-2 sm:space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm sm:text-base font-medium">{t('board.title')}</h3>
        <div className="flex items-center gap-2">
          <CameraButton onCapture={handleCameraCapture} disabled={isRecognizing} />
          {cards.length > 0 && (
            <button
              onClick={onClear}
              className="text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)] touch-manipulation"
            >
              {t('board.clear')}
            </button>
          )}
        </div>
      </div>

      {/* Card slots */}
      <div className="flex gap-1 sm:gap-2">
        {[0, 1, 2].map((i) => (
          <Card key={i} notation={cards[i]} empty={!cards[i]} size="md" onClick={handleCardClick} />
        ))}
        <div className="w-px bg-[var(--border)] mx-0.5 sm:mx-1" />
        <Card notation={cards[3]} empty={!cards[3]} size="md" onClick={handleCardClick} />
        <Card notation={cards[4]} empty={!cards[4]} size="md" onClick={handleCardClick} />
      </div>

      {/* Text input */}
      <input
        type="text"
        placeholder={t('board.placeholder')}
        value={cards.join(' ')}
        onChange={handleInputChange}
        className="w-full px-3 py-2.5 border border-[var(--border)] rounded-[var(--radius-md)] text-sm"
      />

      <div className="text-[10px] sm:text-xs text-[var(--muted-foreground)]">
        {t('board.stages')}
      </div>

      {/* Card Picker Dialog */}
      <CardPickerDialog
        isOpen={showPicker}
        onClose={() => setShowPicker(false)}
        title={t('cardPicker.selectBoard')}
        maxCards={5}
        initialCards={cards}
        usedCards={usedCards}
        onConfirm={handlePickerConfirm}
      />
    </div>
  )
}

export default BoardInput
