import { useState, useCallback, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { X } from 'lucide-react'
import { CardPicker } from './CardPicker'
import { Card } from './Card'

interface CardPickerDialogProps {
  isOpen: boolean
  onClose: () => void
  title: string
  maxCards: number
  initialCards: string[]
  usedCards: string[]
  onConfirm: (cards: string[]) => void
}

export function CardPickerDialog({
  isOpen,
  onClose,
  title,
  maxCards,
  initialCards,
  usedCards,
  onConfirm,
}: CardPickerDialogProps) {
  const { t } = useTranslation()
  const [selectedCards, setSelectedCards] = useState<string[]>(initialCards)

  // Reset selection when dialog opens
  useEffect(() => {
    if (isOpen) {
      setSelectedCards(initialCards)
    }
  }, [isOpen, initialCards])

  const handleCardClick = useCallback((card: string) => {
    setSelectedCards((prev) => {
      const isSelected = prev.includes(card)

      if (isSelected) {
        // Deselect the card
        const newSelection = prev.filter((c) => c !== card)
        return newSelection
      } else {
        // Select the card if under max
        if (prev.length >= maxCards) {
          return prev
        }
        const newSelection = [...prev, card]

        // Auto-close for hole cards when 2 cards are selected
        if (maxCards === 2 && newSelection.length === 2) {
          setTimeout(() => {
            onConfirm(newSelection)
            onClose()
          }, 150)
        }

        return newSelection
      }
    })
  }, [maxCards, onConfirm, onClose])

  const handleClear = useCallback(() => {
    setSelectedCards([])
  }, [])

  const handleDone = useCallback(() => {
    onConfirm(selectedCards)
    onClose()
  }, [selectedCards, onConfirm, onClose])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-[var(--radius-lg)] shadow-xl max-w-[620px] w-full mx-4 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 h-12 border-b border-[var(--border)]">
          <h2 className="font-semibold text-[var(--foreground)]">{title}</h2>
          <button
            onClick={onClose}
            className="w-7 h-7 flex items-center justify-center rounded-[var(--radius-sm)] hover:bg-[var(--muted)] transition-colors"
          >
            <X className="w-4 h-4 text-[var(--muted-foreground)]" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* Card Grid */}
          <CardPicker
            selectedCards={selectedCards}
            usedCards={usedCards}
            onCardClick={handleCardClick}
          />

          {/* Selected Cards Preview + Actions */}
          <div className="flex items-center justify-between pt-2">
            <div className="flex items-center gap-2">
              <span className="text-sm text-[var(--muted-foreground)]">
                {t('cardPicker.selected')}:
              </span>
              <div className="flex gap-1">
                {Array.from({ length: maxCards }).map((_, i) => (
                  <Card
                    key={i}
                    notation={selectedCards[i]}
                    empty={!selectedCards[i]}
                    size="sm"
                    onClick={() => {
                      if (selectedCards[i]) {
                        setSelectedCards((prev) => prev.filter((c) => c !== selectedCards[i]))
                      }
                    }}
                  />
                ))}
              </div>
            </div>

            <div className="flex gap-2">
              <button
                onClick={handleClear}
                className="px-4 py-2 text-sm border border-[var(--border)] rounded-[var(--radius-md)] hover:bg-[var(--muted)] transition-colors"
              >
                {t('actions.clear')}
              </button>
              <button
                onClick={handleDone}
                className="px-4 py-2 text-sm bg-[var(--primary)] text-white rounded-[var(--radius-md)] hover:opacity-90 transition-opacity"
              >
                {t('actions.done')}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default CardPickerDialog
