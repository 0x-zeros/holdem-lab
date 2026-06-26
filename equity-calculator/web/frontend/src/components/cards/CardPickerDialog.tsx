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
    <div className="fixed inset-0 bg-black/50 flex items-end sm:items-center justify-center z-50">
      <div className="bg-white rounded-t-[var(--radius-lg)] sm:rounded-[var(--radius-lg)] shadow-xl w-full sm:max-w-[620px] sm:mx-4 max-h-[85vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 h-12 border-b border-[var(--border)]">
          <h2 className="font-semibold text-[var(--foreground)]">{title}</h2>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center rounded-full hover:bg-[var(--muted)] transition-colors touch-manipulation"
          >
            <X className="w-5 h-5 text-[var(--muted-foreground)]" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4 overflow-y-auto max-h-[calc(85vh-120px)]">
          {/* Card Grid */}
          <div className="overflow-x-auto -mx-4 px-4 sm:mx-0 sm:px-0">
            <CardPicker
              selectedCards={selectedCards}
              usedCards={usedCards}
              onCardClick={handleCardClick}
            />
          </div>

          {/* Selected Cards Preview + Actions */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pt-2 border-t border-[var(--border)]">
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

            <div className="flex gap-2 w-full sm:w-auto">
              <button
                onClick={handleClear}
                className="flex-1 sm:flex-none px-4 py-2.5 text-sm border border-[var(--border)] rounded-[var(--radius-md)] hover:bg-[var(--muted)] transition-colors touch-manipulation"
              >
                {t('actions.clear')}
              </button>
              <button
                onClick={handleDone}
                className="flex-1 sm:flex-none px-4 py-2.5 text-sm bg-[var(--primary)] text-white rounded-[var(--radius-md)] hover:opacity-90 transition-opacity touch-manipulation"
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
