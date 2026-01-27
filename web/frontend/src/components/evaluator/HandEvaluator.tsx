import { useState, useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import { useMutation } from '@tanstack/react-query'
import { apiClient } from '../../api/client'
import type { EvaluateResponse } from '../../api/types'
import { CardPicker } from '../cards/CardPicker'

// Hand type translations for Chinese display
const handTypeTranslations: Record<string, string> = {
  'Royal Flush': '皇家同花顺',
  'Straight Flush': '同花顺',
  'Four of a Kind': '四条',
  'Full House': '葫芦',
  'Flush': '同花',
  'Straight': '顺子',
  'Three of a Kind': '三条',
  'Two Pair': '两对',
  'One Pair': '一对',
  'High Card': '高牌',
}

interface HandEvaluatorProps {
  usedCards: string[]
}

export function HandEvaluator({ usedCards }: HandEvaluatorProps) {
  const { t, i18n } = useTranslation()
  const [selectedCards, setSelectedCards] = useState<string[]>([])
  const [result, setResult] = useState<EvaluateResponse | null>(null)
  const isChinese = i18n.language.startsWith('zh')

  const evaluateMutation = useMutation({
    mutationFn: (cards: string[]) => apiClient.evaluateHand({ cards }),
    onSuccess: (data) => setResult(data),
    onError: () => setResult(null),
  })

  const handleCardClick = (card: string) => {
    setSelectedCards((prev) => {
      if (prev.includes(card)) {
        return prev.filter((c) => c !== card)
      }
      if (prev.length >= 7) {
        return prev
      }
      return [...prev, card]
    })
    setResult(null)
  }

  const handleClear = () => {
    setSelectedCards([])
    setResult(null)
  }

  const handleEvaluate = () => {
    if (selectedCards.length >= 5) {
      evaluateMutation.mutate(selectedCards)
    }
  }

  const canEvaluate = selectedCards.length >= 5 && selectedCards.length <= 7
  const cardCountText = `${selectedCards.length}/7`

  // Combine used cards from parent with locally selected cards for disabling
  const allUsedCards = useMemo(() => {
    const set = new Set(usedCards)
    return Array.from(set)
  }, [usedCards])

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">{t('evaluator.title')}</h3>
        <span className="text-sm text-[var(--muted-foreground)]">
          {cardCountText} {t('evaluator.cards')}
        </span>
      </div>

      {/* Card Picker */}
      <div className="p-4 bg-[var(--muted)] rounded-[var(--radius-lg)]">
        <CardPicker
          selectedCards={selectedCards}
          usedCards={allUsedCards}
          onCardClick={handleCardClick}
        />
      </div>

      {/* Selected Cards Display */}
      {selectedCards.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {selectedCards.map((card) => (
            <span
              key={card}
              className="px-2 py-1 text-sm font-mono bg-[var(--primary)] text-white rounded"
            >
              {card}
            </span>
          ))}
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-2">
        <button
          onClick={handleEvaluate}
          disabled={!canEvaluate || evaluateMutation.isPending}
          className={`
            flex-1 py-2 px-4 rounded-[var(--radius-md)] font-medium transition-colors
            ${canEvaluate && !evaluateMutation.isPending
              ? 'bg-[var(--primary)] text-white hover:opacity-90'
              : 'bg-[var(--muted)] text-[var(--muted-foreground)] cursor-not-allowed'
            }
          `}
        >
          {evaluateMutation.isPending ? t('evaluator.evaluating') : t('evaluator.evaluate')}
        </button>
        <button
          onClick={handleClear}
          disabled={selectedCards.length === 0}
          className={`
            py-2 px-4 rounded-[var(--radius-md)] font-medium transition-colors
            ${selectedCards.length > 0
              ? 'bg-[var(--muted)] text-[var(--foreground)] hover:bg-[var(--border)]'
              : 'bg-[var(--muted)] text-[var(--muted-foreground)] cursor-not-allowed'
            }
          `}
        >
          {t('evaluator.clear')}
        </button>
      </div>

      {/* Hint */}
      {!canEvaluate && selectedCards.length > 0 && selectedCards.length < 5 && (
        <p className="text-sm text-[var(--muted-foreground)]">
          {t('evaluator.needMoreCards', { count: 5 - selectedCards.length })}
        </p>
      )}

      {/* Result */}
      {result && (
        <div className="p-4 bg-white border border-[var(--border)] rounded-[var(--radius-lg)]">
          <div className="text-center">
            <div className="text-2xl font-bold text-[var(--primary)]">
              {result.hand_type}
            </div>
            {isChinese && (
              <div className="text-lg text-[var(--primary)] opacity-80">
                {handTypeTranslations[result.hand_type] || result.hand_type}
              </div>
            )}
            <div className="mt-1 text-[var(--muted-foreground)]">
              {result.description}
            </div>
          </div>
        </div>
      )}

      {/* Error */}
      {evaluateMutation.isError && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-[var(--radius-md)] text-red-700 text-sm">
          {evaluateMutation.error instanceof Error
            ? evaluateMutation.error.message
            : t('evaluator.error')}
        </div>
      )}
    </div>
  )
}

export default HandEvaluator
