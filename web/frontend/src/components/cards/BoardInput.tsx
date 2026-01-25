import { useTranslation } from 'react-i18next'
import { Card } from './Card'

interface BoardInputProps {
  cards: string[]
  onCardsChange: (cards: string[]) => void
  onClear: () => void
}

export function BoardInput({ cards, onCardsChange, onClear }: BoardInputProps) {
  const { t } = useTranslation()

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    const parsed = value
      .split(/[\s,]+/)
      .map((s) => s.trim())
      .filter(Boolean)
      .slice(0, 5)
    onCardsChange(parsed)
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="font-medium">{t('board.title')}</h3>
        {cards.length > 0 && (
          <button
            onClick={onClear}
            className="text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
          >
            {t('board.clear')}
          </button>
        )}
      </div>

      {/* Card slots */}
      <div className="flex gap-2">
        {[0, 1, 2].map((i) => (
          <Card key={i} notation={cards[i]} empty={!cards[i]} size="md" />
        ))}
        <div className="w-px bg-[var(--border)] mx-1" />
        <Card notation={cards[3]} empty={!cards[3]} size="md" />
        <Card notation={cards[4]} empty={!cards[4]} size="md" />
      </div>

      {/* Text input */}
      <input
        type="text"
        placeholder={t('board.placeholder')}
        value={cards.join(' ')}
        onChange={handleInputChange}
        className="w-full px-3 py-2 border border-[var(--border)] rounded-[var(--radius-md)] text-sm"
      />

      <div className="text-xs text-[var(--muted-foreground)]">
        {t('board.stages')}
      </div>
    </div>
  )
}

export default BoardInput
