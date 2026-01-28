import { useMemo } from 'react'

const SUITS = ['s', 'h', 'd', 'c'] as const
const RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'] as const

const SUIT_SYMBOLS: Record<string, string> = {
  s: '♠',
  h: '♥',
  d: '♦',
  c: '♣',
}

const SUIT_COLORS: Record<string, string> = {
  s: 'var(--card-suit-black)',
  h: 'var(--card-suit-red)',
  d: 'var(--card-suit-red)',
  c: 'var(--card-suit-black)',
}

interface CardPickerProps {
  selectedCards: string[]
  usedCards: string[]
  onCardClick: (card: string) => void
}

export function CardPicker({ selectedCards, usedCards, onCardClick }: CardPickerProps) {
  const usedSet = useMemo(() => new Set(usedCards), [usedCards])
  const selectedSet = useMemo(() => new Set(selectedCards), [selectedCards])

  return (
    <div className="flex flex-col gap-0.5 sm:gap-1">
      {SUITS.map((suit) => (
        <div key={suit} className="flex items-center gap-0.5 sm:gap-1">
          {/* Suit label with symbol and letter */}
          <div
            className="w-6 sm:w-10 flex items-center justify-center gap-0.5 text-base sm:text-lg font-bold flex-shrink-0"
            style={{ color: SUIT_COLORS[suit] }}
          >
            <span>{SUIT_SYMBOLS[suit]}</span>
            <span className="hidden sm:inline text-xs opacity-70">{suit}</span>
          </div>

          {/* Cards in this suit */}
          {RANKS.map((rank) => {
            const card = `${rank}${suit}`
            const isSelected = selectedSet.has(card)
            const isUsed = usedSet.has(card) && !isSelected

            return (
              <button
                key={card}
                onClick={() => !isUsed && onCardClick(card)}
                disabled={isUsed}
                className={`
                  w-8 h-8 sm:w-9 sm:h-9 rounded flex flex-col items-center justify-center
                  text-[10px] sm:text-xs font-semibold transition-all touch-manipulation
                  ${isSelected
                    ? 'bg-[var(--primary)] text-white border-2 border-[var(--primary)]'
                    : isUsed
                      ? 'bg-[var(--muted)] border border-[var(--border)] opacity-40 cursor-not-allowed'
                      : 'bg-white border border-[var(--border)] hover:border-[var(--primary)] active:scale-95 cursor-pointer'
                  }
                `}
              >
                <span
                  className="leading-none"
                  style={{ color: isSelected ? 'white' : isUsed ? 'var(--muted-foreground)' : SUIT_COLORS[suit] }}
                >
                  {rank}
                </span>
                <span
                  className="text-[8px] sm:text-[10px] leading-none -mt-0.5"
                  style={{ color: isSelected ? 'white' : isUsed ? 'var(--muted-foreground)' : SUIT_COLORS[suit] }}
                >
                  {SUIT_SYMBOLS[suit]}
                </span>
              </button>
            )
          })}
        </div>
      ))}
    </div>
  )
}

export default CardPicker
