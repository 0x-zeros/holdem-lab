interface CardProps {
  notation?: string  // e.g., "Ah", "Kd"
  empty?: boolean
  selected?: boolean
  onClick?: () => void
  size?: 'sm' | 'md' | 'lg'
}

const SUIT_COLORS: Record<string, string> = {
  h: 'var(--suit-hearts)',
  d: 'var(--suit-diamonds)',
  c: 'var(--suit-clubs)',
  s: 'var(--suit-spades)',
}

const SUIT_SYMBOLS: Record<string, string> = {
  h: '♥',
  d: '♦',
  c: '♣',
  s: '♠',
}

const SIZE_CLASSES = {
  sm: 'w-10 h-14 text-sm',
  md: 'w-14 h-20 text-base',
  lg: 'w-16 h-24 text-lg',
}

export function Card({ notation, empty, selected, onClick, size = 'md' }: CardProps) {
  if (empty || !notation) {
    return (
      <div
        className={`${SIZE_CLASSES[size]} rounded-[var(--radius-md)] border-2 border-dashed border-[var(--border)] flex items-center justify-center text-[var(--muted-foreground)] cursor-pointer hover:border-[var(--primary)] transition-colors`}
        onClick={onClick}
      >
        ?
      </div>
    )
  }

  const rank = notation[0]
  const suit = notation[1]?.toLowerCase()
  const suitColor = SUIT_COLORS[suit] || 'inherit'
  const suitSymbol = SUIT_SYMBOLS[suit] || suit

  return (
    <div
      className={`${SIZE_CLASSES[size]} rounded-[var(--radius-md)] border ${
        selected ? 'border-[var(--primary)] ring-2 ring-[var(--primary)]' : 'border-[var(--border)]'
      } bg-white flex flex-col items-center justify-center cursor-pointer hover:shadow-md transition-all`}
      onClick={onClick}
    >
      <span className="font-bold" style={{ color: suitColor }}>
        {rank}
      </span>
      <span style={{ color: suitColor }}>{suitSymbol}</span>
    </div>
  )
}

export default Card
