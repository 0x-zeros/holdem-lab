import { useTranslation } from 'react-i18next'
import type { PlayerEquityResult } from '../../api/types'
import { EquityBar } from './EquityBar'

interface ResultTableProps {
  players: PlayerEquityResult[]
  totalSimulations: number
  elapsedMs: number
}

export function ResultTable({ players, totalSimulations, elapsedMs }: ResultTableProps) {
  const { t } = useTranslation()

  return (
    <div className="space-y-4">
      {players.map((player) => (
        <div
          key={player.index}
          className="p-4 bg-white border border-[var(--border)] rounded-[var(--radius-lg)]"
        >
          <EquityBar
            equity={player.equity}
            winRate={player.win_rate}
            tieRate={player.tie_rate}
            label={t('player.title', { index: player.index + 1 })}
          />
          <div className="mt-2 text-sm text-[var(--muted-foreground)]">
            {player.hand_description}
            {player.combos > 1 && (
              <span className="ml-2 text-xs">({t('results.combos', { count: player.combos })})</span>
            )}
          </div>
        </div>
      ))}

      {/* Summary */}
      <div className="pt-4 border-t border-[var(--border)] text-sm text-[var(--muted-foreground)]">
        <div className="flex justify-between">
          <span>{t('results.simulations')}:</span>
          <span>{totalSimulations.toLocaleString()}</span>
        </div>
        <div className="flex justify-between">
          <span>{t('results.time')}:</span>
          <span>{elapsedMs.toFixed(1)} ms</span>
        </div>
      </div>
    </div>
  )
}

export default ResultTable
