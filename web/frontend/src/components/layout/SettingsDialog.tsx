import { useTranslation } from 'react-i18next'
import { X } from 'lucide-react'
import { useEquityStore } from '../../store'

interface SettingsDialogProps {
  isOpen: boolean
  onClose: () => void
}

// Format number with commas
const formatNumber = (n: number): string => {
  return n.toLocaleString()
}

export function SettingsDialog({ isOpen, onClose }: SettingsDialogProps) {
  const { t } = useTranslation()
  const { numSimulations, setNumSimulations } = useEquityStore()

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-end sm:items-center justify-center z-50">
      <div className="bg-[var(--background)] rounded-t-[var(--radius-lg)] sm:rounded-[var(--radius-lg)] shadow-xl w-full sm:w-[400px] max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]">
          <h2 className="text-lg font-semibold">{t('settings.title')}</h2>
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-[var(--muted)] rounded-full transition-colors touch-manipulation"
          >
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* Real-time simulations */}
          <div className="space-y-2">
            <label className="text-sm font-medium">
              {t('settings.realTimeSimulations')}: {formatNumber(numSimulations)}
            </label>
            <input
              type="range"
              min={10000}
              max={100000}
              step={10000}
              value={numSimulations}
              onChange={(e) => setNumSimulations(Number(e.target.value))}
              className="w-full h-2 bg-[var(--muted)] rounded-lg appearance-none cursor-pointer accent-[var(--primary)] touch-manipulation"
            />
            <div className="flex justify-between text-xs text-[var(--muted-foreground)]">
              <span>10,000</span>
              <span>100,000</span>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-[var(--border)] flex justify-end">
          <button
            onClick={onClose}
            className="w-full sm:w-auto px-4 py-2.5 bg-[var(--muted)] rounded-[var(--radius-md)] hover:bg-[var(--border)] transition-colors touch-manipulation"
          >
            {t('settings.close')}
          </button>
        </div>
      </div>
    </div>
  )
}

export default SettingsDialog
