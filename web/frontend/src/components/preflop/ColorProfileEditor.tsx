import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { X } from 'lucide-react'
import type { ColorProfile, ColorRule } from '../../data/colorProfiles'
import { DEFAULT_CUSTOM_PROFILE } from '../../data/colorProfiles'

interface ColorProfileEditorProps {
  profile: ColorProfile
  onSave: (profile: ColorProfile) => void
  onClose: () => void
}

export function ColorProfileEditor({ profile, onSave, onClose }: ColorProfileEditorProps) {
  const { t } = useTranslation()

  // Initialize rules from profile or default
  const initialRules = profile.rules.length > 0
    ? profile.rules
    : DEFAULT_CUSTOM_PROFILE.rules

  const [rules, setRules] = useState<ColorRule[]>(initialRules)

  const handleThresholdChange = (index: number, value: string) => {
    const numValue = parseInt(value, 10)
    if (isNaN(numValue) || numValue < 0 || numValue > 100) return

    setRules((prev) => {
      const newRules = [...prev]
      newRules[index] = { ...newRules[index], maxEquity: numValue }
      return newRules
    })
  }

  const handleColorChange = (index: number, color: string) => {
    setRules((prev) => {
      const newRules = [...prev]
      newRules[index] = { ...newRules[index], color }
      return newRules
    })
  }

  const handleReset = () => {
    setRules([...DEFAULT_CUSTOM_PROFILE.rules])
  }

  const handleSave = () => {
    // Sort rules by maxEquity and ensure last one is 100
    const sortedRules = [...rules].sort((a, b) => a.maxEquity - b.maxEquity)

    // Ensure the last rule has maxEquity = 100
    if (sortedRules.length > 0) {
      sortedRules[sortedRules.length - 1].maxEquity = 100
    }

    const customProfile: ColorProfile = {
      id: 'custom',
      name: '自定义',
      nameKey: 'preflop.profile.custom',
      rules: sortedRules,
    }

    onSave(customProfile)
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-end sm:items-center justify-center z-50">
      <div className="bg-white rounded-t-[var(--radius-lg)] sm:rounded-[var(--radius-lg)] shadow-xl w-full sm:w-[360px] max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]">
          <h2 className="text-base sm:text-lg font-semibold">{t('preflop.profile.customTitle')}</h2>
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-[var(--muted)] rounded-full transition-colors touch-manipulation"
          >
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* Instructions */}
          <p className="text-xs text-[var(--muted-foreground)]">
            {t('preflop.profile.instruction')}
          </p>

          {/* Rules table */}
          <div className="space-y-3">
            {/* Header row */}
            <div className="grid grid-cols-[1fr_80px] gap-3 text-xs text-[var(--muted-foreground)]">
              <span>{t('preflop.profile.threshold')}</span>
              <span>{t('preflop.profile.bgColor')}</span>
            </div>

            {/* Rule rows */}
            {rules.map((rule, index) => (
              <div key={index} className="grid grid-cols-[1fr_80px] gap-3 items-center">
                {/* Threshold input */}
                <div className="flex items-center gap-1">
                  <span className="text-sm text-[var(--muted-foreground)]">≤</span>
                  <input
                    type="number"
                    min={0}
                    max={100}
                    value={rule.maxEquity}
                    onChange={(e) => handleThresholdChange(index, e.target.value)}
                    disabled={index === rules.length - 1} // Last rule is always 100
                    className="w-16 px-2 py-1.5 border border-[var(--border)] rounded-[var(--radius-sm)] text-sm text-center disabled:bg-[var(--muted)] disabled:text-[var(--muted-foreground)]"
                  />
                  <span className="text-sm text-[var(--muted-foreground)]">%</span>
                </div>

                {/* Color picker */}
                <div className="flex items-center gap-2">
                  <input
                    type="color"
                    value={rule.color}
                    onChange={(e) => handleColorChange(index, e.target.value)}
                    className="w-8 h-8 p-0 border border-[var(--border)] rounded-[var(--radius-sm)] cursor-pointer"
                  />
                  <span className="text-xs text-[var(--muted-foreground)] font-mono hidden sm:block">
                    {rule.color}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {/* Preview hint */}
          <div className="text-xs text-[var(--muted-foreground)] pt-2">
            {t('preflop.profile.previewHint')}
          </div>
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-[var(--border)] flex gap-2 sm:gap-3">
          <button
            onClick={onClose}
            className="flex-1 py-2.5 border border-[var(--border)] rounded-[var(--radius-md)] hover:bg-[var(--muted)] transition-colors touch-manipulation text-sm"
          >
            {t('preflop.profile.cancel')}
          </button>
          <button
            onClick={handleReset}
            className="flex-1 py-2.5 border border-[var(--border)] rounded-[var(--radius-md)] hover:bg-[var(--muted)] transition-colors touch-manipulation text-sm text-[var(--muted-foreground)]"
          >
            {t('preflop.profile.reset')}
          </button>
          <button
            onClick={handleSave}
            className="flex-1 py-2.5 bg-[var(--primary)] text-white rounded-[var(--radius-md)] hover:opacity-90 transition-opacity touch-manipulation text-sm"
          >
            {t('preflop.profile.save')}
          </button>
        </div>
      </div>
    </div>
  )
}

export default ColorProfileEditor
