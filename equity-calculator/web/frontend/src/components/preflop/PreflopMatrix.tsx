import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ChevronDown, Settings2 } from 'lucide-react'
import type { CanonicalHandInfo } from '../../api/types'
import { useEquityStore } from '../../store'
import {
  BUILTIN_PROFILES,
  getProfileById,
  getProfileColor,
  type ColorProfile,
} from '../../data/colorProfiles'
import { ColorProfileEditor } from './ColorProfileEditor'

// Import split preflop equity files (one per player count)
import equity2 from '../../data/preflop-equity-2.json'
import equity3 from '../../data/preflop-equity-3.json'
import equity4 from '../../data/preflop-equity-4.json'
import equity5 from '../../data/preflop-equity-5.json'
import equity6 from '../../data/preflop-equity-6.json'
import equity7 from '../../data/preflop-equity-7.json'
import equity8 from '../../data/preflop-equity-8.json'
import equity9 from '../../data/preflop-equity-9.json'
import equity10 from '../../data/preflop-equity-10.json'

// Map player count to equity data
const equityDataMap: Record<number, Record<string, number>> = {
  2: equity2,
  3: equity3,
  4: equity4,
  5: equity5,
  6: equity6,
  7: equity7,
  8: equity8,
  9: equity9,
  10: equity10,
}

interface PreflopMatrixProps {
  hands: CanonicalHandInfo[]
}

export function PreflopMatrix({ hands }: PreflopMatrixProps) {
  const { t } = useTranslation()
  const [numPlayers, setNumPlayers] = useState(2)
  const [showProfileDropdown, setShowProfileDropdown] = useState(false)
  const [showEditor, setShowEditor] = useState(false)

  const {
    colorProfileId,
    setColorProfileId,
    customColorProfile,
    setCustomColorProfile,
  } = useEquityStore()

  // Get current profile
  const currentProfile = useMemo(
    () => getProfileById(colorProfileId, customColorProfile),
    [colorProfileId, customColorProfile]
  )

  // Get equity data for current player count
  const equityData = equityDataMap[numPlayers] || {}

  // Build matrix grid
  const matrix = useMemo(() => {
    const grid: (CanonicalHandInfo | null)[][] = Array(13)
      .fill(null)
      .map(() => Array(13).fill(null))

    for (const hand of hands) {
      grid[hand.matrix_row][hand.matrix_col] = hand
    }
    return grid
  }, [hands])

  const handleProfileSelect = (id: string) => {
    if (id === 'custom') {
      setColorProfileId('custom')
      setShowEditor(true)
    } else {
      setColorProfileId(id)
    }
    setShowProfileDropdown(false)
  }

  const handleSaveCustomProfile = (profile: ColorProfile) => {
    setCustomColorProfile(profile)
    setShowEditor(false)
  }

  // Generate legend items based on profile
  const legendItems = useMemo(() => {
    if (currentProfile.isGradient) {
      // 渐变模式：高/中/低
      return [
        { label: t('preflop.high'), color: getProfileColor(80, currentProfile).bg },
        { label: t('preflop.medium'), color: getProfileColor(45, currentProfile).bg },
        { label: t('preflop.low'), color: getProfileColor(10, currentProfile).bg },
      ]
    } else {
      // 阈值模式：根据 rules 生成
      const items: { label: string; color: string }[] = []
      let prevMax = 0
      for (const rule of currentProfile.rules) {
        const label = rule.maxEquity === 100
          ? `≥${prevMax}%`
          : prevMax === 0
            ? `<${rule.maxEquity}%`
            : `${prevMax}-${rule.maxEquity}%`
        items.push({ label, color: rule.color })
        prevMax = rule.maxEquity
      }
      return items
    }
  }, [currentProfile, t])

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Header and selectors */}
      <div className="flex flex-col gap-3">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <h2 className="text-base sm:text-lg font-semibold">{t('preflop.title')}</h2>
          <div className="flex items-center gap-2">
            <span className="text-xs sm:text-sm text-[var(--muted-foreground)]">{t('preflop.players')}</span>
            <div className="flex gap-1 overflow-x-auto pb-1">
              {[2, 3, 4, 5, 6, 7, 8, 9, 10].map((n) => (
                <button
                  key={n}
                  onClick={() => setNumPlayers(n)}
                  className={`flex-shrink-0 px-2 py-1 text-xs rounded-[var(--radius-sm)] transition-colors touch-manipulation ${
                    numPlayers === n
                      ? 'bg-[var(--primary)] text-white'
                      : 'bg-[var(--muted)] text-[var(--muted-foreground)] hover:bg-[var(--border)]'
                  }`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Color profile selector */}
        <div className="flex items-center gap-2">
          <span className="text-xs sm:text-sm text-[var(--muted-foreground)]">{t('preflop.colorProfile')}</span>
          <div className="relative">
            <button
              onClick={() => setShowProfileDropdown(!showProfileDropdown)}
              className="flex items-center gap-1 px-2 py-1 text-xs sm:text-sm bg-[var(--muted)] rounded-[var(--radius-sm)] hover:bg-[var(--border)] transition-colors touch-manipulation"
            >
              <span>{t(currentProfile.nameKey)}</span>
              <ChevronDown className="w-3 h-3 sm:w-4 sm:h-4" />
            </button>

            {showProfileDropdown && (
              <>
                {/* Backdrop */}
                <div
                  className="fixed inset-0 z-10"
                  onClick={() => setShowProfileDropdown(false)}
                />
                {/* Dropdown menu */}
                <div className="absolute top-full left-0 mt-1 bg-white border border-[var(--border)] rounded-[var(--radius-md)] shadow-lg z-20 min-w-[140px]">
                  {BUILTIN_PROFILES.map((profile) => (
                    <button
                      key={profile.id}
                      onClick={() => handleProfileSelect(profile.id)}
                      className={`w-full px-3 py-2 text-left text-xs sm:text-sm hover:bg-[var(--muted)] transition-colors ${
                        colorProfileId === profile.id ? 'bg-[var(--muted)] font-medium' : ''
                      }`}
                    >
                      {t(profile.nameKey)}
                    </button>
                  ))}
                  <div className="border-t border-[var(--border)]" />
                  <button
                    onClick={() => handleProfileSelect('custom')}
                    className={`w-full px-3 py-2 text-left text-xs sm:text-sm hover:bg-[var(--muted)] transition-colors flex items-center gap-1 ${
                      colorProfileId === 'custom' ? 'bg-[var(--muted)] font-medium' : ''
                    }`}
                  >
                    <Settings2 className="w-3 h-3" />
                    <span>{t('preflop.profile.custom')}</span>
                  </button>
                </div>
              </>
            )}
          </div>

          {/* Edit button for custom profile */}
          {colorProfileId === 'custom' && (
            <button
              onClick={() => setShowEditor(true)}
              className="px-2 py-1 text-xs sm:text-sm text-[var(--primary)] hover:underline touch-manipulation"
            >
              {t('preflop.profile.edit')}
            </button>
          )}
        </div>
      </div>

      {/* Description */}
      <p className="text-xs sm:text-sm text-[var(--muted-foreground)]">
        {t('preflop.description')}
      </p>

      {/* Matrix Grid - responsive with horizontal scroll on mobile */}
      <div className="overflow-x-auto -mx-4 px-4 sm:mx-0 sm:px-0 pb-2">
        <div className="inline-grid gap-1 sm:gap-1.5" style={{ gridTemplateColumns: 'repeat(13, minmax(28px, 42px))' }}>
          {matrix.map((row, rowIdx) =>
            row.map((hand, colIdx) => {
              const equity = hand ? equityData[hand.notation] : null
              const colors = equity != null ? getProfileColor(equity, currentProfile) : null
              return (
                <div
                  key={`${rowIdx}-${colIdx}`}
                  className="relative group"
                >
                  <div
                    className="w-[28px] h-[28px] sm:w-[42px] sm:h-[42px] flex flex-col items-center justify-center rounded-[var(--radius-sm)] text-[9px] sm:text-xs cursor-default"
                    style={{
                      backgroundColor: colors?.bg || 'var(--muted)',
                      color: colors?.text || 'var(--muted-foreground)',
                    }}
                  >
                    <span className="font-medium leading-none">{hand?.notation || ''}</span>
                    {equity != null && (
                      <span className="text-[7px] sm:text-[10px] leading-none opacity-90">{equity}%</span>
                    )}
                  </div>
                  {/* Tooltip - hide on mobile */}
                  {hand && (
                    <div className="hidden sm:block absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-[var(--foreground)] text-white text-xs rounded opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap z-20">
                      {hand.notation} · {equity ?? '-'}% · {hand.num_combos} {t('results.combos', { count: hand.num_combos })}
                    </div>
                  )}
                </div>
              )
            })
          )}
        </div>
      </div>

      {/* Legend - dynamic based on profile */}
      <div className="flex flex-wrap items-center gap-2 sm:gap-4 text-[10px] sm:text-xs">
        <span className="text-[var(--muted-foreground)]">{t('preflop.legend')}:</span>
        {legendItems.map((item, idx) => (
          <div key={idx} className="flex items-center gap-1">
            <div
              className="w-3 h-3 sm:w-4 sm:h-4 rounded-[var(--radius-sm)]"
              style={{ backgroundColor: item.color }}
            />
            <span>{item.label}</span>
          </div>
        ))}
      </div>

      {/* Custom profile editor dialog */}
      {showEditor && (
        <ColorProfileEditor
          profile={customColorProfile || currentProfile}
          onSave={handleSaveCustomProfile}
          onClose={() => setShowEditor(false)}
        />
      )}
    </div>
  )
}

export default PreflopMatrix
