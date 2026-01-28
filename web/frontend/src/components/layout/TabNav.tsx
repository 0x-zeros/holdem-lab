import { useTranslation } from 'react-i18next'

interface Tab {
  id: string
  labelKey: string
}

interface TabNavProps {
  tabs: Tab[]
  activeTab: string
  onTabChange: (tabId: string) => void
}

export function TabNav({ tabs, activeTab, onTabChange }: TabNavProps) {
  const { t } = useTranslation()

  return (
    <div className="flex overflow-x-auto -mx-4 px-4 sm:mx-0 sm:px-0 border-b border-[var(--border)] mb-4 sm:mb-6">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          className={`flex-shrink-0 px-4 sm:px-6 py-2.5 sm:py-3 text-sm font-medium transition-colors whitespace-nowrap touch-manipulation
            ${
              activeTab === tab.id
                ? 'text-[var(--primary)] border-b-2 border-[var(--primary)]'
                : 'text-[var(--muted-foreground)] hover:text-[var(--foreground)]'
            }`}
          onClick={() => onTabChange(tab.id)}
        >
          {t(tab.labelKey)}
        </button>
      ))}
    </div>
  )
}

export default TabNav
