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
    <div className="flex border-b border-[var(--border)] mb-6">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          className={`px-6 py-3 text-sm font-medium transition-colors
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
