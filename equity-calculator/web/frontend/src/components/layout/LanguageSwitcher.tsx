import { useTranslation } from 'react-i18next'

export function LanguageSwitcher() {
  const { i18n } = useTranslation()
  const currentLang = i18n.language

  const toggleLanguage = () => {
    const newLang = currentLang.startsWith('zh') ? 'en' : 'zh-CN'
    i18n.changeLanguage(newLang)
  }

  return (
    <button
      onClick={toggleLanguage}
      className="px-3 py-1 text-sm border border-[var(--border)] rounded-[var(--radius-md)] hover:bg-[var(--muted)] transition-colors"
    >
      {currentLang.startsWith('zh') ? 'EN' : '中文'}
    </button>
  )
}

export default LanguageSwitcher
