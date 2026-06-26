import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { X, Eye, EyeOff, Trash2 } from 'lucide-react'
import { apiKeyStorage, maskApiKey } from '../../utils/storage'

interface SettingsDialogProps {
  isOpen: boolean
  onClose: () => void
}

// API Key input component
interface ApiKeyInputProps {
  label: string
  value: string
  onChange: (value: string) => void
  onClear: () => void
  placeholder?: string
}

function ApiKeyInput({ label, value, onChange, onClear, placeholder }: ApiKeyInputProps) {
  const [showKey, setShowKey] = useState(false)

  return (
    <div className="space-y-1.5">
      <label className="text-sm font-medium">{label}</label>
      <div className="flex gap-2">
        <div className="relative flex-1">
          <input
            type={showKey ? 'text' : 'password'}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder={placeholder}
            className="w-full px-3 py-2 pr-10 border border-[var(--border)] rounded-[var(--radius-md)] text-sm bg-[var(--background)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)] focus:ring-opacity-50"
          />
          {value && (
            <button
              type="button"
              onClick={() => setShowKey(!showKey)}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-1 hover:bg-[var(--muted)] rounded transition-colors"
              title={showKey ? 'Hide' : 'Show'}
            >
              {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
            </button>
          )}
        </div>
        {value && (
          <button
            type="button"
            onClick={onClear}
            className="p-2 hover:bg-[var(--muted)] rounded-[var(--radius-md)] transition-colors text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
            title="Clear"
          >
            <Trash2 size={18} />
          </button>
        )}
      </div>
      {value && !showKey && (
        <p className="text-xs text-[var(--muted-foreground)]">
          {maskApiKey(value)}
        </p>
      )}
    </div>
  )
}

export function SettingsDialog({ isOpen, onClose }: SettingsDialogProps) {
  const { t } = useTranslation()

  // API Key states
  const [qwenApiKey, setQwenApiKey] = useState('')
  const [doubaoApiKey, setDoubaoApiKey] = useState('')

  // Load API keys from localStorage on mount
  useEffect(() => {
    if (isOpen) {
      setQwenApiKey(apiKeyStorage.getQwenApiKey() || '')
      setDoubaoApiKey(apiKeyStorage.getDoubaoApiKey() || '')
    }
  }, [isOpen])

  // Save API keys to localStorage
  const handleQwenApiKeyChange = (value: string) => {
    setQwenApiKey(value)
    if (value.trim()) {
      apiKeyStorage.setQwenApiKey(value)
    }
  }

  const handleDoubaoApiKeyChange = (value: string) => {
    setDoubaoApiKey(value)
    if (value.trim()) {
      apiKeyStorage.setDoubaoApiKey(value)
    }
  }

  const handleClearQwenApiKey = () => {
    setQwenApiKey('')
    apiKeyStorage.clearQwenApiKey()
  }

  const handleClearDoubaoApiKey = () => {
    setDoubaoApiKey('')
    apiKeyStorage.clearDoubaoApiKey()
  }

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
        <div className="p-4 space-y-6 max-h-[60vh] overflow-y-auto">
          {/* API Keys Section */}
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-semibold mb-1">{t('settings.apiKeys')}</h3>
              <p className="text-xs text-[var(--muted-foreground)]">
                {t('settings.apiKeyHint')}
              </p>
            </div>

            <ApiKeyInput
              label={t('settings.qwenApiKey')}
              value={qwenApiKey}
              onChange={handleQwenApiKeyChange}
              onClear={handleClearQwenApiKey}
              placeholder={t('settings.apiKeyPlaceholder')}
            />

            <ApiKeyInput
              label={t('settings.doubaoApiKey')}
              value={doubaoApiKey}
              onChange={handleDoubaoApiKeyChange}
              onClear={handleClearDoubaoApiKey}
              placeholder={t('settings.apiKeyPlaceholder')}
            />
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
