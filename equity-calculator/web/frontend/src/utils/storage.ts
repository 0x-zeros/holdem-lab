// localStorage keys for API configuration
const STORAGE_KEYS = {
  QWEN_API_KEY: 'holdem_lab_qwen_api_key',
  DOUBAO_API_KEY: 'holdem_lab_doubao_api_key',
  PREFERRED_PROVIDER: 'holdem_lab_preferred_provider',
} as const

export type VisionProvider = 'qwen' | 'doubao'

export interface ApiKeyStorage {
  getQwenApiKey: () => string | null
  setQwenApiKey: (key: string) => void
  clearQwenApiKey: () => void
  getDoubaoApiKey: () => string | null
  setDoubaoApiKey: (key: string) => void
  clearDoubaoApiKey: () => void
  getPreferredProvider: () => VisionProvider | null
  setPreferredProvider: (provider: VisionProvider) => void
  hasAnyApiKey: () => boolean
  getAvailableProvider: () => VisionProvider | null
}

export const apiKeyStorage: ApiKeyStorage = {
  // Qwen API Key
  getQwenApiKey: () => localStorage.getItem(STORAGE_KEYS.QWEN_API_KEY),
  setQwenApiKey: (key) => {
    if (key.trim()) {
      localStorage.setItem(STORAGE_KEYS.QWEN_API_KEY, key.trim())
    }
  },
  clearQwenApiKey: () => localStorage.removeItem(STORAGE_KEYS.QWEN_API_KEY),

  // Doubao API Key
  getDoubaoApiKey: () => localStorage.getItem(STORAGE_KEYS.DOUBAO_API_KEY),
  setDoubaoApiKey: (key) => {
    if (key.trim()) {
      localStorage.setItem(STORAGE_KEYS.DOUBAO_API_KEY, key.trim())
    }
  },
  clearDoubaoApiKey: () => localStorage.removeItem(STORAGE_KEYS.DOUBAO_API_KEY),

  // Preferred provider
  getPreferredProvider: () => {
    const provider = localStorage.getItem(STORAGE_KEYS.PREFERRED_PROVIDER)
    if (provider === 'qwen' || provider === 'doubao') {
      return provider
    }
    return null
  },
  setPreferredProvider: (provider) => {
    localStorage.setItem(STORAGE_KEYS.PREFERRED_PROVIDER, provider)
  },

  // Check if any API key is configured
  hasAnyApiKey: () => {
    return !!(
      localStorage.getItem(STORAGE_KEYS.QWEN_API_KEY) ||
      localStorage.getItem(STORAGE_KEYS.DOUBAO_API_KEY)
    )
  },

  // Get the first available provider with API key
  getAvailableProvider: () => {
    const preferred = apiKeyStorage.getPreferredProvider()
    if (preferred === 'qwen' && apiKeyStorage.getQwenApiKey()) {
      return 'qwen'
    }
    if (preferred === 'doubao' && apiKeyStorage.getDoubaoApiKey()) {
      return 'doubao'
    }
    // Fallback: return first available
    if (apiKeyStorage.getQwenApiKey()) return 'qwen'
    if (apiKeyStorage.getDoubaoApiKey()) return 'doubao'
    return null
  },
}

/**
 * Mask API key for display (show first 4 and last 4 chars)
 * Example: "sk-xxxxxxxxxxxx1234" -> "sk-x••••••••1234"
 */
export function maskApiKey(key: string): string {
  if (!key) return ''
  if (key.length <= 8) return '••••••••'
  return `${key.slice(0, 4)}${'••••••••'}${key.slice(-4)}`
}
