import type {
  CanonicalHandsResponse,
  ParseCardsResponse,
  EquityRequest,
  EquityResponse,
  DrawsRequest,
  DrawsResponse,
  HealthResponse,
} from './types'

// Environment detection
declare global {
  interface Window {
    __TAURI__?: unknown
    __TAURI_INTERNALS__?: unknown
  }
}

const isTauri = (): boolean => {
  return typeof window !== 'undefined' && ('__TAURI__' in window || '__TAURI_INTERNALS__' in window)
}

const isWasmMode = (): boolean => {
  // Not WASM mode if running in Tauri
  if (isTauri()) return false

  // Check for explicit WASM mode via environment variable
  if (import.meta.env.VITE_USE_WASM === 'true') return true

  // Auto-detect GitHub Pages (no backend available)
  if (typeof window !== 'undefined' && window.location.hostname.includes('github.io')) {
    return true
  }

  return false
}

// Web API client (HTTP)
const API_BASE = '/api'

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

async function fetchApi<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const url = `${API_BASE}${endpoint}`
  console.log('[fetchApi] Request:', options?.method || 'GET', url)
  if (options?.body) {
    console.log('[fetchApi] Body:', options.body)
  }

  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })

  console.log('[fetchApi] Response status:', response.status)

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    console.error('[fetchApi] Error:', error)
    throw new ApiError(response.status, error.detail || 'Request failed')
  }

  const data = await response.json()
  console.log('[fetchApi] Response data:', data)
  return data
}

const webClient = {
  // Health check
  getHealth: () => fetchApi<HealthResponse>('/health'),

  // Cards
  getCanonicalHands: () => fetchApi<CanonicalHandsResponse>('/canonical'),

  parseCards: (input: string) =>
    fetchApi<ParseCardsResponse>('/parse-cards', {
      method: 'POST',
      body: JSON.stringify({ input }),
    }),

  // Equity
  calculateEquity: (request: EquityRequest) =>
    fetchApi<EquityResponse>('/equity', {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  // Draws
  analyzeDraws: (request: DrawsRequest) =>
    fetchApi<DrawsResponse>('/draws', {
      method: 'POST',
      body: JSON.stringify(request),
    }),
}

// Lazy-load Tauri client to avoid issues in web environment
let tauriClientCache: typeof webClient | null = null

async function getTauriClient(): Promise<typeof webClient> {
  if (tauriClientCache) {
    return tauriClientCache
  }

  const { tauriClient } = await import('./tauri')
  tauriClientCache = tauriClient
  return tauriClient
}

// Lazy-load WASM client
let wasmClientCache: typeof webClient | null = null

async function getWasmClient(): Promise<typeof webClient> {
  if (wasmClientCache) {
    return wasmClientCache
  }

  const { wasmClient } = await import('./wasm')
  wasmClientCache = wasmClient
  return wasmClient
}

// Unified API client
export const apiClient = {
  getHealth: async (): Promise<HealthResponse> => {
    if (isTauri()) {
      const client = await getTauriClient()
      return client.getHealth()
    }
    if (isWasmMode()) {
      const client = await getWasmClient()
      return client.getHealth()
    }
    return webClient.getHealth()
  },

  getCanonicalHands: async (): Promise<CanonicalHandsResponse> => {
    if (isTauri()) {
      const client = await getTauriClient()
      return client.getCanonicalHands()
    }
    if (isWasmMode()) {
      const client = await getWasmClient()
      return client.getCanonicalHands()
    }
    return webClient.getCanonicalHands()
  },

  parseCards: async (input: string): Promise<ParseCardsResponse> => {
    if (isTauri()) {
      const client = await getTauriClient()
      return client.parseCards(input)
    }
    if (isWasmMode()) {
      const client = await getWasmClient()
      return client.parseCards(input)
    }
    return webClient.parseCards(input)
  },

  calculateEquity: async (request: EquityRequest): Promise<EquityResponse> => {
    console.log('[API] calculateEquity request:', JSON.stringify(request, null, 2))
    console.log('[API] isTauri:', isTauri(), 'isWasmMode:', isWasmMode())

    if (isTauri()) {
      console.log('[API] Using Tauri client')
      const client = await getTauriClient()
      return client.calculateEquity(request)
    }

    if (isWasmMode()) {
      console.log('[API] Using WASM client')
      const client = await getWasmClient()
      return client.calculateEquity(request)
    }

    console.log('[API] Using Web client, endpoint:', `${API_BASE}/equity`)
    return webClient.calculateEquity(request)
  },

  analyzeDraws: async (request: DrawsRequest): Promise<DrawsResponse> => {
    if (isTauri()) {
      const client = await getTauriClient()
      return client.analyzeDraws(request)
    }
    if (isWasmMode()) {
      const client = await getWasmClient()
      return client.analyzeDraws(request)
    }
    return webClient.analyzeDraws(request)
  },
}

export { ApiError }
