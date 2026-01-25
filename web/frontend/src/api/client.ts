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
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new ApiError(response.status, error.detail || 'Request failed')
  }

  return response.json()
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

// Unified API client
export const apiClient = {
  getHealth: async (): Promise<HealthResponse> => {
    if (isTauri()) {
      const client = await getTauriClient()
      return client.getHealth()
    }
    return webClient.getHealth()
  },

  getCanonicalHands: async (): Promise<CanonicalHandsResponse> => {
    if (isTauri()) {
      const client = await getTauriClient()
      return client.getCanonicalHands()
    }
    return webClient.getCanonicalHands()
  },

  parseCards: async (input: string): Promise<ParseCardsResponse> => {
    if (isTauri()) {
      const client = await getTauriClient()
      return client.parseCards(input)
    }
    return webClient.parseCards(input)
  },

  calculateEquity: async (request: EquityRequest): Promise<EquityResponse> => {
    if (isTauri()) {
      const client = await getTauriClient()
      return client.calculateEquity(request)
    }
    return webClient.calculateEquity(request)
  },

  analyzeDraws: async (request: DrawsRequest): Promise<DrawsResponse> => {
    if (isTauri()) {
      const client = await getTauriClient()
      return client.analyzeDraws(request)
    }
    return webClient.analyzeDraws(request)
  },
}

export { ApiError }
