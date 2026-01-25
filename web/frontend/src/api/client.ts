import type {
  CanonicalHandsResponse,
  ParseCardsResponse,
  EquityRequest,
  EquityResponse,
  DrawsRequest,
  DrawsResponse,
  HealthResponse,
} from './types'

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

export const apiClient = {
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

export { ApiError }
