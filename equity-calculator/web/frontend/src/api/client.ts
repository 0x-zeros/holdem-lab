import type {
  CanonicalHandsResponse,
  ParseCardsResponse,
  EquityRequest,
  EquityResponse,
  DrawsRequest,
  DrawsResponse,
  HealthResponse,
  EvaluateRequest,
  EvaluateResponse,
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

// Lazy-load Tauri client to avoid issues in web environment
let tauriClientCache: ReturnType<typeof createClientInterface> | null = null

async function getTauriClient(): Promise<ReturnType<typeof createClientInterface>> {
  if (tauriClientCache) {
    return tauriClientCache
  }

  const { tauriClient } = await import('./tauri')
  tauriClientCache = tauriClient
  return tauriClient
}

// Lazy-load WASM client
let wasmClientCache: ReturnType<typeof createClientInterface> | null = null

async function getWasmClient(): Promise<ReturnType<typeof createClientInterface>> {
  if (wasmClientCache) {
    return wasmClientCache
  }

  const { wasmClient } = await import('./wasm')
  wasmClientCache = wasmClient
  return wasmClient
}

// Type helper for client interface
function createClientInterface() {
  return {
    getHealth: async (): Promise<HealthResponse> => ({ status: 'ok', version: '0.1.0' }),
    getCanonicalHands: async (): Promise<CanonicalHandsResponse> => ({ hands: [], total: 0 }),
    parseCards: async (_input: string): Promise<ParseCardsResponse> => ({ cards: [], formatted: '', valid: false, error: null }),
    calculateEquity: async (_request: EquityRequest): Promise<EquityResponse> => ({
      players: [],
      total_simulations: 0,
      elapsed_ms: 0,
    }),
    analyzeDraws: async (_request: DrawsRequest): Promise<DrawsResponse> => ({
      has_flush: false,
      has_straight: false,
      flush_draws: [],
      straight_draws: [],
      total_outs: 0,
      all_outs: [],
      is_combo_draw: false,
    }),
    evaluateHand: async (_request: EvaluateRequest): Promise<EvaluateResponse> => ({
      hand_type: '',
      description: '',
      primary_ranks: [],
      kickers: [],
    }),
  }
}

// Unified API client - uses WASM for web, Tauri for desktop
export const apiClient = {
  getHealth: async (): Promise<HealthResponse> => {
    if (isTauri()) {
      const client = await getTauriClient()
      return client.getHealth()
    }
    const client = await getWasmClient()
    return client.getHealth()
  },

  getCanonicalHands: async (): Promise<CanonicalHandsResponse> => {
    if (isTauri()) {
      const client = await getTauriClient()
      return client.getCanonicalHands()
    }
    const client = await getWasmClient()
    return client.getCanonicalHands()
  },

  parseCards: async (input: string): Promise<ParseCardsResponse> => {
    if (isTauri()) {
      const client = await getTauriClient()
      return client.parseCards(input)
    }
    const client = await getWasmClient()
    return client.parseCards(input)
  },

  calculateEquity: async (request: EquityRequest): Promise<EquityResponse> => {
    console.log('[API] calculateEquity request:', JSON.stringify(request, null, 2))
    console.log('[API] isTauri:', isTauri())

    if (isTauri()) {
      console.log('[API] Using Tauri client')
      const client = await getTauriClient()
      return client.calculateEquity(request)
    }

    console.log('[API] Using WASM client')
    const client = await getWasmClient()
    return client.calculateEquity(request)
  },

  analyzeDraws: async (request: DrawsRequest): Promise<DrawsResponse> => {
    if (isTauri()) {
      const client = await getTauriClient()
      return client.analyzeDraws(request)
    }
    const client = await getWasmClient()
    return client.analyzeDraws(request)
  },

  evaluateHand: async (request: EvaluateRequest): Promise<EvaluateResponse> => {
    if (isTauri()) {
      const client = await getTauriClient()
      return client.evaluateHand(request)
    }
    const client = await getWasmClient()
    return client.evaluateHand(request)
  },
}
