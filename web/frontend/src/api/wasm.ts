/**
 * WASM client for holdem-core.
 *
 * This module provides a client interface for the WebAssembly version of holdem-core.
 * The WASM module is lazily loaded on first use.
 */

import type {
  CanonicalHandsResponse,
  ParseCardsResponse,
  EquityRequest,
  EquityResponse,
  DrawsRequest,
  DrawsResponse,
  HealthResponse,
} from './types'

// Type for the WASM module
interface WasmModule {
  default: () => Promise<void>
  wasm_health: () => HealthResponse
  wasm_calculate_equity: (request: EquityRequest) => EquityResponse
  wasm_analyze_draws: (
    hole_cards: string[],
    board: string[],
    dead_cards: string[]
  ) => DrawsResponse
  wasm_get_canonical_hands: () => CanonicalHandsResponse
  wasm_parse_cards: (input: string) => ParseCardsResponse
}

// Lazy-loaded WASM module
let wasmModule: WasmModule | null = null
let wasmLoadPromise: Promise<WasmModule> | null = null

/**
 * Load and initialize the WASM module.
 * Returns cached module if already loaded.
 */
async function loadWasm(): Promise<WasmModule> {
  if (wasmModule) {
    return wasmModule
  }

  if (!wasmLoadPromise) {
    wasmLoadPromise = (async () => {
      console.log('[WASM] Loading holdem-wasm module...')
      try {
        // Dynamic import of the WASM module
        // The module is placed in src/wasm/ by wasm-pack during build
        const wasm = (await import('@/wasm/holdem_wasm')) as WasmModule
        await wasm.default()
        wasmModule = wasm
        console.log('[WASM] Module loaded successfully')
        return wasm
      } catch (error) {
        console.error('[WASM] Failed to load module:', error)
        wasmLoadPromise = null
        throw error
      }
    })()
  }

  return wasmLoadPromise
}

/**
 * Check if WASM module is loaded.
 */
export function isWasmLoaded(): boolean {
  return wasmModule !== null
}

/**
 * Preload the WASM module.
 * Call this early in app initialization for faster first use.
 */
export async function preloadWasm(): Promise<void> {
  await loadWasm()
}

/**
 * WASM client with same interface as HTTP and Tauri clients.
 */
export const wasmClient = {
  getHealth: async (): Promise<HealthResponse> => {
    const wasm = await loadWasm()
    return wasm.wasm_health()
  },

  getCanonicalHands: async (): Promise<CanonicalHandsResponse> => {
    const wasm = await loadWasm()
    return wasm.wasm_get_canonical_hands()
  },

  parseCards: async (input: string): Promise<ParseCardsResponse> => {
    const wasm = await loadWasm()
    return wasm.wasm_parse_cards(input)
  },

  calculateEquity: async (request: EquityRequest): Promise<EquityResponse> => {
    console.log('[WASM] calculateEquity request:', JSON.stringify(request, null, 2))
    const wasm = await loadWasm()
    try {
      const result = wasm.wasm_calculate_equity(request)
      console.log('[WASM] calculateEquity result:', result)
      return result
    } catch (error) {
      // WASM errors come as strings from JsValue::from_str()
      const errorMessage = typeof error === 'string' ? error : String(error)
      console.error('[WASM] calculateEquity error:', errorMessage)
      throw new Error(errorMessage)
    }
  },

  analyzeDraws: async (request: DrawsRequest): Promise<DrawsResponse> => {
    const wasm = await loadWasm()
    try {
      return wasm.wasm_analyze_draws(
        request.hole_cards,
        request.board,
        request.dead_cards || []
      )
    } catch (error) {
      const errorMessage = typeof error === 'string' ? error : String(error)
      console.error('[WASM] analyzeDraws error:', errorMessage)
      throw new Error(errorMessage)
    }
  },
}
