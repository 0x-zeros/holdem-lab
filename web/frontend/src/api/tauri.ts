import { invoke } from '@tauri-apps/api/core'
import type {
  CanonicalHandsResponse,
  ParseCardsResponse,
  EquityRequest,
  EquityResponse,
  DrawsRequest,
  DrawsResponse,
  HealthResponse,
  CanonicalHandInfo,
} from './types'

// Tauri IPC client - calls Rust backend directly
export const tauriClient = {
  // Health check (simulated for Tauri)
  getHealth: async (): Promise<HealthResponse> => {
    return {
      status: 'healthy',
      version: '0.1.0-tauri',
    }
  },

  // Cards
  getCanonicalHands: async (): Promise<CanonicalHandsResponse> => {
    const hands = await invoke<CanonicalHandInfo[]>('get_canonical_hands')
    return {
      hands,
      total: hands.length,
    }
  },

  parseCards: async (input: string): Promise<ParseCardsResponse> => {
    return invoke<ParseCardsResponse>('parse_cards', { input })
  },

  // Equity
  calculateEquity: async (request: EquityRequest): Promise<EquityResponse> => {
    // Transform request to match Rust command format
    const tauriRequest = {
      players: request.players.map((p) => ({
        cards: p.cards,
        range: p.range,
      })),
      board: request.board || [],
      dead_cards: request.dead_cards || [],
      num_simulations: request.num_simulations || 10000,
    }

    return invoke<EquityResponse>('calculate_equity', { request: tauriRequest })
  },

  // Draws
  analyzeDraws: async (request: DrawsRequest): Promise<DrawsResponse> => {
    return invoke<DrawsResponse>('analyze_draws', {
      holeCards: request.hole_cards,
      board: request.board,
      deadCards: request.dead_cards,
    })
  },
}
