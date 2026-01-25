// Card types
export interface CardInfo {
  notation: string
  rank: string
  suit: string
  suit_symbol: string
}

export interface CanonicalHandInfo {
  notation: string
  high_rank: string
  low_rank: string
  suited: boolean
  is_pair: boolean
  num_combos: number
  matrix_row: number
  matrix_col: number
}

export interface CanonicalHandsResponse {
  hands: CanonicalHandInfo[]
  total: number
}

export interface ParseCardsResponse {
  cards: CardInfo[]
  formatted: string
  valid: boolean
  error: string | null
}

// Equity types
export interface PlayerHandInput {
  cards?: string[]
  range?: string[]
}

export interface EquityRequest {
  players: PlayerHandInput[]
  board?: string[]
  dead_cards?: string[]
  num_simulations?: number
}

export interface PlayerEquityResult {
  index: number
  hand_description: string
  equity: number
  win_rate: number
  tie_rate: number
  combos: number
}

export interface EquityResponse {
  players: PlayerEquityResult[]
  total_simulations: number
  elapsed_ms: number
}

// Draws types
export interface DrawsRequest {
  hole_cards: string[]
  board: string[]
  dead_cards?: string[]
}

export interface FlushDrawInfo {
  suit: string
  suit_symbol: string
  cards_held: number
  outs: string[]
  out_count: number
  is_nut: boolean
  draw_type: 'flush_draw' | 'backdoor_flush'
}

export interface StraightDrawInfo {
  draw_type: 'open_ended' | 'gutshot' | 'double_gutshot' | 'backdoor_straight'
  needed_ranks: number[]
  outs: string[]
  out_count: number
  high_card: number
  is_nut: boolean
}

export interface DrawsResponse {
  has_flush: boolean
  has_straight: boolean
  flush_draws: FlushDrawInfo[]
  straight_draws: StraightDrawInfo[]
  total_outs: number
  all_outs: string[]
  is_combo_draw: boolean
}

// Health check
export interface HealthResponse {
  status: string
  version: string
}
