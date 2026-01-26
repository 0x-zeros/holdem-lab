/**
 * TypeScript declarations for holdem-wasm module.
 *
 * The actual WASM module is built by wasm-pack and placed in src/wasm/
 * during the build process.
 */

declare module '@/wasm/holdem_wasm' {
  /**
   * Initialize the WASM module. Must be called before using other functions.
   */
  export default function init(): Promise<void>

  /**
   * Health check endpoint.
   */
  export function wasm_health(): { status: string; version: string }

  /**
   * Calculate equity for multiple players.
   * @param request - EquityRequest object
   * @returns EquityResponse object
   */
  export function wasm_calculate_equity(request: unknown): unknown

  /**
   * Analyze draws for hole cards and board.
   * @param hole_cards - Array of card strings
   * @param board - Array of board card strings
   * @param dead_cards - Array of dead card strings
   * @returns DrawsResponse object
   */
  export function wasm_analyze_draws(
    hole_cards: unknown,
    board: unknown,
    dead_cards: unknown
  ): unknown

  /**
   * Get all 169 canonical starting hands.
   * @returns CanonicalHandsResponse object
   */
  export function wasm_get_canonical_hands(): unknown

  /**
   * Parse cards from string notation.
   * @param input - Card string (e.g., "AhKh")
   * @returns ParseCardsResponse object
   */
  export function wasm_parse_cards(input: string): unknown
}
