/**
 * Parsed result from card recognition
 */
export interface ParsedCardResult {
  /** Each player's hole cards, e.g., [["Ah", "Kh"], ["Qd", "Qc"]] */
  holeCards: string[][]
  /** Community cards on the board, e.g., ["7h", "6c", "2d"] */
  boardCards: string[]
  /** Confidence score from 0 to 1 */
  confidence: number
  /** Raw response from the API for debugging */
  rawResponse: string
}

/**
 * Vision API error
 */
export interface VisionApiError {
  code: string
  message: string
}
