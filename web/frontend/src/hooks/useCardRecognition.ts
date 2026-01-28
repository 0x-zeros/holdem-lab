import { useState, useCallback } from 'react'
import { recognizeCards } from '../api/vision'
import { useEquityStore } from '../store'
import type { ParsedCardResult } from '../api/vision-types'

interface UseCardRecognitionResult {
  /** Execute card recognition on an image */
  recognize: (imageBlob: Blob) => Promise<ParsedCardResult>
  /** Whether recognition is in progress */
  isRecognizing: boolean
  /** Error message if recognition failed */
  error: string | null
  /** Clear the error state */
  clearError: () => void
}

/**
 * Hook for card recognition with automatic state application
 */
export function useCardRecognition(): UseCardRecognitionResult {
  const [isRecognizing, setIsRecognizing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const applyRecognizedCards = useEquityStore((state) => state.applyRecognizedCards)

  const recognize = useCallback(
    async (imageBlob: Blob): Promise<ParsedCardResult> => {
      setIsRecognizing(true)
      setError(null)

      try {
        const result = await recognizeCards(imageBlob)

        console.log('[useCardRecognition] Recognition successful:', {
          holeCards: result.holeCards,
          boardCards: result.boardCards,
          confidence: result.confidence,
        })

        // Apply recognized cards to the store
        applyRecognizedCards(result)

        return result
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Recognition failed'
        setError(message)
        console.error('[useCardRecognition] Error:', err)
        throw err
      } finally {
        setIsRecognizing(false)
      }
    },
    [applyRecognizedCards]
  )

  const clearError = useCallback(() => {
    setError(null)
  }, [])

  return {
    recognize,
    isRecognizing,
    error,
    clearError,
  }
}
