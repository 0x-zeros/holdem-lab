import { apiKeyStorage, type VisionProvider } from '../utils/storage'
import type { ParsedCardResult } from './vision-types'

// Card notation validation regex: rank (2-9, T, J, Q, K, A) + suit (s, h, d, c)
const CARD_REGEX = /^[2-9TJQKA][shdc]$/i

/**
 * Normalize card notation to standard format
 * e.g., "AH" -> "Ah", "10s" -> "Ts"
 */
function normalizeCard(card: string): string | null {
  let normalized = card.trim()

  // Handle "10" as "T"
  normalized = normalized.replace(/^10/, 'T')

  if (!CARD_REGEX.test(normalized)) return null

  // Uppercase rank, lowercase suit
  return normalized.charAt(0).toUpperCase() + normalized.charAt(1).toLowerCase()
}

/**
 * Parse LLM response to extract cards
 */
function parseCardsFromResponse(response: string): ParsedCardResult {
  // Try to extract JSON from response
  const jsonMatch = response.match(/\{[\s\S]*\}/)
  if (jsonMatch) {
    try {
      const parsed = JSON.parse(jsonMatch[0])

      // Parse hole cards
      const holeCards: string[][] = []
      if (Array.isArray(parsed.holeCards)) {
        for (const playerCards of parsed.holeCards) {
          if (Array.isArray(playerCards)) {
            const normalized = playerCards
              .map((c: string) => normalizeCard(String(c)))
              .filter((c): c is string => c !== null)
            if (normalized.length > 0) {
              holeCards.push(normalized)
            }
          }
        }
      }

      // Parse board cards
      const boardCards: string[] = []
      if (Array.isArray(parsed.boardCards)) {
        for (const card of parsed.boardCards) {
          const normalized = normalizeCard(String(card))
          if (normalized) {
            boardCards.push(normalized)
          }
        }
      }

      return {
        holeCards,
        boardCards,
        confidence: typeof parsed.confidence === 'number' ? parsed.confidence : 0.8,
        rawResponse: response,
      }
    } catch {
      // JSON parsing failed, fall through to regex extraction
    }
  }

  // Fallback: extract cards using regex
  const allCards = response.match(/[2-9TJQKA][shdc]/gi) || []
  const normalizedCards = allCards
    .map(normalizeCard)
    .filter((c): c is string => c !== null)

  // Remove duplicates
  const uniqueCards = [...new Set(normalizedCards)]

  return {
    holeCards: [],
    boardCards: uniqueCards,
    confidence: 0.5,
    rawResponse: response,
  }
}

/**
 * Convert Blob to base64 string
 */
function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onloadend = () => {
      const dataUrl = reader.result as string
      // Remove data URL prefix (e.g., "data:image/jpeg;base64,")
      const base64 = dataUrl.split(',')[1]
      resolve(base64)
    }
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
}

// Prompt for card recognition
const CARD_RECOGNITION_PROMPT = `分析这张德州扑克游戏图片，识别所有可见的扑克牌。

以 JSON 格式输出：
{
  "holeCards": [["Ah", "Kh"], ["Qd", "Qc"]],
  "boardCards": ["7h", "6c", "2d"],
  "confidence": 0.9
}

牌面记法规则：
- 点数: 2-9, T(10), J, Q, K, A
- 花色: h(红桃/hearts), d(方块/diamonds), c(梅花/clubs), s(黑桃/spades)
- 示例: Ah = 红桃A, Td = 方块10, Ks = 黑桃K

注意：
- holeCards 是每个玩家的手牌（通常每人2张）
- boardCards 是桌面上的公共牌（最多5张）
- 只输出能清晰识别的牌，不确定的牌请省略
- 如果只能看到部分牌，只返回能看到的牌`

/**
 * Call Qwen-VL API (Alibaba Cloud DashScope)
 */
async function callQwenVL(imageBase64: string): Promise<ParsedCardResult> {
  const apiKey = apiKeyStorage.getQwenApiKey()
  if (!apiKey) {
    throw new Error('Qwen API key not configured')
  }

  const response = await fetch(
    'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation',
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'qwen-vl-max',
        input: {
          messages: [
            {
              role: 'user',
              content: [
                {
                  type: 'image',
                  image: `data:image/jpeg;base64,${imageBase64}`,
                },
                {
                  type: 'text',
                  text: CARD_RECOGNITION_PROMPT,
                },
              ],
            },
          ],
        },
      }),
    }
  )

  if (!response.ok) {
    const errorText = await response.text()
    console.error('[Vision] Qwen API error:', response.status, errorText)
    throw new Error(`Qwen API error: ${response.status}`)
  }

  const data = await response.json()
  const content = data.output?.choices?.[0]?.message?.content || ''

  console.log('[Vision] Qwen raw response:', content)

  return parseCardsFromResponse(content)
}

/**
 * Call Doubao API (ByteDance Volcano Engine)
 */
async function callDoubao(imageBase64: string): Promise<ParsedCardResult> {
  const apiKey = apiKeyStorage.getDoubaoApiKey()
  if (!apiKey) {
    throw new Error('Doubao API key not configured')
  }

  const response = await fetch(
    'https://ark.cn-beijing.volces.com/api/v3/chat/completions',
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'doubao-vision-pro-32k',
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'image_url',
                image_url: {
                  url: `data:image/jpeg;base64,${imageBase64}`,
                },
              },
              {
                type: 'text',
                text: CARD_RECOGNITION_PROMPT,
              },
            ],
          },
        ],
      }),
    }
  )

  if (!response.ok) {
    const errorText = await response.text()
    console.error('[Vision] Doubao API error:', response.status, errorText)
    throw new Error(`Doubao API error: ${response.status}`)
  }

  const data = await response.json()
  const content = data.choices?.[0]?.message?.content || ''

  console.log('[Vision] Doubao raw response:', content)

  return parseCardsFromResponse(content)
}

/**
 * Recognize cards from an image using the configured vision API
 * @param imageBlob - Image as Blob
 * @param provider - Optional provider override
 */
export async function recognizeCards(
  imageBlob: Blob,
  provider?: VisionProvider
): Promise<ParsedCardResult> {
  const base64 = await blobToBase64(imageBlob)

  // Determine which provider to use
  const selectedProvider = provider || apiKeyStorage.getAvailableProvider()

  if (!selectedProvider) {
    throw new Error('No API key configured. Please configure an API key in settings.')
  }

  console.log('[Vision] Using provider:', selectedProvider)
  console.log('[Vision] Image size:', Math.round(base64.length / 1024), 'KB')

  const result =
    selectedProvider === 'qwen'
      ? await callQwenVL(base64)
      : await callDoubao(base64)

  console.log('[Vision] Recognition result:', {
    holeCards: result.holeCards,
    boardCards: result.boardCards,
    confidence: result.confidence,
  })

  return result
}
