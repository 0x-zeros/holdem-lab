import { create } from 'zustand'
import type { EquityResponse, CanonicalHandInfo } from '../api/types'

interface Player {
  id: number
  cards: string[]       // Specific cards like ["Ah", "Kh"]
  range: string[]       // Range notation like ["QQ+", "AKs"]
  useRange: boolean     // Toggle between cards and range input
}

interface EquityStore {
  // Players
  players: Player[]
  addPlayer: () => void
  removePlayer: (id: number) => void
  updatePlayer: (id: number, updates: Partial<Player>) => void

  // Board
  board: string[]
  setBoard: (board: string[]) => void
  addBoardCard: (card: string) => void
  removeBoardCard: (index: number) => void
  clearBoard: () => void

  // Dead cards
  deadCards: string[]
  setDeadCards: (cards: string[]) => void
  toggleDeadCard: (card: string) => void

  // Settings
  numSimulations: number
  setNumSimulations: (n: number) => void

  // Results
  result: EquityResponse | null
  setResult: (result: EquityResponse | null) => void
  isCalculating: boolean
  setIsCalculating: (loading: boolean) => void

  // Canonical hands (cached)
  canonicalHands: CanonicalHandInfo[]
  setCanonicalHands: (hands: CanonicalHandInfo[]) => void

  // Range selection (for dialog)
  activeRangePlayer: number | null
  setActiveRangePlayer: (id: number | null) => void
  selectedRangeHands: Set<string>
  toggleRangeHand: (notation: string) => void
  selectAllHands: () => void
  clearSelectedHands: () => void
  applyRangeSelection: () => void

  // Reset
  reset: () => void
}

const createDefaultPlayer = (id: number): Player => ({
  id,
  cards: [],
  range: [],
  useRange: id > 0, // First player uses specific cards by default
})

export const useEquityStore = create<EquityStore>((set, get) => ({
  // Players
  players: [createDefaultPlayer(0)],
  addPlayer: () => {
    const { players } = get()
    if (players.length >= 10) return
    set({ players: [...players, createDefaultPlayer(players.length)] })
  },
  removePlayer: (id) => {
    const { players } = get()
    if (players.length <= 1) return
    set({ players: players.filter((p) => p.id !== id) })
  },
  updatePlayer: (id, updates) => {
    const { players } = get()
    set({
      players: players.map((p) => (p.id === id ? { ...p, ...updates } : p)),
    })
  },

  // Board
  board: [],
  setBoard: (board) => set({ board }),
  addBoardCard: (card) => {
    const { board } = get()
    if (board.length >= 5) return
    if (!board.includes(card)) {
      set({ board: [...board, card] })
    }
  },
  removeBoardCard: (index) => {
    const { board } = get()
    set({ board: board.filter((_, i) => i !== index) })
  },
  clearBoard: () => set({ board: [] }),

  // Dead cards
  deadCards: [],
  setDeadCards: (deadCards) => set({ deadCards }),
  toggleDeadCard: (card) => {
    const { deadCards } = get()
    if (deadCards.includes(card)) {
      set({ deadCards: deadCards.filter((c) => c !== card) })
    } else {
      set({ deadCards: [...deadCards, card] })
    }
  },

  // Settings
  numSimulations: 10000,
  setNumSimulations: (numSimulations) => set({ numSimulations }),

  // Results
  result: null,
  setResult: (result) => set({ result }),
  isCalculating: false,
  setIsCalculating: (isCalculating) => set({ isCalculating }),

  // Canonical hands
  canonicalHands: [],
  setCanonicalHands: (canonicalHands) => set({ canonicalHands }),

  // Range selection
  activeRangePlayer: null,
  setActiveRangePlayer: (activeRangePlayer) => set({ activeRangePlayer, selectedRangeHands: new Set() }),
  selectedRangeHands: new Set(),
  toggleRangeHand: (notation) => {
    const { selectedRangeHands } = get()
    const newSet = new Set(selectedRangeHands)
    if (newSet.has(notation)) {
      newSet.delete(notation)
    } else {
      newSet.add(notation)
    }
    set({ selectedRangeHands: newSet })
  },
  selectAllHands: () => {
    const { canonicalHands } = get()
    set({ selectedRangeHands: new Set(canonicalHands.map((h) => h.notation)) })
  },
  clearSelectedHands: () => set({ selectedRangeHands: new Set() }),
  applyRangeSelection: () => {
    const { activeRangePlayer, selectedRangeHands, players } = get()
    if (activeRangePlayer === null) return

    const range = Array.from(selectedRangeHands)
    set({
      players: players.map((p) =>
        p.id === activeRangePlayer ? { ...p, range, useRange: true } : p
      ),
      activeRangePlayer: null,
      selectedRangeHands: new Set(),
    })
  },

  // Reset
  reset: () =>
    set({
      players: [createDefaultPlayer(0)],
      board: [],
      deadCards: [],
      result: null,
      isCalculating: false,
      activeRangePlayer: null,
      selectedRangeHands: new Set(),
    }),
}))
