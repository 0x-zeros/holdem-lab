import { useEffect, useState, useMemo } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { apiClient } from './api/client'
import type { EquityRequest } from './api/types'
import { useEquityStore } from './store'
import { BoardInput } from './components/cards'
import { PlayerRow } from './components/players'
import { ResultTable } from './components/results'
import { HandMatrix } from './components/matrix'
import { LanguageSwitcher } from './components/layout'

function App() {
  const { t } = useTranslation()
  const [showRangeDialog, setShowRangeDialog] = useState(false)

  const {
    players,
    updatePlayer,
    addPlayer,
    removePlayer,
    setPlayerMode,
    totalPlayers,
    setTotalPlayers,
    board,
    setBoard,
    clearBoard,
    numSimulations,
    canonicalHands,
    setCanonicalHands,
    selectedRangeHands,
    toggleRangeHand,
    selectAllHands,
    clearSelectedHands,
    activeRangePlayer,
    setActiveRangePlayer,
    applyRangeSelection,
  } = useEquityStore()

  // Fetch canonical hands for matrix
  const { data: canonicalData } = useQuery({
    queryKey: ['canonical'],
    queryFn: () => apiClient.getCanonicalHands(),
  })

  // Store canonical hands in Zustand
  useEffect(() => {
    if (canonicalData) {
      setCanonicalHands(canonicalData.hands)
    }
  }, [canonicalData, setCanonicalHands])

  // Calculate equity mutation
  const equityMutation = useMutation({
    mutationFn: (request: EquityRequest) => apiClient.calculateEquity(request),
  })

  // Calculate used cards (for card picker to disable)
  const getUsedCardsForPlayer = useMemo(() => {
    return (playerId: number) => {
      const usedCards: string[] = [...board]
      players.forEach((p) => {
        if (p.id !== playerId && p.mode === 'cards') {
          usedCards.push(...p.cards)
        }
      })
      return usedCards
    }
  }, [players, board])

  const usedCardsForBoard = useMemo(() => {
    const usedCards: string[] = []
    players.forEach((p) => {
      if (p.mode === 'cards') {
        usedCards.push(...p.cards)
      }
    })
    return usedCards
  }, [players])

  const handleCalculate = () => {
    const validPlayers = players.filter((p) => {
      switch (p.mode) {
        case 'cards': return p.cards.length === 2
        case 'range': return p.range.length > 0
        case 'random': return true
      }
    })

    // 构建请求玩家列表
    const requestPlayers = validPlayers.map((p) => {
      switch (p.mode) {
        case 'cards': return { cards: p.cards }
        case 'range': return { range: p.range }
        case 'random': return { random: true }
      }
    })

    // 自动补充随机玩家到总玩家数
    const playersNeeded = totalPlayers - requestPlayers.length
    for (let i = 0; i < playersNeeded; i++) {
      requestPlayers.push({ random: true })
    }

    const request: EquityRequest = {
      players: requestPlayers,
      board: board.length > 0 ? board : undefined,
      num_simulations: numSimulations,
    }

    equityMutation.mutate(request)
  }

  const handleOpenRangeDialog = (playerId: number) => {
    const player = players.find((p) => p.id === playerId)
    if (player) {
      setActiveRangePlayer(playerId)
      // Pre-populate selection with current range
      player.range.forEach((h) => {
        if (!selectedRangeHands.has(h)) {
          toggleRangeHand(h)
        }
      })
      setShowRangeDialog(true)
    }
  }

  const handleApplyRange = () => {
    applyRangeSelection()
    setShowRangeDialog(false)
  }

  // 只要有1个有效玩家，且总玩家数>=2，就可以计算（剩余玩家自动补充为随机）
  const validPlayerCount = players.filter((p) => {
    switch (p.mode) {
      case 'cards': return p.cards.length === 2
      case 'range': return p.range.length > 0
      case 'random': return true
    }
  }).length
  const canCalculate = validPlayerCount >= 1 && totalPlayers >= 2

  return (
    <div className="min-h-screen bg-[var(--background)]">
      {/* Header */}
      <header className="border-b border-[var(--border)] px-6 py-4 flex items-center justify-between">
        <h1 className="text-xl font-semibold text-[var(--foreground)]">
          {t('app.title')}
        </h1>
        <LanguageSwitcher />
      </header>

      <main className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Panel - Input */}
          <div className="space-y-6">
            {/* Board Input */}
            <section className="bg-[var(--muted)] rounded-[var(--radius-lg)] p-6">
              <BoardInput
                cards={board}
                onCardsChange={setBoard}
                onClear={clearBoard}
                usedCards={usedCardsForBoard}
              />
            </section>

            {/* Total Players Selector */}
            <section className="flex items-center gap-4 py-2">
              <span className="text-sm font-medium">{t('settings.totalPlayers')}</span>
              <div className="flex gap-1">
                {[2, 3, 4, 5, 6, 7, 8, 9, 10].map((n) => (
                  <button
                    key={n}
                    onClick={() => setTotalPlayers(n)}
                    className={`px-2 py-1 text-xs rounded-[var(--radius-sm)] ${
                      totalPlayers === n
                        ? 'bg-[var(--primary)] text-white'
                        : 'bg-[var(--muted)] text-[var(--muted-foreground)] hover:bg-[var(--border)]'
                    }`}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </section>

            {/* Players */}
            <section className="space-y-4">
              {players.map((player) => (
                <PlayerRow
                  key={player.id}
                  index={player.id}
                  cards={player.cards}
                  range={player.range}
                  mode={player.mode}
                  onCardsChange={(cards) => updatePlayer(player.id, { cards })}
                  onRangeClick={() => handleOpenRangeDialog(player.id)}
                  onSetMode={(mode) => setPlayerMode(player.id, mode)}
                  onRemove={() => removePlayer(player.id)}
                  canRemove={players.length > 1}
                  usedCards={getUsedCardsForPlayer(player.id)}
                />
              ))}

              {players.length < totalPlayers && (
                <button
                  onClick={() => addPlayer('random')}
                  className="w-full py-2 border-2 border-dashed border-[var(--border)] rounded-[var(--radius-lg)] text-[var(--muted-foreground)] hover:border-[var(--primary)] hover:text-[var(--primary)] transition-colors"
                >
                  {t('actions.addPlayer')}
                </button>
              )}
            </section>

            {/* Calculate Button */}
            <button
              onClick={handleCalculate}
              disabled={!canCalculate || equityMutation.isPending}
              className="w-full py-3 bg-[var(--primary)] text-white font-medium rounded-[var(--radius-md)] hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {equityMutation.isPending ? t('actions.calculating') : t('actions.evaluate')}
            </button>
          </div>

          {/* Right Panel - Results */}
          <div className="space-y-6">
            {/* Results */}
            {equityMutation.data && (
              <section className="bg-[var(--muted)] rounded-[var(--radius-lg)] p-6">
                <h2 className="text-lg font-medium mb-4">{t('results.title')}</h2>
                <ResultTable
                  players={equityMutation.data.players}
                  totalSimulations={equityMutation.data.total_simulations}
                  elapsedMs={equityMutation.data.elapsed_ms}
                />
              </section>
            )}

            {/* Error */}
            {equityMutation.isError && (
              <div className="bg-red-50 border border-red-200 text-red-700 rounded-[var(--radius-lg)] p-4">
                {t('results.error')}: {(equityMutation.error as Error).message}
              </div>
            )}

          </div>
        </div>
      </main>

      {/* Range Selection Dialog */}
      {showRangeDialog && activeRangePlayer !== null && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-[var(--radius-lg)] p-6 w-fit max-w-[95vw] mx-4 max-h-[90vh] overflow-auto">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-medium">
                {t('matrix.selectRange', { index: activeRangePlayer + 1 })}
              </h2>
              <button
                onClick={() => {
                  setShowRangeDialog(false)
                  setActiveRangePlayer(null)
                }}
                className="text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
              >
                ×
              </button>
            </div>

            <HandMatrix
              hands={canonicalHands}
              selectedHands={selectedRangeHands}
              onToggleHand={toggleRangeHand}
              onSelectAll={selectAllHands}
              onClearAll={clearSelectedHands}
            />

            <div className="mt-4 flex gap-3">
              <button
                onClick={() => {
                  setShowRangeDialog(false)
                  setActiveRangePlayer(null)
                }}
                className="flex-1 py-2 border border-[var(--border)] rounded-[var(--radius-md)] hover:bg-[var(--muted)]"
              >
                {t('actions.cancel')}
              </button>
              <button
                onClick={handleApplyRange}
                className="flex-1 py-2 bg-[var(--primary)] text-white rounded-[var(--radius-md)] hover:opacity-90"
              >
                {t('actions.apply', { count: selectedRangeHands.size })}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
