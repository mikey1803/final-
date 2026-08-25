import { useCallback, useEffect, useState } from 'react'
import {
  batchRun,
  checkHealth,
  compareLive,
  explainFailureType,
  getDemoState,
  injectManual,
  injectNatural,
  resetDemo,
  triggerShift,
} from '../../lib/demoApi'

export function useLiveDemo() {
  const [backendUp, setBackendUp] = useState(null) // null = checking
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [demoState, setDemoState] = useState(null)

  const [compareLoading, setCompareLoading] = useState(false)
  const [compareResult, setCompareResult] = useState(null)
  const [shiftLoading, setShiftLoading] = useState(false)
  const [batchLoading, setBatchLoading] = useState(false)
  const [batchResult, setBatchResult] = useState(null)

  const refreshState = useCallback(async () => {
    try {
      const s = await getDemoState()
      setDemoState(s)
      setBackendUp(true)
      return true
    } catch {
      setBackendUp(false)
      return false
    }
  }, [])

  useEffect(() => {
    checkHealth()
      .then(() => refreshState())
      .catch(() => setBackendUp(false))
  }, [refreshState])

  const run = useCallback(
    async (fn) => {
      setLoading(true)
      setError(null)
      try {
        const data = await fn()
        setResult(data)
        await refreshState()
      } catch (e) {
        setError(e.message || 'Request failed')
      } finally {
        setLoading(false)
      }
    },
    [refreshState],
  )

  const runManual = useCallback((failureType) => run(() => injectManual(failureType)), [run])
  const runNatural = useCallback((text) => run(() => injectNatural(text)), [run])
  const reset = useCallback(
    () =>
      run(async () => {
        await resetDemo()
        setCompareResult(null)
        setBatchResult(null)
        return null
      }),
    [run],
  )

  const runCompare = useCallback(
    async (failureType) => {
      setCompareLoading(true)
      setError(null)
      try {
        const data = await compareLive(failureType)
        setCompareResult(data)
        await refreshState()
      } catch (e) {
        setError(e.message || 'Compare request failed')
      } finally {
        setCompareLoading(false)
      }
    },
    [refreshState],
  )

  const runShift = useCallback(async () => {
    setShiftLoading(true)
    setError(null)
    try {
      await triggerShift()
      await refreshState()
    } catch (e) {
      setError(e.message || 'Trigger-shift request failed')
    } finally {
      setShiftLoading(false)
    }
  }, [refreshState])

  const runBatch = useCallback(async (nEpisodes = 50) => {
    setBatchLoading(true)
    setError(null)
    try {
      const data = await batchRun(nEpisodes)
      setBatchResult(data)
    } catch (e) {
      setError(e.message || 'Batch-run request failed')
    } finally {
      setBatchLoading(false)
    }
  }, [])

  return {
    backendUp,
    loading,
    error,
    result,
    demoState,
    runManual,
    runNatural,
    reset,
    refreshState,
    compareLoading,
    compareResult,
    runCompare,
    shiftLoading,
    runShift,
    batchLoading,
    batchResult,
    runBatch,
    explainFailureType,
  }
}
