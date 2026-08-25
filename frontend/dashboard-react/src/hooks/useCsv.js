import { useEffect, useState } from 'react'
import { loadCsv } from '../lib/csv'

/** Load a CSV from public/data/ into { data, loading, error }. */
export function useCsv(name) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    loadCsv(name)
      .then((rows) => { if (!cancelled) setData(rows) })
      .catch((err) => { if (!cancelled) setError(err) })
    return () => { cancelled = true }
  }, [name])

  return { data, loading: !data && !error, error }
}
