import { useEffect, useState } from 'react'

const cache = new Map()

/** Fetch + cache a JSON file from public/data/<name>. */
export function useJson(name) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    const url = `${import.meta.env.BASE_URL}data/${name}`
    const promise = cache.get(url) ?? fetch(url).then((res) => {
      if (!res.ok) throw new Error(`Failed to load ${name}: ${res.status}`)
      return res.json()
    })
    cache.set(url, promise)
    promise
      .then((json) => { if (!cancelled) setData(json) })
      .catch((err) => { if (!cancelled) setError(err) })
    return () => { cancelled = true }
  }, [name])

  return { data, loading: !data && !error, error }
}
