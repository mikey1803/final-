// Minimal dependency-free CSV parser + loader.
// Every CSV shipped in public/data/ is plain numeric/string data with no
// embedded commas or quoted fields, so a naive split is reliable here and
// avoids pulling in a parsing library for a static results viewer.

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/)
  const headers = lines[0].split(',')
  return lines.slice(1).map((line) => {
    const cells = line.split(',')
    const row = {}
    headers.forEach((h, i) => {
      const raw = cells[i]
      const num = Number(raw)
      row[h] = raw !== '' && !Number.isNaN(num) ? num : raw
    })
    return row
  })
}

const cache = new Map()

/** Fetch + parse a CSV from public/data/<name>.csv, cached per session. */
export async function loadCsv(name) {
  if (cache.has(name)) return cache.get(name)
  const promise = fetch(`${import.meta.env.BASE_URL}data/${name}`)
    .then((res) => {
      if (!res.ok) throw new Error(`Failed to load ${name}: ${res.status}`)
      return res.text()
    })
    .then(parseCsv)
  cache.set(name, promise)
  return promise
}
