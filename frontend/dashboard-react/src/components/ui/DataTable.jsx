import { useEffect, useMemo, useRef, useState } from 'react'
import { motion } from 'motion/react'
import { Search, Download, ChevronUp, ChevronDown, ChevronsUpDown } from 'lucide-react'
import { cn } from '../../lib/utils'

function toCsvCell(value) {
  const str = String(value ?? '')
  return /[",\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str
}

function downloadCsv(columns, rows, filename) {
  const header = columns.map((c) => toCsvCell(c.label)).join(',')
  const lines = rows.map((row) => columns.map((c) => toCsvCell(row[c.key])).join(','))
  const blob = new Blob([[header, ...lines].join('\n')], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

/** Numeric compare when both sides parse as finite numbers, else a locale
 * string compare — covers both raw numeric columns and formatted-string
 * columns (percentages, "13 / 64", method names) without per-column config. */
function compareValues(a, b) {
  const na = Number(a)
  const nb = Number(b)
  if (Number.isFinite(na) && Number.isFinite(nb)) return na - nb
  return String(a ?? '').localeCompare(String(b ?? ''))
}

const rowContainer = {
  hidden: {},
  show: { transition: { staggerChildren: 0.05 } },
}
const rowItem = {
  hidden: { opacity: 0, x: -8 },
  show: { opacity: 1, x: 0, transition: { duration: 0.25, ease: 'easeOut' } },
}

/**
 * Simple styled data table. `columns` = [{ key, label, render?, align? }],
 * `rows` = array of row objects. Pass `animateRows` to reveal rows with a
 * line-by-line stagger the first time the table scrolls into view (opt-in —
 * default stays a plain static table for existing usages). Pass `searchable`
 * to add a filter box above the table that matches against each row's raw
 * column values (not rendered JSX), worthwhile once a table has enough rows
 * that scanning it by eye stops being practical. Pass `downloadable` (with
 * an optional `filename`) to add a CSV export button that downloads exactly
 * the currently-filtered rows, not the full unfiltered set. Pass `sortable`
 * to make every column header clickable (asc -> desc -> asc). Pass
 * `selectedKey`/`selectedValue` to highlight (and auto-scroll to) whichever
 * row has `row[selectedKey] === selectedValue` — for cross-linking a chart
 * click to its table row. `highlightKey` alone still works as a plain
 * boolean-flag highlight for any row where `row[highlightKey]` is truthy.
 */
export function DataTable({
  columns,
  rows,
  highlightKey,
  animateRows = false,
  searchable = false,
  downloadable = false,
  sortable = false,
  filename = 'data.csv',
  selectedKey,
  selectedValue,
}) {
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState(null) // { key, direction: 'asc' | 'desc' }
  const containerRef = useRef(null)

  const filteredRows = useMemo(() => {
    if (!searchable || !query.trim()) return rows
    const q = query.trim().toLowerCase()
    return rows.filter((row) => columns.some((col) => String(row[col.key] ?? '').toLowerCase().includes(q)))
  }, [rows, columns, query, searchable])

  const sortedRows = useMemo(() => {
    if (!sort) return filteredRows
    const dir = sort.direction === 'asc' ? 1 : -1
    return [...filteredRows].sort((a, b) => dir * compareValues(a[sort.key], b[sort.key]))
  }, [filteredRows, sort])

  const toggleSort = (key) => {
    setSort((prev) => {
      if (!prev || prev.key !== key) return { key, direction: 'asc' }
      if (prev.direction === 'asc') return { key, direction: 'desc' }
      return null
    })
  }

  useEffect(() => {
    if (selectedValue === undefined || selectedValue === null || !containerRef.current) return
    const el = containerRef.current.querySelector(`[data-row-value="${CSS.escape(String(selectedValue))}"]`)
    el?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [selectedValue])

  const Body = animateRows ? motion.tbody : 'tbody'
  const Row = animateRows ? motion.tr : 'tr'
  const bodyProps = animateRows
    ? { variants: rowContainer, initial: 'hidden', whileInView: 'show', viewport: { once: true, amount: 0.2 } }
    : {}
  const rowProps = animateRows ? { variants: rowItem } : {}

  return (
    <div>
      {(searchable || downloadable) && (
        <div className="mb-2 flex items-center gap-2">
          {searchable && (
            <div className="flex flex-1 items-center gap-2 rounded-lg border border-lav-border bg-lav-surface px-3 py-1.5">
              <Search size={14} className="shrink-0 text-lav-muted" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Filter rows..."
                className="w-full bg-transparent font-sans text-sm text-lav-text placeholder:text-lav-muted focus:outline-none"
              />
              {query && (
                <span className="shrink-0 font-mono text-xs text-lav-muted">
                  {filteredRows.length} / {rows.length}
                </span>
              )}
            </div>
          )}
          {downloadable && (
            <button
              onClick={() => downloadCsv(columns, sortedRows, filename)}
              className="flex shrink-0 items-center gap-1.5 rounded-lg border border-lav-border bg-lav-surface px-3 py-1.5 font-sans text-sm text-lav-muted transition-colors hover:text-lav-text"
              title="Download the rows shown as CSV"
            >
              <Download size={14} />
              <span className="hidden sm:inline">CSV</span>
            </button>
          )}
        </div>
      )}
      <div ref={containerRef} className="overflow-x-auto rounded-xl border border-lav-border bg-lav-surface">
        <table className="w-full border-collapse font-sans text-sm">
          <thead>
            <tr className="border-b border-lav-border bg-lav-bg/60">
              {columns.map((col) => {
                const active = sort?.key === col.key
                const Icon = active ? (sort.direction === 'asc' ? ChevronUp : ChevronDown) : ChevronsUpDown
                return (
                  <th
                    key={col.key}
                    className={cn(
                      'px-4 py-3 font-sans text-xs font-semibold uppercase tracking-wide text-lav-muted',
                      col.align === 'right' ? 'text-right' : 'text-left',
                    )}
                  >
                    {sortable ? (
                      <button
                        onClick={() => toggleSort(col.key)}
                        className={cn(
                          'inline-flex items-center gap-1 transition-colors hover:text-lav-text',
                          col.align === 'right' && 'flex-row-reverse',
                          active && 'text-lav-primary',
                        )}
                      >
                        {col.label}
                        <Icon size={12} className={cn('shrink-0', !active && 'opacity-40')} />
                      </button>
                    ) : (
                      col.label
                    )}
                  </th>
                )
              })}
            </tr>
          </thead>
          <Body {...bodyProps}>
            {sortedRows.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="px-4 py-6 text-center font-sans text-sm text-lav-muted">
                  No rows match "{query}".
                </td>
              </tr>
            ) : (
              sortedRows.map((row, i) => (
                <Row
                  key={i}
                  {...rowProps}
                  data-row-value={selectedKey ? row[selectedKey] : undefined}
                  className={cn(
                    'border-b border-lav-border/60 last:border-0 transition-colors hover:bg-lav-primary/5',
                    highlightKey && row[highlightKey] && 'bg-lav-primary/10',
                    selectedKey && selectedValue !== undefined && row[selectedKey] === selectedValue && 'bg-lav-primary/15',
                  )}
                >
                  {columns.map((col) => (
                    <td
                      key={col.key}
                      className={cn(
                        'px-4 py-2.5 font-mono text-[13px] text-lav-text',
                        col.align === 'right' ? 'text-right' : 'text-left',
                      )}
                    >
                      {col.render ? col.render(row) : row[col.key]}
                    </td>
                  ))}
                </Row>
              ))
            )}
          </Body>
        </table>
      </div>
    </div>
  )
}
