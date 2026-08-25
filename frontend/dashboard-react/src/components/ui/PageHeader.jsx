export function PageHeader({ title, subtitle }) {
  return (
    <div className="mb-6">
      <h1 className="font-heading text-3xl font-semibold text-lav-text">{title}</h1>
      {subtitle && <p className="mt-1.5 font-sans text-sm text-lav-muted">{subtitle}</p>}
    </div>
  )
}
