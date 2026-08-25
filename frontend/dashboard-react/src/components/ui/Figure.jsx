import { Card } from './Card'

/** Static figure (PNG) in a lavender-styled card, matching Streamlit's stImage container. */
export function Figure({ src, alt, caption }) {
  return (
    <Card className="p-4" hover={false}>
      <img src={src} alt={alt} className="w-full rounded-lg" />
      {caption && <p className="mt-2 text-center font-sans text-xs text-lav-muted">{caption}</p>}
    </Card>
  )
}
