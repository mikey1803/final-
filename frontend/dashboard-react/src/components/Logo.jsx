/** Custom mark for the project, used identically in both navs (they
 * previously used two different unrelated lucide icons, Workflow and
 * Compass, which read as inconsistent branding). A single stroke that dips
 * then recovers past its starting height: the project's actual subject
 * (a failure, then an adaptive recovery), not a generic icon-in-a-box. */
export function LogoMark({ size = 16, className }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      className={className}
      aria-hidden="true"
    >
      <path
        d="M4 8.5 L9.5 18 L20 5.5"
        stroke="currentColor"
        strokeWidth={2.75}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export function LogoBadge({ size = 32, iconSize, className }) {
  return (
    <span
      className={`flex shrink-0 items-center justify-center rounded-lg bg-lav-primary text-lav-bg ${className ?? ''}`}
      style={{ width: size, height: size }}
    >
      <LogoMark size={iconSize ?? Math.round(size * 0.5)} />
    </span>
  )
}
