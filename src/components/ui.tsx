import type { ReactNode } from 'react'

export const CLASS_COLORS = [
  { dot: 'bg-indigo-500', soft: 'bg-indigo-50 text-indigo-700 ring-indigo-200', bar: 'from-indigo-500 to-indigo-400' },
  { dot: 'bg-fuchsia-500', soft: 'bg-fuchsia-50 text-fuchsia-700 ring-fuchsia-200', bar: 'from-fuchsia-500 to-fuchsia-400' },
  { dot: 'bg-emerald-500', soft: 'bg-emerald-50 text-emerald-700 ring-emerald-200', bar: 'from-emerald-500 to-emerald-400' },
  { dot: 'bg-amber-500', soft: 'bg-amber-50 text-amber-700 ring-amber-200', bar: 'from-amber-500 to-amber-400' },
  { dot: 'bg-sky-500', soft: 'bg-sky-50 text-sky-700 ring-sky-200', bar: 'from-sky-500 to-sky-400' },
  { dot: 'bg-rose-500', soft: 'bg-rose-50 text-rose-700 ring-rose-200', bar: 'from-rose-500 to-rose-400' },
  { dot: 'bg-teal-500', soft: 'bg-teal-50 text-teal-700 ring-teal-200', bar: 'from-teal-500 to-teal-400' },
  { dot: 'bg-violet-500', soft: 'bg-violet-50 text-violet-700 ring-violet-200', bar: 'from-violet-500 to-violet-400' },
]
export const classColor = (c: number) => CLASS_COLORS[c % CLASS_COLORS.length]

export function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <svg viewBox="0 0 32 32" className="size-9 drop-shadow-sm" aria-hidden>
        <defs>
          <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#6366f1" />
            <stop offset="1" stopColor="#d946ef" />
          </linearGradient>
        </defs>
        <rect width="32" height="32" rx="9" fill="url(#lg)" />
        <g fill="#fff">
          <rect x="7" y="7" width="8" height="8" rx="2" />
          <rect x="17" y="7" width="8" height="8" rx="2" opacity=".6" />
          <rect x="7" y="17" width="8" height="8" rx="2" opacity=".6" />
          <rect x="17" y="17" width="8" height="8" rx="2" opacity=".85" />
        </g>
      </svg>
      <div className="leading-tight">
        <div className="text-lg font-extrabold tracking-tight text-slate-900">Mosaic</div>
        <div className="text-[11px] font-medium tracking-wide text-slate-500">クラス編成オプティマイザー</div>
      </div>
    </div>
  )
}

export function StepHeader({ n, title, desc, done }: { n: number; title: string; desc?: ReactNode; done?: boolean }) {
  return (
    <div className="mb-5 flex items-start gap-3">
      <div
        className={`grid size-8 shrink-0 place-items-center rounded-full text-sm font-bold ${
          done ? 'bg-emerald-500 text-white' : 'bg-slate-900 text-white'
        }`}
      >
        {done ? '✓' : n}
      </div>
      <div>
        <h2 className="text-base font-bold text-slate-900">{title}</h2>
        {desc && <p className="mt-0.5 text-sm text-slate-500">{desc}</p>}
      </div>
    </div>
  )
}

export function Stat({ label, value, sub, tone = 'default' }: { label: string; value: ReactNode; sub?: ReactNode; tone?: 'default' | 'good' | 'bad' }) {
  const toneCls = tone === 'good' ? 'text-emerald-600' : tone === 'bad' ? 'text-rose-600' : 'text-slate-900'
  return (
    <div className="card p-5">
      <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">{label}</div>
      <div className={`mt-1 text-3xl font-extrabold tabular-nums tracking-tight ${toneCls}`}>{value}</div>
      {sub && <div className="mt-1 text-xs text-slate-500">{sub}</div>}
    </div>
  )
}

export function Segmented<T extends string | number>({
  value,
  options,
  onChange,
}: {
  value: T
  options: { value: T; label: ReactNode }[]
  onChange: (v: T) => void
}) {
  return (
    <div className="inline-flex rounded-xl bg-slate-100 p-1">
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          onClick={() => onChange(o.value)}
          className={`rounded-lg px-3 py-1.5 text-sm font-semibold transition ${
            o.value === value ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-700'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}
