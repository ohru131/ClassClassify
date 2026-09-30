import { useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, CheckCircle2, Download, ExternalLink, GripVertical, Loader2, RotateCcw, Sheet, X } from 'lucide-react'
import type { Problem } from '../solver/types'
import type { ColumnReport, Report } from '../solver/evaluate'
import { pairStatus, rowColor, type PairGroupStatus, type PairTag } from '../solver/pairs'
import { Segmented, Stat, classColor } from './ui'
import { useT } from '../i18n/web'
import { violationText } from '../solver/labels'

type Tab = 'classes' | 'balance' | 'checks'

export function Results({
  problem,
  classOf,
  k,
  report,
  onMove,
  onReset,
  onDownload,
  edited,
  google,
}: {
  problem: Problem
  classOf: number[]
  k: number
  report: Report
  onMove: (student: number, to: number) => void
  onReset: () => void
  onDownload: () => void
  edited: boolean
  /** Google 連携が有効なときだけ渡す */
  google?: { label: string; busy: boolean; url: string | null; onSave: () => void }
}) {
  const { t, className } = useT()
  const [tab, setTab] = useState<Tab>('classes')
  const [selected, setSelected] = useState<number | null>(null)
  const perfect = report.totalExcess === 0
  const sizeGap = Math.max(...report.sizes) - Math.min(...report.sizes)

  return (
    <section className="space-y-6">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label={t('statClasses')} value={k} sub={t('statClassesSub', { n: problem.students.length })} />
        <Stat label={t('statGap')} value={sizeGap} sub={t('statGapSub', { min: Math.min(...report.sizes), max: Math.max(...report.sizes) })} tone={sizeGap <= 1 ? 'good' : 'bad'} />
        <Stat
          label={t('statBalance')}
          value={perfect ? t('perfect') : report.totalExcess}
          sub={perfect ? t('balancePerfectSub') : t('balanceSub')}
          tone={perfect ? 'good' : 'default'}
        />
        <Stat
          label={t('statViolations')}
          value={report.violations.length}
          sub={report.violations.length ? t('violationsSub') : t('violationsOk')}
          tone={report.violations.length ? 'bad' : 'good'}
        />
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <Segmented
          value={tab}
          onChange={setTab}
          options={[
            { value: 'classes', label: t('tabClasses') },
            { value: 'balance', label: t('tabBalance') },
            { value: 'checks', label: report.violations.length ? t('tabChecksN', { n: report.violations.length }) : t('tabChecks') },
          ]}
        />
        <div className="flex gap-2">
          {edited && (
            <button type="button" className="btn-ghost" onClick={onReset}>
              <RotateCcw className="size-4" /> {t('undoMoves')}
            </button>
          )}
          {google && (
            <button type="button" className="btn-ghost" onClick={google.onSave} disabled={google.busy}>
              {google.busy ? <Loader2 className="size-4 animate-spin" /> : <Sheet className="size-4 text-emerald-600" />} {google.label}
            </button>
          )}
          <button type="button" className="btn-primary" onClick={onDownload}>
            <Download className="size-4" /> {t('saveExcel')}
          </button>
        </div>
      </div>

      {google?.url && (
        <a
          href={google.url}
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-2 rounded-2xl border border-emerald-200 bg-emerald-50 px-5 py-3 text-sm font-medium text-emerald-800 hover:bg-emerald-100"
        >
          <CheckCircle2 className="size-4" /> {t('savedToSheet')}
          <ExternalLink className="ml-auto size-4" />
        </a>
      )}

      {tab === 'classes' && (
        <ClassBoard problem={problem} classOf={classOf} k={k} selected={selected} setSelected={setSelected} onMove={onMove} report={report} />
      )}
      {tab === 'balance' && (
        <div className="grid gap-4 lg:grid-cols-2">
          {report.columns.map((c) => (
            <BalanceCard key={c.column} col={c} k={k} />
          ))}
        </div>
      )}
      {tab === 'checks' && <Checks problem={problem} report={report} classOf={classOf} />}

      {selected !== null && (
        <div className="fixed inset-x-0 bottom-[max(1rem,env(safe-area-inset-bottom))] z-30 mx-auto flex w-fit max-w-[calc(100%-2rem)] flex-wrap items-center gap-2 rounded-2xl border border-slate-200 bg-white/95 px-4 py-3 shadow-2xl backdrop-blur">
          <span className="text-sm font-semibold text-slate-800">
            {t('moveTitle', { who: `${problem.students[selected].no}:${problem.students[selected].name}` })}
          </span>
          {Array.from({ length: k }, (_, c) => (
            <button
              key={c}
              type="button"
              disabled={classOf[selected] === c}
              onClick={() => {
                onMove(selected, c)
                setSelected(null)
              }}
              className={`rounded-lg px-2.5 py-1 text-xs font-bold ring-1 transition hover:brightness-95 disabled:opacity-30 ${classColor(c).soft}`}
            >
              {className(c)}
            </button>
          ))}
          <button type="button" onClick={() => setSelected(null)} className="ml-1 rounded-lg p-1 text-slate-400 hover:bg-slate-100" aria-label={t('close')}>
            <X className="size-4" />
          </button>
        </div>
      )}
    </section>
  )
}

function ClassBoard({
  problem,
  classOf,
  k,
  selected,
  setSelected,
  onMove,
  report,
}: {
  problem: Problem
  classOf: number[]
  k: number
  selected: number | null
  setSelected: (s: number | null) => void
  onMove: (student: number, to: number) => void
  report: Report
}) {
  const { t, className, file } = useT()
  const [over, setOver] = useState<number | null>(null)
  const cardRefs = useRef<(HTMLDivElement | null)[]>([])
  const flagged = useMemo(() => new Set(report.violations.flatMap((v) => v.students)), [report])
  const tags = useMemo(() => {
    // 各生徒の「該当」項目を短いタグで表示
    const flagCols = problem.columns.filter((c) => c.enabled && c.kind === 'flag')
    return problem.students.map((s) => flagCols.filter((c) => s.values[c.name] !== '').map((c) => c.name))
  }, [problem])
  const pairs = useMemo(() => pairStatus(problem, classOf, file.tagPrefix), [problem, classOf, file])
  const [colorize, setColorize] = useState(true)
  const [focus, setFocus] = useState<{ kind: 'wanted' | 'unwanted'; group: number } | null>(null)
  // 名簿エディタでグループが変わったら強調を解除（index がずれるため）
  useEffect(() => setFocus(null), [problem.wantedGroups, problem.unwantedGroups])
  const focusMembers = useMemo(() => {
    if (!focus) return null
    const g = (focus.kind === 'wanted' ? problem.wantedGroups : problem.unwantedGroups)[focus.group]
    return g ? new Set(g) : null
  }, [focus, problem])

  return (
    <>
      <p className="text-xs text-slate-500">
        <span className="hidden sm:inline">{t('boardHintDesktop')}</span>
        <span className="sm:hidden">{t('boardHintMobile')}</span>
        {t('boardHintTail')}
      </p>
      {pairs.groups.length > 0 && (
        <PairLegend
          problem={problem}
          classOf={classOf}
          groups={pairs.groups}
          focus={focus}
          setFocus={setFocus}
          colorize={colorize}
          setColorize={setColorize}
        />
      )}
      {/* スマホ: クラスへジャンプ */}
      <div className="-mx-4 flex gap-2 overflow-x-auto px-4 [scrollbar-width:none] sm:hidden">
        {Array.from({ length: k }, (_, c) => (
          <button
            key={c}
            type="button"
            onClick={() => cardRefs.current[c]?.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' })}
            className={`shrink-0 rounded-full px-3 py-1 text-xs font-bold ring-1 ${classColor(c).soft}`}
          >
            {t('classChip', { cls: className(c), n: report.sizes[c] })}
          </button>
        ))}
      </div>
      <div className="-mx-4 flex snap-x snap-mandatory gap-3 overflow-x-auto px-4 pb-2 [scrollbar-width:none] sm:mx-0 sm:grid sm:grid-cols-2 sm:gap-4 sm:overflow-visible sm:px-0 sm:pb-0 xl:grid-cols-4">
        {Array.from({ length: k }, (_, c) => {
          const members = problem.students.map((_, i) => i).filter((i) => classOf[i] === c)
          const color = classColor(c)
          return (
            <div
              key={c}
              ref={(el) => {
                cardRefs.current[c] = el
              }}
              onDragOver={(e) => {
                e.preventDefault()
                setOver(c)
              }}
              onDragLeave={() => setOver(null)}
              onDrop={(e) => {
                e.preventDefault()
                setOver(null)
                const s = Number(e.dataTransfer.getData('text/plain'))
                if (Number.isFinite(s)) onMove(s, c)
              }}
              className={`card w-[85%] shrink-0 snap-center overflow-hidden transition sm:w-auto ${over === c ? 'ring-2 ring-indigo-400' : ''}`}
            >
              <div className={`h-1.5 bg-gradient-to-r ${color.bar}`} />
              <div className="flex items-center justify-between px-5 pb-2 pt-4">
                <div className="flex items-center gap-2">
                  <span className={`size-2.5 rounded-full ${color.dot}`} />
                  <span className="text-lg font-extrabold text-slate-900">{className(c)}</span>
                </div>
                <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-bold tabular-nums text-slate-600">{t('studentsN', { n: members.length })}</span>
              </div>
              <ul className="max-h-[28rem] space-y-1 overflow-y-auto px-3 pb-4">
                {members.map((i) => {
                  const s = problem.students[i]
                  const pt = pairs.tags[i]
                  const bg = colorize ? rowColor(pt) : null
                  const dim = focusMembers && !focusMembers.has(i)
                  return (
                    <li
                      key={i}
                      draggable
                      onDragStart={(e) => e.dataTransfer.setData('text/plain', String(i))}
                      onClick={() => setSelected(selected === i ? null : i)}
                      style={bg && selected !== i ? { backgroundColor: bg.bg } : undefined}
                      className={`group flex cursor-grab items-center gap-2 rounded-xl px-2 py-1.5 text-sm transition active:cursor-grabbing ${
                        selected === i ? 'bg-indigo-50 ring-1 ring-indigo-300' : bg ? 'hover:brightness-95' : 'hover:bg-slate-50'
                      } ${dim ? 'opacity-25' : ''} ${focusMembers?.has(i) && selected !== i ? 'ring-2 ring-slate-900/70' : ''}`}
                    >
                      <GripVertical className="size-3.5 shrink-0 text-slate-300 group-hover:text-slate-400" />
                      <span className="w-7 shrink-0 font-mono text-xs tabular-nums text-slate-400">{s.no}</span>
                      <span className={`min-w-[5.5em] shrink-0 truncate font-medium ${flagged.has(i) ? 'text-rose-600' : 'text-slate-800'}`}>{s.name}</span>
                      <span className="ml-auto flex min-w-0 flex-wrap justify-end gap-0.5">
                        {pt.map((t) => (
                          <PairBadge key={`${t.kind}${t.group}`} tag={t} />
                        ))}
                        {tags[i].slice(0, pt.length ? 2 : 3).map((t) => (
                          <span key={t} className="rounded bg-slate-100 px-1 py-px text-[10px] font-medium text-slate-500">
                            {t}
                          </span>
                        ))}
                      </span>
                    </li>
                  )
                })}
              </ul>
            </div>
          )
        })}
      </div>
    </>
  )
}

function PairBadge({ tag }: { tag: PairTag }) {
  const { t } = useT()
  return (
    <span
      title={`${t(tag.kind === 'wanted' ? 'wantedBadgeTitle' : 'unwantedBadgeTitle', { label: tag.label })}${tag.ok ? '' : t('unmetSuffix')}`}
      className={`rounded px-1 py-px text-[10px] font-bold ring-1 ${tag.ok ? '' : 'ring-2 ring-rose-500'}`}
      style={{ backgroundColor: tag.kind === 'wanted' ? '#ffffffb3' : tag.color.bg, color: tag.color.fg, ['--tw-ring-color' as string]: tag.ok ? `${tag.color.fg}55` : undefined }}
    >
      {tag.label}
      {!tag.ok && '!'}
    </span>
  )
}

function PairLegend({
  problem,
  classOf,
  groups,
  focus,
  setFocus,
  colorize,
  setColorize,
}: {
  problem: Problem
  classOf: number[]
  groups: PairGroupStatus[]
  focus: { kind: 'wanted' | 'unwanted'; group: number } | null
  setFocus: (f: { kind: 'wanted' | 'unwanted'; group: number } | null) => void
  colorize: boolean
  setColorize: (v: boolean) => void
}) {
  const { t, className, file } = useT()
  const ok = groups.filter((g) => g.ok).length
  return (
    <div className="card p-4">
      <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="text-sm font-bold text-slate-900">{t('pairings')}</span>
        <span className={`text-xs font-semibold ${ok === groups.length ? 'text-emerald-600' : 'text-rose-600'}`}>
          {t('pairingsMet', { ok, n: groups.length })}
        </span>
        <span className="text-xs text-slate-400">{t('clickToHighlight')}</span>
        <label className="ml-auto inline-flex cursor-pointer items-center gap-1.5 text-xs font-medium text-slate-500">
          <input type="checkbox" checked={colorize} onChange={(e) => setColorize(e.target.checked)} className="size-4 accent-indigo-600" />
          {t('colorize')}
        </label>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {groups.map((g) => {
          const active = focus?.kind === g.kind && focus.group === g.group
          const classes = [...new Set(g.members.map((i) => classOf[i] + 1))].sort((a, b) => a - b)
          return (
            <button
              key={`${g.kind}${g.group}`}
              type="button"
              onClick={() => setFocus(active ? null : { kind: g.kind, group: g.group })}
              className={`inline-flex max-w-full items-center gap-1.5 rounded-lg px-2 py-1 text-left text-xs transition ${
                active ? 'ring-2 ring-slate-900' : 'ring-1 ring-black/5 hover:ring-black/20'
              } ${g.ok ? '' : 'outline outline-2 outline-rose-500'}`}
              style={{ backgroundColor: g.color.bg, color: g.color.fg }}
            >
              <span className="shrink-0 whitespace-nowrap font-extrabold">{g.label}</span>
              <span className="truncate font-medium text-slate-700">{g.members.map((i) => problem.students[i].name).join(file.joinSep)}</span>
              <span className="shrink-0 whitespace-nowrap font-semibold">
                → {classes.map((c) => className(c - 1)).join('/')} {g.ok ? '✓' : '✗'}
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

function BalanceCard({ col, k }: { col: ColumnReport; k: number }) {
  const { t, className, num } = useT()
  const numeric = col.kind === 'numeric'
  const max = Math.max(1, ...col.rows.flat())
  return (
    <div className="card min-w-0 p-5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
        <div className="font-bold text-slate-900">{col.column}</div>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-slate-400">{t('weightN', { w: col.weight })}</span>
          {numeric ? (
            <span className="rounded-full bg-slate-100 px-2 py-0.5 font-semibold text-slate-600">{t('average')}</span>
          ) : col.excess === 0 ? (
            <span className="rounded-full bg-emerald-50 px-2 py-0.5 font-semibold text-emerald-700">{t('even')}</span>
          ) : (
            <span className="rounded-full bg-amber-50 px-2 py-0.5 font-semibold text-amber-700">{t('offBy', { n: col.excess })}</span>
          )}
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-slate-400">
              <th className="py-1 pr-2 text-left font-medium"></th>
              {Array.from({ length: k }, (_, c) => (
                <th key={c} className="whitespace-nowrap px-1 py-1 text-center font-semibold">
                  {className(c)}
                </th>
              ))}
              <th className="px-1 py-1 text-center font-medium">{t('ideal')}</th>
            </tr>
          </thead>
          <tbody>
            {col.levels.map((level, l) => (
              <tr key={level}>
                <td className="max-w-24 truncate py-1 pr-2 text-xs font-semibold text-slate-600">{numeric ? t('averageLevel') : level}</td>
                {col.rows[l].map((v, c) => {
                  const ideal = col.ideal[l]
                  const ok = numeric || (v >= Math.floor(ideal) && v <= Math.ceil(ideal))
                  const alpha = numeric ? 0.15 : 0.12 + 0.5 * (v / max)
                  return (
                    <td key={c} className="px-1 py-1">
                      <div
                        className={`rounded-lg py-1.5 text-center font-bold tabular-nums ${ok ? 'text-indigo-900' : 'text-amber-900 ring-2 ring-amber-400'}`}
                        style={{ backgroundColor: ok ? `rgb(99 102 241 / ${alpha})` : `rgb(251 191 36 / ${alpha + 0.1})` }}
                      >
                        {numeric ? num(v, 2) : v}
                      </div>
                    </td>
                  )
                })}
                <td className="px-1 py-1 text-center text-xs tabular-nums text-slate-400">{num(col.ideal[l], numeric ? 2 : 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function Checks({ problem, report, classOf }: { problem: Problem; report: Report; classOf: number[] }) {
  const { t, fileLang } = useT()
  const total = problem.wantedGroups.length + problem.unwantedGroups.length
  if (report.violations.length === 0)
    return (
      <div className="card flex items-center gap-4 p-6">
        <CheckCircle2 className="size-10 shrink-0 text-emerald-500" />
        <div>
          <div className="font-bold text-slate-900">{t('allMet')}</div>
          <div className="text-sm text-slate-500">
            {total ? t('allMetBody', { w: problem.wantedGroups.length, u: problem.unwantedGroups.length }) : t('noPairsBody')}
          </div>
        </div>
      </div>
    )
  return (
    <div className="card divide-y divide-slate-100">
      {report.violations.map((v, i) => (
        <div key={i} className="flex items-center gap-3 px-5 py-3.5 text-sm">
          <AlertTriangle className="size-4 shrink-0 text-rose-500" />
          <span className="text-slate-700">{violationText(fileLang, v, problem, classOf)}</span>
        </div>
      ))}
    </div>
  )
}
