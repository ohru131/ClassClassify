import { useRef, useState } from 'react'
import { ArrowLeftRight, ChevronRight, Download, ExternalLink, Pencil, FileSpreadsheet, Loader2, Minus, Plus, Sparkles, Upload, Users } from 'lucide-react'
import type { ColumnSpec, Problem } from '../solver/types'
import { Segmented, StepHeader } from './ui'
import { useT } from '../i18n/web'
import { sampleUrl, templateZipUrl } from '../copy/core'

// サンプルは言語ごとに、その国の学校で配慮される項目で作ってある（public/samples/<lang>/）
const SAMPLES = [
  { id: 'sample1', label: 'sample1' },
  { id: 'sample2', label: 'sample2' },
  { id: 'sample-group', label: 'sampleGroup' },
] as const

/** ファイル名に使えない文字を除く（サンプルの表示名をそのままダウンロード名にするため） */
const safeFileName = (s: string) => s.replace(/[\\/:*?"<>|]/g, '_').trim() || 'sample'

export function DataStep({
  onLoad,
  problem,
  fileName,
  onOpenEditor,
  onExport,
  onGoogle,
  googleBusy,
  onCreateTemplate,
  templateBusy,
  templateUrl,
}: {
  onLoad: (data: ArrayBuffer, name: string) => void
  problem: Problem | null
  fileName: string | null
  onOpenEditor: () => void
  /** 名簿をひな形と同じ形式の Excel で保存 */
  onExport: () => void
  /** Google 連携が有効なときだけ渡す */
  onGoogle?: () => void
  googleBusy?: boolean
  /** Google 連携が有効なときだけ渡す: ひな形スプレッドシートを作成 */
  onCreateTemplate?: () => void
  templateBusy?: boolean
  templateUrl?: string | null
}) {
  const { t, lang } = useT()
  const input = useRef<HTMLInputElement>(null)
  const [drag, setDrag] = useState(false)
  // 名簿を開いているときは名簿カードを出し、「別の名簿を開く」で読み込み画面に切り替える
  // （読み込むたびに App 側で key を変えるので、読み込んだら名簿カードに戻る）
  const [showLoad, setShowLoad] = useState(false)

  const readFile = async (f: File) => onLoad(await f.arrayBuffer(), f.name)
  const loadSample = async (id: (typeof SAMPLES)[number]['id'], label: string) => {
    const res = await fetch(sampleUrl(lang, id))
    onLoad(await res.arrayBuffer(), t('samplePrefix', { label }))
  }

  if (problem && !showLoad)
    return (
      <div>
        {/* 名簿の切り替えは名簿そのものへの操作（編集・保存）とは別の段にする */}
        <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2 px-1">
          <span className="text-xs font-semibold text-slate-500">{t('currentRoster')}</span>
          <button
            type="button"
            onClick={() => setShowLoad(true)}
            className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-semibold text-indigo-600 transition hover:bg-indigo-50"
          >
            <ArrowLeftRight className="size-3.5" /> {t('otherRoster')}
          </button>
        </div>
        <section className="card p-5 sm:p-8">
          <StepHeader n={1} done title={t('step1Title')} />
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex min-w-0 items-center gap-3">
              <div className="grid size-11 shrink-0 place-items-center rounded-2xl bg-gradient-to-br from-indigo-500 to-fuchsia-500 text-white shadow-lg shadow-indigo-500/30">
                <FileSpreadsheet className="size-5" />
              </div>
              <div className="min-w-0">
                <div className="truncate font-semibold text-slate-800">{fileName ?? t('rosterTitle')}</div>
                <div className="text-xs text-slate-500">
                  {t('rosterSummary', { n: problem.students.length, cols: problem.columns.length, w: problem.wantedGroups.length, u: problem.unwantedGroups.length })}
                </div>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <button type="button" className="btn-ghost" onClick={onExport} title={t('saveRosterTitle')}>
                <Download className="size-4" /> {t('saveRoster')}
              </button>
              <button type="button" className="btn-primary" onClick={onOpenEditor}>
                <Pencil className="size-4" /> {t('editRoster')}
              </button>
            </div>
          </div>
        </section>
      </div>
    )

  return (
    <section className="card p-5 sm:p-8">
      <StepHeader
        n={1}
        title={t('step1Title')}
        desc={onGoogle ? t('step1DescGoogle') : t('step1Desc')}
      />
      <div className={onGoogle ? 'grid gap-3 md:grid-cols-[1fr_16rem]' : ''}>
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDrag(true)
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDrag(false)
          const f = e.dataTransfer.files[0]
          if (f) readFile(f)
        }}
        onClick={() => input.current?.click()}
        className={`group flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-10 text-center transition ${
          drag ? 'border-indigo-400 bg-indigo-50/60' : 'border-slate-200 hover:border-indigo-300 hover:bg-slate-50/60'
        }`}
      >
        <div className="grid size-12 place-items-center rounded-2xl bg-gradient-to-br from-indigo-500 to-fuchsia-500 text-white shadow-lg shadow-indigo-500/30 transition group-hover:scale-105">
          <Upload className="size-6" />
        </div>
        <div className="mt-4 font-semibold text-slate-800">{t('dropTitle')}</div>
        <div className="mt-1 text-sm text-slate-500">{t('dropSub')}</div>
        <input
          ref={input}
          type="file"
          accept=".xlsx,.xls"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) readFile(f)
            e.target.value = ''
          }}
        />
      </div>
      {onGoogle && (
        <button
          type="button"
          onClick={onGoogle}
          disabled={googleBusy}
          className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-slate-200 bg-white px-6 py-8 text-center transition hover:border-emerald-300 hover:bg-emerald-50/40 disabled:opacity-60"
        >
          <span className="grid size-12 place-items-center rounded-2xl bg-white shadow-md ring-1 ring-slate-100">
            {googleBusy ? <Loader2 className="size-6 animate-spin text-emerald-600" /> : <SheetsIcon />}
          </span>
          <span className="font-semibold text-slate-800">{t('googleSheets')}</span>
          <span className="text-xs text-slate-500">{t('googleSelect')}</span>
        </button>
      )}
      </div>

      <div className="mt-5 flex flex-wrap items-center gap-2">
        <a href={templateZipUrl(lang)} download className="btn-ghost">
          <Download className="size-4" /> {onCreateTemplate ? t('templateExcel') : t('templateDownload')}
        </a>
        {onCreateTemplate &&
          (templateUrl ? (
            <a href={templateUrl} target="_blank" rel="noreferrer" className="btn-ghost !border-emerald-200 !bg-emerald-50 !text-emerald-800">
              <SheetsIcon small /> {t('openTemplate')} <ExternalLink className="size-3.5" />
            </a>
          ) : (
            <button type="button" className="btn-ghost" onClick={onCreateTemplate} disabled={templateBusy}>
              {templateBusy ? <Loader2 className="size-4 animate-spin" /> : <SheetsIcon small />} {t('createTemplate')}
            </button>
          ))}
        <span className="mx-1 hidden h-5 w-px bg-slate-200 sm:block" />
        <span className="text-xs font-semibold text-slate-400">{t('trySamples')}</span>
        {SAMPLES.map((s) => (
          // サンプルはそのまま読み込むか、Excel をダウンロードして書き換えてから読み込む（無料）
          <span key={s.id} className="inline-flex items-center gap-1">
            <button type="button" onClick={() => loadSample(s.id, t(s.label))} className="btn-ghost !px-3 !py-1.5 !text-xs">
              <Sparkles className="size-3.5 text-fuchsia-500" /> {t(s.label)}
            </button>
            <a
              href={sampleUrl(lang, s.id)}
              download={`${safeFileName(t(s.label))}.xlsx`}
              title={t('sampleExcelTitle', { label: t(s.label) })}
              aria-label={t('sampleExcelTitle', { label: t(s.label) })}
              className="btn-ghost !px-2 !py-1.5 !text-xs"
            >
              <Download className="size-3.5" />
            </a>
          </span>
        ))}
      </div>
      <p className="mt-2 text-xs text-slate-400">{t('samplesExcelHint')}</p>
      {problem && (
        <div className="mt-5 border-t border-slate-100 pt-4">
          <button type="button" className="btn-ghost" onClick={() => setShowLoad(false)}>
            {t('backToRoster')}
          </button>
        </div>
      )}
    </section>
  )
}

export const KIND_KEY = { flag: 'kindFlag', category: 'kindCategory', degree: 'kindDegree', numeric: 'kindNumeric' } as const

export function SettingsStep({
  problem,
  numClasses,
  setNumClasses,
  timeSec,
  setTimeSec,
  onColumnChange,
  onOpenEditor,
}: {
  onOpenEditor: (tab: 'students' | 'wanted' | 'unwanted') => void
  problem: Problem
  numClasses: number
  setNumClasses: (n: number) => void
  timeSec: number
  setTimeSec: (n: number) => void
  onColumnChange: (i: number, patch: Partial<ColumnSpec>) => void
}) {
  const { t, num, lang } = useT()
  const dash = lang === 'ja' ? '〜' : '–'
  const n = problem.students.length
  const lo = Math.floor(n / numClasses)
  const hi = Math.ceil(n / numClasses)
  return (
    <section className="card p-5 sm:p-8">
      <StepHeader n={2} title={t('step2Title')} desc={t('step2Desc')} />

      <div className="grid gap-4 sm:grid-cols-3">
        <button
          type="button"
          onClick={() => onOpenEditor('students')}
          className="group rounded-2xl bg-slate-50 p-4 text-left ring-indigo-200 transition hover:bg-indigo-50/70 hover:ring-1"
        >
          <div className="flex items-center justify-between text-xs font-semibold text-slate-500">
            {t('studentsLabel')}
            <span className="inline-flex items-center gap-0.5 text-indigo-600 opacity-70 transition group-hover:opacity-100">
              {t('openRoster')} <ChevronRight className="size-3.5" />
            </span>
          </div>
          <div className="mt-1 flex items-center gap-2 text-2xl font-extrabold text-slate-900">
            <Users className="size-5 text-indigo-500" />
            {n}
            <span className="text-sm font-medium text-slate-400">{t('personUnit')}</span>
          </div>
          <div className="mt-1 text-xs text-slate-400">{t('rosterHint')}</div>
        </button>
        <div className="rounded-2xl bg-slate-50 p-4">
          <div className="text-xs font-semibold text-slate-500">{t('classCount')}</div>
          <div className="mt-1 flex items-center gap-3">
            <button type="button" className="btn-ghost !p-1.5" onClick={() => setNumClasses(Math.max(2, numClasses - 1))} aria-label={t('decrease')}>
              <Minus className="size-4" />
            </button>
            <span className="w-8 text-center text-2xl font-extrabold tabular-nums text-slate-900">{numClasses}</span>
            <button type="button" className="btn-ghost !p-1.5" onClick={() => setNumClasses(Math.min(n, numClasses + 1))} aria-label={t('increase')}>
              <Plus className="size-4" />
            </button>
          </div>
          <div className="mt-1 text-xs text-slate-400">{t('perClass', { range: lo === hi ? lo : `${lo}${dash}${hi}` })}</div>
        </div>
        <div className="rounded-2xl bg-slate-50 p-4">
          <div className="text-xs font-semibold text-slate-500">{t('searchTime')}</div>
          <div className="mt-2">
            <Segmented
              value={timeSec}
              onChange={setTimeSec}
              options={[
                { value: 3, label: t('quick') },
                { value: 10, label: t('standard') },
                { value: 30, label: t('thorough') },
              ]}
            />
          </div>
          <div className="mt-1 text-xs text-slate-400">{t('timeInfo', { s: timeSec })}</div>
        </div>
      </div>

      <div className="mt-6 overflow-hidden rounded-2xl border border-slate-100">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-left text-xs font-semibold text-slate-500">
            <tr>
              <th className="px-3 py-2.5 sm:px-4">{t('colItem')}</th>
              <th className="hidden px-4 py-2.5 sm:table-cell">{t('colKind')}</th>
              <th className="w-40 px-3 py-2.5 sm:w-56 sm:px-4">{t('colWeight')}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {problem.columns.map((c, i) => (
              <tr key={c.name} className={c.enabled && c.weight > 0 ? '' : 'opacity-45'}>
                <td className="whitespace-nowrap px-3 py-3 font-semibold text-slate-800 sm:px-4">{c.name}</td>
                <td className="hidden px-4 py-3 sm:table-cell">
                  <span className="mr-2 rounded-md bg-slate-100 px-1.5 py-0.5 text-[11px] font-semibold text-slate-500">{t(KIND_KEY[c.kind])}</span>
                  <span className="text-xs text-slate-500">
                    {c.kind === 'degree'
                      ? `1${dash}5`
                      : c.kind === 'numeric'
                        ? `${c.levels[0]}${dash}${c.levels[c.levels.length - 1]}`
                        : c.levels.join(' / ') || t('blankOnly')}
                  </span>
                </td>
                <td className="px-3 py-3 sm:px-4">
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min={0}
                      max={5}
                      step={0.5}
                      value={c.enabled ? c.weight : 0}
                      disabled={c.levels.length === 0}
                      onChange={(e) => {
                        const w = Number(e.target.value)
                        onColumnChange(i, { weight: w, enabled: w > 0 })
                      }}
                      className="h-1.5 w-full cursor-pointer accent-indigo-600"
                    />
                    <span className="w-8 text-right font-mono text-xs font-semibold tabular-nums text-slate-600">
                      {num(c.enabled ? c.weight : 0, 1)}
                    </span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
        <PairBox title={t('pairWanted')} tone="indigo" groups={problem.wantedGroups} problem={problem} onEdit={() => onOpenEditor('wanted')} />
        <PairBox title={t('pairUnwanted')} tone="rose" groups={problem.unwantedGroups} problem={problem} onEdit={() => onOpenEditor('unwanted')} />
      </div>
    </section>
  )
}

function PairBox({
  title,
  groups,
  problem,
  tone,
  onEdit,
}: {
  title: string
  groups: number[][]
  problem: Problem
  tone: 'indigo' | 'rose'
  onEdit: () => void
}) {
  const { t } = useT()
  const cls = tone === 'indigo' ? 'bg-indigo-50 text-indigo-700' : 'bg-rose-50 text-rose-700'
  return (
    <div className="rounded-2xl border border-slate-100 p-4">
      <div className="mb-2 flex items-center justify-between">
        <span className="font-semibold text-slate-700">
          {title} <span className="ml-1 text-xs font-normal text-slate-400">{t('countItems', { n: groups.length })}</span>
        </span>
        <button type="button" onClick={onEdit} className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-semibold text-indigo-600 hover:bg-indigo-50">
          <Pencil className="size-3.5" /> {t('edit')}
        </button>
      </div>
      {groups.length === 0 ? (
        <button type="button" onClick={onEdit} className="text-xs text-slate-400 hover:text-indigo-600">
          {t('noPairsAdd')}
        </button>
      ) : (
        <div className="flex flex-wrap gap-1.5">
          {groups.map((g, i) => (
            <span key={i} className={`rounded-lg px-2 py-1 text-xs font-medium ${cls}`}>
              {g.map((s) => problem.students[s].name || problem.students[s].no).join(' · ')}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function SheetsIcon({ small }: { small?: boolean }) {
  return (
    <svg viewBox="0 0 24 24" className={small ? 'size-4' : 'size-7'} aria-hidden>
      <path fill="#0F9D58" d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2Z" />
      <path fill="#87CEAC" d="M14.5 2v4a1.5 1.5 0 0 0 1.5 1.5h4L14.5 2Z" />
      <path fill="#F1F1F1" d="M7.5 11h9v7h-9v-7Zm1.2 1.2v1.7h2.7v-1.7H8.7Zm3.9 0v1.7h2.7v-1.7h-2.7Zm-3.9 2.9v1.7h2.7v-1.7H8.7Zm3.9 0v1.7h2.7v-1.7h-2.7Z" />
    </svg>
  )
}
