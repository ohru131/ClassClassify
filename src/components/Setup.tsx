import { useRef, useState } from 'react'
import { Download, ExternalLink, FileSpreadsheet, Loader2, Minus, Plus, Sparkles, Upload, Users } from 'lucide-react'
import type { ColumnSpec, Problem } from '../solver/types'
import { Segmented, StepHeader } from './ui'

const SAMPLES = [
  { file: 'sample1.xlsx', label: 'クラス分け（80名・4組）' },
  { file: 'sample2.xlsx', label: 'クラス分け（80名・シンプル）' },
  { file: 'sample-group.xlsx', label: 'グループ分け（30名・6班）' },
]

export function DataStep({
  onLoad,
  fileName,
  onGoogle,
  googleBusy,
  onCreateTemplate,
  templateBusy,
  templateUrl,
}: {
  onLoad: (data: ArrayBuffer, name: string) => void
  fileName: string | null
  /** Google 連携が有効なときだけ渡す */
  onGoogle?: () => void
  googleBusy?: boolean
  /** Google 連携が有効なときだけ渡す: ひな形スプレッドシートを作成 */
  onCreateTemplate?: () => void
  templateBusy?: boolean
  templateUrl?: string | null
}) {
  const input = useRef<HTMLInputElement>(null)
  const [drag, setDrag] = useState(false)

  const readFile = async (f: File) => onLoad(await f.arrayBuffer(), f.name)
  const loadSample = async (file: string, label: string) => {
    const res = await fetch(`./${file}`)
    onLoad(await res.arrayBuffer(), `サンプル: ${label}`)
  }

  return (
    <section className="card p-6 sm:p-8">
      <StepHeader
        n={1}
        done={!!fileName}
        title="名簿を読み込む"
        desc={
          onGoogle
            ? 'ひな形に生徒の特性を記入し、Excel または Google スプレッドシートから読み込み。データはブラウザと Google の間でのみやり取りします。'
            : 'ひな形の Excel に生徒の特性を記入してアップロード。データはブラウザの外に送信されません。'
        }
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
          {fileName ? <FileSpreadsheet className="size-6" /> : <Upload className="size-6" />}
        </div>
        <div className="mt-4 font-semibold text-slate-800">{fileName ?? 'Excel ファイルをドロップ'}</div>
        <div className="mt-1 text-sm text-slate-500">{fileName ? 'クリックして別のファイルを選択' : 'またはクリックして選択（.xlsx）'}</div>
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
          <span className="font-semibold text-slate-800">Google スプレッドシート</span>
          <span className="text-xs text-slate-500">Google アカウントで選択</span>
        </button>
      )}
      </div>

      <div className="mt-5 flex flex-wrap items-center gap-2">
        <a href="./template.zip" download className="btn-ghost">
          <Download className="size-4" /> {onCreateTemplate ? 'Excel ひな形' : 'ひな形をダウンロード'}
        </a>
        {onCreateTemplate &&
          (templateUrl ? (
            <a href={templateUrl} target="_blank" rel="noreferrer" className="btn-ghost !border-emerald-200 !bg-emerald-50 !text-emerald-800">
              <SheetsIcon small /> 作成したひな形を開く <ExternalLink className="size-3.5" />
            </a>
          ) : (
            <button type="button" className="btn-ghost" onClick={onCreateTemplate} disabled={templateBusy}>
              {templateBusy ? <Loader2 className="size-4 animate-spin" /> : <SheetsIcon small />} スプレッドシートでひな形を作成
            </button>
          ))}
        <span className="mx-1 hidden h-5 w-px bg-slate-200 sm:block" />
        <span className="text-xs font-semibold text-slate-400">サンプルで試す</span>
        {SAMPLES.map((s) => (
          <button key={s.file} type="button" onClick={() => loadSample(s.file, s.label)} className="btn-ghost !px-3 !py-1.5 !text-xs">
            <Sparkles className="size-3.5 text-fuchsia-500" /> {s.label}
          </button>
        ))}
      </div>
    </section>
  )
}

const KIND_LABEL: Record<ColumnSpec['kind'], string> = { flag: '該当', category: 'カテゴリ', numeric: '数値' }

export function SettingsStep({
  problem,
  numClasses,
  setNumClasses,
  timeSec,
  setTimeSec,
  onColumnChange,
}: {
  problem: Problem
  numClasses: number
  setNumClasses: (n: number) => void
  timeSec: number
  setTimeSec: (n: number) => void
  onColumnChange: (i: number, patch: Partial<ColumnSpec>) => void
}) {
  const n = problem.students.length
  const lo = Math.floor(n / numClasses)
  const hi = Math.ceil(n / numClasses)
  return (
    <section className="card p-6 sm:p-8">
      <StepHeader n={2} title="条件を調整する" desc="重みが大きい項目ほど優先して均等にします。0 にすると無視します。" />

      <div className="grid gap-4 sm:grid-cols-3">
        <div className="rounded-2xl bg-slate-50 p-4">
          <div className="text-xs font-semibold text-slate-500">生徒数</div>
          <div className="mt-1 flex items-center gap-2 text-2xl font-extrabold text-slate-900">
            <Users className="size-5 text-indigo-500" />
            {n}
            <span className="text-sm font-medium text-slate-400">名</span>
          </div>
        </div>
        <div className="rounded-2xl bg-slate-50 p-4">
          <div className="text-xs font-semibold text-slate-500">クラス（グループ）数</div>
          <div className="mt-1 flex items-center gap-3">
            <button type="button" className="btn-ghost !p-1.5" onClick={() => setNumClasses(Math.max(2, numClasses - 1))} aria-label="減らす">
              <Minus className="size-4" />
            </button>
            <span className="w-8 text-center text-2xl font-extrabold tabular-nums text-slate-900">{numClasses}</span>
            <button type="button" className="btn-ghost !p-1.5" onClick={() => setNumClasses(Math.min(n, numClasses + 1))} aria-label="増やす">
              <Plus className="size-4" />
            </button>
          </div>
          <div className="mt-1 text-xs text-slate-400">1組あたり {lo === hi ? lo : `${lo}〜${hi}`} 名</div>
        </div>
        <div className="rounded-2xl bg-slate-50 p-4">
          <div className="text-xs font-semibold text-slate-500">探索時間</div>
          <div className="mt-2">
            <Segmented
              value={timeSec}
              onChange={setTimeSec}
              options={[
                { value: 3, label: '高速' },
                { value: 10, label: '標準' },
                { value: 30, label: '徹底' },
              ]}
            />
          </div>
          <div className="mt-1 text-xs text-slate-400">{timeSec} 秒 × 並列探索</div>
        </div>
      </div>

      <div className="mt-6 overflow-hidden rounded-2xl border border-slate-100">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-left text-xs font-semibold text-slate-500">
            <tr>
              <th className="px-4 py-2.5">項目</th>
              <th className="hidden px-4 py-2.5 sm:table-cell">種類 / 値</th>
              <th className="w-56 px-4 py-2.5">重み</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {problem.columns.map((c, i) => (
              <tr key={c.name} className={c.enabled && c.weight > 0 ? '' : 'opacity-45'}>
                <td className="px-4 py-3 font-semibold text-slate-800">{c.name}</td>
                <td className="hidden px-4 py-3 sm:table-cell">
                  <span className="mr-2 rounded-md bg-slate-100 px-1.5 py-0.5 text-[11px] font-semibold text-slate-500">{KIND_LABEL[c.kind]}</span>
                  <span className="text-xs text-slate-500">
                    {c.kind === 'numeric' ? `${c.levels[0]}〜${c.levels[c.levels.length - 1]}` : c.levels.join(' / ') || '（空欄のみ）'}
                  </span>
                </td>
                <td className="px-4 py-3">
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
                      {(c.enabled ? c.weight : 0).toFixed(1)}
                    </span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
        <PairBox title="同じ組にする" tone="indigo" groups={problem.wantedGroups} problem={problem} />
        <PairBox title="別の組にする" tone="rose" groups={problem.unwantedGroups} problem={problem} />
      </div>
    </section>
  )
}

function PairBox({ title, groups, problem, tone }: { title: string; groups: number[][]; problem: Problem; tone: 'indigo' | 'rose' }) {
  const cls = tone === 'indigo' ? 'bg-indigo-50 text-indigo-700' : 'bg-rose-50 text-rose-700'
  return (
    <div className="rounded-2xl border border-slate-100 p-4">
      <div className="mb-2 flex items-center justify-between">
        <span className="font-semibold text-slate-700">{title}</span>
        <span className="text-xs text-slate-400">{groups.length} 件</span>
      </div>
      {groups.length === 0 ? (
        <div className="text-xs text-slate-400">指定なし</div>
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
