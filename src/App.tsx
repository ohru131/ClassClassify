import { useEffect, useMemo, useRef, useState } from 'react'
import { AlertCircle, Cpu, Loader2, X, Lock, Play, Square, Zap } from 'lucide-react'
import { parseWorkbook } from './solver/parse'
import { compile } from './solver/compile'
import { evaluate } from './solver/evaluate'
import { buildResultSheets, exportWorkbook } from './solver/export'
import { createTemplateSpreadsheet } from './google/template'
import { downloadAsXlsx, googleEnabled, pickSpreadsheet, writeResults, type GoogleFile } from './google/google'
import { runParallel } from './solver/run'
import type { ColumnSpec, Problem } from './solver/types'
import { DataStep, SettingsStep } from './components/Setup'
import { RosterEditor, type EditorTab } from './components/RosterEditor'
import { exportRoster } from './solver/roster'
import { Results } from './components/Results'
import { Logo, StepHeader } from './components/ui'

interface Solution {
  classOf: number[]
  original: number[]
  k: number
  iterations: number
  workers: number
}

export default function App() {
  const [problem, setProblem] = useState<Problem | null>(null)
  const [fileName, setFileName] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [numClasses, setNumClasses] = useState(4)
  const [timeSec, setTimeSec] = useState(10)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [solution, setSolution] = useState<Solution | null>(null)
  const [googleFile, setGoogleFile] = useState<GoogleFile | null>(null)
  const [googleBusy, setGoogleBusy] = useState(false)
  const [savingGoogle, setSavingGoogle] = useState(false)
  const [templateBusy, setTemplateBusy] = useState(false)
  const [templateUrl, setTemplateUrl] = useState<string | null>(null)
  const [savedUrl, setSavedUrl] = useState<string | null>(null)
  const [editorTab, setEditorTab] = useState<EditorTab | null>(null)
  const cancelRef = useRef<() => void>(() => {})
  const problemRef = useRef<Problem | null>(null)
  useEffect(() => {
    problemRef.current = problem
  }, [problem])
  const resultRef = useRef<HTMLDivElement>(null)

  const onLoad = (data: ArrayBuffer, name: string, source: GoogleFile | null = null) => {
    try {
      const p = parseWorkbook(data)
      setGoogleFile(source)
      setSavedUrl(null)
      setProblem(p)
      setNumClasses(p.numClasses)
      setFileName(name)
      setSolution(null)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const showError = (e: unknown) => {
    if (e instanceof Error && e.message === 'cancelled') return
    setError(e instanceof Error ? e.message : String(e))
  }

  const loadFromGoogle = async () => {
    setGoogleBusy(true)
    try {
      const file = await pickSpreadsheet()
      if (file) onLoad(await downloadAsXlsx(file), file.name, file)
    } catch (e) {
      showError(e)
    } finally {
      setGoogleBusy(false)
    }
  }

  const createTemplate = async () => {
    setTemplateBusy(true)
    // ポップアップブロック回避のため、クリック直後に空のタブを開いておく
    const tab = window.open('', '_blank')
    try {
      const file = await createTemplateSpreadsheet()
      setTemplateUrl(file.url)
      if (tab) {
        tab.opener = null
        tab.location.href = file.url
      }
    } catch (e) {
      tab?.close()
      showError(e)
    } finally {
      setTemplateBusy(false)
    }
  }

  const saveToGoogle = async () => {
    if (!problem || !solution || !report) return
    setSavingGoogle(true)
    try {
      const sheets = buildResultSheets(problem, solution.classOf, solution.k, report)
      const title = `クラス編成結果 ${new Date().toLocaleString('ja-JP')}`
      setSavedUrl(await writeResults(sheets, googleFile, title))
    } catch (e) {
      showError(e)
    } finally {
      setSavingGoogle(false)
    }
  }

  const onProblemChange = (next: Problem) => {
    // 生徒の追加・削除で index が変わるため、既存の編成結果は破棄
    if (problem && next.students.length !== problem.students.length) setSolution(null)
    setProblem(next)
  }

  const saveBlob = (blob: Blob, name: string) => {
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = name
    document.body.appendChild(a)
    a.click()
    a.remove()
    setTimeout(() => URL.revokeObjectURL(a.href), 1000)
  }

  const onColumnChange = (i: number, patch: Partial<ColumnSpec>) =>
    setProblem((p) => (p ? { ...p, columns: p.columns.map((c, j) => (i === j ? { ...c, ...patch } : c)) } : p))

  const run = async () => {
    if (!problem) return
    const { compiled } = compile(problem, numClasses)
    setRunning(true)
    setProgress(0)
    setError(null)
    const job = runParallel(compiled, timeSec * 1000, (f) => setProgress(f))
    cancelRef.current = job.cancel
    const start = problem
    try {
      const res = await job.promise
      const now = problemRef.current
      // 実行中に生徒やペア指定が変わった場合、結果の index が合わないので破棄
      if (!now || now.students !== start.students || now.wantedGroups !== start.wantedGroups || now.unwantedGroups !== start.unwantedGroups) {
        setError('実行中に名簿が変更されたため、結果を破棄しました。もう一度実行してください。')
        return
      }
      setSavedUrl(null)
      setSolution({ classOf: res.classOf, original: res.classOf, k: numClasses, iterations: res.iterations, workers: res.workers })
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50)
    } catch (e) {
      if (!(e instanceof Error && e.message === 'cancelled')) setError(e instanceof Error ? e.message : String(e))
    } finally {
      setRunning(false)
    }
  }

  const report = useMemo(
    () => (problem && solution && solution.classOf.length === problem.students.length ? evaluate(problem, solution.classOf, solution.k) : null),
    [problem, solution],
  )

  const download = () => {
    if (!problem || !solution || !report) return
    saveBlob(exportWorkbook(problem, solution.classOf, solution.k, report), `クラス編成結果_${new Date().toISOString().slice(0, 10)}.xlsx`)
  }

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-white/60 bg-white/60 backdrop-blur-xl">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-3">
          <Logo />
          <a
            href="https://github.com/ohru131/ClassClassify"
            target="_blank"
            rel="noreferrer"
            className="rounded-xl p-2 text-slate-500 transition hover:bg-slate-100 hover:text-slate-900"
            aria-label="GitHub"
          >
            <svg viewBox="0 0 24 24" className="size-5" fill="currentColor" aria-hidden><path d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37-.52-1.33-1.28-1.69-1.28-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.41-2.69 5.39-5.25 5.67.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5Z"/></svg>
          </a>
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-6 px-5 pb-24 pt-10">
        <div className="max-w-3xl">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-indigo-100 bg-white/70 px-3 py-1 text-xs font-semibold text-indigo-700">
            <Zap className="size-3.5" /> 焼きなまし法 × 並列マルチスタート
          </div>
          <h1 className="text-4xl font-black leading-tight tracking-tight text-slate-900 sm:text-5xl">
            個性が響き合う、
            <br />
            <span className="bg-gradient-to-r from-indigo-600 via-violet-600 to-fuchsia-600 bg-clip-text text-transparent">
              バランスの良いクラスを。
            </span>
          </h1>
          <p className="mt-4 text-base leading-relaxed text-slate-600">
            性別・学力・支援の必要性など、生徒の特性が各クラスに均等に散らばるよう自動で編成します。
            「同じ組にしたい」「別の組にしたい」組み合わせも考慮。結果はその場で手直しできます。
          </p>
          <div className="mt-5 flex flex-wrap gap-4 text-xs font-medium text-slate-500">
            <span className="inline-flex items-center gap-1.5">
              <Lock className="size-3.5 text-emerald-500" /> データはブラウザ内だけで処理
            </span>
            <span className="inline-flex items-center gap-1.5">
              <Cpu className="size-3.5 text-indigo-500" /> 登録・トークン不要、完全無料
            </span>
          </div>
        </div>

        {error && (
          <div className="flex items-center gap-3 rounded-2xl border border-rose-200 bg-rose-50 px-5 py-3 text-sm text-rose-700">
            <AlertCircle className="size-4 shrink-0" /> {error}
          </div>
        )}

        <DataStep
          onLoad={(d, n) => onLoad(d, n)}
          fileName={fileName}
          onGoogle={googleEnabled ? loadFromGoogle : undefined}
          googleBusy={googleBusy}
          onCreateTemplate={googleEnabled ? createTemplate : undefined}
          templateBusy={templateBusy}
          templateUrl={templateUrl}
        />

        {problem && problem.warnings.length > 0 && (
          <div className="rounded-2xl border border-amber-200 bg-amber-50 px-5 py-3 text-sm text-amber-800">
            <div className="mb-1 flex items-center justify-between font-semibold">
              読み込み時の注意
              <button type="button" onClick={() => setProblem({ ...problem, warnings: [] })} className="rounded p-0.5 hover:bg-amber-100" aria-label="閉じる">
                <X className="size-4" />
              </button>
            </div>
            <ul className="list-inside list-disc space-y-0.5">
              {problem.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        )}

        {problem && (
          <SettingsStep
            problem={problem}
            numClasses={numClasses}
            setNumClasses={setNumClasses}
            timeSec={timeSec}
            setTimeSec={setTimeSec}
            onColumnChange={onColumnChange}
            onOpenEditor={setEditorTab}
          />
        )}

        {problem && (
          <section className="card p-6 sm:p-8">
            <StepHeader n={3} done={!!solution && !running} title="編成する" desc="複数の CPU コアで同時に探索し、最もバランスの良い案を採用します。" />
            {running ? (
              <div className="space-y-3">
                <div className="h-3 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className="shimmer h-full rounded-full bg-gradient-to-r from-indigo-500 via-fuchsia-500 to-indigo-500 transition-[width] duration-200"
                    style={{ width: `${Math.max(3, progress * 100)}%` }}
                  />
                </div>
                <div className="flex items-center justify-between text-sm text-slate-500">
                  <span className="inline-flex items-center gap-2">
                    <Loader2 className="size-4 animate-spin" /> 最適な組み合わせを探索中… {Math.round(progress * 100)}%
                  </span>
                  <button type="button" className="btn-ghost !py-1.5" onClick={() => cancelRef.current()}>
                    <Square className="size-3.5" /> 中止
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex flex-wrap items-center gap-4">
                <button type="button" className="btn-primary !px-6 !py-3 !text-base" onClick={run}>
                  <Play className="size-4 fill-current" /> {solution ? 'もう一度編成する' : 'クラス編成を実行'}
                </button>
                {solution && (
                  <span className="text-xs text-slate-400">
                    前回: {solution.workers} 並列 · {(solution.iterations / 1e6).toFixed(1)}M 回の探索
                  </span>
                )}
              </div>
            )}
          </section>
        )}

        <div ref={resultRef} className="scroll-mt-24">
          {problem && solution && report && (
            <Results
              problem={problem}
              classOf={solution.classOf}
              k={solution.k}
              report={report}
              edited={solution.classOf !== solution.original}
              onMove={(s, to) =>
                setSolution((sol) => (sol ? { ...sol, classOf: sol.classOf.map((c, i) => (i === s ? to : c)) } : sol))
              }
              onReset={() => setSolution((sol) => (sol ? { ...sol, classOf: sol.original } : sol))}
              onDownload={download}
              google={
                googleEnabled
                  ? {
                      label: googleFile?.mimeType === 'application/vnd.google-apps.spreadsheet' ? '元のシートに書き出す' : 'スプレッドシートに保存',
                      busy: savingGoogle,
                      url: savedUrl,
                      onSave: saveToGoogle,
                    }
                  : undefined
              }
            />
          )}
        </div>
      </main>

      {problem && editorTab && (
        <RosterEditor
          problem={problem}
          tab={editorTab}
          setTab={setEditorTab}
          onChange={onProblemChange}
          onClose={() => setEditorTab(null)}
          onExport={() => saveBlob(exportRoster(problem, numClasses), `名簿_${new Date().toISOString().slice(0, 10)}.xlsx`)}
        />
      )}

      <footer className="border-t border-slate-200/70 py-8 text-center text-xs text-slate-400">
        Mosaic · クラス編成オプティマイザー — MIT License
      </footer>
    </div>
  )
}
