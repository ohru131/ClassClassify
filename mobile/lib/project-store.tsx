import AsyncStorage from '@react-native-async-storage/async-storage'
import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'

import { clearAppFiles } from './app-files'
import { useI18n } from './language-provider'
import { stepClassCountByMax } from './class-size'
import { CancelledError, runSliced } from './runner'
import { compile, evaluate, type ColumnSpec, type Problem, type Report } from './solver'
import { isStoredProject, type StoredProject } from './stored-project'

export interface Solution {
  classOf: number[]
  /** 自動編成の直後の結果（手動移動を戻すため） */
  original: number[]
  k: number
  iterations: number
  starts: number
}

type ProjectContextValue = {
  hydrated: boolean
  problem: Problem | null
  fileName: string | null
  numClasses: number
  timeSec: number
  solution: Solution | null
  report: Report | null
  edited: boolean
  running: boolean
  progress: number
  error: string | null
  loadProblem: (p: Problem, name: string) => void
  updateProblem: (next: Problem) => void
  /** 最新の名簿に対して変更を当てる（続けて編集しても古い名簿から上書きしない） */
  modifyProblem: (f: (p: Problem) => Problem) => void
  updateColumn: (i: number, patch: Partial<ColumnSpec>) => void
  setNumClasses: (k: number) => void
  /** 最大人数を1段階増やす（+1）／減らす（-1）。クラス数はそれに合わせて決まる */
  stepMaxPerClass: (dir: 1 | -1) => void
  setTimeSec: (s: number) => void
  dismissWarnings: () => void
  clearProject: () => void
  run: () => Promise<boolean>
  cancel: () => void
  moveStudent: (student: number, to: number) => void
  resetMoves: () => void
  setError: (e: string | null) => void
}

// 旧名 Mosaic のままにしてある（変えると保存済みの名簿・結果を読めなくなる）
const STORAGE_KEY = 'mosaic.project.v1'
const ProjectContext = createContext<ProjectContextValue | null>(null)

export function ProjectProvider({ children }: { children: ReactNode }) {
  const { t } = useI18n()
  const [hydrated, setHydrated] = useState(false)
  const [problem, setProblem] = useState<Problem | null>(null)
  const [fileName, setFileName] = useState<string | null>(null)
  const [numClasses, setNumClassesState] = useState(4)
  const numClassesRef = useRef(4)
  useEffect(() => {
    numClassesRef.current = numClasses
  }, [numClasses])
  const [timeSec, setTimeSec] = useState(10)
  const [solution, setSolution] = useState<Solution | null>(null)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const cancelRef = useRef<() => void>(() => {})
  const problemRef = useRef<Problem | null>(null)
  const runningRef = useRef(false)
  useEffect(() => {
    problemRef.current = problem
  }, [problem])

  // --- 端末内への保存（AsyncStorage）。データはどこにも送信しない ---
  useEffect(() => {
    let active = true
    AsyncStorage.getItem(STORAGE_KEY)
      .then((raw) => {
        if (!active || !raw) return
        const data: unknown = JSON.parse(raw)
        if (!isStoredProject(data)) return
        problemRef.current = data.problem
        setProblem(data.problem)
        setFileName(data.fileName)
        setNumClassesState(data.numClasses)
        setTimeSec(data.timeSec)
        setSolution(data.solution)
      })
      .catch(() => undefined)
      .finally(() => {
        if (active) setHydrated(true)
      })
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    // 読み込み前に保存すると、保存済みのデータを空で上書きしてしまう
    if (!hydrated) return
    const data: StoredProject = { version: 1, problem, fileName, numClasses, timeSec, solution }
    const t = setTimeout(() => {
      void AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(data)).catch(() => undefined)
    }, 400)
    return () => clearTimeout(t)
  }, [hydrated, problem, fileName, numClasses, timeSec, solution])

  const loadProblem = useCallback((p: Problem, name: string) => {
    cancelRef.current()
    problemRef.current = p
    setProblem(p)
    setFileName(name)
    setNumClassesState(Math.max(2, Math.min(p.numClasses, Math.max(2, p.students.length))))
    setSolution(null)
    setError(null)
  }, [])

  const updateProblem = useCallback((next: Problem) => {
    const prev = problemRef.current
    // 生徒の追加・削除で index が変わるので、既存の編成結果は破棄（Web 版と同じ）
    if (prev && next.students.length !== prev.students.length) setSolution(null)
    problemRef.current = next
    setProblem(next)
  }, [])

  const modifyProblem = useCallback(
    (f: (p: Problem) => Problem) => {
      const p = problemRef.current
      if (p) updateProblem(f(p))
    },
    [updateProblem],
  )

  const updateColumn = useCallback((i: number, patch: Partial<ColumnSpec>) => {
    modifyProblem((p) => ({ ...p, columns: p.columns.map((c, j) => (i === j ? { ...c, ...patch } : c)) }))
  }, [modifyProblem])

  const setNumClasses = useCallback((k: number) => {
    setNumClassesState(k)
    // クラス数を直接決めたら、最大人数の指定はクラス数から決まる値に任せる
    const p = problemRef.current
    if (p && p.maxPerClass !== null) updateProblem({ ...p, maxPerClass: null })
  }, [updateProblem])

  const stepMaxPerClass = useCallback(
    (dir: 1 | -1) => {
      const p = problemRef.current
      if (!p) return
      const next = stepClassCountByMax(p.students.length, numClassesRef.current, dir)
      if (!next) return
      numClassesRef.current = next.k
      setNumClassesState(next.k)
      // 保存する最大人数は表示と同じ値（ceil(n / k)）
      updateProblem({ ...p, maxPerClass: next.max })
    },
    [updateProblem],
  )

  const dismissWarnings = useCallback(() => modifyProblem((p) => ({ ...p, warnings: [] })), [modifyProblem])

  const clearProject = useCallback(() => {
    cancelRef.current()
    problemRef.current = null
    setProblem(null)
    setFileName(null)
    setSolution(null)
    setError(null)
    // 保存の遅延（400ms）を待たずに端末から消す。その間にアプリが終了しても名簿が残らないように
    void AsyncStorage.removeItem(STORAGE_KEY).catch(() => undefined)
    // 書き出した Excel・PDF とピッカーのコピーも消す
    void clearAppFiles()
  }, [])

  const run = useCallback(async () => {
    const start = problemRef.current
    if (!start || runningRef.current) return false
    if (start.students.length < numClasses) {
      setError(t('tooFew', { n: start.students.length, k: numClasses }))
      return false
    }
    runningRef.current = true
    setRunning(true)
    setProgress(0)
    setError(null)
    try {
      // compile も try の中で（例外で実行中のまま固まらないように）
      const { compiled } = compile(start, numClasses)
      // 1% 刻みに丸める（同じ値なら React が再描画を省くので、計算中の UI の負担が減る）
      const job = runSliced(compiled, { timeMs: timeSec * 1000, onProgress: (f) => setProgress(Math.round(f * 100) / 100) })
      cancelRef.current = job.cancel
      const res = await job.promise
      const now = problemRef.current
      // 実行中に生徒やペア指定が変わった場合は index が合わないので破棄（Web 版と同じ）
      if (!now || now.students !== start.students || now.wantedGroups !== start.wantedGroups || now.unwantedGroups !== start.unwantedGroups) {
        setError(t('discarded'))
        return false
      }
      setSolution({ classOf: res.classOf, original: res.classOf, k: numClasses, iterations: res.iterations, starts: res.starts })
      return true
    } catch (e) {
      if (!(e instanceof CancelledError)) setError(e instanceof Error ? e.message : String(e))
      return false
    } finally {
      cancelRef.current = () => {}
      runningRef.current = false
      setRunning(false)
    }
  }, [numClasses, timeSec, t])

  const cancel = useCallback(() => cancelRef.current(), [])

  const moveStudent = useCallback((student: number, to: number) => {
    setSolution((s) => (s ? { ...s, classOf: s.classOf.map((c, i) => (i === student ? to : c)) } : s))
  }, [])
  const resetMoves = useCallback(() => setSolution((s) => (s ? { ...s, classOf: s.original } : s)), [])

  const report = useMemo(
    () => (problem && solution && solution.classOf.length === problem.students.length ? evaluate(problem, solution.classOf, solution.k) : null),
    [problem, solution],
  )
  const edited = !!solution && solution.classOf.some((c, i) => c !== solution.original[i])

  const value = useMemo<ProjectContextValue>(
    () => ({
      hydrated,
      problem,
      fileName,
      numClasses,
      timeSec,
      solution,
      report,
      edited,
      running,
      progress,
      error,
      loadProblem,
      updateProblem,
      modifyProblem,
      updateColumn,
      setNumClasses,
      stepMaxPerClass,
      setTimeSec,
      dismissWarnings,
      clearProject,
      run,
      cancel,
      moveStudent,
      resetMoves,
      setError,
    }),
    [hydrated, problem, fileName, numClasses, timeSec, solution, report, edited, running, progress, error, loadProblem, updateProblem, modifyProblem, updateColumn, setNumClasses, stepMaxPerClass, dismissWarnings, clearProject, run, cancel, moveStudent, resetMoves],
  )
  return <ProjectContext.Provider value={value}>{children}</ProjectContext.Provider>
}

export function useProject() {
  const v = useContext(ProjectContext)
  if (!v) throw new Error('ProjectProvider の内部で使用してください。')
  return v
}
