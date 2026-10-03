import { useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  Check,
  ChevronDown,
  Download,
  Filter,
  Link2,
  Pencil,
  Plus,
  RefreshCw,
  Search,
  Settings2,
  Split,
  Trash2,
  UserPlus,
  Users,
  X,
} from 'lucide-react'
import type { ColumnKind, ColumnSpec, Problem } from '../solver/types'
import {
  addColumn,
  addGroup,
  addLevel,
  addStudent,
  findConflicts,
  groupsOf,
  isNoTaken,
  removeColumn,
  removeGroup,
  removeLevel,
  removeStudents,
  renameLevel,
  setGroup,
  setValueFor,
  updateStudent,
  type GroupKind,
} from '../solver/roster'
import { DEGREE_LEVELS } from '../solver/columns'
import { useT } from '../i18n/web'
import { KIND_KEY } from './Setup'

export type EditorTab = 'students' | GroupKind

/** 画面幅が md（768px）以上か。表とカードのどちらか一方だけを描画するために使う */
function useIsDesktop() {
  const query = '(min-width: 768px)'
  const [match, setMatch] = useState(() => window.matchMedia(query).matches)
  useEffect(() => {
    const mq = window.matchMedia(query)
    const on = () => setMatch(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])
  return match
}

const EMPTY = '__empty__'

export function RosterEditor({
  problem,
  tab,
  setTab,
  onChange,
  onClose,
  onExport,
}: {
  problem: Problem
  tab: EditorTab
  setTab: (t: EditorTab) => void
  onChange: (p: Problem) => void
  onClose: () => void
  onExport: () => void
}) {
  const { t, file } = useT()
  const conflicts = useMemo(() => findConflicts(problem), [problem])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && !(e.target instanceof HTMLInputElement) && onClose()
    window.addEventListener('keydown', onKey)
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', onKey)
      document.body.style.overflow = ''
    }
  }, [onClose])

  const tabs: { id: EditorTab; label: string; count: number; icon: ReactNode }[] = [
    { id: 'students', label: t('tabStudentsList'), count: problem.students.length, icon: <Users className="size-4" /> },
    { id: 'wanted', label: t('tabWanted'), count: problem.wantedGroups.length, icon: <Link2 className="size-4" /> },
    { id: 'unwanted', label: t('tabUnwanted'), count: problem.unwantedGroups.length, icon: <Split className="size-4" /> },
  ]

  return (
    <div
      className="fixed inset-0 z-40 flex items-stretch justify-center bg-slate-900/40 p-0 backdrop-blur-sm sm:p-4"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        role="dialog"
        aria-modal
        aria-label={t('rosterEditorAria')}
        className="flex w-full max-w-[90rem] flex-col overflow-hidden bg-white shadow-2xl sm:rounded-3xl"
      >
        <div className="flex flex-wrap items-center gap-2 border-b border-slate-100 px-3 py-2 sm:gap-3 sm:px-5 sm:py-3">
          <div className="mr-2 text-lg font-extrabold text-slate-900">{t('rosterTitle')}</div>
          <div className="order-last flex w-full rounded-xl bg-slate-100 p-1 sm:order-none sm:inline-flex sm:w-auto">
            {tabs.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => setTab(t.id)}
                className={`inline-flex flex-1 items-center justify-center gap-1.5 whitespace-nowrap rounded-lg px-2 py-1.5 text-sm font-semibold transition sm:flex-none sm:px-3 ${
                  tab === t.id ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                }`}
              >
                <span className="hidden sm:inline">{t.icon}</span>
                {t.label}
                <span className="rounded-full bg-slate-200/70 px-1.5 text-[11px] tabular-nums text-slate-600">{t.count}</span>
              </button>
            ))}
          </div>
          <div className="ml-auto flex items-center gap-2">
            <button type="button" className="btn-ghost !py-2" onClick={onExport} title={t('saveRosterTitle')}>
              <Download className="size-4" /> <span className="hidden sm:inline">{t('saveRoster')}</span><span className="sm:hidden">{t('saveShort')}</span>
            </button>
            <button type="button" onClick={onClose} className="rounded-xl p-2 text-slate-500 hover:bg-slate-100" aria-label={t('close')}>
              <X className="size-5" />
            </button>
          </div>
        </div>

        {conflicts.length > 0 && (
          <div className="flex items-start gap-2 border-b border-amber-100 bg-amber-50 px-5 py-2.5 text-sm text-amber-800">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" />
            <div>
              {t('conflict')}
              {conflicts.map(([a, b]) => `${label(problem, a)} ⇔ ${label(problem, b)}`).join(file.listSep)}
            </div>
          </div>
        )}

        <div className="min-h-0 flex-1">
          {tab === 'students' ? (
            <StudentsTab problem={problem} onChange={onChange} openGroups={setTab} />
          ) : (
            <GroupsTab key={tab} kind={tab} problem={problem} onChange={onChange} conflicts={conflicts} />
          )}
        </div>
      </div>
    </div>
  )
}

const label = (p: Problem, i: number) => `${p.students[i].no} ${p.students[i].name}`

/* ------------------------------------------------------------------ */
/* 生徒一覧                                                            */
/* ------------------------------------------------------------------ */

type Filters = Record<string, Set<string>>
type PairFilter = 'all' | 'any' | 'wanted' | 'unwanted' | 'none'

function StudentsTab({ problem, onChange, openGroups }: { problem: Problem; onChange: (p: Problem) => void; openGroups: (t: EditorTab) => void }) {
  const { t, file, compare } = useT()
  const [query, setQuery] = useState('')
  const [filters, setFilters] = useState<Filters>({})
  const [pairFilter, setPairFilter] = useState<PairFilter>('all')
  const [sort, setSort] = useState<{ col: string; dir: 1 | -1 } | null>(null)
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [newCol, setNewCol] = useState<string | null>(null)
  const [newKind, setNewKind] = useState<ColumnKind>('flag')
  const desktop = useIsDesktop()
  const scrollRef = useRef<HTMLDivElement>(null)
  const { wanted, unwanted } = useMemo(() => groupsOf(problem), [problem])

  // 絞り込み・並べ替えは条件を変えたときだけ適用する（編集中の行が消えたり動いたりしないように）
  const problemRef = useRef(problem)
  problemRef.current = problem
  const [applyTick, setApplyTick] = useState(0)
  const [dirty, setDirty] = useState(false)
  const order = useMemo(() => {
    const p = problemRef.current
    const { wanted, unwanted } = groupsOf(p)
    const q = query.trim().toLowerCase()
    let rows = p.students
      .map((s, i) => ({ s, i }))
      .filter(({ s, i }) => {
        if (q && !(s.name.toLowerCase().includes(q) || `${s.no}`.startsWith(q))) return false
        for (const [col, set] of Object.entries(filters)) {
          if (set.size === 0) continue
          const v = s.values[col] ?? ''
          if (!set.has(v === '' ? EMPTY : v)) return false
        }
        const w = wanted[i].length > 0
        const u = unwanted[i].length > 0
        if (pairFilter === 'any' && !w && !u) return false
        if (pairFilter === 'wanted' && !w) return false
        if (pairFilter === 'unwanted' && !u) return false
        if (pairFilter === 'none' && (w || u)) return false
        return true
      })
    if (sort) {
      const get = (s: (typeof rows)[number]['s']) => (sort.col === 'NO' ? s.no : sort.col === '名前' ? s.name : (s.values[sort.col] ?? ''))
      rows = [...rows].sort((a, b) => {
        const x = get(a.s)
        const y = get(b.s)
        const nx = Number(x)
        const ny = Number(y)
        if (x === '' && y !== '') return 1
        if (y === '' && x !== '') return -1
        const c = Number.isFinite(nx) && Number.isFinite(ny) && x !== '' && y !== '' ? nx - ny : compare(String(x), String(y))
        return c * sort.dir
      })
    }
    return rows.map((r) => r.i)
    // 生徒数・ペア指定が変わったとき（index が変わりうる）も再計算
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, filters, pairFilter, sort, applyTick, problem.students.length, problem.wantedGroups, problem.unwantedGroups, compare])
  useEffect(() => setDirty(false), [order])
  const visible = useMemo(
    () => order.filter((i) => i < problem.students.length).map((i) => ({ s: problem.students[i], i })),
    [order, problem.students],
  )
  const filtering = !!query || pairFilter !== 'all' || !!sort || Object.values(filters).some((f) => f.size > 0)
  const change = (p: Problem) => {
    if (filtering) setDirty(true)
    onChange(p)
  }

  const activeFilterCount = Object.values(filters).filter((s) => s.size > 0).length + (pairFilter !== 'all' ? 1 : 0) + (query ? 1 : 0)
  const clearFilters = () => {
    setFilters({})
    setPairFilter('all')
    setQuery('')
  }
  const toggleSort = (col: string) =>
    setSort((s) => (s?.col !== col ? { col, dir: 1 } : s.dir === 1 ? { col, dir: -1 } : null))

  const allVisibleSelected = visible.length > 0 && visible.every(({ i }) => selected.has(i))
  const toggleAll = () =>
    setSelected((sel) => {
      const next = new Set(sel)
      if (allVisibleSelected) visible.forEach(({ i }) => next.delete(i))
      else visible.forEach(({ i }) => next.add(i))
      return next
    })
  const toggleOne = (i: number) =>
    setSelected((sel) => {
      const next = new Set(sel)
      if (next.has(i)) next.delete(i)
      else next.add(i)
      return next
    })

  // 非表示になった生徒は一括操作の対象にしない
  const visibleSet = new Set(visible.map((v) => v.i))
  const sel = [...selected].filter((i) => visibleSet.has(i))
  const hiddenSelected = selected.size - sel.length

  return (
    <div className="flex h-full flex-col">
      {/* ツールバー */}
      <div className="flex flex-col gap-2 border-b border-slate-100 px-3 py-3 sm:px-5 lg:flex-row lg:flex-wrap lg:items-center">
        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-slate-400" />
          <input
            autoFocus={desktop}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t('searchPlaceholder')}
            className="w-full rounded-xl border border-slate-200 bg-white py-2 pl-9 pr-3 text-sm outline-none lg:w-52 transition focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100"
          />
        </div>
        {/* スマホでは横スクロール */}
        <div className="-mx-3 flex items-center gap-2 overflow-x-auto px-3 pb-1 [scrollbar-width:none] sm:-mx-5 sm:px-5 lg:mx-0 lg:flex-wrap lg:overflow-visible lg:px-0 lg:pb-0">
        {problem.columns.map((c) => (
          <FilterMenu
            key={c.name}
            column={c}
            problem={problem}
            value={filters[c.name] ?? new Set()}
            onChange={(set) => setFilters((f) => ({ ...f, [c.name]: set }))}
          />
        ))}
        <Popover
          button={(open) => (
            <FilterButton active={pairFilter !== 'all'} open={open}>
              {t('pairFilter')}{pairFilter !== 'all' && `: ${t(PAIR_LABEL[pairFilter])}`}
            </FilterButton>
          )}
        >
          {(close) => (
            <div className="w-44 p-1">
              {(Object.keys(PAIR_LABEL) as PairFilter[]).map((k) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => {
                    setPairFilter(k)
                    close()
                  }}
                  className="flex w-full items-center justify-between rounded-lg px-3 py-1.5 text-left text-sm hover:bg-slate-50"
                >
                  {t(PAIR_LABEL[k])}
                  {pairFilter === k && <Check className="size-4 text-indigo-600" />}
                </button>
              ))}
            </div>
          )}
        </Popover>
        {activeFilterCount > 0 && (
          <button type="button" onClick={clearFilters} className="text-xs font-semibold text-indigo-600 hover:underline">
            {t('clearFilters')}
          </button>
        )}
        </div>
        <div className="flex flex-wrap items-center gap-2 lg:ml-auto">
          {dirty && (
            <button
              type="button"
              onClick={() => setApplyTick((t) => t + 1)}
              className="inline-flex items-center gap-1 rounded-lg bg-amber-50 px-2 py-1 text-xs font-semibold text-amber-700 ring-1 ring-amber-200 hover:bg-amber-100"
              title={t('reapplyTitle')}
            >
              <RefreshCw className="size-3.5" /> {t('reapply')}
            </button>
          )}
          <span className="mr-auto text-xs tabular-nums text-slate-500 lg:mr-0">
            {visible.length === problem.students.length ? t('countAll', { n: visible.length }) : t('countSome', { shown: visible.length, total: problem.students.length })}
          </span>
          {newCol !== null ? (
            <form
              className="flex w-full flex-wrap items-center gap-1 sm:w-auto"
              onSubmit={(e) => {
                e.preventDefault()
                const name = newCol.trim()
                if (name) onChange(addColumn(problem, name, newKind))
                setNewCol(null)
              }}
            >
              <input
                autoFocus
                value={newCol}
                onChange={(e) => setNewCol(e.target.value)}
                onKeyDown={(e) => e.key === 'Escape' && (e.stopPropagation(), setNewCol(null))}
                placeholder={t('newColPlaceholder')}
                className="min-w-0 flex-1 rounded-xl border border-indigo-300 px-3 py-2 text-sm outline-none ring-2 ring-indigo-100 sm:w-44 sm:flex-none"
              />
              <select
                value={newKind}
                onChange={(e) => setNewKind(e.target.value as ColumnKind)}
                className="rounded-xl border border-slate-200 px-2 py-2 text-sm outline-none focus:border-indigo-400"
                title={t('kindTitle')}
              >
                <option value="flag">{t('kindOptFlag')}</option>
                <option value="category">{t('kindOptCategory')}</option>
                <option value="degree">{t('kindOptDegree')}</option>
                <option value="numeric">{t('kindOptNumeric')}</option>
              </select>
              <button type="button" className="btn-ghost !py-2" onClick={() => setNewCol(null)}>
                {t('cancelShort')}
              </button>
              <button type="submit" className="btn-primary !py-2">
                {t('add')}
              </button>
            </form>
          ) : (
            <button type="button" className="btn-ghost !py-2" onClick={() => setNewCol('')}>
              <Plus className="size-4" /> {t('addColumn')}
            </button>
          )}
          <button
            type="button"
            className="btn-ghost !py-2"
            onClick={() => {
              onChange(addStudent(problem))
              clearFilters()
              setSort(null)
              setTimeout(() => {
                scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
                const inputs = scrollRef.current?.querySelectorAll<HTMLInputElement>('input[data-name]')
                inputs?.[inputs.length - 1]?.focus()
              }, 50)
            }}
          >
            <UserPlus className="size-4" /> {t('addStudent')}
          </button>
        </div>
      </div>

      {/* 表 */}
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-auto">
        {/* スマホ: カード表示 */}
        {!desktop && (
        <div className="space-y-2 p-3">
          {/* 項目ごとの設定（リストの選択肢の編集・項目の削除） */}
          {problem.columns.length > 0 && (
            <div className="-mx-3 flex gap-2 overflow-x-auto px-3 pb-1 [scrollbar-width:none]">
              {problem.columns.map((c) => (
                <ColumnMenu key={c.name} column={c} problem={problem} onChange={change} chip />
              ))}
            </div>
          )}
          {visible.length > 0 && (
            <label className="flex items-center gap-2 px-1 text-xs font-semibold text-slate-500">
              <Checkbox checked={allVisibleSelected} onChange={toggleAll} /> {t('selectAllVisible')}
            </label>
          )}
          {visible.map(({ s, i }) => {
            const isSel = selected.has(i)
            return (
              <div key={i} className={`rounded-2xl border p-3 transition ${isSel ? 'border-indigo-300 bg-indigo-50/60' : 'border-slate-200 bg-white'}`}>
                <div className="flex items-center gap-2">
                  <Checkbox checked={isSel} onChange={() => toggleOne(i)} />
                  <NoInput
                    value={s.no}
                    validate={(no) => (isNoTaken(problem, no, i) ? t('noTaken', { no }) : null)}
                    onCommit={(no) => change(updateStudent(problem, i, { no }))}
                  />
                  <input
                    data-name
                    value={s.name}
                    placeholder={t('namePlaceholder')}
                    onChange={(e) => change(updateStudent(problem, i, { name: e.target.value }))}
                    className="min-w-0 flex-1 rounded-lg border border-transparent bg-transparent px-2 py-1 text-base font-semibold text-slate-800 outline-none placeholder:text-slate-300 focus:border-indigo-400 focus:bg-white"
                  />
                  <button
                    type="button"
                    aria-label={t('delete')}
                    onClick={() => {
                      if (confirm(t('confirmDeleteStudent', { who: label(problem, i) }))) {
                        onChange(removeStudents(problem, [i]))
                        setSelected(new Set())
                      }
                    }}
                    className="rounded-lg p-2 text-slate-300 hover:bg-rose-50 hover:text-rose-500"
                  >
                    <Trash2 className="size-4" />
                  </button>
                </div>
                {(wanted[i].length > 0 || unwanted[i].length > 0) && (
                  <div className="mt-1 flex flex-wrap gap-1 pl-7">
                    {wanted[i].map((g) => (
                      <button key={`w${g}`} type="button" onClick={() => openGroups('wanted')} className="rounded-md bg-indigo-50 px-2 py-1 text-xs font-bold text-indigo-600">
                        {file.tagPrefix.wanted}{g + 1}: {problem.wantedGroups[g].filter((j) => j !== i).map((j) => problem.students[j].name).join(file.joinSep)}
                      </button>
                    ))}
                    {unwanted[i].map((g) => (
                      <button key={`u${g}`} type="button" onClick={() => openGroups('unwanted')} className="rounded-md bg-rose-50 px-2 py-1 text-xs font-bold text-rose-600">
                        {file.tagPrefix.unwanted}{g + 1}: {problem.unwantedGroups[g].filter((j) => j !== i).map((j) => problem.students[j].name).join(file.joinSep)}
                      </button>
                    ))}
                  </div>
                )}
                <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1.5">
                  {problem.columns.map((c) => (
                    // チェックは2列に並べ、選択肢を並べる項目は1行を使う
                    <div key={c.name} className={`flex min-w-0 items-center justify-between gap-2 ${c.kind === 'flag' ? '' : 'col-span-2'}`}>
                      <span className="min-w-0 flex-1 truncate text-xs font-medium text-slate-500">{c.name}</span>
                      <ValueCell
                        column={c}
                        value={s.values[c.name] ?? ''}
                        onChange={(v) => change(setValueFor(problem, isSel ? sel : [i], c.name, v))}
                        onAddOption={(v) => change(setValueFor(addLevel(problem, c.name, v), isSel ? sel : [i], c.name, v))}
                      />
                    </div>
                  ))}
                </div>
              </div>
            )
          })}
        </div>

        )}

        {desktop && (
        <table className="w-full border-separate border-spacing-0 text-sm">
          <thead className="sticky top-0 z-10 bg-white/95 backdrop-blur">
            <tr className="text-left text-xs text-slate-500">
              <th className="sticky left-0 z-10 w-10 border-b border-slate-200 bg-white px-3 py-2">
                <Checkbox checked={allVisibleSelected} onChange={toggleAll} />
              </th>
              <SortHeader id="NO" label={t('colNo')} sort={sort} onSort={toggleSort} className="w-20" />
              <SortHeader id="名前" label={t('colName')} sort={sort} onSort={toggleSort} className="min-w-40" />
              {problem.columns.map((c) => (
                <SortHeader key={c.name} label={c.name} sort={sort} onSort={toggleSort} sub={<Distribution column={c} problem={problem} />}>
                  <ColumnMenu column={c} problem={problem} onChange={change} />
                </SortHeader>
              ))}
              <th className="border-b border-slate-200 px-3 py-2 font-semibold">{t('colPair')}</th>
              <th className="w-10 border-b border-slate-200" />
            </tr>
          </thead>
          <tbody>
            {visible.map(({ s, i }) => {
              const isSel = selected.has(i)
              return (
                <tr key={i} className={`group/row ${isSel ? 'bg-indigo-50/70' : 'hover:bg-slate-50/80'}`}>
                  <td className={`sticky left-0 border-b border-slate-100 px-3 py-1 ${isSel ? 'bg-indigo-50' : 'bg-white group-hover/row:bg-slate-50'}`}>
                    <Checkbox checked={isSel} onChange={() => toggleOne(i)} />
                  </td>
                  <td className="border-b border-slate-100 px-2 py-1">
                    <NoInput
                      value={s.no}
                      validate={(no) => (isNoTaken(problem, no, i) ? t('noTaken', { no }) : null)}
                      onCommit={(no) => change(updateStudent(problem, i, { no }))}
                    />
                  </td>
                  <td className="border-b border-slate-100 px-2 py-1">
                    <input
                      data-name
                      value={s.name}
                      placeholder={t('namePlaceholder')}
                      onChange={(e) => change(updateStudent(problem, i, { name: e.target.value }))}
                      className="w-full min-w-32 rounded-lg border border-transparent bg-transparent px-2 py-1 font-medium text-slate-800 outline-none placeholder:text-slate-300 hover:border-slate-200 focus:border-indigo-400 focus:bg-white"
                    />
                  </td>
                  {problem.columns.map((c) => (
                    <td key={c.name} className="border-b border-slate-100 px-2 py-1">
                      <ValueCell
                        column={c}
                        value={s.values[c.name] ?? ''}
                        onChange={(v) => change(setValueFor(problem, isSel ? sel : [i], c.name, v))}
                        onAddOption={(v) => change(setValueFor(addLevel(problem, c.name, v), isSel ? sel : [i], c.name, v))}
                      />
                    </td>
                  ))}
                  <td className="border-b border-slate-100 px-3 py-1">
                    <div className="flex gap-1">
                      {wanted[i].map((g) => (
                        <button
                          key={`w${g}`}
                          type="button"
                          onClick={() => openGroups('wanted')}
                          title={t('wantedTitleAttr', { names: problem.wantedGroups[g].map((j) => label(problem, j)).join(file.joinSep) })}
                          className="rounded-md bg-indigo-50 px-1.5 py-0.5 text-[11px] font-bold text-indigo-600 hover:bg-indigo-100"
                        >
                          {file.tagPrefix.wanted}
                          {g + 1}
                        </button>
                      ))}
                      {unwanted[i].map((g) => (
                        <button
                          key={`u${g}`}
                          type="button"
                          onClick={() => openGroups('unwanted')}
                          title={t('unwantedTitleAttr', { names: problem.unwantedGroups[g].map((j) => label(problem, j)).join(file.joinSep) })}
                          className="rounded-md bg-rose-50 px-1.5 py-0.5 text-[11px] font-bold text-rose-600 hover:bg-rose-100"
                        >
                          {file.tagPrefix.unwanted}
                          {g + 1}
                        </button>
                      ))}
                    </div>
                  </td>
                  <td className="border-b border-slate-100 pr-3">
                    <button
                      type="button"
                      aria-label={t('delete')}
                      onClick={() => {
                        if (confirm(t('confirmDeleteStudent', { who: label(problem, i) }))) {
                          onChange(removeStudents(problem, [i]))
                          setSelected(new Set())
                        }
                      }}
                      className="rounded-lg p-1.5 text-slate-300 opacity-0 transition hover:bg-rose-50 hover:text-rose-500 group-hover/row:opacity-100"
                    >
                      <Trash2 className="size-4" />
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        )}
        {visible.length === 0 && (
          <div className="py-16 text-center text-sm text-slate-400">
            {t('noMatch')}
            <button type="button" onClick={clearFilters} className="ml-1 font-semibold text-indigo-600 hover:underline">
              {t('clearFilters')}
            </button>
          </div>
        )}
      </div>

      {/* 一括操作バー */}
      {selected.size > 0 && (
        <div className="flex flex-wrap items-center gap-2 border-t border-slate-200 bg-slate-900 px-3 py-2.5 pb-[max(0.625rem,env(safe-area-inset-bottom))] text-sm text-white sm:px-5 sm:py-3">
          <span className="mr-1 font-semibold tabular-nums">
            {t('selectedN', { n: sel.length })}
            {hiddenSelected > 0 && <span className="ml-1 text-xs font-normal text-slate-400">{t('hiddenSelected', { n: hiddenSelected })}</span>}
          </span>
          <button
            type="button"
            disabled={sel.length < 2}
            onClick={() => {
              onChange(addGroup(problem, 'wanted', sel))
              setSelected(new Set())
            }}
            className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-500 px-3 py-1.5 font-semibold hover:bg-indigo-400 disabled:opacity-40"
          >
            <Link2 className="size-4" /> {t('makeWanted')}
          </button>
          <button
            type="button"
            disabled={sel.length < 2}
            onClick={() => {
              onChange(addGroup(problem, 'unwanted', sel))
              setSelected(new Set())
            }}
            className="inline-flex items-center gap-1.5 rounded-lg bg-rose-500 px-3 py-1.5 font-semibold hover:bg-rose-400 disabled:opacity-40"
          >
            <Split className="size-4" /> {t('makeUnwanted')}
          </button>
          {sel.length > 0 && <BulkSet problem={problem} onApply={(col, v) => change(setValueFor(problem, sel, col, v))} />}
          <button
            type="button"
            disabled={sel.length === 0}
            onClick={() => {
              if (confirm(t('confirmDeleteN', { n: sel.length }))) {
                onChange(removeStudents(problem, sel))
                setSelected(new Set())
              }
            }}
            className="inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 font-semibold text-rose-300 hover:bg-white/10 disabled:opacity-40"
          >
            <Trash2 className="size-4" /> {t('delete')}
          </button>
          <span className="ml-auto hidden text-xs text-slate-400 lg:inline">{t('bulkHint')}</span>
          <button type="button" onClick={() => setSelected(new Set())} className="rounded-lg p-1.5 hover:bg-white/10" aria-label={t('clearSelection')}>
            <X className="size-4" />
          </button>
        </div>
      )}
    </div>
  )
}

const PAIR_LABEL = {
  all: 'pairAll',
  any: 'pairAny',
  wanted: 'pairWantedF',
  unwanted: 'pairUnwantedF',
  none: 'pairNone',
} as const satisfies Record<PairFilter, string>

function SortHeader({
  id,
  label,
  sort,
  onSort,
  className = '',
  sub,
  children,
}: {
  /** 並べ替えのキー（省略時は label。NO・名前は表示名と別に固定のキーを使う） */
  id?: string
  label: string
  sort: { col: string; dir: 1 | -1 } | null
  onSort: (c: string) => void
  className?: string
  sub?: ReactNode
  children?: ReactNode
}) {
  const key = id ?? label
  const active = sort?.col === key
  return (
    <th className={`group border-b border-slate-200 px-3 py-2 align-bottom font-semibold ${className}`}>
      <div className="flex items-center gap-1">
        <button type="button" onClick={() => onSort(key)} className={`inline-flex items-center gap-1 whitespace-nowrap hover:text-slate-900 ${active ? 'text-indigo-600' : ''}`}>
          {label}
          {active ? sort.dir === 1 ? <ArrowUp className="size-3" /> : <ArrowDown className="size-3" /> : null}
        </button>
        {children}
      </div>
      {sub}
    </th>
  )
}

function Distribution({ column, problem }: { column: ColumnSpec; problem: Problem }) {
  const { t, num } = useT()
  if (column.kind === 'numeric') {
    const xs = problem.students.map((s) => Number(s.values[column.name])).filter((x, i) => problem.students[i].values[column.name] !== '' && Number.isFinite(x))
    const avg = xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0
    return <div className="mt-0.5 text-[10px] font-normal text-slate-400">{t('avg', { v: num(avg, 1) })}</div>
  }
  const counts = column.levels.map((l) => problem.students.filter((s) => s.values[column.name] === l).length)
  return (
    <div className="mt-0.5 whitespace-nowrap text-[10px] font-normal tabular-nums text-slate-400">
      {column.levels.length === 0 ? t('notEntered') : column.levels.map((l, j) => `${column.kind === 'flag' ? '' : `${l}:`}${counts[j]}`).join(' ')}
    </div>
  )
}

/** 選択肢のチップ（選んでいるものは塗りつぶし） */
function Chip({ selected, onClick, title, children }: { selected?: boolean; onClick: () => void; title?: string; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={selected}
      title={title}
      className={`inline-flex h-8 min-w-8 items-center justify-center whitespace-nowrap rounded-lg px-2 text-xs font-bold transition md:h-7 md:min-w-7 ${
        selected ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-500/30' : 'border border-slate-200 bg-white text-slate-500 hover:border-indigo-300 hover:text-indigo-600'
      }`}
    >
      {children}
    </button>
  )
}

/**
 * 値の入力。チェック → チェックボックス、リスト → 選択肢のチップ（選んでいるものをもう一度押すと空欄）と選択肢の追加、
 * 程度 → 1〜5 のチップ、数値 → 数値の入力欄
 */
function ValueCell({ column, value, onChange, onAddOption }: { column: ColumnSpec; value: string; onChange: (v: string) => void; onAddOption: (v: string) => void }) {
  const { t } = useT()
  const [adding, setAdding] = useState<string | null>(null)
  // Esc でやめたときは、入力欄が消えるときの blur で追加しない
  const cancelled = useRef(false)
  if (column.kind === 'flag') {
    const checked = value !== ''
    return (
      <button
        type="button"
        role="checkbox"
        aria-checked={checked}
        aria-label={column.name}
        onClick={() => onChange(checked ? '' : (column.levels[0] ?? t('flagMark')))}
        className="grid size-9 place-items-center rounded-lg transition hover:bg-indigo-50 md:size-7"
      >
        <span
          className={`grid size-5 place-items-center rounded-md transition ${
            checked ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-500/30' : 'border-2 border-slate-300 bg-white'
          }`}
        >
          {checked && <Check className="size-3.5" strokeWidth={3} />}
        </span>
      </button>
    )
  }
  if (column.kind === 'numeric')
    return (
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-20 rounded-lg border border-slate-200 bg-white px-2 py-1 text-right text-sm tabular-nums outline-none focus:border-indigo-400"
      />
    )
  // 単一選択: 選んでいるものをもう一度押すと空欄に戻る
  const levels = column.kind === 'degree' ? DEGREE_LEVELS : column.levels
  const addOption = () => {
    const v = cancelled.current ? '' : adding?.trim()
    cancelled.current = false
    // 足した選択肢をそのまま選ぶ（既にあれば選ぶだけ）
    if (v) {
      if (column.levels.includes(v)) onChange(v)
      else onAddOption(v)
    }
    setAdding(null)
  }
  return (
    // 表では1行に並べる（行の高さをそろえる）。スマホのカードでは折り返す
    <div className="flex flex-wrap items-center justify-end gap-1 md:flex-nowrap md:justify-start">
      {levels.map((l) => (
        <Chip key={l} selected={value === l} onClick={() => onChange(value === l ? '' : l)}>
          {l}
        </Chip>
      ))}
      {column.kind === 'category' &&
        (adding === null ? (
          <Chip onClick={() => setAdding('')} title={t('addOption')}>
            <Plus className="size-3.5" />
          </Chip>
        ) : (
          <input
            autoFocus
            value={adding}
            onChange={(e) => setAdding(e.target.value)}
            onBlur={addOption}
            onKeyDown={(e) => {
              // 確定は blur にまとめる（Enter で blur → 追加）
              if (e.key === 'Enter') e.currentTarget.blur()
              if (e.key === 'Escape') {
                e.stopPropagation()
                cancelled.current = true
                e.currentTarget.blur()
              }
            }}
            placeholder={t('newOptionPlaceholder')}
            aria-label={t('addOption')}
            className="h-8 w-36 rounded-lg border border-indigo-300 bg-white px-2 text-xs outline-none ring-2 ring-indigo-100 md:h-7"
          />
        ))}
    </div>
  )
}

/** 項目の設定（種類の表示・リストの選択肢の編集・項目の削除）。chip はスマホのカード表示用の見た目 */
function ColumnMenu({ column, problem, onChange, chip }: { column: ColumnSpec; problem: Problem; onChange: (p: Problem) => void; chip?: boolean }) {
  const { t } = useT()
  return (
    <Popover
      button={(open) =>
        chip ? (
          <span
            className={`inline-flex items-center gap-1 whitespace-nowrap rounded-xl border px-3 py-1.5 text-xs font-semibold transition ${
              open ? 'border-indigo-300 bg-indigo-50 text-indigo-700' : 'border-slate-200 bg-white text-slate-600'
            }`}
          >
            <Settings2 className="size-3.5 opacity-60" />
            {column.name}
            <span className="font-normal text-slate-400">· {t(KIND_KEY[column.kind])}</span>
          </span>
        ) : (
          <span
            title={t('columnSettings', { name: column.name })}
            className={`grid place-items-center rounded p-0.5 transition hover:bg-slate-100 hover:text-slate-700 ${open ? 'bg-slate-100 text-slate-700' : 'text-slate-300'}`}
          >
            <Settings2 className="size-3.5" />
          </span>
        )
      }
    >
      {(close) => (
        <div className="max-h-[70vh] w-72 overflow-auto p-3 text-sm font-normal text-slate-700">
          <div className="flex items-center gap-2">
            <span className="min-w-0 flex-1 truncate font-bold text-slate-900">{column.name}</span>
            <span className="shrink-0 rounded-md bg-slate-100 px-1.5 py-0.5 text-[11px] font-semibold text-slate-500">{t(KIND_KEY[column.kind])}</span>
          </div>
          {column.kind === 'category' && <OptionsEditor column={column} problem={problem} onChange={onChange} />}
          <button
            type="button"
            onClick={() => {
              if (!confirm(t('confirmDeleteColumn', { name: column.name }))) return
              close()
              onChange(removeColumn(problem, column.name))
            }}
            className="mt-3 inline-flex w-full items-center gap-1.5 rounded-lg px-2 py-1.5 text-xs font-semibold text-rose-600 hover:bg-rose-50"
          >
            <Trash2 className="size-3.5" /> {t('deleteColumnTitle', { name: column.name })}
          </button>
        </div>
      )}
    </Popover>
  )
}

/** リストの選択肢の一覧（選んでいる人数）・追加・名前の変更・削除 */
function OptionsEditor({ column, problem, onChange }: { column: ColumnSpec; problem: Problem; onChange: (p: Problem) => void }) {
  const { t } = useT()
  const [editing, setEditing] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [added, setAdded] = useState('')
  const name = column.name
  const count = (l: string) => problem.students.filter((s) => s.values[name] === l).length
  const v = added.trim()
  const addDup = column.levels.includes(v)
  const add = () => {
    if (!v || addDup) return
    onChange(addLevel(problem, name, v))
    setAdded('')
  }
  const d = draft.trim()
  const renameDup = editing !== null && d !== editing && column.levels.includes(d)
  const rename = () => {
    if (renameDup) return
    if (editing !== null && d) onChange(renameLevel(problem, name, editing, d))
    setEditing(null)
  }
  const inputCls = 'min-w-0 flex-1 rounded-lg border border-slate-200 bg-white px-2 py-1 text-sm outline-none focus:border-indigo-400'
  const iconBtn = 'rounded-lg p-1.5 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700 disabled:opacity-40'
  return (
    <div className="mt-3 space-y-1">
      <div className="text-xs font-semibold text-slate-500">{t('listOptions')}</div>
      {column.levels.length === 0 && <div className="py-1 text-xs text-slate-400">{t('noOptions')}</div>}
      {column.levels.map((l) =>
        editing === l ? (
          <form
            key={l}
            className="flex items-center gap-1"
            onSubmit={(e) => {
              e.preventDefault()
              rename()
            }}
          >
            <input autoFocus value={draft} onChange={(e) => setDraft(e.target.value)} aria-label={t('editOption', { v: l })} className={inputCls} />
            <button type="submit" disabled={!d || renameDup} className={iconBtn} aria-label={t('edit')}>
              <Check className="size-4" />
            </button>
            <button type="button" onClick={() => setEditing(null)} className={iconBtn} aria-label={t('cancel')}>
              <X className="size-4" />
            </button>
          </form>
        ) : (
          <div key={l} className="flex items-center gap-1 rounded-lg bg-slate-50 pl-3">
            <span className="min-w-0 flex-1 truncate font-semibold text-slate-800">{l}</span>
            <span className="text-xs tabular-nums text-slate-400">{t('countAll', { n: count(l) })}</span>
            <button
              type="button"
              title={t('editOption', { v: l })}
              aria-label={t('editOption', { v: l })}
              onClick={() => {
                setEditing(l)
                setDraft(l)
              }}
              className={iconBtn}
            >
              <Pencil className="size-3.5" />
            </button>
            <button
              type="button"
              title={t('deleteOption', { v: l })}
              aria-label={t('deleteOption', { v: l })}
              onClick={() => {
                const n = count(l)
                if (n === 0 || confirm(t('confirmDeleteOption', { v: l, n }))) onChange(removeLevel(problem, name, l))
              }}
              className="rounded-lg p-1.5 text-slate-400 transition hover:bg-rose-50 hover:text-rose-500"
            >
              <Trash2 className="size-3.5" />
            </button>
          </div>
        ),
      )}
      {renameDup && <div className="text-xs text-rose-600">{t('duplicateOption')}</div>}
      <form
        className="flex items-center gap-1 pt-1"
        onSubmit={(e) => {
          e.preventDefault()
          add()
        }}
      >
        <input value={added} onChange={(e) => setAdded(e.target.value)} placeholder={t('newOptionPlaceholder')} aria-label={t('addOption')} className={inputCls} />
        <button type="submit" disabled={!v || addDup} className="btn-primary !px-2 !py-1" aria-label={t('addOption')}>
          <Plus className="size-4" />
        </button>
      </form>
      {addDup && v && <div className="text-xs text-rose-600">{t('duplicateOption')}</div>}
      <p className="pt-1 text-[11px] leading-snug text-slate-400">{t('listOptionsHint')}</p>
    </div>
  )
}

function BulkSet({ problem, onApply }: { problem: Problem; onApply: (col: string, v: string) => void }) {
  const { t } = useT()
  return (
    <Popover
      dark
      up
      button={(open) => (
        <span className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 font-semibold hover:bg-white/10 ${open ? 'bg-white/10' : ''}`}>
          {t('bulkSet')} <ChevronDown className="size-4" />
        </span>
      )}
    >
      {(close) => (
        <div className="max-h-80 w-64 overflow-auto p-2 text-slate-700">
          {problem.columns.filter((c) => c.kind !== 'numeric').map((c) => (
            <div key={c.name} className="flex items-center justify-between gap-2 rounded-lg px-2 py-1.5 hover:bg-slate-50">
              <span className="truncate text-sm font-semibold">{c.name}</span>
              <div className="flex max-w-40 shrink-0 flex-wrap justify-end gap-1">
                {(c.kind === 'degree' ? DEGREE_LEVELS : c.kind === 'flag' ? [c.levels[0] ?? t('flagMark')] : c.levels).map((l) => (
                  <button
                    key={l}
                    type="button"
                    onClick={() => {
                      onApply(c.name, l)
                      close()
                    }}
                    className="rounded-md bg-indigo-50 px-1.5 py-0.5 text-xs font-bold text-indigo-700 hover:bg-indigo-100"
                  >
                    {l}
                  </button>
                ))}
                <button
                  type="button"
                  onClick={() => {
                    onApply(c.name, '')
                    close()
                  }}
                  className="rounded-md bg-slate-100 px-1.5 py-0.5 text-xs font-bold text-slate-500 hover:bg-slate-200"
                  title={t('setBlankTitle')}
                >
                  {t('blankShort')}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </Popover>
  )
}

function FilterMenu({ column, problem, value, onChange }: { column: ColumnSpec; problem: Problem; value: Set<string>; onChange: (s: Set<string>) => void }) {
  const { t, file } = useT()
  const options = useMemo(() => {
    const levels = column.kind === 'numeric' ? [...column.levels] : column.levels
    const count = (l: string) => problem.students.filter((s) => (s.values[column.name] ?? '') === l).length
    return [...levels.map((l) => ({ key: l, label: l, n: count(l) })), { key: EMPTY, label: t('blankLabel'), n: count('') }].filter((o) => o.n > 0)
  }, [column, problem, t])
  const toggle = (k: string) => {
    const next = new Set(value)
    if (next.has(k)) next.delete(k)
    else next.add(k)
    onChange(next)
  }
  const names = [...value].map((k) => (k === EMPTY ? t('blankName') : k))
  const summary =
    value.size === 0
      ? ''
      : column.kind === 'flag' && value.size === 1
        ? `: ${value.has(EMPTY) ? t('filterNo') : t('filterYes')}`
        : value.size <= 2
          ? `: ${names.join(file.joinSep)}`
          : `: ${t('filterCount', { n: value.size })}`
  return (
    <Popover
      button={(open) => (
        <FilterButton active={value.size > 0} open={open}>
          {column.name}
          {summary}
        </FilterButton>
      )}
    >
      {() => (
        <div className="max-h-72 w-52 overflow-auto p-1">
          {options.map((o) => (
            <label key={o.key} className="flex cursor-pointer items-center gap-2 rounded-lg px-3 py-1.5 text-sm hover:bg-slate-50">
              <Checkbox checked={value.has(o.key)} onChange={() => toggle(o.key)} />
              <span className="flex-1">{column.kind === 'flag' && o.key !== EMPTY ? t('flagOption', { v: o.label }) : o.label}</span>
              <span className="text-xs tabular-nums text-slate-400">{o.n}</span>
            </label>
          ))}
          {value.size > 0 && (
            <button type="button" onClick={() => onChange(new Set())} className="mt-1 w-full rounded-lg px-3 py-1.5 text-left text-xs font-semibold text-indigo-600 hover:bg-slate-50">
              {t('clearThis')}
            </button>
          )}
        </div>
      )}
    </Popover>
  )
}

function FilterButton({ active, open, children }: { active: boolean; open: boolean; children: ReactNode }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-xl border px-3 py-1.5 text-xs font-semibold transition ${
        active ? 'border-indigo-300 bg-indigo-50 text-indigo-700' : `border-slate-200 bg-white text-slate-600 hover:border-slate-300 ${open ? 'border-slate-300' : ''}`
      }`}
    >
      {active ? <Filter className="size-3" /> : null}
      {children}
      <ChevronDown className="size-3 opacity-60" />
    </span>
  )
}

/**
 * ポップオーバー。body 直下に fixed 配置で描画するため、横スクロールするツールバー等の中でも切れない。
 */
function Popover({
  button,
  children,
  dark,
  up,
}: {
  button: (open: boolean) => ReactNode
  children: (close: () => void) => ReactNode
  dark?: boolean
  up?: boolean
}) {
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState<CSSProperties>({})
  const ref = useRef<HTMLDivElement>(null)
  const panel = useRef<HTMLDivElement>(null)

  const place = () => {
    const r = ref.current?.getBoundingClientRect()
    if (!r) return
    const width = panel.current?.offsetWidth ?? 220
    const height = panel.current?.offsetHeight ?? 0
    const left = Math.max(8, Math.min(r.left, window.innerWidth - width - 8))
    // 下に収まらなければ上に開く
    const flip = up || (r.bottom + 4 + height > window.innerHeight - 8 && r.top - 8 - height > 8)
    setPos(flip ? { left, bottom: window.innerHeight - r.top + 8 } : { left, top: r.bottom + 4 })
  }

  useLayoutEffect(() => {
    if (!open || !panel.current) return
    place()
    // 中身の増減（「この条件をクリア」の表示など）で位置を再計算
    const ro = new ResizeObserver(() => place())
    ro.observe(panel.current)
    return () => ro.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  useEffect(() => {
    if (!open) return
    const inside = (t: EventTarget | null) => ref.current?.contains(t as Node) || panel.current?.contains(t as Node)
    const onDown = (e: Event) => !inside(e.target) && setOpen(false)
    // Esc はポップオーバーだけを閉じ、名簿エディタ自体には伝えない
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      e.stopPropagation()
      setOpen(false)
    }
    // スクロール時は閉じずに位置を追従（スマホでボタンを押すと横スクロールが起きるため）
    const onScroll = (e: Event) => !panel.current?.contains(e.target as Node) && place()
    document.addEventListener('pointerdown', onDown, true)
    document.addEventListener('keydown', onKey, true)
    window.addEventListener('scroll', onScroll, true)
    window.addEventListener('resize', place)
    return () => {
      document.removeEventListener('pointerdown', onDown, true)
      document.removeEventListener('keydown', onKey, true)
      window.removeEventListener('scroll', onScroll, true)
      window.removeEventListener('resize', place)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  return (
    <div ref={ref} className="relative shrink-0">
      <button type="button" onClick={() => setOpen((o) => !o)} className={dark ? 'text-white' : ''}>
        {button(open)}
      </button>
      {open &&
        createPortal(
          <div ref={panel} style={{ position: 'fixed', ...pos }} className="z-50 max-w-[calc(100vw-1rem)] rounded-xl border border-slate-200 bg-white shadow-xl">
            {children(() => setOpen(false))}
          </div>,
          document.body,
        )}
    </div>
  )
}

function Checkbox({ checked, onChange }: { checked: boolean; onChange: () => void }) {
  return (
    <input
      type="checkbox"
      checked={checked}
      onChange={onChange}
      className="size-5 cursor-pointer rounded border-slate-300 accent-indigo-600 md:size-4"
    />
  )
}

/* ------------------------------------------------------------------ */
/* 同じ組 / 別の組                                                      */
/* ------------------------------------------------------------------ */

function GroupsTab({ kind, problem, onChange, conflicts }: { kind: GroupKind; problem: Problem; onChange: (p: Problem) => void; conflicts: [number, number][] }) {
  const { t, file } = useT()
  const prefix = file.tagPrefix[kind]
  const groups = kind === 'wanted' ? problem.wantedGroups : problem.unwantedGroups
  const [draft, setDraft] = useState<number[] | null>(null)
  const tone = kind === 'wanted' ? TONE.wanted : TONE.unwanted
  const conflictSet = useMemo(() => new Set(conflicts.flat()), [conflicts])

  return (
    <div className="h-full overflow-auto px-3 py-4 sm:px-5 sm:py-5">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="font-bold text-slate-900">{kind === 'wanted' ? t('groupsWantedTitle') : t('groupsUnwantedTitle')}</div>
          <p className="mt-0.5 text-sm text-slate-500">
            {kind === 'wanted' ? t('groupsWantedDesc') : t('groupsUnwantedDesc')}
            <span className="text-slate-400">{t('groupsBulkHint')}</span>
          </p>
        </div>
        {draft === null && (
          <button type="button" className="btn-primary" onClick={() => setDraft([])}>
            <Plus className="size-4" /> {t('newGroup')}
          </button>
        )}
      </div>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {draft !== null && (
          <div className={`rounded-2xl border-2 border-dashed p-4 ${tone.border}`}>
            <div className="mb-2 text-xs font-bold text-slate-500">{t('newGroupLabel')}</div>
            <Members problem={problem} members={draft} tone={tone} onChange={setDraft} autoFocus />
            <div className="mt-3 flex justify-end gap-2">
              <button type="button" className="btn-ghost !py-1.5" onClick={() => setDraft(null)}>
                {t('cancel')}
              </button>
              <button
                type="button"
                className="btn-primary !py-1.5"
                disabled={draft.length < 2}
                onClick={() => {
                  onChange(addGroup(problem, kind, draft))
                  setDraft(null)
                }}
              >
                {t('create')}
              </button>
            </div>
          </div>
        )}
        {groups.map((g, gi) => {
          const bad = g.some((i) => conflictSet.has(i))
          return (
            <div key={gi} className={`group rounded-2xl border bg-white p-4 transition hover:shadow-md ${bad ? 'border-amber-300' : 'border-slate-200'}`}>
              <div className="mb-2 flex items-center justify-between">
                <span className={`rounded-md px-2 py-0.5 text-xs font-bold ${tone.badge}`}>
                  {prefix}
                  {gi + 1}
                </span>
                <div className="flex items-center gap-1">
                  {bad && <AlertTriangle className="size-4 text-amber-500" />}
                  <button
                    type="button"
                    onClick={() => confirm(t('confirmDeleteGroup', { label: `${prefix}${gi + 1}` })) && onChange(removeGroup(problem, kind, gi))}
                    className="rounded-lg p-1 text-slate-300 opacity-0 transition hover:bg-rose-50 hover:text-rose-500 group-hover:opacity-100"
                    aria-label={t('deleteGroup')}
                  >
                    <Trash2 className="size-4" />
                  </button>
                </div>
              </div>
              <Members
                problem={problem}
                members={g}
                tone={tone}
                onChange={(m) => {
                  if (m.length < 2 && !confirm(t('confirmGroupDissolve'))) return
                  onChange(setGroup(problem, kind, gi, m))
                }}
              />
            </div>
          )
        })}
      </div>
      {groups.length === 0 && draft === null && (
        <div className="rounded-2xl border border-dashed border-slate-200 py-16 text-center text-sm text-slate-400">
          {t('noGroups')}
        </div>
      )}
    </div>
  )
}

const TONE = {
  wanted: { chip: 'bg-indigo-50 text-indigo-700 ring-indigo-200', badge: 'bg-indigo-100 text-indigo-700', border: 'border-indigo-200' },
  unwanted: { chip: 'bg-rose-50 text-rose-700 ring-rose-200', badge: 'bg-rose-100 text-rose-700', border: 'border-rose-200' },
}

function Members({
  problem,
  members,
  tone,
  onChange,
  autoFocus,
}: {
  problem: Problem
  members: number[]
  tone: (typeof TONE)['wanted']
  onChange: (m: number[]) => void
  autoFocus?: boolean
}) {
  const { t } = useT()
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {members.map((i) => (
        <span key={i} className={`inline-flex items-center gap-1 rounded-lg py-1 pl-2 pr-1 text-sm font-medium ring-1 ${tone.chip}`}>
          <span className="font-mono text-[11px] opacity-60">{problem.students[i].no}</span>
          {problem.students[i].name || t('noName')}
          <button
            type="button"
            onClick={() => onChange(members.filter((x) => x !== i))}
            className="rounded p-0.5 opacity-50 hover:bg-white/60 hover:opacity-100"
            aria-label={t('removeMember')}
          >
            <X className="size-3" />
          </button>
        </span>
      ))}
      <StudentPicker problem={problem} exclude={members} onPick={(i) => onChange([...members, i])} autoFocus={autoFocus} />
    </div>
  )
}

/** NO・名前で検索して生徒を追加するコンボボックス */
function StudentPicker({ problem, exclude, onPick, autoFocus }: { problem: Problem; exclude: number[]; onPick: (i: number) => void; autoFocus?: boolean }) {
  const { t } = useT()
  const [q, setQ] = useState('')
  const [open, setOpen] = useState(false)
  const [cursor, setCursor] = useState(0)
  const ex = useMemo(() => new Set(exclude), [exclude])
  const results = useMemo(() => {
    const t = q.trim().toLowerCase()
    return problem.students
      .map((s, i) => ({ s, i }))
      .filter(({ s, i }) => !ex.has(i) && (!t || `${s.no}`.startsWith(t) || s.name.toLowerCase().includes(t)))
      .slice(0, 8)
  }, [q, problem, ex])
  const pick = (i: number) => {
    onPick(i)
    setQ('')
    setCursor(0)
  }
  return (
    <div className="relative">
      <input
        autoFocus={autoFocus}
        value={q}
        onChange={(e) => {
          setQ(e.target.value)
          setOpen(true)
          setCursor(0)
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 120)}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown') {
            e.preventDefault()
            setCursor((c) => Math.min(c + 1, results.length - 1))
          } else if (e.key === 'ArrowUp') {
            e.preventDefault()
            setCursor((c) => Math.max(c - 1, 0))
          } else if (e.key === 'Enter' && results[cursor]) {
            e.preventDefault()
            pick(results[cursor].i)
          }
        }}
        placeholder={t('addMemberPlaceholder')}
        className="w-44 rounded-lg border border-dashed border-slate-300 bg-transparent px-2 py-1 text-sm outline-none placeholder:text-slate-400 focus:border-solid focus:border-indigo-400 focus:bg-white"
      />
      {open && results.length > 0 && (
        <ul className="absolute left-0 top-full z-30 mt-1 w-56 overflow-hidden rounded-xl border border-slate-200 bg-white py-1 shadow-xl">
          {results.map(({ s, i }, j) => (
            <li key={i}>
              <button
                type="button"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => pick(i)}
                onMouseEnter={() => setCursor(j)}
                className={`flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm ${j === cursor ? 'bg-indigo-50 text-indigo-800' : 'text-slate-700'}`}
              >
                <span className="w-8 font-mono text-xs text-slate-400">{s.no}</span>
                {s.name || t('noName')}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/** NO 入力: 入力中は下書きとして保持し、確定（Enter / フォーカスアウト）時に 1 以上の整数・重複なしを検証 */
function NoInput({ value, validate, onCommit }: { value: number; validate: (no: number) => string | null; onCommit: (no: number) => void }) {
  const { t } = useT()
  const [draft, setDraft] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined)
  useEffect(() => () => clearTimeout(timer.current), [])
  const commit = () => {
    if (draft === null) return
    const no = Number(draft)
    const err = !Number.isInteger(no) || no < 1 ? t('noInvalid') : validate(no)
    if (err) {
      setError(err)
      clearTimeout(timer.current)
      timer.current = setTimeout(() => setError(null), 2500)
    } else if (no !== value) onCommit(no)
    setDraft(null)
  }
  return (
    <div className="relative">
      <input
        inputMode="numeric"
        value={draft ?? String(value)}
        onChange={(e) => setDraft(e.target.value.replace(/[^0-9]/g, ''))}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
          if (e.key === 'Escape') {
            e.stopPropagation()
            setDraft(null)
          }
        }}
        className={`w-16 rounded-lg border bg-transparent px-2 py-1 font-mono text-xs tabular-nums text-slate-500 outline-none focus:bg-white ${
          error ? 'border-rose-400 bg-rose-50' : 'border-transparent hover:border-slate-200 focus:border-indigo-400'
        }`}
      />
      {error && (
        <div className="absolute left-0 top-full z-20 mt-1 whitespace-nowrap rounded-lg bg-rose-600 px-2 py-1 text-xs font-medium text-white shadow-lg">{error}</div>
      )}
    </div>
  )
}
