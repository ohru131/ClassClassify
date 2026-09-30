import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  Check,
  ChevronDown,
  Download,
  Filter,
  Link2,
  Plus,
  RefreshCw,
  Search,
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
  addStudent,
  findConflicts,
  groupsOf,
  isNoTaken,
  removeColumn,
  removeGroup,
  removeStudents,
  setGroup,
  setValueFor,
  updateStudent,
  type GroupKind,
} from '../solver/roster'

export type EditorTab = 'students' | GroupKind

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
    { id: 'students', label: '生徒一覧', count: problem.students.length, icon: <Users className="size-4" /> },
    { id: 'wanted', label: '同じ組', count: problem.wantedGroups.length, icon: <Link2 className="size-4" /> },
    { id: 'unwanted', label: '別の組', count: problem.unwantedGroups.length, icon: <Split className="size-4" /> },
  ]

  return (
    <div
      className="fixed inset-0 z-40 flex items-stretch justify-center bg-slate-900/40 p-0 backdrop-blur-sm sm:p-4"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        role="dialog"
        aria-modal
        aria-label="名簿エディタ"
        className="flex w-full max-w-[90rem] flex-col overflow-hidden bg-white shadow-2xl sm:rounded-3xl"
      >
        <div className="flex flex-wrap items-center gap-3 border-b border-slate-100 px-5 py-3">
          <div className="mr-2 text-lg font-extrabold text-slate-900">名簿</div>
          <div className="inline-flex rounded-xl bg-slate-100 p-1">
            {tabs.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => setTab(t.id)}
                className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-semibold transition ${
                  tab === t.id ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                }`}
              >
                {t.icon}
                {t.label}
                <span className="rounded-full bg-slate-200/70 px-1.5 text-[11px] tabular-nums text-slate-600">{t.count}</span>
              </button>
            ))}
          </div>
          <div className="ml-auto flex items-center gap-2">
            <button type="button" className="btn-ghost !py-2" onClick={onExport} title="ひな形と同じ形式で保存（再読み込み可能）">
              <Download className="size-4" /> 名簿を保存
            </button>
            <button type="button" onClick={onClose} className="rounded-xl p-2 text-slate-500 hover:bg-slate-100" aria-label="閉じる">
              <X className="size-5" />
            </button>
          </div>
        </div>

        {conflicts.length > 0 && (
          <div className="flex items-start gap-2 border-b border-amber-100 bg-amber-50 px-5 py-2.5 text-sm text-amber-800">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" />
            <div>
              矛盾する指定があります（同じ組でつながる生徒が別の組にも指定）:{' '}
              {conflicts.map(([a, b]) => `${label(problem, a)} ⇔ ${label(problem, b)}`).join('、')}
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
  const [query, setQuery] = useState('')
  const [filters, setFilters] = useState<Filters>({})
  const [pairFilter, setPairFilter] = useState<PairFilter>('all')
  const [sort, setSort] = useState<{ col: string; dir: 1 | -1 } | null>(null)
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [newCol, setNewCol] = useState<string | null>(null)
  const [newKind, setNewKind] = useState<ColumnKind>('flag')
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
        const c = Number.isFinite(nx) && Number.isFinite(ny) && x !== '' && y !== '' ? nx - ny : String(x).localeCompare(String(y), 'ja')
        return c * sort.dir
      })
    }
    return rows.map((r) => r.i)
    // 生徒数・ペア指定が変わったとき（index が変わりうる）も再計算
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, filters, pairFilter, sort, applyTick, problem.students.length, problem.wantedGroups, problem.unwantedGroups])
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
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-100 px-5 py-3">
        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-slate-400" />
          <input
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="NO・名前で検索"
            className="w-52 rounded-xl border border-slate-200 bg-white py-2 pl-9 pr-3 text-sm outline-none transition focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100"
          />
        </div>
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
              ペア{pairFilter !== 'all' && `: ${PAIR_LABEL[pairFilter]}`}
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
                  {PAIR_LABEL[k]}
                  {pairFilter === k && <Check className="size-4 text-indigo-600" />}
                </button>
              ))}
            </div>
          )}
        </Popover>
        {activeFilterCount > 0 && (
          <button type="button" onClick={clearFilters} className="text-xs font-semibold text-indigo-600 hover:underline">
            条件をクリア
          </button>
        )}
        <div className="ml-auto flex items-center gap-2">
          {dirty && (
            <button
              type="button"
              onClick={() => setApplyTick((t) => t + 1)}
              className="inline-flex items-center gap-1 rounded-lg bg-amber-50 px-2 py-1 text-xs font-semibold text-amber-700 ring-1 ring-amber-200 hover:bg-amber-100"
              title="編集内容に合わせて絞り込み・並べ替えをやり直す"
            >
              <RefreshCw className="size-3.5" /> 絞り込みを再適用
            </button>
          )}
          <span className="text-xs tabular-nums text-slate-500">
            {visible.length === problem.students.length ? `${visible.length} 名` : `${visible.length} / ${problem.students.length} 名`}
          </span>
          {newCol !== null ? (
            <form
              className="flex items-center gap-1"
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
                placeholder="項目名（例: リーダー）"
                className="w-44 rounded-xl border border-indigo-300 px-3 py-2 text-sm outline-none ring-2 ring-indigo-100"
              />
              <select
                value={newKind}
                onChange={(e) => setNewKind(e.target.value as ColumnKind)}
                className="rounded-xl border border-slate-200 px-2 py-2 text-sm outline-none focus:border-indigo-400"
                title="値の種類"
              >
                <option value="flag">○ / 空欄</option>
                <option value="category">段階・カテゴリ</option>
                <option value="numeric">数値（点数など）</option>
              </select>
              <button type="button" className="btn-ghost !py-2" onClick={() => setNewCol(null)}>
                取消
              </button>
              <button type="submit" className="btn-primary !py-2">
                追加
              </button>
            </form>
          ) : (
            <button type="button" className="btn-ghost !py-2" onClick={() => setNewCol('')}>
              <Plus className="size-4" /> 項目
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
            <UserPlus className="size-4" /> 生徒
          </button>
        </div>
      </div>

      {/* 表 */}
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-auto">
        <table className="w-full border-separate border-spacing-0 text-sm">
          <thead className="sticky top-0 z-10 bg-white/95 backdrop-blur">
            <tr className="text-left text-xs text-slate-500">
              <th className="sticky left-0 z-10 w-10 border-b border-slate-200 bg-white px-3 py-2">
                <Checkbox checked={allVisibleSelected} onChange={toggleAll} />
              </th>
              <SortHeader label="NO" sort={sort} onSort={toggleSort} className="w-20" />
              <SortHeader label="名前" sort={sort} onSort={toggleSort} className="min-w-40" />
              {problem.columns.map((c) => (
                <SortHeader key={c.name} label={c.name} sort={sort} onSort={toggleSort} sub={<Distribution column={c} problem={problem} />}>
                  <button
                    type="button"
                    title={`「${c.name}」列を削除`}
                    onClick={(e) => {
                      e.stopPropagation()
                      if (confirm(`「${c.name}」列を削除しますか？`)) onChange(removeColumn(problem, c.name))
                    }}
                    className="rounded p-0.5 text-slate-300 opacity-0 transition hover:bg-rose-50 hover:text-rose-500 group-hover:opacity-100"
                  >
                    <Trash2 className="size-3.5" />
                  </button>
                </SortHeader>
              ))}
              <th className="border-b border-slate-200 px-3 py-2 font-semibold">ペア</th>
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
                      validate={(no) => (isNoTaken(problem, no, i) ? `NO ${no} は他の生徒が使っています` : null)}
                      onCommit={(no) => change(updateStudent(problem, i, { no }))}
                    />
                  </td>
                  <td className="border-b border-slate-100 px-2 py-1">
                    <input
                      data-name
                      value={s.name}
                      placeholder="名前を入力"
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
                          title={`同じ組: ${problem.wantedGroups[g].map((j) => label(problem, j)).join('・')}`}
                          className="rounded-md bg-indigo-50 px-1.5 py-0.5 text-[11px] font-bold text-indigo-600 hover:bg-indigo-100"
                        >
                          同{g + 1}
                        </button>
                      ))}
                      {unwanted[i].map((g) => (
                        <button
                          key={`u${g}`}
                          type="button"
                          onClick={() => openGroups('unwanted')}
                          title={`別の組: ${problem.unwantedGroups[g].map((j) => label(problem, j)).join('・')}`}
                          className="rounded-md bg-rose-50 px-1.5 py-0.5 text-[11px] font-bold text-rose-600 hover:bg-rose-100"
                        >
                          別{g + 1}
                        </button>
                      ))}
                    </div>
                  </td>
                  <td className="border-b border-slate-100 pr-3">
                    <button
                      type="button"
                      aria-label="削除"
                      onClick={() => {
                        if (confirm(`${label(problem, i)} を名簿から削除しますか？`)) {
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
        {visible.length === 0 && (
          <div className="py-16 text-center text-sm text-slate-400">
            条件に合う生徒がいません。
            <button type="button" onClick={clearFilters} className="ml-1 font-semibold text-indigo-600 hover:underline">
              条件をクリア
            </button>
          </div>
        )}
      </div>

      {/* 一括操作バー */}
      {selected.size > 0 && (
        <div className="flex flex-wrap items-center gap-2 border-t border-slate-200 bg-slate-900 px-5 py-3 text-sm text-white">
          <span className="mr-1 font-semibold tabular-nums">
            {sel.length} 名を選択
            {hiddenSelected > 0 && <span className="ml-1 text-xs font-normal text-slate-400">（非表示の {hiddenSelected} 名は対象外）</span>}
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
            <Link2 className="size-4" /> 同じ組にする
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
            <Split className="size-4" /> 別の組にする
          </button>
          <BulkSet problem={problem} onApply={(col, v) => change(setValueFor(problem, sel, col, v))} />
          <button
            type="button"
            onClick={() => {
              if (confirm(`${sel.length} 名を名簿から削除しますか？`)) {
                onChange(removeStudents(problem, sel))
                setSelected(new Set())
              }
            }}
            className="inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 font-semibold text-rose-300 hover:bg-white/10"
          >
            <Trash2 className="size-4" /> 削除
          </button>
          <span className="ml-auto hidden text-xs text-slate-400 lg:inline">選択中はセルの変更が選択した全員に反映されます</span>
          <button type="button" onClick={() => setSelected(new Set())} className="rounded-lg p-1.5 hover:bg-white/10" aria-label="選択解除">
            <X className="size-4" />
          </button>
        </div>
      )}
    </div>
  )
}

const PAIR_LABEL: Record<PairFilter, string> = {
  all: 'すべて',
  any: '指定あり',
  wanted: '同じ組の指定あり',
  unwanted: '別の組の指定あり',
  none: '指定なし',
}

function SortHeader({
  label,
  sort,
  onSort,
  className = '',
  sub,
  children,
}: {
  label: string
  sort: { col: string; dir: 1 | -1 } | null
  onSort: (c: string) => void
  className?: string
  sub?: ReactNode
  children?: ReactNode
}) {
  const active = sort?.col === label
  return (
    <th className={`group border-b border-slate-200 px-3 py-2 align-bottom font-semibold ${className}`}>
      <div className="flex items-center gap-1">
        <button type="button" onClick={() => onSort(label)} className={`inline-flex items-center gap-1 whitespace-nowrap hover:text-slate-900 ${active ? 'text-indigo-600' : ''}`}>
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
  if (column.kind === 'numeric') {
    const xs = problem.students.map((s) => Number(s.values[column.name])).filter((x, i) => problem.students[i].values[column.name] !== '' && Number.isFinite(x))
    const avg = xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0
    return <div className="mt-0.5 text-[10px] font-normal text-slate-400">平均 {avg.toFixed(1)}</div>
  }
  const counts = column.levels.map((l) => problem.students.filter((s) => s.values[column.name] === l).length)
  return (
    <div className="mt-0.5 whitespace-nowrap text-[10px] font-normal tabular-nums text-slate-400">
      {column.levels.length === 0 ? '未入力' : column.levels.map((l, j) => `${column.kind === 'flag' ? '' : `${l}:`}${counts[j]}`).join(' ')}
    </div>
  )
}

function ValueCell({ column, value, onChange }: { column: ColumnSpec; value: string; onChange: (v: string) => void }) {
  // 値が1種類（○など）または未入力の列はトグル
  if (column.kind === 'flag' && (column.levels.length <= 1)) {
    const mark = column.levels[0] ?? '○'
    const on = value !== ''
    return (
      <button
        type="button"
        onClick={() => onChange(on ? '' : mark)}
        aria-pressed={on}
        className={`grid h-7 min-w-10 place-items-center rounded-lg px-2 text-xs font-bold transition ${
          on ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-500/30' : 'border border-dashed border-slate-200 text-slate-300 hover:border-indigo-300 hover:text-indigo-400'
        }`}
      >
        {on ? value : '—'}
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
  return <CategorySelect levels={column.levels} value={value} onChange={onChange} />
}

function CategorySelect({ levels, value, onChange }: { levels: string[]; value: string; onChange: (v: string) => void }) {
  return (
    <select
      value={value}
      onChange={(e) => {
        if (e.target.value === '__new__') {
          const v = prompt('新しい値を入力')?.trim()
          if (v) onChange(v)
        } else onChange(e.target.value)
      }}
      className={`h-7 rounded-lg border px-2 text-sm outline-none focus:border-indigo-400 ${value === '' ? 'border-dashed border-slate-200 text-slate-300' : 'border-slate-200 bg-white font-semibold text-slate-700'}`}
    >
      <option value="">—</option>
      {levels.map((l) => (
        <option key={l} value={l}>
          {l}
        </option>
      ))}
      <option value="__new__">＋ 新しい値…</option>
    </select>
  )
}

function BulkSet({ problem, onApply }: { problem: Problem; onApply: (col: string, v: string) => void }) {
  return (
    <Popover
      dark
      up
      button={(open) => (
        <span className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 font-semibold hover:bg-white/10 ${open ? 'bg-white/10' : ''}`}>
          項目を一括設定 <ChevronDown className="size-4" />
        </span>
      )}
    >
      {(close) => (
        <div className="max-h-80 w-64 overflow-auto p-2 text-slate-700">
          {problem.columns.filter((c) => c.kind !== 'numeric').map((c) => (
            <div key={c.name} className="flex items-center justify-between gap-2 rounded-lg px-2 py-1.5 hover:bg-slate-50">
              <span className="truncate text-sm font-semibold">{c.name}</span>
              <div className="flex shrink-0 gap-1">
                {(c.levels.length ? c.levels : c.kind === 'flag' ? ['○'] : []).slice(0, 6).map((l) => (
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
                  title="空欄にする"
                >
                  空
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
  const options = useMemo(() => {
    const levels = column.kind === 'numeric' ? [...column.levels] : column.levels
    const count = (l: string) => problem.students.filter((s) => (s.values[column.name] ?? '') === l).length
    return [...levels.map((l) => ({ key: l, label: l, n: count(l) })), { key: EMPTY, label: '（空欄）', n: count('') }].filter((o) => o.n > 0)
  }, [column, problem])
  const toggle = (k: string) => {
    const next = new Set(value)
    if (next.has(k)) next.delete(k)
    else next.add(k)
    onChange(next)
  }
  const names = [...value].map((k) => (k === EMPTY ? '空欄' : k))
  const summary =
    value.size === 0
      ? ''
      : column.kind === 'flag' && value.size === 1
        ? `: ${value.has(EMPTY) ? 'なし' : 'あり'}`
        : value.size <= 2
          ? `: ${names.join('・')}`
          : `: ${value.size}件`
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
              <span className="flex-1">{column.kind === 'flag' && o.key !== EMPTY ? `${o.label}（該当）` : o.label}</span>
              <span className="text-xs tabular-nums text-slate-400">{o.n}</span>
            </label>
          ))}
          {value.size > 0 && (
            <button type="button" onClick={() => onChange(new Set())} className="mt-1 w-full rounded-lg px-3 py-1.5 text-left text-xs font-semibold text-indigo-600 hover:bg-slate-50">
              この条件をクリア
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
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false)
    // Esc はポップオーバーだけを閉じ、名簿エディタ自体には伝えない
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      e.stopPropagation()
      setOpen(false)
    }
    document.addEventListener('mousedown', onDown, true)
    document.addEventListener('keydown', onKey, true)
    return () => {
      document.removeEventListener('mousedown', onDown, true)
      document.removeEventListener('keydown', onKey, true)
    }
  }, [open])
  return (
    <div ref={ref} className="relative">
      <button type="button" onClick={() => setOpen((o) => !o)} className={dark ? 'text-white' : ''}>
        {button(open)}
      </button>
      {open && (
        <div className={`absolute left-0 z-30 rounded-xl border border-slate-200 bg-white shadow-xl ${up ? 'bottom-full mb-2' : 'top-full mt-1'}`}>
          {children(() => setOpen(false))}
        </div>
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
      className="size-4 cursor-pointer rounded border-slate-300 accent-indigo-600"
    />
  )
}

/* ------------------------------------------------------------------ */
/* 同じ組 / 別の組                                                      */
/* ------------------------------------------------------------------ */

function GroupsTab({ kind, problem, onChange, conflicts }: { kind: GroupKind; problem: Problem; onChange: (p: Problem) => void; conflicts: [number, number][] }) {
  const groups = kind === 'wanted' ? problem.wantedGroups : problem.unwantedGroups
  const [draft, setDraft] = useState<number[] | null>(null)
  const tone = kind === 'wanted' ? TONE.wanted : TONE.unwanted
  const conflictSet = useMemo(() => new Set(conflicts.flat()), [conflicts])

  return (
    <div className="h-full overflow-auto px-5 py-5">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="font-bold text-slate-900">{kind === 'wanted' ? '同じ組にするグループ' : '別の組にするグループ'}</div>
          <p className="mt-0.5 text-sm text-slate-500">
            {kind === 'wanted'
              ? 'グループ内の生徒は必ず同じ組に配置されます。'
              : 'グループ内の生徒はできる限り互いに別の組に配置されます（3人以上なら全員が別々）。'}
            <span className="text-slate-400"> 生徒一覧でチェックして一括作成もできます。</span>
          </p>
        </div>
        {draft === null && (
          <button type="button" className="btn-primary" onClick={() => setDraft([])}>
            <Plus className="size-4" /> 新しいグループ
          </button>
        )}
      </div>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {draft !== null && (
          <div className={`rounded-2xl border-2 border-dashed p-4 ${tone.border}`}>
            <div className="mb-2 text-xs font-bold text-slate-500">新しいグループ（2人以上）</div>
            <Members problem={problem} members={draft} tone={tone} onChange={setDraft} autoFocus />
            <div className="mt-3 flex justify-end gap-2">
              <button type="button" className="btn-ghost !py-1.5" onClick={() => setDraft(null)}>
                キャンセル
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
                作成
              </button>
            </div>
          </div>
        )}
        {groups.map((g, gi) => {
          const bad = g.some((i) => conflictSet.has(i))
          return (
            <div key={`${gi}:${g.join('-')}`} className={`group rounded-2xl border bg-white p-4 transition hover:shadow-md ${bad ? 'border-amber-300' : 'border-slate-200'}`}>
              <div className="mb-2 flex items-center justify-between">
                <span className={`rounded-md px-2 py-0.5 text-xs font-bold ${tone.badge}`}>
                  {kind === 'wanted' ? '同' : '別'}
                  {gi + 1}
                </span>
                <div className="flex items-center gap-1">
                  {bad && <AlertTriangle className="size-4 text-amber-500" />}
                  <button
                    type="button"
                    onClick={() => confirm(`${kind === 'wanted' ? '同' : '別'}${gi + 1} のグループを削除しますか？`) && onChange(removeGroup(problem, kind, gi))}
                    className="rounded-lg p-1 text-slate-300 opacity-0 transition hover:bg-rose-50 hover:text-rose-500 group-hover:opacity-100"
                    aria-label="グループを削除"
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
                  if (m.length < 2 && !confirm('メンバーが1人になるため、このグループは削除されます。よろしいですか？')) return
                  onChange(setGroup(problem, kind, gi, m))
                }}
              />
            </div>
          )
        })}
      </div>
      {groups.length === 0 && draft === null && (
        <div className="rounded-2xl border border-dashed border-slate-200 py-16 text-center text-sm text-slate-400">
          まだ指定はありません
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
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {members.map((i) => (
        <span key={i} className={`inline-flex items-center gap-1 rounded-lg py-1 pl-2 pr-1 text-sm font-medium ring-1 ${tone.chip}`}>
          <span className="font-mono text-[11px] opacity-60">{problem.students[i].no}</span>
          {problem.students[i].name || '（名前なし）'}
          <button
            type="button"
            onClick={() => onChange(members.filter((x) => x !== i))}
            className="rounded p-0.5 opacity-50 hover:bg-white/60 hover:opacity-100"
            aria-label="外す"
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
        placeholder="＋ NO か名前で追加"
        className="w-40 rounded-lg border border-dashed border-slate-300 bg-transparent px-2 py-1 text-sm outline-none placeholder:text-slate-400 focus:border-solid focus:border-indigo-400 focus:bg-white"
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
                {s.name || '（名前なし）'}
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
  const [draft, setDraft] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const commit = () => {
    if (draft === null) return
    const no = Number(draft)
    const err = !Number.isInteger(no) || no < 1 ? 'NO は 1 以上の整数で入力してください' : validate(no)
    if (err) {
      setError(err)
      setTimeout(() => setError(null), 2500)
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
