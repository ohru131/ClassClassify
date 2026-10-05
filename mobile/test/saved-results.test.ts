import { describe, expect, it } from 'vitest'
import XLSX from 'xlsx-js-style'
import { makeI18n } from '../lib/i18n-core'
import { runSliced } from '../lib/runner'
import { loadSample } from '../lib/samples'
import {
  canSaveMore,
  countedSaves,
  defaultSaveName,
  findPreviousClassColumn,
  FREE_SAVE_LIMIT,
  isSavedMetaList,
  isSavedResult,
  readSavedMetaList,
  metaOf,
  placementOfSaved,
  previousClassValues,
  PREVIOUS_CLASS_WEIGHT,
  setColumnEnabled,
  withPreviousClass,
  type SavedResult,
} from '../lib/saved-results'
import { compile, evaluate, parsePlacement, resultWorkbook, rosterWorkbook, writeXlsx, type Problem } from '../lib/solver'

const ja = makeI18n('ja')
const saved = (problem: Problem, classOf: number[], k: number): SavedResult => ({ version: 1, id: 'x', name: 'テスト', savedAt: '2026-03-20T09:00:00.000Z', problem, classOf, k })
const solve = async (p: Problem, seed: number) => (await runSliced(compile(p).compiled, { timeMs: 800, starts: 1, seed, yieldToUi: async () => {} }).promise).classOf

/** 新しい組の中で、前回も同じ組だった2人の組み合わせの数 */
function pairsAgain(prev: number[], next: number[]): number {
  let n = 0
  for (let i = 0; i < prev.length; i++) for (let j = i + 1; j < prev.length; j++) if (prev[i] === prev[j] && next[i] === next[j]) n++
  return n
}

describe('保存した編成', () => {
  const { problem } = loadSample('ja', 'sample1')
  const classOf = problem.students.map((_, i) => i % 4)

  it('形を確かめる（壊れたデータは開かない）', () => {
    const s = saved(problem, classOf, 4)
    expect(isSavedResult(s)).toBe(true)
    expect(isSavedResult({ ...s, classOf: classOf.slice(1) })).toBe(false)
    expect(isSavedResult({ ...s, classOf: classOf.map(() => 4) })).toBe(false)
    expect(isSavedResult({ ...s, version: 2 })).toBe(false)
    expect(isSavedMetaList([metaOf(s)])).toBe(true)
    expect(metaOf(s)).toEqual({ id: 'x', name: 'テスト', savedAt: s.savedAt, n: problem.students.length, k: 4 })
    expect(isSavedMetaList([{ id: 1 }])).toBe(false)
    // 1件壊れていても、ほかの保存は一覧に残す。配列でなければ読めなかった扱い
    expect(readSavedMetaList([metaOf(s), { id: 1 }])).toEqual([metaOf(s)])
    expect(readSavedMetaList({})).toBeNull()
  })

  it('Excel に書き出したときに残したものは書き出し元を持ち、件数の上限に数えない', () => {
    const s: SavedResult = { ...saved(problem, classOf, 4), exported: { fileName: 'クラス編成結果_2026-10-05.xlsx' } }
    expect(isSavedResult(s)).toBe(true)
    expect(isSavedResult({ ...s, exported: { fileName: 1 } })).toBe(false)
    expect(metaOf(s).exported).toEqual({ fileName: 'クラス編成結果_2026-10-05.xlsx' })
    expect(isSavedMetaList([metaOf(s)])).toBe(true)
    expect(readSavedMetaList([metaOf(s), { ...metaOf(s), exported: 'x' }])).toEqual([metaOf(s)])
    // 書き出し元の無い古い保存はそのまま読める
    expect('exported' in metaOf(saved(problem, classOf, 4))).toBe(false)
    expect(countedSaves([metaOf(s), metaOf(saved(problem, classOf, 4)), metaOf(s)])).toBe(1)
  })

  it('無料版は3件まで、Pro は無制限に保存できる', () => {
    expect(FREE_SAVE_LIMIT).toBe(3)
    expect(canSaveMore(false, 2)).toBe(true)
    expect(canSaveMore(false, 3)).toBe(false)
    expect(canSaveMore(true, 300)).toBe(true)
  })

  it('保存名の既定値は「年月 · 名簿の名前」。開き直して保存しても年月を重ねない', () => {
    const d = new Date(2026, 9, 2)
    expect(defaultSaveName('ja', d, '3年生.xlsx')).toBe('2026年10月 · 3年生')
    expect(defaultSaveName('ja', d, '2026年3月 · 3年生')).toBe('2026年10月 · 3年生')
    expect(defaultSaveName('ja', d, null)).toBe('2026年10月')
    // 名簿の名前にある年（年度など）は外さない
    expect(defaultSaveName('ja', d, '2026年度 · 6年1組.xlsx')).toBe('2026年10月 · 2026年度 · 6年1組')
    expect(defaultSaveName('en', d, 'March 2026 · Grade 3')).toBe('October 2026 · Grade 3')
    expect(defaultSaveName('en', d, 'Grade 3')).toBe('October 2026 · Grade 3')
  })

  it('前回の組を名前で照らし合わせ、名前が無ければ NO で照らす', () => {
    const s = saved(problem, classOf, 4)
    // 並び順を変え、1人は名前を変える（NO で見つかる）、1人は名簿に無い生徒にする
    const students = [...problem.students].reverse().map((st, i) => (i === 0 ? { ...st, name: '改名' } : i === 1 ? { ...st, name: '転入生', no: 999 } : st))
    const { values, matched } = previousClassValues({ ...problem, students }, placementOfSaved(s, ja.className))
    expect(matched).toBe(students.length - 1)
    expect(values[1]).toBe('')
    const last = problem.students.length - 1
    expect(values[0]).toBe(ja.className(classOf[last]))
    expect(values[2]).toBe(ja.className(classOf[last - 2]))
  })

  it('NO を振り直した名簿では、名前で見つからない生徒を NO で照らさない', () => {
    const s = saved(problem, classOf, 4)
    // 全員の NO をずらし、1人だけ名前を変える → 名簿が続いていないので、その子は空欄
    const students = problem.students.map((st, i) => ({ ...st, no: st.no + 100, name: i === 0 ? '別の子' : st.name }))
    const { values, matched } = previousClassValues({ ...problem, students }, placementOfSaved(s, ja.className))
    expect(values[0]).toBe('')
    expect(matched).toBe(students.length - 1)
  })

  it('名前で一致したのが1〜2人だけなら、名簿が続いているとはみなさない', () => {
    const s = saved(problem, classOf, 4)
    // NO はそのままだが、名前が一致するのは1人だけ → 残りの子を NO で照らさない
    const students = problem.students.map((st, i) => (i === 0 ? st : { ...st, name: `新入生${i}` }))
    const { matched } = previousClassValues({ ...problem, students }, placementOfSaved(s, ja.className))
    expect(matched).toBe(1)
  })

  it('今の名簿で同じ名前が2人いるときは、名前では照らさない', () => {
    const s = saved(problem, classOf, 4)
    const dup = problem.students[1].name
    // NO も振り直す（名簿が続いていない）ので、同じ名前の2人はどちらも空欄
    const students = problem.students.map((st, i) => ({ ...st, no: st.no + 100, name: i === 2 ? dup : st.name }))
    const { values } = previousClassValues({ ...problem, students }, placementOfSaved(s, ja.className))
    expect(values[1]).toBe('')
    expect(values[2]).toBe('')
    expect(values[3]).toBe(ja.className(classOf[3]))
  })

  it('「前回の組」をカテゴリの列として入れ、チェックを外すと無効にする', () => {
    const s = saved(problem, classOf, 4)
    const { values } = previousClassValues(problem, placementOfSaved(s, ja.className))
    const order = [0, 1, 2, 3].map(ja.className)
    const p = withPreviousClass(problem, '前回の組', values, order)
    const col = p.columns.find((c) => c.name === '前回の組')!
    expect(col).toEqual({ name: '前回の組', kind: 'category', levels: order, weight: PREVIOUS_CLASS_WEIGHT, enabled: true })
    expect(p.students[5].values['前回の組']).toBe(ja.className(1))
    // 入れ直しても列は1本のまま
    expect(withPreviousClass(p, '前回の組', values, order).columns.filter((c) => c.name === '前回の組')).toHaveLength(1)
    // 日本語で入れたあと英語表示に切り替えても、同じ列として見つかる
    expect(findPreviousClassColumn(p)?.name).toBe('前回の組')
    expect(findPreviousClassColumn(withPreviousClass(problem, 'Last class', values, order))?.name).toBe('Last class')
    expect(findPreviousClassColumn(problem)).toBeUndefined()
    // 重み 0 のまま有効にしても使われないので、有効にするときは既定の重みに戻す
    const zero = { ...p, columns: p.columns.map((c) => (c.name === '前回の組' ? { ...c, weight: 0, enabled: false } : c)) }
    expect(setColumnEnabled(zero, '前回の組', true).columns.find((c) => c.name === '前回の組')).toMatchObject({ enabled: true, weight: PREVIOUS_CLASS_WEIGHT })
    expect(setColumnEnabled(p, '前回の組', false).columns.find((c) => c.name === '前回の組')!.enabled).toBe(false)
  })

  it('前回の組を入れて編成すると、前回同じ組だった2人が同じ組になる組み合わせが減る', async () => {
    const first = await solve(problem, 1)
    const plain = await solve(problem, 2)
    const { values } = previousClassValues(problem, placementOfSaved(saved(problem, first, 4), ja.className))
    const mixed = withPreviousClass(problem, '前回の組', values, [0, 1, 2, 3].map(ja.className))
    const next = await solve(mixed, 2)
    expect(pairsAgain(first, next)).toBeLessThan(pairsAgain(first, plain))
    // 同じ組の指定（必ず守る）以外は、前回の組がどの組にもほぼ均等に散る
    const report = evaluate(mixed, next, 4)
    expect(report.violations).toEqual([])
    const prevCol = report.columns.find((c) => c.column === '前回の組')!
    expect(prevCol).toBeDefined()
  }, 20000)

  it('書き出した結果の Excel（どの言語でも）から前回の組を読み、保存した編成と同じように使える', () => {
    // 10組以上でも「1組・2組…10組・11組」の順に並べる
    const k = 11
    const cls = problem.students.map((_, i) => i % k)
    const xlsx = (lang: 'ja' | 'en') => writeXlsx(resultWorkbook(problem, cls, k, evaluate(problem, cls, k), lang), 'array')
    const fromSaved = previousClassValues(problem, placementOfSaved(saved(problem, cls, k), ja.className))

    const pja = parsePlacement(xlsx('ja'))!
    expect(pja.order).toEqual(Array.from({ length: k }, (_, c) => ja.className(c)))
    expect(previousClassValues(problem, pja)).toEqual(fromSaved)

    // 英語で書き出したファイルは組の名前が英語のまま（散らすのに名前の言語は関係ない）
    const pen = parsePlacement(xlsx('en'))!
    expect(pen.order[0]).toBe('Class 1')
    expect(pen.order[10]).toBe('Class 11')
    const en = previousClassValues(problem, pen)
    expect(en.matched).toBe(problem.students.length)
    expect(en.values[12]).toBe('Class 2')

    // 名簿だけのファイル（組分けのシートが無い）は読めない
    expect(parsePlacement(writeXlsx(rosterWorkbook(problem, 4), 'array'))).toBeNull()
    // Excel でないデータ・名前の列を消した組分けのシートも読めない（誰とも照らせないので）
    expect(parsePlacement(new TextEncoder().encode('not an excel file').buffer as ArrayBuffer)).toBeNull()
    const wb = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet([['NO', '組'], [1, '1組'], [2, '2組']]), '組分け')
    expect(parsePlacement(writeXlsx(wb, 'array'))).toBeNull()
  })
})
