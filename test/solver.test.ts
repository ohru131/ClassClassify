import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { parseWorkbook } from '../src/solver/parse'
import { compile } from '../src/solver/compile'
import { anneal, createAnnealer } from '../src/solver/anneal'
import { evaluate } from '../src/solver/evaluate'
import { exportRoster, exportWorkbook } from '../src/solver/export'

const load = (f: string) => {
  const b = readFileSync(new URL(`../public/${f}`, import.meta.url))
  return parseWorkbook(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength))
}

describe.each(['sample1.xlsx', 'sample2.xlsx', 'sample-group.xlsx'])('%s', (file) => {
  it('parses and solves without violations', () => {
    const p = load(file)
    expect(p.students.length).toBeGreaterThan(0)
    const { compiled } = compile(p)
    const res = anneal(compiled, { timeMs: 1500, seed: 42 })
    const report = evaluate(p, res.classOf, p.numClasses)
    console.log(file, p.students.length, p.numClasses, 'sizes', report.sizes, 'excess', report.totalExcess, 'cost', res.cost.toFixed(2), 'iters', res.iterations, p.warnings)
    expect(report.violations).toEqual([])
    expect(Math.max(...report.sizes) - Math.min(...report.sizes)).toBeLessThanOrEqual(1)
    expect(exportWorkbook(p, res.classOf, p.numClasses, report).size).toBeGreaterThan(0)
  })
})

describe('Google ひな形', () => {
  it('ひな形の内容が sample1.xlsx と同じ問題として読み込める', async () => {
    const XLSX = await import('xlsx-js-style')
    const { buildTemplateSheets } = await import('../src/google/template')
    const b = readFileSync(new URL('../public/sample1.xlsx', import.meta.url))
    const buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)
    const wb = XLSX.utils.book_new()
    for (const s of buildTemplateSheets(buf)) XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(s.rows), s.name)
    const out = XLSX.write(wb, { bookType: 'xlsx', type: 'array' }) as ArrayBuffer
    const a = parseWorkbook(buf)
    const t = parseWorkbook(out)
    expect(t.students).toEqual(a.students)
    expect(t.columns).toEqual(a.columns)
    expect(t.numClasses).toBe(a.numClasses)
    expect(t.wantedGroups).toEqual(a.wantedGroups)
    expect(t.unwantedGroups).toEqual(a.unwantedGroups)
  })
})

describe('名簿編集', async () => {
  const r = await import('../src/solver/roster')
  it('生徒削除でペアの index を詰め直す', () => {
    const p = load('sample1.xlsx')
    const target = p.wantedGroups[0][0]
    const q = r.removeStudents(p, [target])
    expect(q.students.length).toBe(p.students.length - 1)
    for (const g of [...q.wantedGroups, ...q.unwantedGroups]) for (const i of g) expect(i).toBeLessThan(q.students.length)
    const names = (pp: typeof p, gs: number[][]) => gs.map((g) => g.map((i) => pp.students[i].no))
    const removedNo = p.students[target].no
    expect(names(q, q.wantedGroups)).toEqual(
      names(p, p.wantedGroups).map((g) => g.filter((n) => n !== removedNo)).filter((g) => g.length >= 2),
    )
  })
  it('矛盾する指定を検出する', () => {
    let p = load('sample2.xlsx')
    p = { ...p, wantedGroups: [], unwantedGroups: [] }
    p = r.addGroup(p, 'wanted', [0, 1])
    p = r.addGroup(p, 'wanted', [1, 2])
    p = r.addGroup(p, 'unwanted', [0, 2])
    expect(r.findConflicts(p)).toEqual([[0, 2]])
  })
  it('値の編集で項目の種類を再判定し、名簿 Excel を再読み込みできる', () => {
    let p = load('sample1.xlsx')
    p = r.addColumn(p, 'リーダー')
    p = r.setValueFor(p, [0, 5, 9], 'リーダー', '○')
    const col = p.columns.find((c) => c.name === 'リーダー')!
    expect(col).toMatchObject({ kind: 'flag', levels: ['○'], enabled: true })
    p = r.updateStudent(p, 3, { name: '新しい名前' })
    const buf = exportRoster(p, 4)
    return buf.arrayBuffer().then((ab) => {
      const q = parseWorkbook(ab)
      expect(q.students).toEqual(p.students)
      expect(q.columns).toEqual(p.columns)
      expect(q.wantedGroups).toEqual(p.wantedGroups)
      expect(q.unwantedGroups).toEqual(p.unwantedGroups)
    })
  })
})

describe('レビュー指摘の回帰テスト', async () => {
  const r = await import('../src/solver/roster')
  it('数値列は編集で水準が減っても数値のまま、カテゴリ列もカテゴリのまま', () => {
    let p = load('sample1.xlsx')
    p = r.addColumn(p, '点数', 'numeric')
    p = r.addColumn(p, '係', 'category')
    expect(p.columns.find((c) => c.name === '点数')!.kind).toBe('numeric')
    p = r.setValueFor(p, [0, 1, 2, 3, 4, 5, 6], '点数', '50')
    p = r.setValueFor(p, [0], '点数', '90')
    p = r.setValueFor(p, [0], '点数', '')
    expect(p.columns.find((c) => c.name === '点数')!.kind).toBe('numeric')
    p = r.setValueFor(p, [0, 1], '係', '図書')
    expect(p.columns.find((c) => c.name === '係')).toMatchObject({ kind: 'category', levels: ['図書'], enabled: true })
  })
  it('toCell は正規の数値文字列だけ数値化する', () => {
    expect(r.toCell('3')).toBe(3)
    expect(r.toCell('2.5')).toBe(2.5)
    expect(r.toCell('007')).toBe('007')
    expect(r.toCell('0x10')).toBe('0x10')
    expect(r.toCell('○')).toBe('○')
  })
  it('NO の重複を検出する', () => {
    const p = load('sample1.xlsx')
    expect(r.isNoTaken(p, p.students[1].no, 0)).toBe(true)
    expect(r.isNoTaken(p, p.students[0].no, 0)).toBe(false)
  })
})

describe('無効にした項目', async () => {
  it('名簿 Excel の往復で無効のまま', async () => {
    let p = load('sample1.xlsx')
    p = { ...p, columns: p.columns.map((c, i) => (i === 0 ? { ...c, enabled: false, weight: 0 } : c)) }
    const q = parseWorkbook(await exportRoster(p, 4).arrayBuffer())
    expect(q.columns[0].enabled).toBe(false)
  })
})

describe('設定シート', () => {
  it('クラス数が計算結果なしの数式でも、生徒人数と最大人数から求める', async () => {
    const XLSX = await import('xlsx-js-style')
    const wb = XLSX.utils.book_new()
    const settings = XLSX.utils.aoa_to_sheet([['生徒人数', 10], ['1クラスの最大人数', 4], ['クラス数', null]])
    settings['B3'] = { t: 'n', f: 'CEILING(B1/B2,1)' } as import('xlsx-js-style').CellObject
    XLSX.utils.book_append_sheet(wb, settings, '設定')
    XLSX.utils.book_append_sheet(
      wb,
      XLSX.utils.aoa_to_sheet([['', '重み', 1], ['NO', '名前', 'A'], ...Array.from({ length: 10 }, (_, i) => [i + 1, `s${i}`, i % 2 ? '○' : ''])]),
      '生徒名簿',
    )
    const p = parseWorkbook(XLSX.write(wb, { bookType: 'xlsx', type: 'array' }) as ArrayBuffer)
    expect(p.numClasses).toBe(3)
    expect(p.warnings).toEqual([])
  })
})

describe('ペア指定の出力', () => {
  it('組分けシートにペア指定列と色が入り、ペア指定シートに判定が出る', async () => {
    const XLSX = await import('xlsx-js-style')
    const { pairStatus } = await import('../src/solver/pairs')
    const p = load('sample1.xlsx')
    const { compiled } = compile(p)
    const res = anneal(compiled, { timeMs: 500, seed: 1 })
    // 同1 を意図的に崩す
    const classOf = [...res.classOf]
    const [a, b] = p.wantedGroups[0]
    classOf[b] = (classOf[a] + 1) % p.numClasses
    const report = evaluate(p, classOf, p.numClasses)
    const wb = XLSX.read(await exportWorkbook(p, classOf, p.numClasses, report).arrayBuffer(), { cellStyles: true })
    expect(wb.SheetNames).toContain('ペア指定')
    const ws = wb.Sheets['組分け']
    expect(ws['D1'].v).toBe('ペア指定')
    const row = a + 2
    expect(String(ws[`D${row}`].v)).toContain('同1(×)')
    expect(ws[`A${row}`].s?.fgColor?.rgb ?? ws[`A${row}`].s?.fill?.fgColor?.rgb).toMatch(/E0F2FE$/)
    const pairs = XLSX.utils.sheet_to_json<string[]>(wb.Sheets['ペア指定'], { header: 1 })
    expect(pairs[1][0]).toBe('同1')
    expect(pairs[1][4]).toBe('×')
    const st = pairStatus(p, classOf)
    expect(st.groups.find((g) => g.label === '同1')!.ok).toBe(false)
  })
})

describe('createAnnealer', () => {
  it('run(0) や run(NaN) でも探索が進み、いずれ終わる', () => {
    const { compiled } = compile(load('sample1.xlsx'))
    const a = createAnnealer(compiled, { timeMs: 30, seed: 1 })
    let done = false
    const t0 = Date.now()
    for (let i = 0; i < 1e6 && !done && Date.now() - t0 < 5000; i++) done = a.run(i % 2 ? 0 : NaN)
    expect(done).toBe(true)
    expect(a.result().classOf.length).toBe(compiled.n)
  })
})
