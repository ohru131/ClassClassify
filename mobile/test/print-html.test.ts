import { describe, expect, it } from 'vitest'
import { makeI18n } from '../lib/i18n-core'
import { buildResultPrintHtml, escapeHtml } from '../lib/print-html'
import { isWebProPreview } from '../lib/pro-preview'
import { loadSample } from '../lib/samples'
import { compile, evaluate, roster } from '../lib/solver'
import { runSliced } from '../lib/runner'

const solve = async (id: string) => {
  const { problem } = loadSample('ja', id)
  const res = await runSliced(compile(problem).compiled, { timeMs: 400, yieldToUi: async () => {} }).promise
  return { problem, classOf: res.classOf, k: problem.numClasses, report: evaluate(problem, res.classOf, problem.numClasses) }
}

describe('印刷用 HTML', () => {
  it('利用者の入力（名前・項目名・値）を HTML エスケープする', async () => {
    let { problem } = loadSample('ja', 'sample-group')
    problem = roster.updateStudent(problem, 0, { name: '<script>alert("x")</script>&太郎' })
    problem = roster.addColumn(problem, '<b>項目</b>')
    problem = roster.setValueFor(problem, [0], '<b>項目</b>', '○')
    problem = roster.addGroup(problem, 'wanted', [0, 1])
    const classOf = problem.students.map((_, i) => i % problem.numClasses)
    const html = buildResultPrintHtml({ problem, classOf, k: problem.numClasses, report: evaluate(problem, classOf, problem.numClasses), createdAt: new Date(2026, 8, 30, 9, 5), i18n: makeI18n('ja'), title: 'A&B <組>' })
    expect(html).not.toContain('<script>alert')
    expect(html).not.toContain('<b>項目</b>')
    expect(html).toContain('&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;&amp;太郎')
    expect(html).toContain('&lt;b&gt;項目&lt;/b&gt;')
    expect(html).toContain('<title>A&amp;B &lt;組&gt;</title>')
    expect(html).toMatch(/作成日: 2026年9月30日 0?9:05/)
  })

  it('A4 縦・クラスごとの名簿・集計・ペア指定の凡例・改ページを含む', async () => {
    const { problem, classOf, k, report } = await solve('sample1')
    const html = buildResultPrintHtml({ problem, classOf, k, report, createdAt: new Date(), i18n: makeI18n('ja') })
    expect(html).toContain('size: A4 portrait')
    for (let c = 1; c <= k; c++) expect(html).toContain(`>${c}組<span>`)
    for (const s of problem.students) expect(html).toContain(`<td class="name">${escapeHtml(s.name)}</td>`)
    expect(html).toContain('ペア指定（')
    expect(html).toContain('同1')
    expect(html).toContain('別1')
    expect(html).toContain('集計・バランス')
    expect(html).toContain('break-inside: avoid')
    expect(html).toContain('break-before: page')
    for (const col of report.columns) expect(html).toContain(escapeHtml(col.column))
  })

  it('ペア指定が無ければ凡例を出さない', async () => {
    const { problem } = loadSample('ja', 'sample2')
    const p = { ...problem, wantedGroups: [], unwantedGroups: [] }
    const classOf = p.students.map((_, i) => i % 4)
    expect(buildResultPrintHtml({ problem: p, classOf, k: 4, report: evaluate(p, classOf, 4), createdAt: new Date(), i18n: makeI18n('en') })).not.toContain('class="legend"')
  })

  it('escapeHtml', () => {
    expect(escapeHtml(`<a href="x">'&'</a>`)).toBe('&lt;a href=&quot;x&quot;&gt;&#39;&amp;&#39;&lt;/a&gt;')
  })

  it.each(['en', 'ko', 'es', 'de', 'pt-BR'] as const)('%s: 見出し・組の名前・日付をその言語で出し、日本語が混ざらない', async (lang) => {
    const { problem } = loadSample(lang, 'sample1')
    const res = await runSliced(compile(problem).compiled, { timeMs: 300, yieldToUi: async () => {} }).promise
    const report = evaluate(problem, res.classOf, problem.numClasses)
    const i18n = makeI18n(lang)
    let p2 = roster.updateStudent(problem, 0, { name: '<img src=x onerror=alert(1)>' })
    const html = buildResultPrintHtml({ problem: p2, classOf: res.classOf, k: problem.numClasses, report, createdAt: new Date(2026, 8, 30, 9, 5), i18n })
    expect(html).toContain(`<html lang="${i18n.lang === 'es' ? 'es-419' : i18n.lang === 'pt-BR' ? 'pt-BR' : i18n.lang === 'en' ? 'en-US' : i18n.lang === 'ko' ? 'ko-KR' : 'de-DE'}">`)
    expect(html).toContain(escapeHtml(i18n.className(0)))
    expect(html).toContain(escapeHtml(i18n.t('printClassLists')))
    expect(html).toContain('2026')
    expect(html).not.toContain('<img src=x')
    expect(html).not.toMatch(/[ぁ-んァ-ヶ一-龥]/)
    p2 = problem
  })
})

describe('Web 限定の Pro プレビュー', () => {
  it('Web で ?pro=preview のときだけ有効、ネイティブでは常に無効', () => {
    expect(isWebProPreview('web', '?pro=preview')).toBe(true)
    expect(isWebProPreview('web', '')).toBe(false)
    expect(isWebProPreview('android', '?pro=preview')).toBe(false)
    expect(isWebProPreview('ios', '?pro=preview')).toBe(false)
  })
})
