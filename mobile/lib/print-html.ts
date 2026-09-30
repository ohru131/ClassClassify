import type { I18n } from './i18n-core'
import { LANGUAGE_META } from './i18n'
import { pairStatus, rowColor, violationText, type PairTag, type Problem, type Report } from './solver'

// 印刷・PDF 用の結果の HTML（純関数）。ネイティブは expo-print、Web は新しいウィンドウで印刷する。
// 名前・項目名・値はすべて利用者の入力なので、必ず esc() を通してから埋め込むこと。

export const escapeHtml = (s: string | number) =>
  String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!)
const esc = escapeHtml

const CLASS_DOT = ['#6366F1', '#D946EF', '#10B981', '#F59E0B', '#0EA5E9', '#F43F5E', '#14B8A6', '#8B5CF6']
const dot = (c: number) => CLASS_DOT[c % CLASS_DOT.length]

const badge = (t: PairTag) =>
  `<span class="badge${t.ok ? '' : ' bad'}" style="color:${t.color.fg};background:${t.kind === 'wanted' ? '#fff' : t.color.bg}">${esc(t.label)}${t.ok ? '' : '×'}</span>`

export interface PrintInput {
  problem: Problem
  classOf: number[]
  k: number
  report: Report
  createdAt: Date
  /** 見出し・組の名前・日付と数値の書式（選択中の言語） */
  i18n: I18n
  title?: string
}

export function buildResultPrintHtml({ problem: p, classOf, k, report, createdAt, i18n, title = i18n.t('printTitle') }: PrintInput): string {
  const { t, num, className, file } = i18n
  const { tags, groups } = pairStatus(p, classOf, file.tagPrefix)
  const flagCols = p.columns.filter((c) => c.enabled && c.kind === 'flag')
  const sizeGap = Math.max(...report.sizes) - Math.min(...report.sizes)
  const perfect = report.totalExcess === 0

  const summary = `
    <div class="stats">
      <div><b>${k}</b><span>${esc(t('printClasses', { n: p.students.length }))}</span></div>
      <div><b>${sizeGap}</b><span>${esc(t('printGap', { min: Math.min(...report.sizes), max: Math.max(...report.sizes) }))}</span></div>
      <div><b>${perfect ? esc(t('perfect')) : esc(num(report.totalExcess, report.totalExcess % 1 ? 1 : 0))}</b><span>${esc(perfect ? t('balancePerfectSub') : t('balanceSub'))}</span></div>
      <div><b class="${report.violations.length ? 'ng' : 'ok'}">${report.violations.length}</b><span>${esc(t('printViolations'))}</span></div>
    </div>`

  const legend = groups.length
    ? `<section class="legend"><h2>${esc(t('printPairings', { ok: groups.filter((g) => g.ok).length, n: groups.length }))}</h2><ul>${groups
        .map((g) => {
          const classes = [...new Set(g.members.map((i) => classOf[i] + 1))].sort((a, b) => a - b)
          return `<li class="${g.ok ? '' : 'bad'}" style="background:${g.color.bg};color:${g.color.fg}"><b>${esc(g.label)}</b> ${esc(g.kind === 'wanted' ? file.wantedKind : file.unwantedKind)}: <span class="members">${g.members
            .map((i) => esc(p.students[i].name || `${file.no} ${p.students[i].no}`))
            .join(esc(file.joinSep))}</span> → ${classes.map((c) => esc(className(c - 1))).join('/')} ${g.ok ? '✓' : `✗ ${esc(t('printNotMet'))}`}</li>`
        })
        .join('')}</ul><p class="note">${esc(t('printLegendNote', { w: file.tagPrefix.wanted, u: file.tagPrefix.unwanted }))}</p></section>`
    : ''

  const classCards = Array.from({ length: k }, (_, c) => {
    const members = p.students.map((_, i) => i).filter((i) => classOf[i] === c)
    const rows = members
      .map((i) => {
        const s = p.students[i]
        const bg = rowColor(tags[i])
        const marks = flagCols.filter((col) => (s.values[col.name] ?? '') !== '').map((col) => esc(col.name))
        return `<tr${bg ? ` style="background:${bg.bg}"` : ''}><td class="no">${esc(s.no)}</td><td class="name">${esc(s.name)}</td><td>${tags[i].map(badge).join(' ')}</td><td class="marks">${marks.join(esc(file.joinSep))}</td></tr>`
      })
      .join('')
    return `<div class="cls"><h3><i style="background:${dot(c)}"></i>${esc(className(c))}<span>${esc(t('studentsN', { n: members.length }))}</span></h3><table><thead><tr><th>${esc(file.no)}</th><th>${esc(file.name)}</th><th>${esc(t('printColPair'))}</th><th>${esc(t('printColMarks'))}</th></tr></thead><tbody>${rows}</tbody></table></div>`
  }).join('')

  const head = (label: string) => `<tr><th class="lv">${esc(label)}</th>${Array.from({ length: k }, (_, c) => `<th>${esc(className(c))}</th>`).join('')}<th>${esc(t('target'))}</th></tr>`
  const balance = report.columns
    .map((col) => {
      const numeric = col.kind === 'numeric'
      const body = col.levels
        .map((level, l) => {
          const ideal = col.ideal[l]
          const cells = col.rows[l]
            .map((v) => {
              const ok = numeric || (v >= Math.floor(ideal) && v <= Math.ceil(ideal))
              return `<td class="${ok ? '' : 'out'}">${numeric ? num(v, 2) : v}</td>`
            })
            .join('')
          return `<tr><th class="lv">${esc(numeric ? t('average') : level)}</th>${cells}<td class="ideal">${num(ideal, numeric ? 2 : 1)}</td></tr>`
        })
        .join('')
      const state = numeric ? t('average') : col.excess === 0 ? t('even') : t('offBy', { n: num(col.excess, col.excess % 1 ? 1 : 0) })
      return `<div class="bal"><h3>${esc(col.column)} <span>${esc(t('weightN', { w: num(col.weight, col.weight % 1 ? 1 : 0) }))} · ${esc(state)}</span></h3><table><thead>${head('')}</thead><tbody>${body}</tbody></table></div>`
    })
    .join('')
  const sizes = `<div class="bal"><h3>${esc(file.count)}</h3><table><thead>${head('')}</thead><tbody><tr><th class="lv">${esc(file.count)}</th>${report.sizes.map((s) => `<td>${s}</td>`).join('')}<td class="ideal">${num(p.students.length / k, 1)}</td></tr></tbody></table></div>`
  const violations = report.violations.length
    ? `<section><h2>${esc(t('printUnmet'))}</h2><ul class="viol">${report.violations.map((v) => `<li>${esc(violationText(i18n.fileLang, v, p, classOf))}</li>`).join('')}</ul></section>`
    : ''

  return `<!DOCTYPE html>
<html lang="${esc(LANGUAGE_META[i18n.lang].intl)}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title)}</title>
<style>
@page { size: A4 portrait; margin: 12mm; }
* { box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: -apple-system, "Segoe UI", Roboto, "Hiragino Sans", "Noto Sans JP", "Noto Sans KR", "Apple SD Gothic Neo", "Noto Sans CJK JP", "Yu Gothic", sans-serif; color: #0f172a; font-size: 10.5pt; margin: 0; }
header { display: flex; justify-content: space-between; align-items: baseline; border-bottom: 2px solid #4f46e5; padding-bottom: 4px; margin-bottom: 8px; }
h1 { font-size: 16pt; margin: 0; } header .date { color: #475569; font-size: 9pt; }
h2 { font-size: 11pt; margin: 10px 0 4px; } h3 { font-size: 11pt; margin: 0 0 4px; display: flex; align-items: center; gap: 6px; }
h3 span { margin-left: auto; font-weight: 400; color: #475569; font-size: 9pt; } h3 i { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-bottom: 6px; }
.stats div { border: 1px solid #e2e8f0; border-radius: 6px; padding: 4px 8px; } .stats b { font-size: 14pt; display: block; } .stats span { font-size: 8pt; color: #475569; }
.ok { color: #059669; } .ng { color: #e11d48; }
.legend ul { list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 4px; }
.legend li { border-radius: 5px; padding: 2px 6px; font-size: 9pt; border: 1px solid rgba(0,0,0,.08); } .legend li.bad { border: 2px solid #e11d48; }
.legend .members { color: #334155; } .note { font-size: 8pt; color: #64748b; margin: 4px 0 0; }
.classes { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
.cls { border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px; break-inside: avoid; page-break-inside: avoid; }
table { width: 100%; border-collapse: collapse; } th, td { border-bottom: 1px solid #e2e8f0; padding: 2px 4px; text-align: left; font-size: 9pt; }
thead th { background: #f1f5f9; white-space: nowrap; } td.no { width: 2.6em; color: #64748b; } td.name { font-weight: 600; } td.marks { font-size: 8pt; color: #475569; }
.badge { border: 1px solid currentColor; border-radius: 3px; padding: 0 3px; font-size: 8pt; font-weight: 700; } .badge.bad { outline: 2px solid #e11d48; }
.balance { break-before: page; page-break-before: always; }
.bal { break-inside: avoid; page-break-inside: avoid; margin-bottom: 8px; } .bal td, .bal thead th { text-align: center; } .bal th.lv { text-align: left; width: 7em; }
.bal td.out { background: #fef3c7; font-weight: 700; } .bal td.ideal { color: #64748b; }
.viol li { color: #be123c; }
@media screen { body { max-width: 210mm; margin: 0 auto; padding: 12mm; } }
</style></head><body>
<header><h1>${esc(title)}</h1><span class="date">${esc(t('printCreated', { date: i18n.date(createdAt) }))}</span></header>
${summary}
${legend}
<section><h2>${esc(t('printClassLists'))}</h2><div class="classes">${classCards}</div></section>
<section class="balance"><h2>${esc(t('printSummary'))}</h2>${sizes}${balance}${violations}</section>
</body></html>`
}
