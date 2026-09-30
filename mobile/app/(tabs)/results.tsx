import { useRouter } from 'expo-router'
import { useMemo, useState } from 'react'
import { Pressable, ScrollView, Switch, Text, View } from 'react-native'

import { C, classColor } from '@/components/theme'
import { Btn, Card, Chip, Notice, Screen, Segmented, Stat, styles } from '@/components/ui'
import { useLayout } from '@/lib/layout'
import { useProject } from '@/lib/project-store'
import { pairStatus, resultWorkbook, rowColor, type ColumnReport, type PairTag, type Problem, type Report } from '@/lib/solver'
import { useProExport } from '@/lib/use-pro-export'

type Tab = 'classes' | 'balance' | 'checks'
type Focus = { kind: 'wanted' | 'unwanted'; group: number } | null

export default function ResultsScreen() {
  const { problem, solution, report, edited, moveStudent, resetMoves, error, setError } = useProject()
  const { exportXlsx, busy, isPro } = useProExport()
  const router = useRouter()
  const { isWide } = useLayout()
  const [tab, setTab] = useState<Tab>('classes')
  const [selected, setSelected] = useState<number | null>(null)

  if (!problem || !solution || !report)
    return (
      <Screen>
        <Text style={{ fontSize: 24, fontWeight: '900', color: C.text }}>結果</Text>
        <Card style={{ gap: 10 }}>
          <Text style={{ color: C.sub }}>まだ編成していません。「設定・実行」タブで実行すると、ここに結果が表示されます。</Text>
          <Btn variant="primary" icon="play" label="設定・実行へ" onPress={() => router.navigate(problem ? '/run' : '/')} />
        </Card>
      </Screen>
    )

  const k = solution.k
  const perfect = report.totalExcess === 0
  const sizeGap = Math.max(...report.sizes) - Math.min(...report.sizes)
  const sel = selected !== null && selected < problem.students.length ? selected : null

  return (
    <View style={{ flex: 1 }}>
      <Screen>
        <Text style={{ fontSize: 24, fontWeight: '900', color: C.text }}>結果</Text>
        {error ? (
          <Notice tone="error" onClose={() => setError(null)}>
            {error}
          </Notice>
        ) : null}
        <View style={styles.wrap}>
          <Stat label="クラス" value={k} sub={`${problem.students.length} 名を編成`} />
          <Stat label="人数差" value={sizeGap} sub={`${Math.min(...report.sizes)}〜${Math.max(...report.sizes)} 名`} tone={sizeGap <= 1 ? 'good' : 'bad'} />
          <Stat label="バランス" value={perfect ? '完全' : report.totalExcess} sub={perfect ? '全項目が理想の範囲内' : '理想範囲からのずれ（人）'} tone={perfect ? 'good' : 'default'} />
          <Stat
            label="条件違反"
            value={report.violations.length}
            sub={report.violations.length ? 'ペア条件を満たせていません' : 'ペア条件をすべて満たしています'}
            tone={report.violations.length ? 'bad' : 'good'}
          />
        </View>
        <View style={[styles.row, { flexWrap: 'wrap' }]}>
          {edited ? <Btn small icon="arrow-undo-outline" label="手動変更を戻す" onPress={resetMoves} /> : null}
          <Btn
            small
            variant="primary"
            icon={isPro ? 'share-outline' : 'lock-closed-outline'}
            label="結果を Excel で共有"
            busy={busy}
            onPress={() => exportXlsx(() => resultWorkbook(problem, solution.classOf, k, report), 'クラス編成結果')}
          />
          {!isPro ? <Text style={{ fontSize: 12, color: C.muted }}>Excel での保存・共有は Pro の機能です</Text> : null}
        </View>
        <Segmented
          value={tab}
          onChange={setTab}
          options={[
            { value: 'classes', label: 'クラス一覧' },
            { value: 'balance', label: 'バランス分析' },
            { value: 'checks', label: `条件チェック${report.violations.length ? ` (${report.violations.length})` : ''}` },
          ]}
        />
        {tab === 'classes' && <ClassBoard problem={problem} classOf={solution.classOf} k={k} report={report} selected={sel} setSelected={setSelected} />}
        {tab === 'balance' && (
          <View style={{ flexDirection: isWide ? 'row' : 'column', flexWrap: 'wrap', gap: 12 }}>
            {report.columns.map((c) => (
              <BalanceCard key={c.column} col={c} k={k} wide={isWide} />
            ))}
            {report.columns.length === 0 ? <Text style={{ color: C.muted }}>均等にする項目がありません。</Text> : null}
          </View>
        )}
        {tab === 'checks' && <Checks problem={problem} report={report} />}
        {/* 移動パネルの下に隠れないための余白 */}
        {sel !== null ? <View style={{ height: 120 }} /> : null}
      </Screen>

      {sel !== null ? (
        <View style={moveBar} accessibilityLiveRegion="polite">
          <View style={[styles.row, { justifyContent: 'space-between' }]}>
            <Text style={{ fontWeight: '800', color: C.text, flexShrink: 1 }} numberOfLines={1}>
              {problem.students[sel].no}:{problem.students[sel].name} を移動 →
            </Text>
            <Btn small icon="close" accessibilityLabel="閉じる" onPress={() => setSelected(null)} />
          </View>
          <View style={[styles.wrap, { marginTop: 8 }]}>
            {Array.from({ length: k }, (_, c) => {
              const color = classColor(c)
              const here = solution.classOf[sel] === c
              return (
                <Btn
                  key={c}
                  small
                  label={`${c + 1}組`}
                  disabled={here}
                  accessibilityLabel={`${c + 1}組へ移動`}
                  style={{ backgroundColor: color.soft, borderColor: color.dot }}
                  onPress={() => {
                    moveStudent(sel, c)
                    setSelected(null)
                  }}
                />
              )
            })}
          </View>
        </View>
      ) : null}
    </View>
  )
}

const moveBar = {
  position: 'absolute' as const,
  left: 12,
  right: 12,
  bottom: 12,
  maxWidth: 720,
  alignSelf: 'center' as const,
  marginHorizontal: 'auto' as const,
  backgroundColor: '#fff',
  borderRadius: 18,
  borderWidth: 1,
  borderColor: C.border,
  padding: 14,
  shadowColor: '#000',
  shadowOpacity: 0.15,
  shadowRadius: 16,
  shadowOffset: { width: 0, height: 6 },
  elevation: 8,
}

function ClassBoard({
  problem,
  classOf,
  k,
  report,
  selected,
  setSelected,
}: {
  problem: Problem
  classOf: number[]
  k: number
  report: Report
  selected: number | null
  setSelected: (s: number | null) => void
}) {
  const { isWide, classColumns } = useLayout()
  const [current, setCurrent] = useState(0)
  const [colorize, setColorize] = useState(true)
  const [focus, setFocus] = useState<Focus>(null)
  const pairs = useMemo(() => pairStatus(problem, classOf), [problem, classOf])
  const flagged = useMemo(() => new Set(report.violations.flatMap((v) => v.students)), [report])
  const flagTags = useMemo(() => {
    const flagCols = problem.columns.filter((c) => c.enabled && c.kind === 'flag')
    return problem.students.map((s) => flagCols.filter((c) => s.values[c.name] !== '').map((c) => c.name))
  }, [problem])
  const focusMembers = useMemo(() => {
    if (!focus) return null
    const g = (focus.kind === 'wanted' ? problem.wantedGroups : problem.unwantedGroups)[focus.group]
    return g ? new Set(g) : null
  }, [focus, problem])
  const cur = Math.min(current, k - 1)

  const card = (c: number) => {
    const members = problem.students.map((_, i) => i).filter((i) => classOf[i] === c)
    const color = classColor(c)
    return (
      <View key={c} style={{ backgroundColor: '#fff', borderRadius: 18, borderWidth: 1, borderColor: C.border, overflow: 'hidden', flex: isWide ? 1 : undefined, minWidth: 0 }}>
        <View style={{ height: 5, backgroundColor: color.dot }} />
        <View style={[styles.row, { justifyContent: 'space-between', padding: 14, paddingBottom: 6 }]}>
          <View style={styles.row}>
            <View style={{ width: 10, height: 10, borderRadius: 5, backgroundColor: color.dot }} />
            <Text style={{ fontSize: 18, fontWeight: '900', color: C.text }}>{c + 1}組</Text>
          </View>
          <Text style={{ fontSize: 12, fontWeight: '800', color: C.sub, backgroundColor: '#F1F5F9', paddingHorizontal: 8, paddingVertical: 2, borderRadius: 999 }}>{members.length} 名</Text>
        </View>
        <View style={{ paddingHorizontal: 8, paddingBottom: 10 }}>
          {members.map((i) => {
            const s = problem.students[i]
            const pt = pairs.tags[i]
            const bg = colorize ? rowColor(pt) : null
            const dim = focusMembers && !focusMembers.has(i)
            const isSel = selected === i
            return (
              <Pressable
                key={i}
                accessibilityRole="button"
                accessibilityLabel={`${s.no} ${s.name}、${c + 1}組。タップして別の組へ移動`}
                accessibilityState={{ selected: isSel }}
                onPress={() => setSelected(isSel ? null : i)}
                style={(st: { pressed: boolean; hovered?: boolean; focused?: boolean }) => [
                  { flexDirection: 'row', alignItems: 'center', gap: 6, paddingVertical: 7, paddingHorizontal: 8, borderRadius: 10, marginVertical: 1, minHeight: 36 },
                  bg && !isSel ? { backgroundColor: bg.bg } : null,
                  (st.hovered || st.pressed || st.focused) && !isSel && !bg ? { backgroundColor: C.hover } : null,
                  isSel ? { backgroundColor: C.primarySoft, borderWidth: 1, borderColor: '#A5B4FC' } : null,
                  focusMembers?.has(i) && !isSel ? { borderWidth: 2, borderColor: '#334155' } : null,
                  dim ? { opacity: 0.25 } : null,
                ]}
              >
                <Text style={{ width: 28, fontSize: 12, color: C.muted, fontVariant: ['tabular-nums'] }}>{s.no}</Text>
                <Text style={{ flexShrink: 1, fontSize: 14, fontWeight: '600', color: flagged.has(i) ? C.danger : C.text }} numberOfLines={1}>
                  {s.name || '（名前なし）'}
                </Text>
                <View style={{ flex: 1, flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'flex-end', gap: 3 }}>
                  {pt.map((t) => (
                    <PairBadge key={`${t.kind}${t.group}`} tag={t} />
                  ))}
                  {flagTags[i].slice(0, pt.length ? 2 : 3).map((t) => (
                    <Text key={t} style={{ fontSize: 10, color: C.sub, backgroundColor: '#F1F5F9', paddingHorizontal: 4, borderRadius: 4, overflow: 'hidden' }}>
                      {t}
                    </Text>
                  ))}
                </View>
              </Pressable>
            )
          })}
        </View>
      </View>
    )
  }

  const rows: number[][] = []
  for (let c = 0; c < k; c += classColumns) rows.push(Array.from({ length: Math.min(classColumns, k - c) }, (_, j) => c + j))

  return (
    <View style={{ gap: 12 }}>
      <Text style={{ fontSize: 12, color: C.sub }}>
        {isWide ? '生徒をクリック（タップ）すると別の組へ移動できます。' : '上の組を選んで切り替え。生徒をタップすると別の組へ移動できます。'}集計は即座に再計算されます。
      </Text>
      {pairs.groups.length > 0 ? (
        <Card style={{ gap: 8, padding: 12 }}>
          <View style={[styles.row, { flexWrap: 'wrap', justifyContent: 'space-between' }]}>
            <Text style={{ fontWeight: '800', color: C.text }}>
              ペア指定{' '}
              <Text style={{ color: pairs.groups.every((g) => g.ok) ? C.good : C.danger, fontSize: 12 }}>
                {pairs.groups.filter((g) => g.ok).length} / {pairs.groups.length} 件を満たしています
              </Text>
            </Text>
            <View style={styles.row}>
              <Text style={{ fontSize: 12, color: C.sub }}>同じ組を色分け</Text>
              <Switch value={colorize} onValueChange={setColorize} accessibilityLabel="同じ組を色分け" />
            </View>
          </View>
          <View style={styles.wrap}>
            {pairs.groups.map((g) => {
              const active = focus?.kind === g.kind && focus.group === g.group
              const classes = [...new Set(g.members.map((i) => classOf[i] + 1))].sort((a, b) => a - b)
              return (
                <Pressable
                  key={`${g.kind}${g.group}`}
                  accessibilityRole="button"
                  accessibilityState={{ selected: active }}
                  onPress={() => setFocus(active ? null : { kind: g.kind, group: g.group })}
                  style={{
                    flexDirection: 'row',
                    gap: 6,
                    alignItems: 'center',
                    backgroundColor: g.color.bg,
                    borderRadius: 10,
                    paddingHorizontal: 8,
                    paddingVertical: 6,
                    borderWidth: active ? 2 : g.ok ? 1 : 2,
                    borderColor: active ? C.text : g.ok ? 'rgba(0,0,0,0.06)' : C.danger,
                    maxWidth: '100%',
                  }}
                >
                  <Text style={{ fontWeight: '900', color: g.color.fg, fontSize: 12 }}>{g.label}</Text>
                  <Text style={{ color: C.sub, fontSize: 12, flexShrink: 1 }} numberOfLines={1}>
                    {g.members.map((i) => problem.students[i].name).join('・')}
                  </Text>
                  <Text style={{ color: g.color.fg, fontSize: 12, fontWeight: '700' }}>
                    → {classes.map((c) => `${c}組`).join('/')} {g.ok ? '✓' : '✗'}
                  </Text>
                </Pressable>
              )
            })}
          </View>
        </Card>
      ) : null}
      {isWide ? (
        rows.map((r) => (
          <View key={r[0]} style={{ flexDirection: 'row', gap: 12, alignItems: 'flex-start' }}>
            {r.map(card)}
            {Array.from({ length: classColumns - r.length }, (_, j) => (
              <View key={`pad${j}`} style={{ flex: 1 }} />
            ))}
          </View>
        ))
      ) : (
        <>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ gap: 6 }}>
            {Array.from({ length: k }, (_, c) => (
              <Chip key={c} label={`${c + 1}組 ${report.sizes[c]}名`} selected={cur === c} onPress={() => setCurrent(c)} />
            ))}
          </ScrollView>
          {card(cur)}
        </>
      )}
    </View>
  )
}

function PairBadge({ tag }: { tag: PairTag }) {
  return (
    <Text
      accessibilityLabel={`${tag.kind === 'wanted' ? '同じ組' : '別の組'}指定 ${tag.label}${tag.ok ? '' : '、満たせていません'}`}
      style={{
        fontSize: 10,
        fontWeight: '900',
        color: tag.color.fg,
        backgroundColor: tag.kind === 'wanted' ? 'rgba(255,255,255,0.7)' : tag.color.bg,
        paddingHorizontal: 4,
        borderRadius: 4,
        borderWidth: tag.ok ? 1 : 2,
        borderColor: tag.ok ? `${tag.color.fg}55` : C.danger,
        overflow: 'hidden',
      }}
    >
      {tag.label}
      {tag.ok ? '' : '!'}
    </Text>
  )
}

function BalanceCard({ col, k, wide }: { col: ColumnReport; k: number; wide: boolean }) {
  const numeric = col.kind === 'numeric'
  const max = Math.max(1, ...col.rows.flat())
  return (
    <Card style={{ gap: 8, flexBasis: wide ? '48%' : undefined, flexGrow: 1 }}>
      <View style={[styles.row, { justifyContent: 'space-between' }]}>
        <Text style={{ fontWeight: '800', color: C.text, fontSize: 15 }}>{col.column}</Text>
        <View style={styles.row}>
          <Text style={{ fontSize: 12, color: C.muted }}>重み {col.weight}</Text>
          <Text
            style={{
              fontSize: 12,
              fontWeight: '800',
              paddingHorizontal: 8,
              paddingVertical: 2,
              borderRadius: 999,
              overflow: 'hidden',
              color: numeric ? C.sub : col.excess === 0 ? C.good : C.warn,
              backgroundColor: numeric ? '#F1F5F9' : col.excess === 0 ? C.goodSoft : C.warnSoft,
            }}
          >
            {numeric ? '平均値' : col.excess === 0 ? '均等' : `ずれ ${col.excess}`}
          </Text>
        </View>
      </View>
      <ScrollView horizontal>
        <View>
          <View style={{ flexDirection: 'row' }}>
            <Text style={{ width: 76 }} />
            {Array.from({ length: k }, (_, c) => (
              <Text key={c} style={{ width: 52, textAlign: 'center', fontSize: 12, fontWeight: '800', color: C.muted }}>
                {c + 1}組
              </Text>
            ))}
            <Text style={{ width: 52, textAlign: 'center', fontSize: 12, color: C.muted }}>理想</Text>
          </View>
          {col.levels.map((level, l) => (
            <View key={level} style={{ flexDirection: 'row', alignItems: 'center', marginTop: 4 }}>
              <Text style={{ width: 76, fontSize: 12, fontWeight: '700', color: C.sub }} numberOfLines={1}>
                {level}
              </Text>
              {col.rows[l].map((v, c) => {
                const ideal = col.ideal[l]
                const ok = numeric || (v >= Math.floor(ideal) && v <= Math.ceil(ideal))
                const alpha = numeric ? 0.15 : 0.12 + 0.5 * (v / max)
                return (
                  <View key={c} style={{ width: 52, paddingHorizontal: 2 }}>
                    <Text
                      style={{
                        textAlign: 'center',
                        paddingVertical: 6,
                        borderRadius: 8,
                        overflow: 'hidden',
                        fontWeight: '800',
                        fontVariant: ['tabular-nums'],
                        color: ok ? '#312E81' : '#78350F',
                        backgroundColor: ok ? `rgba(99,102,241,${alpha})` : `rgba(251,191,36,${alpha + 0.1})`,
                        borderWidth: ok ? 0 : 2,
                        borderColor: '#FBBF24',
                      }}
                    >
                      {numeric ? v.toFixed(2) : v}
                    </Text>
                  </View>
                )
              })}
              <Text style={{ width: 52, textAlign: 'center', fontSize: 12, color: C.muted }}>{col.ideal[l].toFixed(numeric ? 2 : 1)}</Text>
            </View>
          ))}
        </View>
      </ScrollView>
    </Card>
  )
}

function Checks({ problem, report }: { problem: Problem; report: Report }) {
  const total = problem.wantedGroups.length + problem.unwantedGroups.length
  if (report.violations.length === 0)
    return (
      <Notice tone="good">
        <View>
          <Text style={{ fontWeight: '800', color: C.good }}>すべての条件を満たしています</Text>
          <Text style={{ color: C.sub, fontSize: 13, marginTop: 2 }}>
            {total ? `同じ組 ${problem.wantedGroups.length} 件・別の組 ${problem.unwantedGroups.length} 件の指定をすべて反映しました。` : 'ペアの指定はありません。'}
          </Text>
        </View>
      </Notice>
    )
  return (
    <Card style={{ gap: 8 }}>
      {report.violations.map((v, i) => (
        <Text key={i} style={{ color: C.text, fontSize: 14 }}>
          ⚠ {v.message}
        </Text>
      ))}
    </Card>
  )
}
