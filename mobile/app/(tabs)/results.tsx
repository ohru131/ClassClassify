import { useRouter } from 'expo-router'
import { type ReactElement, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { FlatList, type NativeScrollEvent, type NativeSyntheticEvent, Pressable, ScrollView, Switch, Text, View } from 'react-native'

import { C, classColor } from '@/components/theme'
import { Btn, Card, Chip, Notice, Screen, Segmented, Stat, styles } from '@/components/ui'
import { useI18n } from '@/lib/language-provider'
import { useLayout } from '@/lib/layout'
import { useProject } from '@/lib/project-store'
import { pairStatus, resultWorkbook, rowColor, violationText, type ColumnReport, type PairTag, type Problem, type Report } from '@/lib/solver'
import { clampPage, offsetForPage, pageFromOffset, pageLayout, resolvePagerScroll, stepPage } from '@/lib/class-pager'
import { canSharePdf } from '@/lib/print'
import { buildResultPrintHtml } from '@/lib/print-html'
import { useProExport } from '@/lib/use-pro-export'

type Tab = 'classes' | 'balance' | 'checks'
type Focus = { kind: 'wanted' | 'unwanted'; group: number } | null

export default function ResultsScreen() {
  const { problem, solution, report, edited, moveStudent, resetMoves, error, setError } = useProject()
  const { exportXlsx, print, exportPdf, busy, isPro } = useProExport()
  const router = useRouter()
  const { isWide } = useLayout()
  const i18n = useI18n()
  const { t, className, fileLang, num } = i18n
  const [tab, setTab] = useState<Tab>('classes')
  const [selected, setSelected] = useState<number | null>(null)

  if (!problem || !solution || !report)
    return (
      <Screen>
        <Text style={{ fontSize: 24, fontWeight: '900', color: C.text }}>{t('resultsTitle')}</Text>
        <Card style={{ gap: 10 }}>
          <Text style={{ color: C.sub }}>{t('notYet')}</Text>
          <Btn variant="primary" icon="play" label={t('toRunBtn')} onPress={() => router.navigate(problem ? '/run' : '/')} />
        </Card>
      </Screen>
    )

  const k = solution.k
  const perfect = report.totalExcess === 0
  const sizeGap = Math.max(...report.sizes) - Math.min(...report.sizes)
  const sel = selected !== null && selected < problem.students.length ? selected : null
  const printHtmlFor = () => buildResultPrintHtml({ problem, classOf: solution.classOf, k, report, createdAt: new Date(), i18n })

  return (
    <View style={{ flex: 1 }}>
      <Screen>
        <Text style={{ fontSize: 24, fontWeight: '900', color: C.text }}>{t('resultsTitle')}</Text>
        {error ? (
          <Notice tone="error" onClose={() => setError(null)}>
            {error}
          </Notice>
        ) : null}
        <View style={styles.wrap}>
          <Stat label={t('statClasses')} value={k} sub={t('statClassesSub', { n: problem.students.length })} />
          <Stat label={t('statGap')} value={sizeGap} sub={t('statGapSub', { min: Math.min(...report.sizes), max: Math.max(...report.sizes) })} tone={sizeGap <= 1 ? 'good' : 'bad'} />
          <Stat
            label={t('statBalance')}
            value={perfect ? t('perfect') : num(report.totalExcess, report.totalExcess % 1 ? 1 : 0)}
            sub={perfect ? t('balancePerfectSub') : t('balanceSub')}
            tone={perfect ? 'good' : 'default'}
          />
          <Stat
            label={t('statViolations')}
            value={report.violations.length}
            sub={report.violations.length ? t('violationsSub') : t('violationsOk')}
            tone={report.violations.length ? 'bad' : 'good'}
          />
        </View>
        <View style={[styles.row, { flexWrap: 'wrap' }]}>
          {edited ? <Btn small icon="arrow-undo-outline" label={t('undoMoves')} onPress={resetMoves} /> : null}
          <Btn
            small
            variant="primary"
            icon={isPro ? 'share-outline' : 'lock-closed-outline'}
            label={t('shareXlsx')}
            busy={busy === 'xlsx'}
            onPress={() => exportXlsx(() => resultWorkbook(problem, solution.classOf, k, report, fileLang), t('fileResults'))}
          />
          <Btn
            small
            icon={isPro ? 'print-outline' : 'lock-closed-outline'}
            label={canSharePdf ? t('print') : t('printPdf')}
            busy={busy === 'print'}
            onPress={() => print(printHtmlFor)}
          />
          {canSharePdf ? (
            <Btn small icon={isPro ? 'document-outline' : 'lock-closed-outline'} label={t('sharePdf')} busy={busy === 'pdf'} onPress={() => exportPdf(printHtmlFor, t('fileResults'))} />
          ) : null}
          {!isPro ? <Text style={{ fontSize: 12, color: C.muted }}>{t('proFeaturesNote')}</Text> : null}
        </View>
        <Segmented
          value={tab}
          onChange={setTab}
          options={[
            { value: 'classes', label: t('tabClasses') },
            { value: 'balance', label: t('tabBalance') },
            { value: 'checks', label: report.violations.length ? t('tabChecksN', { n: report.violations.length }) : t('tabChecks') },
          ]}
        />
        {tab === 'classes' && <ClassBoard problem={problem} classOf={solution.classOf} k={k} report={report} selected={sel} setSelected={setSelected} />}
        {tab === 'balance' && (
          <View style={{ flexDirection: isWide ? 'row' : 'column', flexWrap: 'wrap', gap: 12 }}>
            {report.columns.map((c) => (
              <BalanceCard key={c.column} col={c} k={k} wide={isWide} />
            ))}
            {report.columns.length === 0 ? <Text style={{ color: C.muted }}>{t('noBalanceItems')}</Text> : null}
          </View>
        )}
        {tab === 'checks' && <Checks problem={problem} report={report} classOf={solution.classOf} />}
        {/* 移動パネルの下に隠れないための余白 */}
        {sel !== null ? <View style={{ height: 120 }} /> : null}
      </Screen>

      {sel !== null ? (
        <View style={moveBar} accessibilityLiveRegion="polite">
          <View style={[styles.row, { justifyContent: 'space-between' }]}>
            <Text style={{ fontWeight: '800', color: C.text, flexShrink: 1 }} numberOfLines={1}>
              {t('moveTitle', { who: `${problem.students[sel].no}:${problem.students[sel].name}` })}
            </Text>
            <Btn small icon="close" accessibilityLabel={t('close')} onPress={() => setSelected(null)} />
          </View>
          <View style={[styles.wrap, { marginTop: 8 }]}>
            {Array.from({ length: k }, (_, c) => {
              const color = classColor(c)
              const here = solution.classOf[sel] === c
              return (
                <Btn
                  key={c}
                  small
                  label={className(c)}
                  disabled={here}
                  accessibilityLabel={t('moveTo', { cls: className(c) })}
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
  const { t, className, file } = useI18n()
  const [current, setCurrent] = useState(0)
  const [colorize, setColorize] = useState(true)
  const [focus, setFocus] = useState<Focus>(null)
  const pairs = useMemo(() => pairStatus(problem, classOf, file.tagPrefix), [problem, classOf, file])
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
            <Text style={{ fontSize: 18, fontWeight: '900', color: C.text }}>{className(c)}</Text>
          </View>
          <Text style={{ fontSize: 12, fontWeight: '800', color: C.sub, backgroundColor: '#F1F5F9', paddingHorizontal: 8, paddingVertical: 2, borderRadius: 999 }}>{t('studentsN', { n: members.length })}</Text>
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
                accessibilityLabel={t('studentA11y', { no: s.no, name: s.name, cls: className(c) })}
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
                  {s.name || t('noName')}
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
        {isWide ? t('boardHintWide') : t('boardHintNarrow')}
      </Text>
      {pairs.groups.length > 0 ? (
        <Card style={{ gap: 8, padding: 12 }}>
          <View style={[styles.row, { flexWrap: 'wrap', justifyContent: 'space-between' }]}>
            <Text style={{ fontWeight: '800', color: C.text }}>
              {t('pairingsHeader')}{' '}
              <Text style={{ color: pairs.groups.every((g) => g.ok) ? C.good : C.danger, fontSize: 12 }}>
                {t('pairingsMet', { ok: pairs.groups.filter((g) => g.ok).length, n: pairs.groups.length })}
              </Text>
            </Text>
            <View style={styles.row}>
              <Text style={{ fontSize: 12, color: C.sub }}>{t('colorize')}</Text>
              <Switch value={colorize} onValueChange={setColorize} accessibilityLabel={t('colorize')} />
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
                    {g.members.map((i) => problem.students[i].name).join(file.joinSep)}
                  </Text>
                  <Text style={{ color: g.color.fg, fontSize: 12, fontWeight: '700' }}>
                    → {classes.map((c) => className(c - 1)).join('/')} {g.ok ? '✓' : '✗'}
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
        <ClassPager k={k} current={cur} onChange={setCurrent} sizes={report.sizes} renderPage={card} extraData={[classOf, selected, colorize, focus, pairs]} />
      )}
    </View>
  )
}

/**
 * スマホ幅の結果画面: クラスを1ページずつ横に並べ、左右のスワイプ・上のタブ・前後の矢印で切り替える。
 * ネイティブモジュールは足さず、横向きの FlatList（pagingEnabled）だけで組む（Web 書き出しでも同じ）。
 * 縦のスクロールは画面全体の ScrollView が受け持ち、ページの中では縦に入れ子にしない
 * （向きの違うスクロールは OS が方向で振り分けるので、生徒の一覧を縦に送る指と左右のスワイプが干渉しない）。
 * 今どのページかは onScroll の位置から求め、止まったら（onMomentumScrollEnd、Web は最後の scroll から一定時間後）
 * 実際の位置で選び直す（react-native-web は onMomentumScrollEnd を出さない）。判定は lib/class-pager.ts の resolvePagerScroll。
 */
function ClassPager({
  k,
  current,
  onChange,
  sizes,
  renderPage,
  extraData,
}: {
  k: number
  current: number
  onChange: (c: number) => void
  sizes: number[]
  renderPage: (c: number) => ReactElement
  extraData: unknown
}) {
  const { t, className } = useI18n()
  const listRef = useRef<FlatList<number>>(null)
  const tabsRef = useRef<ScrollView>(null)
  const [width, setWidth] = useState(0)
  // スクロールで決まったページ。タブ・矢印で動かすときはスクロールが終わるまでこの値を追わない
  const settling = useRef<number | null>(null)
  const pages = useMemo(() => Array.from({ length: k }, (_, c) => c), [k])
  const cur = clampPage(current, k)
  // タブ（クラスのチップ）の位置。クラスが多くてタブが画面からはみ出すとき、今のクラスのタブを見える位置へ送る
  const chipX = useRef<number[]>([])
  useEffect(() => {
    const x = chipX.current[cur]
    if (x !== undefined) tabsRef.current?.scrollTo({ x: Math.max(0, x - 24), animated: true })
  }, [cur])

  const goTo = useCallback(
    (c: number, animated = true) => {
      const target = clampPage(c, k)
      settling.current = animated ? target : null
      onChange(target)
      if (width > 0) listRef.current?.scrollToOffset({ offset: offsetForPage(target, width), animated })
    },
    [k, width, onChange],
  )

  // ページ幅が変わった（回転・ウィンドウのリサイズ）・クラス数が変わったときは、保留中の移動を捨てて今のクラスの位置へ戻す
  useEffect(() => {
    settling.current = null
    if (width > 0) listRef.current?.scrollToOffset({ offset: offsetForPage(cur, width), animated: false })
    // cur はここでは追わない（スワイプのたびに位置を戻すと指の動きと喧嘩する）
  }, [width, k]) // eslint-disable-line react-hooks/exhaustive-deps

  // スクロールが止まったことの検出（Web には onMomentumScrollEnd が無いので、最後の scroll から一定時間で止まったとみなす）
  const endTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => () => {
    if (endTimer.current) clearTimeout(endTimer.current)
  }, [])
  const handle = (phase: 'drag' | 'scroll' | 'end', x: number) => {
    const r = resolvePagerScroll({ settling: settling.current, current: cur }, { phase, page: pageFromOffset(x, width, k) })
    settling.current = r.settling
    if (r.select !== null) onChange(r.select)
  }
  const onScroll = (e: NativeSyntheticEvent<NativeScrollEvent>) => {
    const x = e.nativeEvent.contentOffset.x
    handle('scroll', x)
    if (endTimer.current) clearTimeout(endTimer.current)
    endTimer.current = setTimeout(() => handle('end', x), 160)
  }

  const prev = stepPage(cur, -1, k)
  const next = stepPage(cur, 1, k)
  return (
    <View style={{ gap: 8 }} onLayout={(e) => setWidth(Math.round(e.nativeEvent.layout.width))}>
      <ScrollView ref={tabsRef} horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ gap: 6 }}>
        {pages.map((c) => (
          <View key={c} onLayout={(e) => (chipX.current[c] = e.nativeEvent.layout.x)}>
            <Chip label={t('classChip', { cls: className(c), n: sizes[c] ?? 0 })} selected={cur === c} onPress={() => goTo(c)} />
          </View>
        ))}
      </ScrollView>
      <View style={[styles.row, { justifyContent: 'space-between' }]}>
        <Btn small icon="chevron-back" accessibilityLabel={t('prevClass')} disabled={prev === null} onPress={() => prev !== null && goTo(prev)} />
        <View style={[styles.row, { gap: 5 }]} accessibilityLabel={t('classPositionA11y', { cls: className(cur), i: cur + 1, n: k })} accessibilityLiveRegion="polite">
          {k <= 12
            ? pages.map((c) => <View key={c} style={{ width: c === cur ? 16 : 6, height: 6, borderRadius: 3, backgroundColor: c === cur ? classColor(c).dot : C.border }} />)
            : null}
          <Text style={{ fontSize: 12, fontWeight: '800', color: C.sub, marginLeft: 4, fontVariant: ['tabular-nums'] }}>
            {cur + 1} / {k}
          </Text>
        </View>
        <Btn small icon="chevron-forward" accessibilityLabel={t('nextClass')} disabled={next === null} onPress={() => next !== null && goTo(next)} />
      </View>
      {width > 0 ? (
        <FlatList
          ref={listRef}
          data={pages}
          extraData={extraData}
          keyExtractor={(c) => String(c)}
          horizontal
          pagingEnabled
          nestedScrollEnabled
          showsHorizontalScrollIndicator={false}
          initialScrollIndex={cur}
          getItemLayout={(_, index) => pageLayout(width, index)}
          onScroll={onScroll}
          // 利用者が触ったら、タブ・矢印で始めた移動の行き先を捨てて指の動きに従う
          onScrollBeginDrag={(e) => handle('drag', e.nativeEvent.contentOffset.x)}
          onMomentumScrollEnd={(e) => handle('end', e.nativeEvent.contentOffset.x)}
          scrollEventThrottle={16}
          // 高さはいちばん長いクラスに揃う（人数差はふつう1人以内なので、短いクラスの下の余白はわずか）
          style={{ width }}
          renderItem={({ item }) => <View style={{ width }}>{renderPage(item)}</View>}
        />
      ) : null}
    </View>
  )
}

function PairBadge({ tag }: { tag: PairTag }) {
  const { t } = useI18n()
  return (
    <Text
      accessibilityLabel={`${t(tag.kind === 'wanted' ? 'wantedBadgeA11y' : 'unwantedBadgeA11y', { label: tag.label })}${tag.ok ? '' : t('notMetA11y')}`}
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
  const { t, className, num } = useI18n()
  const numeric = col.kind === 'numeric'
  const max = Math.max(1, ...col.rows.flat())
  return (
    <Card style={{ gap: 8, flexBasis: wide ? '48%' : undefined, flexGrow: 1 }}>
      <View style={[styles.row, { justifyContent: 'space-between' }]}>
        <Text style={{ fontWeight: '800', color: C.text, fontSize: 15 }}>{col.column}</Text>
        <View style={styles.row}>
          <Text style={{ fontSize: 12, color: C.muted }}>{t('weightN', { w: num(col.weight, col.weight % 1 ? 1 : 0) })}</Text>
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
            {numeric ? t('average') : col.excess === 0 ? t('even') : t('offBy', { n: num(col.excess, col.excess % 1 ? 1 : 0) })}
          </Text>
        </View>
      </View>
      <ScrollView horizontal>
        <View>
          <View style={{ flexDirection: 'row' }}>
            <Text style={{ width: 76 }} />
            {Array.from({ length: k }, (_, c) => (
              <Text key={c} style={{ width: 64, textAlign: 'center', fontSize: 11, fontWeight: '800', color: C.muted }} numberOfLines={1}>
                {className(c)}
              </Text>
            ))}
            <Text style={{ width: 64, textAlign: 'center', fontSize: 11, color: C.muted }}>{t('target')}</Text>
          </View>
          {col.levels.map((level, l) => (
            <View key={level} style={{ flexDirection: 'row', alignItems: 'center', marginTop: 4 }}>
              <Text style={{ width: 76, fontSize: 12, fontWeight: '700', color: C.sub }} numberOfLines={1}>
                {numeric ? t('average') : level}
              </Text>
              {col.rows[l].map((v, c) => {
                const ideal = col.ideal[l]
                const ok = numeric || (v >= Math.floor(ideal) && v <= Math.ceil(ideal))
                const alpha = numeric ? 0.15 : 0.12 + 0.5 * (v / max)
                return (
                  <View key={c} style={{ width: 64, paddingHorizontal: 2 }}>
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
                      {numeric ? num(v, 2) : v}
                    </Text>
                  </View>
                )
              })}
              <Text style={{ width: 64, textAlign: 'center', fontSize: 12, color: C.muted }}>{num(col.ideal[l], numeric ? 2 : 1)}</Text>
            </View>
          ))}
        </View>
      </ScrollView>
    </Card>
  )
}

function Checks({ problem, report, classOf }: { problem: Problem; report: Report; classOf: number[] }) {
  const { t, fileLang } = useI18n()
  const total = problem.wantedGroups.length + problem.unwantedGroups.length
  if (report.violations.length === 0)
    return (
      <Notice tone="good">
        <View>
          <Text style={{ fontWeight: '800', color: C.good }}>{t('allMet')}</Text>
          <Text style={{ color: C.sub, fontSize: 13, marginTop: 2 }}>
            {total ? t('allMetBody', { w: problem.wantedGroups.length, u: problem.unwantedGroups.length }) : t('noPairsBody')}
          </Text>
        </View>
      </Notice>
    )
  return (
    <Card style={{ gap: 8 }}>
      {report.violations.map((v, i) => (
        <Text key={i} style={{ color: C.text, fontSize: 14 }}>
          ⚠ {violationText(fileLang, v, problem, classOf)}
        </Text>
      ))}
    </Card>
  )
}
