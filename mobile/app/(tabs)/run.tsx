import { useFocusEffect, useRouter } from 'expo-router'
import { useCallback, useRef, useState } from 'react'
import { Switch, Text, View } from 'react-native'

import { C } from '@/components/theme'
import { Btn, Card, Chip, Notice, Screen, Segmented, Stepper, styles } from '@/components/ui'
import { useI18n } from '@/lib/language-provider'
import { useLayout } from '@/lib/layout'
import { useProject } from '@/lib/project-store'
import { defaultStarts } from '@/lib/runner'
import { findPreviousClassColumn, placementOfSaved, previousClassValues, setColumnEnabled, withPreviousClass } from '@/lib/saved-results'
import { useSavedResults } from '@/lib/saved-results-store'
import { parsePlacement, type Placement } from '@/lib/solver'
import { pickXlsx } from '@/lib/xlsx-files'

export default function RunScreen() {
  const { problem, numClasses, setNumClasses, stepMaxPerClass, timeSec, setTimeSec, run, cancel, running, progress, solution, error, setError, modifyProblem } = useProject()
  const router = useRouter()
  const { isWide } = useLayout()
  const { t, num, className, file } = useI18n()
  const saved = useSavedResults()
  // 「前回とできるだけ入れ替える」の元にした編成（保存済みの id、書き出した Excel なら null）と、前回の組が分かった人数
  const [source, setSource] = useState<{ id: string | null; name: string; matched: number; students: unknown } | null>(null)
  // 前回の編成を読んでいる間は編成を始めない（読み終えて名簿を変えると、編成の結果が捨てられる）
  const [applying, setApplying] = useState<null | 'saved' | 'file'>(null)
  // 実行中に別のタブへ移った人を、終わった瞬間に結果画面へ引き戻さない
  const focusedRef = useRef(true)
  useFocusEffect(
    useCallback(() => {
      focusedRef.current = true
      return () => {
        focusedRef.current = false
      }
    }, []),
  )

  if (!problem)
    return (
      <Screen>
        <Title />
        <Card style={{ gap: 10 }}>
          <Text style={{ color: C.sub }}>{t('loadFirst')}</Text>
          <Btn variant="primary" icon="people" label={t('loadRosterBtn')} onPress={() => router.navigate('/')} />
        </Card>
      </Screen>
    )

  const n = problem.students.length
  const lo = Math.floor(n / numClasses)
  const hi = Math.ceil(n / numClasses)
  const enabled = problem.columns.filter((c) => c.enabled && c.weight > 0)
  const starts = defaultStarts(timeSec * 1000)

  // 既にある列はその名前のまま使う（入れたあとで表示の言語を変えても同じ列を扱う）
  const prev = findPreviousClassColumn(problem)
  const prevCol = prev?.name ?? t('prevClassColumn')
  const mixOn = !!prev && prev.enabled && prev.weight > 0
  // 選んだ編成の表示は、それを入れた名簿のときだけ（別の名簿を開いたら出さない）
  const shownSource = source && source.students === problem.students ? source : null
  // 前回の組分けから各生徒の前回の組を名簿に入れ、その列を均等に散らす（前回同じ組だった子が重ならないように）
  const applyPlacement = (placement: Placement, id: string | null, name: string) => {
    let matched = 0
    let students: unknown = null
    modifyProblem((p) => {
      const r = previousClassValues(p, placement)
      matched = r.matched
      const next = withPreviousClass(p, prevCol, r.values, placement.order)
      students = next.students
      return next
    })
    setSource({ id, name, matched, students })
  }
  const applyFrom = async (id: string) => {
    // 編成中に名簿を変えると、終わった結果が捨てられる
    if (running || applying) return
    setApplying('saved')
    try {
      const s = await saved.load(id)
      if (!s) {
        setError(t('openFailed'))
        return
      }
      applyPlacement(placementOfSaved(s, className), id, s.name)
    } catch {
      setError(t('openFailed'))
    } finally {
      setApplying(null)
    }
  }
  // 結果画面から書き出した Excel（Web 版・どの言語のものでも）の「組分け」シートを前回の組分けとして読む
  const applyFromFile = async () => {
    if (running || applying) return
    setApplying('file')
    try {
      const f = await pickXlsx()
      if (!f) return
      let placement: Placement | null = null
      try {
        placement = parsePlacement(f.data)
      } catch {
        placement = null
      }
      if (!placement) {
        setError(t('mixFileInvalid', { sheet: file.sheets.assign }))
        return
      }
      applyPlacement(placement, null, f.name)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setApplying(null)
    }
  }
  const toggleMix = (on: boolean) => {
    if (running || applying) return
    if (!on) modifyProblem((p) => setColumnEnabled(p, prevCol, false))
    else if (prev) modifyProblem((p) => setColumnEnabled(p, prevCol, true))
    else if (saved.list[0]) void applyFrom(saved.list[0].id)
    else void applyFromFile()
  }

  const start = async () => {
    if ((await run()) && focusedRef.current) router.navigate('/results')
  }

  return (
    <Screen>
      <Title />
      {error ? (
        <Notice tone="error" onClose={() => setError(null)}>
          {error}
        </Notice>
      ) : null}
      <View style={{ flexDirection: isWide ? 'row' : 'column', gap: 14 }}>
        <Card style={[{ gap: 6 }, isWide && { flex: 1 }]}>
          <Text style={styles.label}>{t('classCount')}</Text>
          <Stepper label={t('classCountA11y')} value={numClasses} min={2} max={Math.max(2, n)} onChange={setNumClasses} />
          <Text style={{ fontSize: 12, color: C.muted }}>{t('perClass', { range: lo === hi ? lo : `${lo}–${hi}`, n })}</Text>
        </Card>
        <Card style={[{ gap: 6 }, isWide && { flex: 1 }]}>
          <Text style={styles.label}>{t('maxPerClass')}</Text>
          <Stepper label={t('maxPerClass')} value={hi} min={1} max={Math.max(1, Math.ceil(n / 2))} onChange={(v) => stepMaxPerClass(v > hi ? 1 : -1)} />
          <Text style={{ fontSize: 12, color: C.muted }}>{t('maxHint')}</Text>
        </Card>
        <Card style={[{ gap: 6 }, isWide && { flex: 1.3 }]}>
          <Text style={styles.label}>{t('searchTime')}</Text>
          <Segmented
            value={timeSec}
            onChange={setTimeSec}
            options={[
              { value: 3, label: t('quick') },
              { value: 10, label: t('standard') },
              { value: 30, label: t('thorough') },
            ]}
          />
          <Text style={{ fontSize: 12, color: C.muted }}>
            {starts > 1 ? t('timeInfoStarts', { s: timeSec, n: starts }) : t('timeInfo', { s: timeSec })}
          </Text>
        </Card>
      </View>

      <Card style={{ gap: 8 }}>
        <View style={[styles.row, { justifyContent: 'space-between' }]}>
          <Text style={{ fontSize: 16, fontWeight: '800', color: C.text, flexShrink: 1 }}>{t('balancedItems')}</Text>
          <Btn small icon="create-outline" label={t('editAttributes')} onPress={() => router.navigate('/')} />
        </View>
        {enabled.length === 0 ? (
          <Text style={{ color: C.warn }}>{t('noEnabled')}</Text>
        ) : (
          <View style={styles.wrap}>
            {enabled.map((c) => (
              <View key={c.name} style={{ backgroundColor: C.primarySoft, borderRadius: 10, paddingHorizontal: 10, paddingVertical: 6 }}>
                <Text style={{ color: C.primaryText, fontWeight: '700', fontSize: 13 }}>
                  {c.name} ×{num(c.weight, c.weight % 1 ? 1 : 0)}
                </Text>
              </View>
            ))}
          </View>
        )}
        <Text style={{ fontSize: 13, color: C.sub }}>
          {t('pairsSummary', { w: problem.wantedGroups.length, u: problem.unwantedGroups.length })}
        </Text>
      </Card>

      <Card style={{ gap: 8 }}>
        <View style={[styles.row, { justifyContent: 'space-between' }]}>
          <Text style={{ fontSize: 16, fontWeight: '800', color: C.text, flexShrink: 1 }}>{t('mixTitle')}</Text>
          <Switch value={mixOn} onValueChange={toggleMix} disabled={running || !!applying} accessibilityLabel={t('mixTitle')} />
        </View>
        <Text style={{ fontSize: 13, color: C.sub, lineHeight: 19 }}>{t('mixHelp')}</Text>
        {!prev && saved.list.length === 0 ? <Text style={{ fontSize: 12, color: C.muted }}>{t('mixNoSaved')}</Text> : null}
        {mixOn ? <Text style={[styles.label, { marginTop: 4 }]}>{t('mixPick')}</Text> : null}
        {mixOn && saved.list.length > 0 ? (
          <View style={styles.wrap} accessibilityRole="radiogroup">
            {saved.list.map((m) => (
              <Chip key={m.id} label={m.name} selected={shownSource?.id === m.id} onPress={() => void applyFrom(m.id)} />
            ))}
          </View>
        ) : null}
        {/* 結果画面から書き出した Excel も前回の編成として選べる（保存していなくても、別の端末・Web 版で作ったものでも） */}
        {mixOn || (!prev && saved.list.length === 0) ? (
          <View style={styles.wrap}>
            <Btn small icon="document-attach-outline" label={t('mixPickFile')} busy={applying === 'file'} disabled={running || applying === 'saved'} onPress={() => void applyFromFile()} />
          </View>
        ) : null}
        {mixOn && shownSource ? (
          <Text style={{ fontSize: 12, color: C.muted }}>
            {t('mixFrom', { name: shownSource.name })} · {t('mixMatched', { m: shownSource.matched, n })}
          </Text>
        ) : null}
      </Card>

      <Card style={{ gap: 12 }}>
        {running ? (
          <>
            <View style={{ height: 12, borderRadius: 6, backgroundColor: '#EEF0F5', overflow: 'hidden' }} accessibilityRole="progressbar" accessibilityValue={{ min: 0, max: 100, now: Math.round(progress * 100) }}>
              <View style={{ height: '100%', width: `${Math.max(3, progress * 100)}%`, backgroundColor: C.primary, borderRadius: 6 }} />
            </View>
            <View style={[styles.row, { justifyContent: 'space-between' }]}>
              <Text style={{ color: C.sub, flexShrink: 1 }}>{t('searching', { p: Math.round(progress * 100) })}</Text>
              <Btn small icon="stop" label={t('stop')} onPress={cancel} />
            </View>
          </>
        ) : (
          <>
            <Btn variant="primary" icon="play" label={solution ? t('rerun') : t('runBtn')} disabled={!!applying} onPress={start} />
            {/* 保存した編成を開いただけのときは、編成の記録（案の数など）が無いので出さない */}
            {solution && solution.starts > 0 ? (
              <Text style={{ fontSize: 12, color: C.muted }}>
                {t('lastRun', { k: solution.k, starts: solution.starts, m: num(solution.iterations / 1e6, 2) })}
              </Text>
            ) : null}
          </>
        )}
      </Card>
    </Screen>
  )
}

function Title() {
  const { t } = useI18n()
  return <Text style={{ fontSize: 24, fontWeight: '900', color: C.text }}>{t('runTitle')}</Text>
}
