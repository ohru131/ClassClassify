import { useRouter } from 'expo-router'
import { Text, View } from 'react-native'

import { C } from '@/components/theme'
import { Btn, Card, Notice, Screen, Segmented, Stepper, styles } from '@/components/ui'
import { useLayout } from '@/lib/layout'
import { useProject } from '@/lib/project-store'
import { defaultStarts } from '@/lib/runner'

export default function RunScreen() {
  const { problem, numClasses, setNumClasses, setMaxPerClass, timeSec, setTimeSec, run, cancel, running, progress, solution, error, setError } = useProject()
  const router = useRouter()
  const { isWide } = useLayout()

  if (!problem)
    return (
      <Screen>
        <Title />
        <Card style={{ gap: 10 }}>
          <Text style={{ color: C.sub }}>まず「名簿」タブで名簿を読み込んでください。</Text>
          <Btn variant="primary" icon="people" label="名簿を読み込む" onPress={() => router.navigate('/')} />
        </Card>
      </Screen>
    )

  const n = problem.students.length
  const lo = Math.floor(n / numClasses)
  const hi = Math.ceil(n / numClasses)
  const enabled = problem.columns.filter((c) => c.enabled && c.weight > 0)
  const starts = defaultStarts(timeSec * 1000)

  const start = async () => {
    if (await run()) router.navigate('/results')
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
          <Text style={styles.label}>クラス（グループ）数</Text>
          <Stepper label="クラス数" value={numClasses} min={2} max={Math.max(2, n)} onChange={setNumClasses} />
          <Text style={{ fontSize: 12, color: C.muted }}>1組あたり {lo === hi ? lo : `${lo}〜${hi}`} 名（{n} 名）</Text>
        </Card>
        <Card style={[{ gap: 6 }, isWide && { flex: 1 }]}>
          <Text style={styles.label}>1クラスの最大人数</Text>
          <Stepper label="最大人数" value={hi} min={1} max={Math.max(1, Math.ceil(n / 2))} onChange={setMaxPerClass} />
          <Text style={{ fontSize: 12, color: C.muted }}>変えるとクラス数を自動で決めます</Text>
        </Card>
        <Card style={[{ gap: 6 }, isWide && { flex: 1.3 }]}>
          <Text style={styles.label}>探索時間</Text>
          <Segmented
            value={timeSec}
            onChange={setTimeSec}
            options={[
              { value: 3, label: '高速 3秒' },
              { value: 10, label: '標準 10秒' },
              { value: 30, label: '徹底 30秒' },
            ]}
          />
          <Text style={{ fontSize: 12, color: C.muted }}>
            {timeSec} 秒{starts > 1 ? ` · ${starts} 回探索して最良の案を採用` : ''}
          </Text>
        </Card>
      </View>

      <Card style={{ gap: 8 }}>
        <View style={[styles.row, { justifyContent: 'space-between' }]}>
          <Text style={{ fontSize: 16, fontWeight: '800', color: C.text }}>均等にする項目</Text>
          <Btn small icon="create-outline" label="項目・重みを編集" onPress={() => router.navigate('/')} />
        </View>
        {enabled.length === 0 ? (
          <Text style={{ color: C.warn }}>有効な項目がありません（人数だけを揃えます）。</Text>
        ) : (
          <View style={styles.wrap}>
            {enabled.map((c) => (
              <View key={c.name} style={{ backgroundColor: C.primarySoft, borderRadius: 10, paddingHorizontal: 10, paddingVertical: 6 }}>
                <Text style={{ color: C.primaryText, fontWeight: '700', fontSize: 13 }}>
                  {c.name} ×{c.weight}
                </Text>
              </View>
            ))}
          </View>
        )}
        <Text style={{ fontSize: 13, color: C.sub }}>
          同じ組にする {problem.wantedGroups.length} 件 · 別の組にする {problem.unwantedGroups.length} 件
        </Text>
      </Card>

      <Card style={{ gap: 12 }}>
        {running ? (
          <>
            <View style={{ height: 12, borderRadius: 6, backgroundColor: '#EEF0F5', overflow: 'hidden' }} accessibilityRole="progressbar" accessibilityValue={{ min: 0, max: 100, now: Math.round(progress * 100) }}>
              <View style={{ height: '100%', width: `${Math.max(3, progress * 100)}%`, backgroundColor: C.primary, borderRadius: 6 }} />
            </View>
            <View style={[styles.row, { justifyContent: 'space-between' }]}>
              <Text style={{ color: C.sub }}>最適な組み合わせを探索中… {Math.round(progress * 100)}%</Text>
              <Btn small icon="stop" label="中止" onPress={cancel} />
            </View>
          </>
        ) : (
          <>
            <Btn variant="primary" icon="play" label={solution ? 'もう一度編成する' : 'クラス編成を実行'} onPress={start} />
            {solution ? (
              <Text style={{ fontSize: 12, color: C.muted }}>
                前回: {solution.k} クラス · {solution.starts} 回 · {(solution.iterations / 1e6).toFixed(2)}M 回の探索
              </Text>
            ) : null}
          </>
        )}
      </Card>
    </Screen>
  )
}

function Title() {
  return <Text style={{ fontSize: 24, fontWeight: '900', color: C.text }}>設定・実行</Text>
}
