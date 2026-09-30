import { useRouter } from 'expo-router'
import { useState } from 'react'
import { Text, View } from 'react-native'

import { useLayout } from '@/lib/layout'
import { useProject } from '@/lib/project-store'
import { blankProblem, loadSample, SAMPLES } from '@/lib/samples'
import { parseWorkbook } from '@/lib/solver'
import { pickXlsx } from '@/lib/xlsx-files'
import { C } from './theme'
import { Btn, Card, styles, Title } from './ui'

/** 名簿の読み込み（Excel・サンプル・新規作成） */
export function LoadPanel({ onDone }: { onDone?: () => void }) {
  const { loadProblem, setError } = useProject()
  const { isWide } = useLayout()
  const router = useRouter()
  const [busy, setBusy] = useState(false)

  const pick = async () => {
    setBusy(true)
    try {
      const f = await pickXlsx()
      if (!f) return
      loadProblem(parseWorkbook(f.data), f.name)
      onDone?.()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card>
      <Title sub="ひな形の Excel（Web 版と同じ形式）に生徒の特性を記入して選ぶか、サンプル・新規作成から始めます。名簿はこの端末の中だけに保存されます。">
        名簿を読み込む
      </Title>
      <View style={[styles.wrap, { marginTop: 4 }]}>
        <Btn variant="primary" icon="document-attach" label="Excel ファイルを選ぶ（.xlsx）" busy={busy} onPress={pick} style={isWide ? undefined : { flexGrow: 1 }} />
        <Btn
          icon="create-outline"
          label="新しい名簿を作る"
          onPress={() => {
            loadProblem(blankProblem(), '新しい名簿')
            onDone?.()
          }}
          style={isWide ? undefined : { flexGrow: 1 }}
        />
      </View>
      <Text style={[styles.label, { marginTop: 16 }]}>サンプルで試す</Text>
      <View style={styles.wrap}>
        {SAMPLES.map((s) => (
          <Btn
            key={s.id}
            small
            variant="soft"
            icon="sparkles-outline"
            label={s.label}
            onPress={() => {
              try {
                const { problem, name } = loadSample(s.id)
                loadProblem(problem, name)
                onDone?.()
              } catch (e) {
                setError(e instanceof Error ? e.message : String(e))
              }
            }}
          />
        ))}
      </View>
      <Text style={{ fontSize: 12, color: C.muted, marginTop: 14 }} onPress={() => router.push('/privacy')} accessibilityRole="link">
        データの扱い（プライバシーポリシー）›
      </Text>
    </Card>
  )
}
