import { useRouter } from 'expo-router'
import { useState } from 'react'
import { ActivityIndicator, Text, View } from 'react-native'

import { ColumnEditor } from '@/components/column-editor'
import { GroupEditor } from '@/components/group-editor'
import { LoadPanel } from '@/components/load-panel'
import { StudentList } from '@/components/student-list'
import { C } from '@/components/theme'
import { Btn, Card, Notice, Screen, Segmented, styles } from '@/components/ui'
import { useProject } from '@/lib/project-store'
import { rosterWorkbook } from '@/lib/solver'
import { useProExport } from '@/lib/use-pro-export'

type Tab = 'students' | 'columns' | 'wanted' | 'unwanted'

export default function RosterScreen() {
  const { hydrated, problem, fileName, numClasses, modifyProblem, dismissWarnings, error, setError } = useProject()
  const { exportXlsx, busy, isPro } = useProExport()
  const router = useRouter()
  const [tab, setTab] = useState<Tab>('students')
  const [showLoad, setShowLoad] = useState(false)

  if (!hydrated)
    return (
      <Screen>
        <ActivityIndicator color={C.primary} style={{ marginTop: 40 }} />
      </Screen>
    )

  return (
    <Screen>
      <View>
        <Text style={{ fontSize: 13, fontWeight: '800', color: C.primary }}>Mosaic · クラス編成</Text>
        <Text style={{ fontSize: 24, fontWeight: '900', color: C.text, marginTop: 2 }}>名簿</Text>
      </View>
      {error ? (
        <Notice tone="error" onClose={() => setError(null)}>
          {error}
        </Notice>
      ) : null}

      {!problem || showLoad ? (
        <>
          <LoadPanel
            onDone={() => {
              setShowLoad(false)
              setTab('students')
            }}
          />
          {problem ? <Btn label="キャンセル（今の名簿に戻る）" onPress={() => setShowLoad(false)} /> : null}
        </>
      ) : (
        <>
          <Card style={{ gap: 10 }}>
            <View style={[styles.row, { flexWrap: 'wrap', justifyContent: 'space-between' }]}>
              <View style={{ flexShrink: 1 }}>
                <Text style={{ fontSize: 16, fontWeight: '800', color: C.text }} numberOfLines={1}>
                  {fileName ?? '名簿'}
                </Text>
                <Text style={{ fontSize: 13, color: C.sub }}>
                  {problem.students.length} 名 · 項目 {problem.columns.length} · 同じ組 {problem.wantedGroups.length} 件 · 別の組 {problem.unwantedGroups.length} 件
                </Text>
              </View>
              <View style={[styles.row, { flexWrap: 'wrap' }]}>
                <Btn small icon="folder-open-outline" label="別の名簿" onPress={() => setShowLoad(true)} />
                <Btn
                  small
                  icon={isPro ? 'share-outline' : 'lock-closed-outline'}
                  label="名簿を Excel で保存"
                  busy={busy === 'xlsx'}
                  onPress={() => exportXlsx(() => rosterWorkbook(problem, numClasses), '名簿')}
                />
                <Btn small variant="primary" icon="arrow-forward" label="設定・実行へ" onPress={() => router.navigate('/run')} />
              </View>
            </View>
            {!isPro ? <Text style={{ fontSize: 12, color: C.muted }}>Excel での保存・共有は Pro（買い切り）の機能です。</Text> : null}
          </Card>

          {problem.warnings.length ? (
            <Notice tone="warn" onClose={dismissWarnings}>
              <View>
                <Text style={{ color: C.warn, fontWeight: '800', marginBottom: 2 }}>読み込み時の注意</Text>
                {problem.warnings.map((w, i) => (
                  <Text key={i} style={{ color: C.warn, fontSize: 13 }}>
                    ・{w}
                  </Text>
                ))}
              </View>
            </Notice>
          ) : null}

          <Segmented
            value={tab}
            onChange={setTab}
            options={[
              { value: 'students', label: `生徒 ${problem.students.length}` },
              { value: 'columns', label: '項目・重み' },
              { value: 'wanted', label: `同じ組 ${problem.wantedGroups.length}` },
              { value: 'unwanted', label: `別の組 ${problem.unwantedGroups.length}` },
            ]}
          />
          {tab === 'students' && <StudentList problem={problem} onChange={modifyProblem} />}
          {tab === 'columns' && <ColumnEditor problem={problem} onChange={modifyProblem} />}
          {(tab === 'wanted' || tab === 'unwanted') && <GroupEditor key={tab} problem={problem} kind={tab} onChange={modifyProblem} />}
        </>
      )}
    </Screen>
  )
}
