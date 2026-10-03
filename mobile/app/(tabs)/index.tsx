import Ionicons from '@expo/vector-icons/Ionicons'
import { useLocalSearchParams, useRouter } from 'expo-router'
import { useEffect, useState } from 'react'
import { ActivityIndicator, Pressable, Text, View } from 'react-native'

import { ColumnEditor } from '@/components/column-editor'
import { ExportButton } from '@/components/export-button'
import { GroupEditor } from '@/components/group-editor'
import { LoadPanel } from '@/components/load-panel'
import { SavedList } from '@/components/saved-list'
import { StudentList } from '@/components/student-list'
import { C } from '@/components/theme'
import { Btn, Card, Notice, Screen, Segmented, styles } from '@/components/ui'
import { useI18n } from '@/lib/language-provider'
import { useProject } from '@/lib/project-store'
import { rosterWorkbook } from '@/lib/solver'
import { useProExport } from '@/lib/use-pro-export'

type Tab = 'students' | 'columns' | 'wanted' | 'unwanted'

export default function RosterScreen() {
  const { hydrated, problem, fileName, numClasses, modifyProblem, dismissWarnings, error, setError } = useProject()
  const { exportXlsx, saveToFile, canSaveToFile, busy, isPro, fileSaved, dismissFileSaved } = useProExport()
  const { t, fileLang } = useI18n()
  const router = useRouter()
  const [tab, setTab] = useState<Tab>('students')
  const [showLoad, setShowLoad] = useState(false)
  // 結果画面の「保存した編成を見る」から来たら、一覧（読み込みパネルの下）を出す
  const { saved } = useLocalSearchParams<{ saved?: string }>()
  useEffect(() => {
    if (saved === '1') {
      setShowLoad(true)
      router.setParams({ saved: undefined })
    }
  }, [saved, router])

  if (!hydrated)
    return (
      <Screen>
        <ActivityIndicator color={C.primary} style={{ marginTop: 40 }} />
      </Screen>
    )

  return (
    <Screen>
      <View>
        <Text style={{ fontSize: 13, fontWeight: '800', color: C.primary }}>{t('appEyebrow')}</Text>
        <Text style={{ fontSize: 24, fontWeight: '900', color: C.text, marginTop: 2 }}>{t('rosterTitle')}</Text>
      </View>
      {error ? (
        <Notice tone="error" onClose={() => setError(null)}>
          {error}
        </Notice>
      ) : null}
      {fileSaved ? (
        <Notice tone="good" onClose={dismissFileSaved}>
          {t('fileSavedDone')}
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
          <SavedList onOpened={() => setShowLoad(false)} />
          {problem ? <Btn label={t('backToRoster')} onPress={() => setShowLoad(false)} /> : null}
        </>
      ) : (
        <>
          {/* 名簿の切り替えは名簿そのものへの操作（書き出し・実行）とは別の段にする */}
          <View style={[styles.row, { justifyContent: 'space-between', flexWrap: 'wrap' }]}>
            <Text style={{ fontSize: 12, fontWeight: '700', color: C.sub }}>{t('currentRoster')}</Text>
            <Pressable
              accessibilityRole="button"
              onPress={() => setShowLoad(true)}
              style={(st: { pressed: boolean; hovered?: boolean }) => [styles.row, { gap: 4, paddingVertical: 6, paddingHorizontal: 4, borderRadius: 8 }, (st.pressed || st.hovered) && { backgroundColor: C.hover }]}
            >
              <Ionicons name="swap-horizontal" size={16} color={C.primary} />
              <Text style={{ fontSize: 13, fontWeight: '700', color: C.primary }}>{t('otherRoster')}</Text>
            </Pressable>
          </View>
          <Card style={{ gap: 12, marginTop: -8 }}>
            <View>
              <Text style={{ fontSize: 16, fontWeight: '800', color: C.text }} numberOfLines={1}>
                {fileName ?? t('rosterTitle')}
              </Text>
              <Text style={{ fontSize: 13, color: C.sub }}>
                {t('rosterSummary', { n: problem.students.length, cols: problem.columns.length, w: problem.wantedGroups.length, u: problem.unwantedGroups.length })}
              </Text>
            </View>
            {/* 長い言語でもカードからはみ出さないよう、行の幅に収めて折り返す */}
            <View style={[styles.row, { flexWrap: 'wrap' }]}>
              <ExportButton
                icon={isPro ? 'grid-outline' : 'lock-closed-outline'}
                label={t('saveRosterXlsx')}
                title={t('exportSheetTitle', { format: t('exportExcel') })}
                isPro={isPro}
                busy={busy === 'xlsx' || busy === 'save'}
                choices={[
                  { icon: 'share-social-outline', label: t('shareVia'), sub: t('shareViaSub'), onPress: () => exportXlsx(() => rosterWorkbook(problem, numClasses, fileLang), t('fileRoster')) },
                  ...(canSaveToFile ? [{ icon: 'save-outline' as const, label: t('saveToFile'), sub: t('saveToFileSub'), onPress: () => saveToFile(() => rosterWorkbook(problem, numClasses, fileLang), t('fileRoster')) }] : []),
                ]}
              />
              <Btn small variant="primary" icon="arrow-forward" label={t('toRun')} onPress={() => router.navigate('/run')} />
            </View>
            {!isPro ? <Text style={{ fontSize: 12, color: C.muted }}>{t('proFeaturesNote')}</Text> : null}
          </Card>

          {problem.warnings.length ? (
            <Notice tone="warn" onClose={dismissWarnings}>
              <View>
                <Text style={{ color: C.warn, fontWeight: '800', marginBottom: 2 }}>{t('loadWarnings')}</Text>
                {problem.warnings.map((w, i) => (
                  <Text key={i} style={{ color: C.warn, fontSize: 13 }}>
                    • {w}
                  </Text>
                ))}
              </View>
            </Notice>
          ) : null}

          <Segmented
            value={tab}
            onChange={setTab}
            options={[
              { value: 'students', label: t('tabStudents', { n: problem.students.length }) },
              { value: 'columns', label: t('tabColumns') },
              { value: 'wanted', label: t('tabWanted', { n: problem.wantedGroups.length }) },
              { value: 'unwanted', label: t('tabUnwanted', { n: problem.unwantedGroups.length }) },
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
