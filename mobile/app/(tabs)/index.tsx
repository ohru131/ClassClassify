import { useRouter } from 'expo-router'
import { useState } from 'react'
import { ActivityIndicator, Text, View } from 'react-native'

import { ColumnEditor } from '@/components/column-editor'
import { GroupEditor } from '@/components/group-editor'
import { LoadPanel } from '@/components/load-panel'
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
  const { exportXlsx, saveToFile, canSaveToFile, busy, isPro } = useProExport()
  const { t, fileLang } = useI18n()
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
        <Text style={{ fontSize: 13, fontWeight: '800', color: C.primary }}>{t('appEyebrow')}</Text>
        <Text style={{ fontSize: 24, fontWeight: '900', color: C.text, marginTop: 2 }}>{t('rosterTitle')}</Text>
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
          {problem ? <Btn label={t('backToRoster')} onPress={() => setShowLoad(false)} /> : null}
        </>
      ) : (
        <>
          <Card style={{ gap: 10 }}>
            <View style={[styles.row, { flexWrap: 'wrap', justifyContent: 'space-between' }]}>
              <View style={{ flexShrink: 1 }}>
                <Text style={{ fontSize: 16, fontWeight: '800', color: C.text }} numberOfLines={1}>
                  {fileName ?? t('rosterTitle')}
                </Text>
                <Text style={{ fontSize: 13, color: C.sub }}>
                  {t('rosterSummary', { n: problem.students.length, cols: problem.columns.length, w: problem.wantedGroups.length, u: problem.unwantedGroups.length })}
                </Text>
              </View>
              {/* 長い言語でもカードからはみ出さないよう、行の幅に収めて折り返す */}
              <View style={[styles.row, { flexWrap: 'wrap', flexShrink: 1 }]}>
                <Btn small icon="folder-open-outline" label={t('otherRoster')} onPress={() => setShowLoad(true)} />
                <Btn
                  small
                  icon={isPro ? 'share-outline' : 'lock-closed-outline'}
                  label={t('saveRosterXlsx')}
                  busy={busy === 'xlsx'}
                  onPress={() => exportXlsx(() => rosterWorkbook(problem, numClasses, fileLang), t('fileRoster'))}
                />
                {canSaveToFile ? (
                  <Btn
                    small
                    icon={isPro ? 'save-outline' : 'lock-closed-outline'}
                    label={t('saveToFile')}
                    busy={busy === 'save'}
                    onPress={() => saveToFile(() => rosterWorkbook(problem, numClasses, fileLang), t('fileRoster'))}
                  />
                ) : null}
                <Btn small variant="primary" icon="arrow-forward" label={t('toRun')} onPress={() => router.navigate('/run')} />
              </View>
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
