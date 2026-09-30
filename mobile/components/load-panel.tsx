import { useRouter } from 'expo-router'
import { useState } from 'react'
import { Text, View } from 'react-native'

import { useI18n } from '@/lib/language-provider'
import { useLayout } from '@/lib/layout'
import { useProject } from '@/lib/project-store'
import { blankProblem, loadSample, samplesFor } from '@/lib/samples'
import { parseWorkbook, rosterWorkbook } from '@/lib/solver'
import { pickXlsx, shareXlsx } from '@/lib/xlsx-files'
import { C } from './theme'
import { Btn, Card, styles, Title } from './ui'

/** 名簿の読み込み（Excel・サンプル・新規作成）と、ひな形の入手（無料） */
export function LoadPanel({ onDone }: { onDone?: () => void }) {
  const { loadProblem, setError } = useProject()
  const { lang, t, parseMessages, fileLang } = useI18n()
  const { isWide } = useLayout()
  const router = useRouter()
  const [busy, setBusy] = useState<null | 'pick' | 'template'>(null)
  const fail = (e: unknown) => setError(e instanceof Error ? e.message : String(e))

  const pick = async () => {
    setBusy('pick')
    try {
      const f = await pickXlsx()
      if (!f) return
      loadProblem(parseWorkbook(f.data, parseMessages), f.name)
      onDone?.()
    } catch (e) {
      fail(e)
    } finally {
      setBusy(null)
    }
  }

  const template = () => {
    setBusy('template')
    // ひな形はその言語のシート名・見出しで作る（どの言語のファイルも読み込める）
    shareXlsx(rosterWorkbook(blankProblem(lang, 5), 2, fileLang), `${t('fileTemplate')}.xlsx`, t('sharingUnavailable'))
      .catch(fail)
      .finally(() => setBusy(null))
  }

  const grow = isWide ? undefined : { flexGrow: 1 }
  return (
    <Card>
      <Title sub={t('loadDesc')}>{t('loadTitle')}</Title>
      <View style={[styles.wrap, { marginTop: 4 }]}>
        <Btn variant="primary" icon="document-attach" label={t('pickFile')} busy={busy === 'pick'} onPress={pick} style={grow} />
        <Btn
          icon="create-outline"
          label={t('newRoster')}
          onPress={() => {
            loadProblem(blankProblem(lang), t('newRosterName'))
            onDone?.()
          }}
          style={grow}
        />
        <Btn icon="download-outline" label={t('getTemplate')} busy={busy === 'template'} onPress={template} style={grow} />
      </View>
      <Text style={{ fontSize: 12, color: C.muted, marginTop: 6 }}>{t('templateHint')}</Text>
      <Text style={[styles.label, { marginTop: 16 }]}>{t('trySamples')}</Text>
      <View style={styles.wrap}>
        {samplesFor(lang).map((s) => (
          <Btn
            key={s.id}
            small
            variant="soft"
            icon="sparkles-outline"
            label={s.label}
            onPress={() => {
              try {
                const { problem, label } = loadSample(lang, s.id)
                loadProblem(problem, t('samplePrefix', { label }))
                onDone?.()
              } catch (e) {
                fail(e)
              }
            }}
          />
        ))}
      </View>
      <Text style={{ fontSize: 12, color: C.muted, marginTop: 14 }} onPress={() => router.push('/privacy')} accessibilityRole="link">
        {t('privacyLink')}
      </Text>
    </Card>
  )
}
