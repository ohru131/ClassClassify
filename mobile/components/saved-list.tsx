import Ionicons from '@expo/vector-icons/Ionicons'
import { useRouter } from 'expo-router'
import { useState } from 'react'
import { Text, View } from 'react-native'

import { confirmAction } from '@/lib/confirm'
import { useI18n } from '@/lib/language-provider'
import { useProject } from '@/lib/project-store'
import { usePro } from '@/lib/revenuecat-provider'
import { countedSaves, FREE_SAVE_LIMIT } from '@/lib/saved-results'
import { useSavedResults } from '@/lib/saved-results-store'
import { C } from './theme'
import { Btn, Card, styles, Title } from './ui'

/**
 * 名前を付けて保存した編成の一覧（開く・削除）。開くと名簿と結果がそのまま戻り、結果画面へ移る。
 * Excel に書き出したときに自動で残したものも並べる（書き出した時点の内容。Excel を直したものは「Excel ファイルを選ぶ」から読み込む）
 */
export function SavedList({ onOpened }: { onOpened?: () => void }) {
  const { list, load, remove } = useSavedResults()
  const { openSaved, setError } = useProject()
  const { t, date } = useI18n()
  const { isPro } = usePro()
  const router = useRouter()
  const [busy, setBusy] = useState<string | null>(null)

  const open = async (id: string) => {
    if (busy) return
    setBusy(id)
    try {
      const saved = await load(id)
      if (!saved) {
        setError(t('openFailed'))
        return
      }
      openSaved(saved)
      onOpened?.()
      router.navigate('/results')
    } catch {
      setError(t('openFailed'))
    } finally {
      setBusy(null)
    }
  }

  return (
    <Card style={{ gap: 10 }}>
      <Title sub={isPro ? undefined : t('savedCount', { n: countedSaves(list), max: FREE_SAVE_LIMIT })}>{t('savedTitle')}</Title>
      {list.length === 0 ? <Text style={{ fontSize: 13, color: C.muted, lineHeight: 19 }}>{t('savedEmpty')}</Text> : null}
      {list.some((m) => m.exported) ? <Text style={{ fontSize: 12, color: C.muted, lineHeight: 17 }}>{t('savedExcelHint')}</Text> : null}
      {list.map((m) => (
        <View key={m.id} style={[styles.row, { justifyContent: 'space-between', borderTopWidth: 1, borderTopColor: C.border, paddingTop: 10 }]}>
          <View style={{ flexShrink: 1 }}>
            <Text style={{ fontSize: 15, fontWeight: '800', color: C.text }} numberOfLines={2}>
              {m.name}
            </Text>
            <Text style={{ fontSize: 12, color: C.sub }}>{t('savedMeta', { date: date(new Date(m.savedAt)), n: m.n, k: m.k })}</Text>
            {m.exported ? (
              <View style={[styles.row, { gap: 4, marginTop: 2 }]}>
                <Ionicons name="grid-outline" size={12} color={C.primary} />
                <Text style={{ fontSize: 12, color: C.primary, fontWeight: '700', flexShrink: 1 }}>{t('savedFromExcel', { file: m.exported.fileName })}</Text>
              </View>
            ) : null}
          </View>
          <View style={styles.row}>
            <Btn small variant="soft" icon="folder-open-outline" label={t('openSaved')} busy={busy === m.id} onPress={() => open(m.id)} />
            <Btn
              small
              icon="trash-outline"
              accessibilityLabel={`${t('delete')}: ${m.name}`}
              onPress={async () => {
                if (await confirmAction(t('deleteSavedTitle'), t('deleteSavedBody', { name: m.name }), t('delete'), t('cancel'))) {
                  remove(m.id).catch(() => setError(t('saveFailed')))
                }
              }}
            />
          </View>
        </View>
      ))}
    </Card>
  )
}
