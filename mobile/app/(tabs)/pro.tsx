import { useRouter } from 'expo-router'
import { useState } from 'react'
import { ActivityIndicator, Text, View } from 'react-native'

import { C } from '@/components/theme'
import { Btn, Card, Chip, Notice, Screen, styles } from '@/components/ui'
import { confirmAction } from '@/lib/confirm'
import { APP_LANGUAGES, LANGUAGE_META } from '@/lib/i18n'
import { useI18n } from '@/lib/language-provider'
import { useProject } from '@/lib/project-store'
import { usePro } from '@/lib/pro-provider'
import { FREE_SAVE_LIMIT } from '@/lib/saved-results'
import { useSavedResults } from '@/lib/saved-results-store'

const FEATURES = [
  ['feat1Title', 'feat1Body'],
  ['feat2Title', 'feat2Body'],
  ['featSaveTitle', 'featSaveBody'],
  ['feat3Title', 'feat3Body'],
] as const

export default function ProScreen() {
  const { isPro, isReady, purchaseMessage, priceLabel, isPurchasing, purchasePro, restorePurchases, isNativePurchaseAvailable } = usePro()
  const { problem, clearProject } = useProject()
  const saved = useSavedResults()
  const [eraseFailed, setEraseFailed] = useState(false)
  const { t, choice, setChoice } = useI18n()
  const router = useRouter()

  return (
    <Screen>
      <Text style={{ fontSize: 24, fontWeight: '900', color: C.text }}>{t('proTitle')}</Text>
      <Card style={{ gap: 12 }}>
        <View style={[styles.row, { justifyContent: 'space-between' }]}>
          <Text style={{ fontSize: 20, fontWeight: '900', color: C.text }}>{t('proName')}</Text>
          {isPro ? <Text style={{ color: C.good, fontWeight: '800' }}>{t('active')}</Text> : null}
        </View>
        <Text style={{ fontSize: 13, color: C.sub }}>{t('proIntro')}</Text>
        {FEATURES.map(([title, body]) => (
          <View key={title} style={{ flexDirection: 'row', gap: 10 }}>
            <Text style={{ color: C.primary, fontWeight: '900' }}>✓</Text>
            <View style={{ flex: 1 }}>
              <Text style={{ fontWeight: '800', color: C.text }}>{t(title)}</Text>
              <Text style={{ fontSize: 13, color: C.sub }}>{t(body, { max: FREE_SAVE_LIMIT })}</Text>
            </View>
          </View>
        ))}
        {!isReady ? (
          <ActivityIndicator color={C.primary} />
        ) : isPro ? (
          <Notice tone="good">{t('thanksPro')}</Notice>
        ) : (
          // 価格はストアのローカライズ済み文字列をそのまま出す（コードでは決めない）
          <Btn
            variant="primary"
            icon="star"
            label={priceLabel ? t('buyWithPrice', { price: priceLabel }) : t('buy')}
            busy={isPurchasing}
            disabled={!isNativePurchaseAvailable}
            onPress={purchasePro}
          />
        )}
        <Btn icon="refresh" label={t('restore')} disabled={!isReady || isPurchasing || !isNativePurchaseAvailable} onPress={restorePurchases} />
        {purchaseMessage ? <Notice tone="info">{purchaseMessage}</Notice> : null}
      </Card>

      <Card style={{ gap: 10 }}>
        <Text style={{ fontSize: 16, fontWeight: '800', color: C.text }}>{t('language')}</Text>
        <Text style={{ fontSize: 12, color: C.muted }}>{t('languageHint')}</Text>
        <View style={styles.wrap} accessibilityRole="radiogroup">
          <Chip label={t('followDevice')} selected={choice === 'device'} onPress={() => setChoice('device')} />
          {APP_LANGUAGES.map((l) => (
            // 言語名はその言語自身の表記（どの言語で表示していても自分の言語を見つけられる）
            <Chip key={l} label={LANGUAGE_META[l].endonym} selected={choice === l} onPress={() => setChoice(l)} />
          ))}
        </View>
      </Card>

      <Card style={{ gap: 10 }}>
        <Text style={{ fontSize: 16, fontWeight: '800', color: C.text }}>{t('dataTitle')}</Text>
        <Text style={{ fontSize: 13, color: C.sub, lineHeight: 19 }}>{t('dataBody')}</Text>
        {eraseFailed ? (
          <Notice tone="error" onClose={() => setEraseFailed(false)}>
            {t('eraseFailed')}
          </Notice>
        ) : null}
        <View style={[styles.row, { flexWrap: 'wrap' }]}>
          <Btn small icon="shield-checkmark-outline" label={t('privacyTitle')} onPress={() => router.push('/privacy')} />
          <Btn
            small
            variant="danger"
            icon="trash-outline"
            label={t('clearData')}
            disabled={!problem && saved.list.length === 0}
            onPress={async () => {
              if (await confirmAction(t('clearTitle'), t('clearBody'), t('erase'), t('cancel'))) {
                clearProject()
                setEraseFailed(false)
                // 消し損ねたら知らせる（一覧は残るので、もう一度押せばやり直せる）
                await saved.removeAll().catch(() => setEraseFailed(true))
              }
            }}
          />
        </View>
      </Card>
      <Text style={{ fontSize: 12, color: C.muted, textAlign: 'center' }}>{t('footer')}</Text>
    </Screen>
  )
}
