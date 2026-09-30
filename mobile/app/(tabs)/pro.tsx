import { useRouter } from 'expo-router'
import { ActivityIndicator, Text, View } from 'react-native'

import { C } from '@/components/theme'
import { Btn, Card, Notice, Screen, styles } from '@/components/ui'
import { confirmAction } from '@/lib/confirm'
import { useProject } from '@/lib/project-store'
import { usePro } from '@/lib/revenuecat-provider'

const FEATURES = [
  { title: '広告を非表示', body: '画面上部のバナー広告が出なくなります。' },
  { title: '結果を Excel で共有', body: '組分け・クラス別名簿・各組・ペア指定・集計のシートを .xlsx で書き出し、Excel・Google ドライブ・メールなどへ送れます。' },
  { title: '名簿を Excel で保存', body: '編集した名簿をひな形と同じ形式で保存できます（Web 版でもそのまま読み込めます）。' },
]

export default function ProScreen() {
  const { isPro, isReady, purchaseMessage, priceLabel, isPurchasing, purchasePro, restorePurchases, isNativePurchaseAvailable } = usePro()
  const { problem, clearProject } = useProject()
  const router = useRouter()

  return (
    <Screen>
      <Text style={{ fontSize: 24, fontWeight: '900', color: C.text }}>Pro・設定</Text>
      <Card style={{ gap: 12 }}>
        <View style={[styles.row, { justifyContent: 'space-between' }]}>
          <Text style={{ fontSize: 20, fontWeight: '900', color: C.text }}>Mosaic Pro</Text>
          {isPro ? <Text style={{ color: C.good, fontWeight: '800' }}>✓ 利用中</Text> : null}
        </View>
        <Text style={{ fontSize: 13, color: C.sub }}>買い切り（1回のお支払い）です。定期購入（サブスクリプション）ではないので、継続して請求されることはありません。</Text>
        {FEATURES.map((f) => (
          <View key={f.title} style={{ flexDirection: 'row', gap: 10 }}>
            <Text style={{ color: C.primary, fontWeight: '900' }}>✓</Text>
            <View style={{ flex: 1 }}>
              <Text style={{ fontWeight: '800', color: C.text }}>{f.title}</Text>
              <Text style={{ fontSize: 13, color: C.sub }}>{f.body}</Text>
            </View>
          </View>
        ))}
        {!isReady ? (
          <ActivityIndicator color={C.primary} />
        ) : isPro ? (
          <Notice tone="good">Pro をご利用いただきありがとうございます。すべての機能が使えます。</Notice>
        ) : (
          <Btn
            variant="primary"
            icon="star"
            label={priceLabel ? `Pro を購入（${priceLabel}・買い切り）` : 'Pro を購入（買い切り）'}
            busy={isPurchasing}
            disabled={!isNativePurchaseAvailable}
            onPress={purchasePro}
          />
        )}
        <Btn icon="refresh" label="購入を復元" disabled={!isReady || isPurchasing || !isNativePurchaseAvailable} onPress={restorePurchases} />
        {purchaseMessage ? <Notice tone="info">{purchaseMessage}</Notice> : null}
      </Card>

      <Card style={{ gap: 10 }}>
        <Text style={{ fontSize: 16, fontWeight: '800', color: C.text }}>データについて</Text>
        <Text style={{ fontSize: 13, color: C.sub, lineHeight: 19 }}>
          名簿と編成結果はこの端末の中だけに保存され、どこにも送信されません。書き出した Excel をどこへ送るかは、共有先をご自身で選べます。
        </Text>
        <View style={[styles.row, { flexWrap: 'wrap' }]}>
          <Btn small icon="shield-checkmark-outline" label="プライバシーポリシー" onPress={() => router.push('/privacy')} />
          <Btn
            small
            variant="danger"
            icon="trash-outline"
            label="この端末の名簿と結果を消去"
            disabled={!problem}
            onPress={async () => {
              if (await confirmAction('名簿と結果を消去', 'この端末に保存している名簿と編成結果を削除します。元に戻せません。', '消去')) clearProject()
            }}
          />
        </View>
      </Card>
      <Text style={{ fontSize: 12, color: C.muted, textAlign: 'center' }}>Mosaic · クラス編成オプティマイザー（Web 版は無料・https://ohru131.github.io/ClassClassify/）</Text>
    </Screen>
  )
}
