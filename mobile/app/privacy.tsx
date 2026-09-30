import { ScrollView, Text, View } from 'react-native'

import { C } from '@/components/theme'
import { useLayout } from '@/lib/layout'

// 実態と食い違わないこと。「何も一切送らない」とは書かない（購入の確認のため RevenueCat が
// 匿名の識別子とレシートを受け取る）。変えたらストアのデータセーフティの申告も直す。
const SECTIONS: { title: string; body: string[] }[] = [
  {
    title: '名簿・編成結果',
    body: [
      '読み込んだ名簿（生徒の名前・特性・ペア指定）と編成結果は、この端末の中（アプリの保存領域）だけに保存します。開発者のサーバーやその他の外部へ送信することはありません。',
      '名簿の最適化（クラス編成の計算）も端末の中で行います。',
      'Excel ファイルとして書き出した場合、そのファイルを送る先（Excel、Google ドライブ、メールなど）は OS の共有画面で利用者が選びます。送信先での取り扱いは各サービスの規約に従います。',
      '「Pro・設定」タブの「この端末の名簿と結果を消去」、またはアプリの削除で、保存したデータを消去できます。',
    ],
  },
  {
    title: '購入（Pro）と外部への送信',
    body: [
      '広告は表示しません（広告 SDK を組み込んでいません）。',
      'Pro は買い切り（非消費型）の購入です。決済は Apple App Store または Google Play が行い、購入状態の管理に RevenueCat を利用しています。',
      'RevenueCat は、購入を確認するために端末で生成された匿名の識別子と、購入・レシートの情報を受け取ります。このアプリから名前やメールアドレス、名簿の内容を送ることはありません。購入の確認以外の目的（広告・行動分析など）で外部へ情報を送ることもありません。',
      'RevenueCat のプライバシーポリシー: https://www.revenuecat.com/privacy',
    ],
  },
  {
    title: 'お問い合わせ',
    body: ['ご質問は GitHub（https://github.com/ohru131/ClassClassify）の Issues からお寄せください。'],
  },
]

export default function PrivacyScreen() {
  const { contentMaxWidth } = useLayout()
  return (
    <ScrollView style={{ flex: 1, backgroundColor: C.bg }} contentContainerStyle={{ padding: 16, paddingBottom: 40 }}>
      <View style={{ width: '100%', maxWidth: contentMaxWidth ?? 720, alignSelf: 'center', gap: 18 }}>
        <Text style={{ fontSize: 22, fontWeight: '900', color: C.text }}>プライバシーポリシー</Text>
        {SECTIONS.map((s) => (
          <View key={s.title} style={{ gap: 6 }}>
            <Text style={{ fontSize: 16, fontWeight: '800', color: C.text }} accessibilityRole="header">
              {s.title}
            </Text>
            {s.body.map((b, i) => (
              <Text key={i} style={{ fontSize: 14, color: C.sub, lineHeight: 21 }} selectable>
                {b}
              </Text>
            ))}
          </View>
        ))}
      </View>
    </ScrollView>
  )
}
