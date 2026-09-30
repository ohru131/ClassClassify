import { ScrollView, Text, View } from 'react-native'

import { C } from '@/components/theme'
import { PRIVACY } from '@/lib/copy/privacy'
import { useI18n } from '@/lib/language-provider'
import { useLayout } from '@/lib/layout'

// 本文は lib/copy/privacy.ts（6言語）。実態と食い違わないこと
export default function PrivacyScreen() {
  const { contentMaxWidth } = useLayout()
  const { lang, t } = useI18n()
  return (
    <ScrollView style={{ flex: 1, backgroundColor: C.bg }} contentContainerStyle={{ padding: 16, paddingBottom: 40 }}>
      <View style={{ width: '100%', maxWidth: contentMaxWidth ?? 720, alignSelf: 'center', gap: 18 }}>
        <Text style={{ fontSize: 22, fontWeight: '900', color: C.text }}>{t('privacyTitle')}</Text>
        {PRIVACY[lang].map((s) => (
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
