import Ionicons from '@expo/vector-icons/Ionicons'
import { Tabs } from 'expo-router'
import { View } from 'react-native'
import { useSafeAreaInsets } from 'react-native-safe-area-context'

import { C } from '@/components/theme'
import { useI18n } from '@/lib/language-provider'
import { useLayout } from '@/lib/layout'
import { useProject } from '@/lib/project-store'

export default function TabsLayout() {
  const insets = useSafeAreaInsets()
  const { isWide } = useLayout()
  const { report } = useProject()
  const { t } = useI18n()
  return (
    <View style={{ flex: 1, paddingTop: insets.top, backgroundColor: C.bg }}>
      <Tabs
        screenOptions={{
          headerShown: false,
          tabBarActiveTintColor: C.primary,
          tabBarInactiveTintColor: C.muted,
          // 広い画面ではアイコンとラベルを横に並べる
          tabBarLabelPosition: isWide ? 'beside-icon' : 'below-icon',
          // 既定の高さ（49dp）はラベル 10px 前提で、12px 太字のラベルだと下が欠ける（Web 版のストア用
          // スクリーンショットで判明）。高さを指定すると下端の安全領域は自動で足されないので自分で足す
          tabBarStyle: { backgroundColor: '#fff', borderTopColor: C.border, height: 58 + insets.bottom },
          tabBarLabelStyle: { fontSize: 12, fontWeight: '700' },
        }}
      >
        <Tabs.Screen name="index" options={{ title: t('tabRoster'), tabBarIcon: ({ color, size }) => <Ionicons name="people" color={color} size={size} /> }} />
        <Tabs.Screen name="run" options={{ title: t('tabRun'), tabBarIcon: ({ color, size }) => <Ionicons name="options" color={color} size={size} /> }} />
        <Tabs.Screen
          name="results"
          options={{
            title: t('tabResults'),
            tabBarIcon: ({ color, size }) => <Ionicons name="grid" color={color} size={size} />,
            tabBarBadge: report && report.violations.length ? report.violations.length : undefined,
          }}
        />
        <Tabs.Screen name="pro" options={{ title: t('tabPro'), tabBarIcon: ({ color, size }) => <Ionicons name="star" color={color} size={size} /> }} />
      </Tabs>
    </View>
  )
}
