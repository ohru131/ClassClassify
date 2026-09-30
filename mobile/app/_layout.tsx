import { Stack } from 'expo-router'
import { StatusBar } from 'expo-status-bar'
import { SafeAreaProvider } from 'react-native-safe-area-context'

import { ProjectProvider } from '@/lib/project-store'
import { RevenueCatProvider } from '@/lib/revenuecat-provider'

export default function RootLayout() {
  return (
    <SafeAreaProvider>
      <RevenueCatProvider>
        <ProjectProvider>
          <Stack screenOptions={{ headerShown: false }}>
            <Stack.Screen name="(tabs)" />
            <Stack.Screen name="privacy" options={{ headerShown: true, title: 'プライバシーポリシー' }} />
          </Stack>
          <StatusBar style="dark" />
        </ProjectProvider>
      </RevenueCatProvider>
    </SafeAreaProvider>
  )
}
