import { Stack } from 'expo-router'
import { StatusBar } from 'expo-status-bar'
import { SafeAreaProvider } from 'react-native-safe-area-context'

import { LanguageProvider, useI18n } from '@/lib/language-provider'
import { ProjectProvider } from '@/lib/project-store'
import { RevenueCatProvider } from '@/lib/revenuecat-provider'
import { SavedResultsProvider } from '@/lib/saved-results-store'

export default function RootLayout() {
  return (
    <SafeAreaProvider>
      <LanguageProvider>
        <RevenueCatProvider>
          <ProjectProvider>
            <SavedResultsProvider>
              <RootStack />
            </SavedResultsProvider>
          </ProjectProvider>
        </RevenueCatProvider>
      </LanguageProvider>
    </SafeAreaProvider>
  )
}

function RootStack() {
  const { t } = useI18n()
  return (
    <>
      <Stack screenOptions={{ headerShown: false }}>
        <Stack.Screen name="(tabs)" />
        <Stack.Screen name="privacy" options={{ headerShown: true, title: t('privacyTitle') }} />
      </Stack>
      <StatusBar style="dark" />
    </>
  )
}
