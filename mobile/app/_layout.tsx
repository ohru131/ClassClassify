import { Stack } from 'expo-router'
import { StatusBar } from 'expo-status-bar'
import { SafeAreaProvider } from 'react-native-safe-area-context'

import { LanguageProvider, useI18n } from '@/lib/language-provider'
import { ProjectProvider } from '@/lib/project-store'
import { ProProvider } from '@/lib/pro-provider'
import { SavedResultsProvider } from '@/lib/saved-results-store'

export default function RootLayout() {
  return (
    <SafeAreaProvider>
      <LanguageProvider>
        <ProProvider>
          <ProjectProvider>
            <SavedResultsProvider>
              <RootStack />
            </SavedResultsProvider>
          </ProjectProvider>
        </ProProvider>
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
