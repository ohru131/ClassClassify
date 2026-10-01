import AsyncStorage from '@react-native-async-storage/async-storage'
import { useLocales } from 'expo-localization'
import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useState } from 'react'

import { isAppLanguage, resolveDeviceLanguage } from './i18n'
import { makeI18n, type I18n, type LanguageChoice } from './i18n-core'

export type { I18n, LanguageChoice }

// 旧名 Mosaic のままにしてある（変えると保存済みの言語設定を読めなくなる）
const STORAGE_KEY = 'mosaic.language.v1'
const Ctx = createContext<I18n | null>(null)

/** 表示言語。既定は端末の言語（未対応なら英語）。設定で選んだら端末に保存する */
export function LanguageProvider({ children }: { children: ReactNode }) {
  const locales = useLocales()
  const deviceLang = resolveDeviceLanguage(locales)
  const [choice, setChoiceState] = useState<LanguageChoice>('device')

  useEffect(() => {
    AsyncStorage.getItem(STORAGE_KEY)
      .then((v) => {
        if (v && (v === 'device' || isAppLanguage(v))) setChoiceState(v)
      })
      .catch(() => undefined)
  }, [])

  const setChoice = useCallback((c: LanguageChoice) => {
    setChoiceState(c)
    void AsyncStorage.setItem(STORAGE_KEY, c).catch(() => undefined)
  }, [])

  const lang = choice === 'device' ? deviceLang : choice
  const value = useMemo(() => makeI18n(lang, choice, setChoice), [lang, choice, setChoice])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useI18n(): I18n {
  const v = useContext(Ctx)
  if (!v) throw new Error('LanguageProvider の内部で使用してください。')
  return v
}
