import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

import { LANGUAGE_STORAGE_KEY, makeWebI18n, resolveInitialLanguage, type WebI18n } from '../copy/core'
import type { AppLanguage } from './languages'

type Ctx = WebI18n & { setLang: (l: AppLanguage) => void }
const I18nContext = createContext<Ctx | null>(null)

const readStored = () => {
  try {
    return localStorage.getItem(LANGUAGE_STORAGE_KEY)
  } catch {
    return null
  }
}

/** Web 版の表示言語。<html lang>・タイトル・meta description も言語に合わせる */
export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<AppLanguage>(() => resolveInitialLanguage(window.location.search, readStored(), navigator.languages ?? [navigator.language]))

  const setLang = useCallback((l: AppLanguage) => {
    setLangState(l)
    try {
      localStorage.setItem(LANGUAGE_STORAGE_KEY, l)
    } catch {
      // 保存できなくても表示は切り替える
    }
    // URL に ?lang= があれば合わせて書き換える（再読み込みしても選んだ言語のまま）
    const url = new URL(window.location.href)
    if (url.searchParams.has('lang')) {
      url.searchParams.set('lang', l)
      window.history.replaceState(null, '', url)
    }
  }, [])

  const value = useMemo(() => ({ ...makeWebI18n(lang), setLang }), [lang, setLang])

  useEffect(() => {
    document.documentElement.lang = lang
    document.title = value.t('docTitle')
    document.querySelector('meta[name="description"]')?.setAttribute('content', value.t('metaDescription'))
  }, [lang, value])

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useT(): Ctx {
  const v = useContext(I18nContext)
  if (!v) throw new Error('I18nProvider の内部で使用してください。')
  return v
}
