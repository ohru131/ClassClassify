import { createContext, type ReactNode, useContext, useEffect, useMemo, useState } from 'react'
import { Platform } from 'react-native'

import { initializeMobileAds } from './ads-native-init'
import { usePro } from './revenuecat-provider'

type AdsContextValue = {
  /** いまバナーを出してよいか（iOS/Android・Pro 復元済み・Pro でない・同意取得と SDK 初期化済み） */
  isBannerVisible: boolean
}

const AdsContext = createContext<AdsContextValue | null>(null)

export function AdsProvider({ children }: { children: ReactNode }) {
  const { isPro, isReady } = usePro()
  const platformOk = Platform.OS === 'ios' || Platform.OS === 'android'
  const [canRequestAds, setCanRequestAds] = useState(false)

  useEffect(() => {
    // Pro と確定した人には同意フォームも SDK 初期化も行わない
    if (!platformOk || !isReady || isPro) return
    let active = true
    void initializeMobileAds().then((ok) => {
      if (active) setCanRequestAds(ok)
    })
    return () => {
      active = false
    }
  }, [isPro, isReady, platformOk])

  const value = useMemo(() => ({ isBannerVisible: platformOk && isReady && !isPro && canRequestAds }), [platformOk, isReady, isPro, canRequestAds])
  return <AdsContext.Provider value={value}>{children}</AdsContext.Provider>
}

export function useAds() {
  const v = useContext(AdsContext)
  if (!v) throw new Error('AdsProvider の内部で使用してください。')
  return v
}
