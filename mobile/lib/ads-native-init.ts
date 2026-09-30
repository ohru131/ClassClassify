import mobileAds, { AdsConsent } from 'react-native-google-mobile-ads'

/**
 * iOS/Android のみ（Web は ads-native-init.web.ts に差し替わる）。
 * 同意が必要な地域では同意フォームを出し、広告リクエストが許可されるまで SDK を初期化しない。
 */
export async function initializeMobileAds(): Promise<boolean> {
  try {
    const consent = await AdsConsent.gatherConsent()
    if (!consent.canRequestAds) return false
    await mobileAds().initialize()
    return true
  } catch {
    return false
  }
}
