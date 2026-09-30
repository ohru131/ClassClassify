// react-native-google-mobile-ads はネイティブ専用で、import するだけで Web バンドルが壊れる
export async function initializeMobileAds(): Promise<boolean> {
  return false
}
