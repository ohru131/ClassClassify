// react-native-purchases を読み込まない純関数（テストから読めるように分けてある）
/**
 * 動作確認・スクリーンショット用: **Web 版に限り** URL に ?pro=preview を付けると Pro の画面を
 * 表示する（購入はできず、課金も起きない）。ネイティブでは Platform.OS で必ず無効。
 */
export function isWebProPreview(platformOS: string, search: string | undefined): boolean {
  return platformOS === 'web' && typeof search === 'string' && new URLSearchParams(search).get('pro') === 'preview'
}
