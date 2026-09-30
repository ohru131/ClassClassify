import * as FileSystem from 'expo-file-system/legacy'

// 書き出したファイルは cacheDirectory 直下ではなく専用のサブディレクトリに置く。
// 「データを消去」でこのディレクトリごと消せば、自アプリが作ったものだけを確実に掃除できる。
const EXPORT_DIR = `${FileSystem.cacheDirectory}exports/`
// expo-document-picker が copyToCacheDirectory で名簿をコピーする先
const PICKER_DIR = `${FileSystem.cacheDirectory}DocumentPicker/`

/**
 * 書き出し先の file:// URI。表示名はそのまま共有シートの題に使い、パスの方は
 * encodeURIComponent で組む（日本語や # ? % を含む名前が URI として壊れないように）。
 */
export async function exportUri(fileName: string): Promise<string> {
  await FileSystem.makeDirectoryAsync(EXPORT_DIR, { intermediates: true }).catch(() => {})
  const safe = fileName.replace(/[\\/:*?"<>|\s]+/g, '_')
  return `${EXPORT_DIR}${encodeURIComponent(safe)}`
}

/**
 * 1つ消す。失敗しても止めない（掃除のために本来の操作を失敗させない）。
 * 自アプリのキャッシュの中だけを消す（ピッカーが content:// の元ファイルを返した場合などに触らない）。
 */
export async function removeFile(uri: string | null | undefined): Promise<void> {
  if (!uri || !FileSystem.cacheDirectory || !uri.startsWith(FileSystem.cacheDirectory)) return
  await FileSystem.deleteAsync(uri, { idempotent: true }).catch(() => {})
}

/** 「データを消去」のとき: 書き出したファイルとピッカーのコピーをまとめて消す */
export async function clearAppFiles(): Promise<void> {
  await Promise.all([removeFile(EXPORT_DIR), removeFile(PICKER_DIR)])
}
