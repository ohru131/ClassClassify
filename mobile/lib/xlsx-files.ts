import * as DocumentPicker from 'expo-document-picker'
import * as FileSystem from 'expo-file-system/legacy'
import * as Sharing from 'expo-sharing'
import type { WorkBook } from 'xlsx-js-style'

import { base64ToArrayBuffer } from './base64'
import { writeXlsx } from './solver'

const XLSX_MIME = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

/** .xlsx を1つ選んで読み込む。キャンセルなら null */
export async function pickXlsx(): Promise<{ data: ArrayBuffer; name: string } | null> {
  const res = await DocumentPicker.getDocumentAsync({
    // 端末によっては .xlsx に正しい MIME が付かないので、octet-stream も受け付ける
    type: [XLSX_MIME, 'application/vnd.ms-excel', 'application/octet-stream'],
    copyToCacheDirectory: true,
    multiple: false,
  })
  if (res.canceled || !res.assets?.[0]) return null
  const asset = res.assets[0]
  const b64 = await FileSystem.readAsStringAsync(asset.uri, { encoding: FileSystem.EncodingType.Base64 })
  return { data: base64ToArrayBuffer(b64), name: asset.name }
}

/** ファイル名に使えない文字を落とす */
export const safeFileName = (s: string) => s.replace(/[\\/:*?"<>|\s]+/g, '_')

/** ブックを .xlsx としてキャッシュに書き、OS の共有シートを開く（Excel・Google ドライブ・メール等へ） */
export async function shareXlsx(wb: WorkBook, fileName: string, unavailableMessage: string): Promise<void> {
  if (!(await Sharing.isAvailableAsync())) throw new Error(unavailableMessage)
  const uri = `${FileSystem.cacheDirectory}${safeFileName(fileName)}`
  await FileSystem.writeAsStringAsync(uri, writeXlsx(wb, 'base64'), { encoding: FileSystem.EncodingType.Base64 })
  await Sharing.shareAsync(uri, { mimeType: XLSX_MIME, UTI: 'org.openxmlformats.spreadsheetml.sheet', dialogTitle: fileName })
}
