import { requireOptionalNativeModule } from 'expo'
import * as DocumentPicker from 'expo-document-picker'
import * as FileSystem from 'expo-file-system/legacy'
import * as Sharing from 'expo-sharing'
import { Platform } from 'react-native'
import type { WorkBook } from 'xlsx-js-style'

import { exportUri, removeFile } from './app-files'
import { base64ToArrayBuffer } from './base64'
import { writeXlsx } from './solver'

const XLSX_MIME = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
const GOOGLE_SHEET_MIME = 'application/vnd.google-apps.spreadsheet'

// OS の開く・保存ダイアログを直接呼ぶ自前モジュール（modules/saf-files）。Android のみ。
// 古い開発ビルドなど未搭載のときは null になり、従来の経路に戻る。
const SafFiles =
  Platform.OS === 'android'
    ? requireOptionalNativeModule<{
        openDocumentAsync(mimeTypes: string[]): Promise<{ name: string; base64: string } | null>
        createDocumentAsync(fileName: string, mimeType: string, base64: string): Promise<string | null>
      }>('SafFiles')
    : null

/** 「ファイルとして保存」（Google ドライブ等を選べる保存ダイアログ）が使えるか */
export const canSaveToFile = SafFiles !== null

/** .xlsx（または Google スプレッドシート）を1つ選んで読み込む。キャンセルなら null */
export async function pickXlsx(): Promise<{ data: ArrayBuffer; name: string } | null> {
  if (SafFiles) {
    const f = await SafFiles.openDocumentAsync([XLSX_MIME, 'application/vnd.ms-excel', GOOGLE_SHEET_MIME, 'application/octet-stream'])
    return f ? { data: base64ToArrayBuffer(f.base64), name: f.name } : null
  }
  const res = await DocumentPicker.getDocumentAsync({
    // 端末によっては .xlsx に正しい MIME が付かないので、octet-stream も受け付ける
    type: [XLSX_MIME, 'application/vnd.ms-excel', 'application/octet-stream'],
    copyToCacheDirectory: true,
    multiple: false,
  })
  if (res.canceled || !res.assets?.[0]) return null
  const asset = res.assets[0]
  try {
    const b64 = await FileSystem.readAsStringAsync(asset.uri, { encoding: FileSystem.EncodingType.Base64 })
    return { data: base64ToArrayBuffer(b64), name: asset.name }
  } finally {
    // 名簿（個人情報）のコピーをキャッシュに残さない。読み込んだ中身はメモリと保存データにある
    await removeFile(asset.uri)
  }
}

/** ファイル名に使えない文字を落とす */
export const safeFileName = (s: string) => s.replace(/[\\/:*?"<>|\s]+/g, '_')

/** ブックを OS の保存ダイアログで好きな場所（Google ドライブ等）へ .xlsx として保存する。キャンセルなら false */
export async function saveXlsx(wb: WorkBook, fileName: string): Promise<boolean> {
  if (!SafFiles) return false
  return (await SafFiles.createDocumentAsync(fileName, XLSX_MIME, writeXlsx(wb, 'base64'))) !== null
}

/** ブックを .xlsx としてキャッシュに書き、OS の共有シートを開く（Excel・Google ドライブ・メール等へ） */
export async function shareXlsx(wb: WorkBook, fileName: string, unavailableMessage: string): Promise<void> {
  if (!(await Sharing.isAvailableAsync())) throw new Error(unavailableMessage)
  const uri = await exportUri(fileName)
  try {
    await FileSystem.writeAsStringAsync(uri, writeXlsx(wb, 'base64'), { encoding: FileSystem.EncodingType.Base64 })
    await Sharing.shareAsync(uri, { mimeType: XLSX_MIME, UTI: 'org.openxmlformats.spreadsheetml.sheet', dialogTitle: fileName })
  } finally {
    // 共有シートが閉じたら消す（渡した先のアプリは既に自分の側へコピーしている）
    await removeFile(uri)
  }
}
