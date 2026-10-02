import * as FileSystem from 'expo-file-system/legacy'
import * as Print from 'expo-print'
import * as Sharing from 'expo-sharing'

import { exportUri, removeFile } from './app-files'
import { saveBase64As } from './xlsx-files'

// A4 縦（ポイント）。expo-print の既定は US Letter
const A4 = { width: 595, height: 842 }

/** OS の印刷画面を開く */
export async function printHtml(html: string, _messages?: { popupBlocked: string }): Promise<void> {
  await Print.printAsync({ html, ...A4 })
}

/** PDF にして共有シートを開く */
export async function sharePdf(html: string, fileName: string, messages: { sharingUnavailable: string; popupBlocked?: string }): Promise<void> {
  if (!(await Sharing.isAvailableAsync())) throw new Error(messages.sharingUnavailable)
  const { uri } = await Print.printToFileAsync({ html, ...A4 })
  const dest = await exportUri(fileName)
  try {
    await FileSystem.deleteAsync(dest, { idempotent: true })
    await FileSystem.moveAsync({ from: uri, to: dest })
    await Sharing.shareAsync(dest, { mimeType: 'application/pdf', UTI: 'com.adobe.pdf', dialogTitle: fileName })
  } finally {
    // 共有が終わったら PDF を残さない（移動に失敗したときは元の一時ファイルも消す）
    await Promise.all([removeFile(dest), removeFile(uri)])
  }
}

/** PDF にして OS の保存ダイアログ（Google ドライブ等）で保存する。キャンセルなら false */
export async function savePdf(html: string, fileName: string): Promise<boolean> {
  const { uri } = await Print.printToFileAsync({ html, ...A4 })
  try {
    const b64 = await FileSystem.readAsStringAsync(uri, { encoding: FileSystem.EncodingType.Base64 })
    return await saveBase64As(fileName, 'application/pdf', b64)
  } finally {
    await removeFile(uri)
  }
}

export const canSharePdf = true
