import * as FileSystem from 'expo-file-system/legacy'
import * as Print from 'expo-print'
import * as Sharing from 'expo-sharing'

import { safeFileName } from './xlsx-files'

// A4 縦（ポイント）。expo-print の既定は US Letter
const A4 = { width: 595, height: 842 }

/** OS の印刷画面を開く */
export async function printHtml(html: string): Promise<void> {
  await Print.printAsync({ html, ...A4 })
}

/** PDF にして共有シートを開く */
export async function sharePdf(html: string, fileName: string): Promise<void> {
  if (!(await Sharing.isAvailableAsync())) throw new Error('この端末ではファイルの共有を利用できません。')
  const { uri } = await Print.printToFileAsync({ html, ...A4 })
  const dest = `${FileSystem.cacheDirectory}${safeFileName(fileName)}`
  await FileSystem.deleteAsync(dest, { idempotent: true })
  await FileSystem.moveAsync({ from: uri, to: dest })
  await Sharing.shareAsync(dest, { mimeType: 'application/pdf', UTI: 'com.adobe.pdf', dialogTitle: fileName })
}

export const canSharePdf = true
