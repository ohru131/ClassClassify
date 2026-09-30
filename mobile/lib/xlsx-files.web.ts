import * as DocumentPicker from 'expo-document-picker'
import type { WorkBook } from 'xlsx-js-style'

import { writeXlsx } from './solver'

const XLSX_MIME = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

export async function pickXlsx(): Promise<{ data: ArrayBuffer; name: string } | null> {
  const res = await DocumentPicker.getDocumentAsync({ type: [XLSX_MIME, '.xlsx'], multiple: false })
  if (res.canceled || !res.assets?.[0]) return null
  const asset = res.assets[0]
  const data = asset.file ? await asset.file.arrayBuffer() : await (await fetch(asset.uri)).arrayBuffer()
  return { data, name: asset.name }
}

export const safeFileName = (s: string) => s.replace(/[\\/:*?"<>|\s]+/g, '_')

/** Web（動作確認用）はダウンロードする */
export async function shareXlsx(wb: WorkBook, fileName: string, _unavailableMessage?: string): Promise<void> {
  const blob = new Blob([writeXlsx(wb, 'array')], { type: XLSX_MIME })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = safeFileName(fileName)
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(a.href), 1000)
}
