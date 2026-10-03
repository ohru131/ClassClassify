import * as DocumentPicker from 'expo-document-picker'
import type { WorkBook } from 'xlsx-js-style'

import { base64ToArrayBuffer } from './base64'
import { safeFileName, writeXlsx } from './solver'

const XLSX_MIME = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

export async function pickXlsx(): Promise<{ data: ArrayBuffer; name: string } | null> {
  const res = await DocumentPicker.getDocumentAsync({ type: [XLSX_MIME, '.xlsx'], multiple: false })
  if (res.canceled || !res.assets?.[0]) return null
  const asset = res.assets[0]
  const data = asset.file ? await asset.file.arrayBuffer() : await (await fetch(asset.uri)).arrayBuffer()
  return { data, name: asset.name }
}

export const canSaveToFile = false
export const saveXlsx = async (_wb: WorkBook, _fileName: string): Promise<boolean> => false
export const saveBase64As = async (_fileName: string, _mimeType: string, _base64: string): Promise<boolean> => false

export { safeFileName } from './solver'

/** Web（動作確認用）はダウンロードする */
export const shareXlsx = async (wb: WorkBook, fileName: string, _unavailableMessage?: string): Promise<void> => download(new Blob([writeXlsx(wb, 'array')], { type: XLSX_MIME }), fileName)

export const shareXlsxBase64 = async (base64: string, fileName: string, _unavailableMessage?: string): Promise<void> =>
  download(new Blob([base64ToArrayBuffer(base64)], { type: XLSX_MIME }), fileName)

function download(blob: Blob, fileName: string) {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = safeFileName(fileName)
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(a.href), 1000)
}
