import { useRouter } from 'expo-router'
import { useRef, useState } from 'react'
import type { WorkBook } from 'xlsx-js-style'

import { useI18n } from './language-provider'
import { printHtml, sharePdf } from './print'
import { useProject } from './project-store'
import { usePro } from './revenuecat-provider'
import { shareXlsx } from './xlsx-files'

const today = () => {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
}

/**
 * Excel・印刷・PDF（Pro 限定）。無料版でもボタンは見せ、押すと Pro の画面へ案内する。
 * 中身（ブック・HTML）は押したときに作る。Web の印刷は window.open をクリックと同じ流れで
 * 呼ばないとポップアップとして止められるので、await を挟まずに呼び出す。
 * ファイル名とシート名は選択中の言語で出す。
 */
export function useProExport() {
  const { isPro } = usePro()
  const { setError } = useProject()
  const { t } = useI18n()
  const router = useRouter()
  const [busy, setBusy] = useState<null | 'xlsx' | 'print' | 'pdf'>(null)
  // busy は再描画後の値なので、同じフレームの2回目のタップはすり抜ける。同期のロックで止める
  const lockRef = useRef(false)
  const messages = { sharingUnavailable: t('sharingUnavailable'), popupBlocked: t('popupBlocked') }

  const run = (kind: 'xlsx' | 'print' | 'pdf', task: () => Promise<void>) => {
    if (!isPro) {
      router.navigate('/pro')
      return
    }
    if (lockRef.current) return
    lockRef.current = true
    setBusy(kind)
    // task() は同期で呼ぶ（Web の印刷をタップと同じ流れに保つ）。同期的に投げてもロックは必ず外す
    let pending: Promise<void>
    try {
      pending = task()
    } catch (e) {
      pending = Promise.reject(e)
    }
    pending
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => {
        lockRef.current = false
        setBusy(null)
      })
  }

  const exportXlsx = (build: () => WorkBook, baseName: string) => run('xlsx', () => shareXlsx(build(), `${baseName}_${today()}.xlsx`, messages.sharingUnavailable))
  const print = (build: () => string) => run('print', () => printHtml(build(), messages))
  const exportPdf = (build: () => string, baseName: string) => run('pdf', () => sharePdf(build(), `${baseName}_${today()}.pdf`, messages))

  return { exportXlsx, print, exportPdf, busy, isPro }
}
