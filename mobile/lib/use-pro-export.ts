import { useRouter } from 'expo-router'
import { useCallback, useState } from 'react'
import type { WorkBook } from 'xlsx-js-style'

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
 */
export function useProExport() {
  const { isPro } = usePro()
  const { setError } = useProject()
  const router = useRouter()
  const [busy, setBusy] = useState<null | 'xlsx' | 'print' | 'pdf'>(null)

  const run = useCallback(
    (kind: 'xlsx' | 'print' | 'pdf', task: () => Promise<void>) => {
      if (!isPro) {
        router.navigate('/pro')
        return
      }
      setBusy(kind)
      task()
        .catch((e) => setError(e instanceof Error ? e.message : String(e)))
        .finally(() => setBusy(null))
    },
    [isPro, router, setError],
  )

  const exportXlsx = useCallback((build: () => WorkBook, baseName: string) => run('xlsx', () => shareXlsx(build(), `${baseName}_${today()}.xlsx`)), [run])
  const print = useCallback((build: () => string) => run('print', () => printHtml(build())), [run])
  const exportPdf = useCallback((build: () => string, baseName: string) => run('pdf', () => sharePdf(build(), `${baseName}_${today()}.pdf`)), [run])

  return { exportXlsx, print, exportPdf, busy, isPro }
}
