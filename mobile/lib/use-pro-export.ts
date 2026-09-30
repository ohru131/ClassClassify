import { useRouter } from 'expo-router'
import { useCallback, useState } from 'react'
import type { WorkBook } from 'xlsx-js-style'

import { useProject } from './project-store'
import { usePro } from './revenuecat-provider'
import { shareXlsx } from './xlsx-files'

const today = () => {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
}

/**
 * Excel での書き出し（Pro 限定）。無料版でもボタンは見せ、押すと Pro の画面へ案内する。
 * ブックは押したときに作る（大きい名簿で毎レンダー作らないため）。
 */
export function useProExport() {
  const { isPro } = usePro()
  const { setError } = useProject()
  const router = useRouter()
  const [busy, setBusy] = useState(false)
  const exportXlsx = useCallback(
    async (build: () => WorkBook, baseName: string) => {
      if (!isPro) {
        router.navigate('/pro')
        return
      }
      setBusy(true)
      try {
        await shareXlsx(build(), `${baseName}_${today()}.xlsx`)
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      } finally {
        setBusy(false)
      }
    },
    [isPro, router, setError],
  )
  return { exportXlsx, busy, isPro }
}
