import AsyncStorage from '@react-native-async-storage/async-storage'
import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'

import { isSavedMetaList, isSavedResult, metaOf, newSavedId, type SavedMeta, type SavedResult } from './saved-results'
import type { Problem } from './solver'

// 名前を付けて保存した編成の置き場所（端末内の AsyncStorage。Android では中身が SQLite）。
// 一覧（名前・日時・人数・組数）は1つのキーにまとめ、名簿と結果の本体は1件ずつ別のキーに置く
// （一覧を出すたびに全件の名簿を読まないように）。どこにも送信しない。
const INDEX_KEY = 'fairclass.saved.index.v1'
const itemKey = (id: string) => `fairclass.saved.v1.${id}`

type SavedContextValue = {
  /** 新しい順 */
  list: SavedMeta[]
  loaded: boolean
  save: (name: string, problem: Problem, classOf: number[], k: number) => Promise<SavedMeta>
  load: (id: string) => Promise<SavedResult | null>
  remove: (id: string) => Promise<void>
  removeAll: () => Promise<void>
}

const SavedContext = createContext<SavedContextValue | null>(null)

const byNewest = (a: SavedMeta, b: SavedMeta) => b.savedAt.localeCompare(a.savedAt)

export function SavedResultsProvider({ children }: { children: ReactNode }) {
  const [list, setListState] = useState<SavedMeta[]>([])
  const [loaded, setLoaded] = useState(false)
  // 書き込みは最新の一覧から作る（setState の更新関数は後で呼ばれるので、そこから値を取り出さない）
  const listRef = useRef<SavedMeta[]>([])
  const setList = (next: SavedMeta[]) => {
    listRef.current = next
    setListState(next)
  }

  useEffect(() => {
    let active = true
    AsyncStorage.getItem(INDEX_KEY)
      .then((raw) => {
        if (!active || !raw) return
        const data: unknown = JSON.parse(raw)
        if (isSavedMetaList(data)) setList([...data].sort(byNewest))
      })
      .catch(() => undefined)
      .finally(() => {
        if (active) setLoaded(true)
      })
    return () => {
      active = false
    }
  }, [])

  const writeIndex = (next: SavedMeta[]) => AsyncStorage.setItem(INDEX_KEY, JSON.stringify(next))

  const save = useCallback(async (name: string, problem: Problem, classOf: number[], k: number) => {
    const now = new Date()
    const item: SavedResult = { version: 1, id: newSavedId(now), name, savedAt: now.toISOString(), problem, classOf, k }
    // 本体を先に書く（一覧だけあって本体が無い状態を作らない）
    await AsyncStorage.setItem(itemKey(item.id), JSON.stringify(item))
    const meta = metaOf(item)
    const next = [meta, ...listRef.current].sort(byNewest)
    setList(next)
    await writeIndex(next)
    return meta
  }, [])

  const load = useCallback(async (id: string) => {
    const raw = await AsyncStorage.getItem(itemKey(id))
    if (!raw) return null
    const data: unknown = JSON.parse(raw)
    return isSavedResult(data) ? data : null
  }, [])

  const remove = useCallback(async (id: string) => {
    const next = listRef.current.filter((m) => m.id !== id)
    setList(next)
    await writeIndex(next)
    await AsyncStorage.removeItem(itemKey(id))
  }, [])

  const removeAll = useCallback(async () => {
    const keys = (await AsyncStorage.getAllKeys()).filter((key) => key === INDEX_KEY || key.startsWith('fairclass.saved.v1.'))
    setList([])
    await AsyncStorage.multiRemove(keys)
  }, [])

  const value = useMemo(() => ({ list, loaded, save, load, remove, removeAll }), [list, loaded, save, load, remove, removeAll])
  return <SavedContext.Provider value={value}>{children}</SavedContext.Provider>
}

export function useSavedResults() {
  const v = useContext(SavedContext)
  if (!v) throw new Error('SavedResultsProvider の内部で使用してください。')
  return v
}
