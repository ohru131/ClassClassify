import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { Platform } from 'react-native'

import { useI18n } from './language-provider'
import { hasPendingPro, ownsPro, PRO_PRODUCT_ID, purchaseErrorKind, unacknowledgedPro, type PurchaseLike } from './play-billing'
import { isWebProPreview } from './pro-preview'
import { resolvePurchaseMessageKey } from './purchase-message'

// Pro（買い切り）を Google Play Billing で直接扱う（expo-iap）。外部の課金サービスは使わず、
// 購入の状態は端末の Play ストアに問い合わせるだけで、アプリからどこにも送信しない。
// 判定は lib/play-billing.ts（買い切り1本・consume しない・acknowledge する）。

type PurchaseMessageKey =
  | 'purchaseStoreOnly'
  | 'purchaseStatusFailed'
  | 'purchaseSucceeded'
  | 'purchasePending'
  | 'purchaseFailed'
  | 'productLoadFailed'
  | 'proRestored'
  | 'noRestorablePurchase'
  | 'restoreFailed'

type ProContextValue = {
  isPro: boolean
  /** Pro 状態の復元が終わったか */
  isReady: boolean
  isNativePurchaseAvailable: boolean
  purchaseMessage: string | null
  priceLabel: string | null
  isPurchasing: boolean
  purchasePro: () => Promise<void>
  restorePurchases: () => Promise<void>
}

const ProContext = createContext<ProContextValue | null>(null)

// ネイティブモジュールは Android のときだけ読み込む（Web 版の画面確認では読まない）
type Iap = typeof import('expo-iap')
let iapPromise: Promise<Iap> | null = null
const loadIap = () => (iapPromise ??= import('expo-iap'))

export function ProProvider({ children }: { children: ReactNode }) {
  // 文言は表示の直前に選択中の言語で引く（state にはキーだけを持つ）
  const { t } = useI18n()
  const [isEntitled, setIsPro] = useState(false)
  const [webPreview] = useState(() => isWebProPreview(Platform.OS, typeof window !== 'undefined' ? window.location?.search : undefined))
  const isPro = isEntitled || webPreview
  const [isNativeReady, setIsNativeReady] = useState(false)
  const [purchaseMessageKey, setPurchaseMessageKey] = useState<PurchaseMessageKey | null>(null)
  const [priceLabel, setPriceLabel] = useState<string | null>(null)
  const [isPurchasing, setIsPurchasing] = useState(false)
  // state と Pressable の disabled はコミット後の値なので、同じフレームの二重タップをすり抜ける。
  // 課金 API を叩く経路なので同期フラグで直列化する（購入と復元で共有）。
  const purchaseLockRef = useRef(false)
  // ストアとの接続（initConnection）が済んだか。済む前の購入・復元は接続から始める
  const connectedRef = useRef<Promise<Iap> | null>(null)
  const isNativePurchaseAvailable = Platform.OS === 'android'

  // Android 以外（Web）では購入できない。購入・復元はこの理由を出して受け付けない
  const blockedReasonKey: PurchaseMessageKey | null = isNativePurchaseAvailable ? null : 'purchaseStoreOnly'
  const isReady = blockedReasonKey !== null ? true : isNativeReady
  const messageKey = resolvePurchaseMessageKey(purchaseMessageKey, blockedReasonKey, isPro)
  const purchaseMessage = messageKey ? t(messageKey) : null

  const connect = useCallback(() => {
    connectedRef.current ??= loadIap().then(async (iap) => {
      await iap.initConnection()
      return iap
    })
    // 失敗したら次の操作で接続し直す
    connectedRef.current.catch(() => {
      connectedRef.current = null
    })
    return connectedRef.current
  }, [])

  /**
   * 購入の一覧を Pro の状態に反映する（起動時・購入・復元・リスナーのどこから来ても同じ処理）。
   * acknowledge していない購入はここで済ませる（3日以内に済ませないと Play が払い戻す）。
   * 戻り値は Pro か（保留中の支払いは Pro にしない）
   */
  const applyPurchases = useCallback(async (iap: Iap, purchases: readonly PurchaseLike[]) => {
    for (const p of unacknowledgedPro(purchases)) {
      // 失敗しても Pro の表示は止めない（次の起動・復元でもう一度 acknowledge する）
      await iap.finishTransaction({ purchase: p as Parameters<Iap['finishTransaction']>[0]['purchase'], isConsumable: false }).catch(() => undefined)
    }
    const owned = ownsPro(purchases)
    if (owned) setIsPro(true)
    return owned
  }, [])

  /** 端末の Play ストアにある購入を読み直す。返金されていれば Pro を外す */
  const refresh = useCallback(
    async (iap: Iap) => {
      const purchases = await iap.getAvailablePurchases()
      const owned = await applyPurchases(iap, purchases)
      if (!owned) setIsPro(false)
      return { owned, pending: hasPendingPro(purchases) }
    },
    [applyPurchases],
  )

  const loadPrice = useCallback(async (iap: Iap) => {
    const products = await iap.fetchProducts({ skus: [PRO_PRODUCT_ID], type: 'in-app' })
    const product = products?.find((p) => p.id === PRO_PRODUCT_ID)
    // 価格はストアのローカライズ済み文字列をそのまま出す
    if (product) setPriceLabel(product.displayPrice)
    return !!product
  }, [])

  useEffect(() => {
    if (blockedReasonKey !== null) return
    let active = true
    const subs: { remove: () => void }[] = []
    const start = async () => {
      let iap: Iap
      try {
        iap = await connect()
      } catch {
        if (active) {
          setPurchaseMessageKey('purchaseStatusFailed')
          setIsNativeReady(true)
        }
        return
      }
      if (!active) return
      // 購入の画面の外で届く更新（保留中だった支払いが済んだ など）も Pro に反映する
      subs.push(
        iap.purchaseUpdatedListener((purchase) => {
          if (!active) return
          void applyPurchases(iap, [purchase]).then((owned) => {
            if (active && owned) setPurchaseMessageKey('purchaseSucceeded')
          })
        }),
      )
      try {
        await refresh(iap)
      } catch {
        if (active) setPurchaseMessageKey('purchaseStatusFailed')
      }
      // 価格の取得失敗で Pro 表示まで巻き添えにしない
      try {
        await loadPrice(iap)
      } catch {
        // 購入時にもう一度取りに行く
      } finally {
        if (active) setIsNativeReady(true)
      }
    }
    void start()
    return () => {
      active = false
      subs.forEach((s) => s.remove())
    }
  }, [blockedReasonKey, connect, applyPurchases, refresh, loadPrice])

  const restoreWith = useCallback(
    async (iap: Iap) => {
      const { owned, pending } = await refresh(iap)
      setPurchaseMessageKey(owned ? 'proRestored' : pending ? 'purchasePending' : 'noRestorablePurchase')
    },
    [refresh],
  )

  const purchasePro = useCallback(async () => {
    if (blockedReasonKey !== null) {
      setPurchaseMessageKey(blockedReasonKey)
      return
    }
    if (!isReady || purchaseLockRef.current) return
    purchaseLockRef.current = true
    setIsPurchasing(true)
    setPurchaseMessageKey(null)
    try {
      let iap: Iap
      try {
        iap = await connect()
      } catch {
        setPurchaseMessageKey('productLoadFailed')
        return
      }
      // 商品が取れないときは購入させない（Play Console で商品が有効になっていない・圏外 など）
      const found = await loadPrice(iap).catch(() => false)
      if (!found) {
        setPurchaseMessageKey('productLoadFailed')
        return
      }
      try {
        const result = await iap.requestPurchase({ request: { google: { skus: [PRO_PRODUCT_ID] } }, type: 'in-app' })
        const purchases = (Array.isArray(result) ? result : result ? [result] : []) as PurchaseLike[]
        // 支払いが済んだことを確かめられたときだけ成功と言う（保留中の支払いはまだ Pro にしない）
        if (await applyPurchases(iap, purchases)) setPurchaseMessageKey('purchaseSucceeded')
        else if (hasPendingPro(purchases)) setPurchaseMessageKey('purchasePending')
      } catch (error) {
        const kind = purchaseErrorKind(error)
        // 購入済み（別の端末で買った・入れ直した）なら、復元と同じように Pro に戻す
        if (kind === 'alreadyOwned') await restoreWith(iap).catch(() => setPurchaseMessageKey('restoreFailed'))
        // ユーザーによるキャンセルはエラーではない（「失敗しました」を出さない）
        else if (kind !== 'cancelled') setPurchaseMessageKey('purchaseFailed')
      }
    } finally {
      purchaseLockRef.current = false
      setIsPurchasing(false)
    }
  }, [blockedReasonKey, isReady, connect, loadPrice, applyPurchases, restoreWith])

  const restorePurchases = useCallback(async () => {
    if (blockedReasonKey !== null) {
      setPurchaseMessageKey(blockedReasonKey)
      return
    }
    if (!isReady || purchaseLockRef.current) return
    purchaseLockRef.current = true
    setIsPurchasing(true)
    setPurchaseMessageKey(null)
    try {
      await restoreWith(await connect())
    } catch {
      setPurchaseMessageKey('restoreFailed')
    } finally {
      purchaseLockRef.current = false
      setIsPurchasing(false)
    }
  }, [blockedReasonKey, isReady, connect, restoreWith])

  const value = useMemo(
    () => ({ isPro, isReady, isNativePurchaseAvailable, purchaseMessage, priceLabel, isPurchasing, purchasePro, restorePurchases }),
    [isPro, isReady, isNativePurchaseAvailable, purchaseMessage, priceLabel, isPurchasing, purchasePro, restorePurchases],
  )
  return <ProContext.Provider value={value}>{children}</ProContext.Provider>
}

export function usePro() {
  const v = useContext(ProContext)
  if (!v) throw new Error('ProProvider の内部で使用してください。')
  return v
}
