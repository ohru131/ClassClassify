import AsyncStorage from '@react-native-async-storage/async-storage'
import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { AppState, Platform } from 'react-native'

import { useI18n } from './language-provider'
import { hasPendingPro, ownsPro, PRO_PRODUCT_ID, purchaseErrorKind, unacknowledgedPro, type PurchaseLike } from './play-billing'
import { isWebProPreview } from './pro-preview'
import { resolvePurchaseMessageKey } from './purchase-message'

// Pro（買い切り）を Google Play Billing で直接扱う（expo-iap）。外部の課金サービスは使わず、
// 購入の状態は端末の Play ストアに問い合わせるだけで、アプリからどこにも送信しない。
// 判定は lib/play-billing.ts（買い切り1本・consume しない・acknowledge する）。
//
// Play の購入を読み直すのは、起動時・アプリが前面に戻ったとき・購入・復元のとき
// （保留中だった支払いがバックグラウンドの間に済んだ、起動時に読めなかった などを拾う）。

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

// 最後に Play で確かめた Pro の状態（Play の購入を読めないときに、買った人を無料版に戻さないため）。
// 読み直せたら必ず上書きする（返金されていれば消える）
const PRO_CACHE_KEY = 'fairclass.pro.v1'

export function ProProvider({ children }: { children: ReactNode }) {
  // 文言は表示の直前に選択中の言語で引く（state にはキーだけを持つ）
  const { t } = useI18n()
  const [isEntitled, setIsEntitled] = useState(false)
  const [webPreview] = useState(() => isWebProPreview(Platform.OS, typeof window !== 'undefined' ? window.location?.search : undefined))
  const isPro = isEntitled || webPreview
  const [isNativeReady, setIsNativeReady] = useState(false)
  const [purchaseMessageKey, setPurchaseMessageKey] = useState<PurchaseMessageKey | null>(null)
  const [priceLabel, setPriceLabel] = useState<string | null>(null)
  const [isPurchasing, setIsPurchasing] = useState(false)
  // state と Pressable の disabled はコミット後の値なので、同じフレームの二重タップをすり抜ける。
  // 課金 API を叩く経路なので同期フラグで直列化する（購入と復元で共有）。
  const purchaseLockRef = useRef(false)
  // ストアとの接続（initConnection）。済む前の購入・復元は接続から始める
  const connectedRef = useRef<Promise<Iap> | null>(null)
  // 購入の更新のリスナー（接続できたときに1つだけ登録する）
  const listenerRef = useRef<{ remove: () => void } | null>(null)
  // Pro にした回数。読み直しの最中に別の経路（リスナー・購入）で Pro になったら、
  // 読み直しの古い一覧で Pro を外さない
  const proSeqRef = useRef(0)
  // Play で確かめた結果が出たら、端末のキャッシュ（起動直後に読む）で上書きしない
  const checkedRef = useRef(false)
  const mountedRef = useRef(true)
  const isNativePurchaseAvailable = Platform.OS === 'android'

  // Android 以外（Web）では購入できない。購入・復元はこの理由を出して受け付けない
  const blockedReasonKey: PurchaseMessageKey | null = isNativePurchaseAvailable ? null : 'purchaseStoreOnly'
  const isReady = blockedReasonKey !== null ? true : isNativeReady
  const messageKey = resolvePurchaseMessageKey(purchaseMessageKey, blockedReasonKey, isPro)
  const purchaseMessage = messageKey ? t(messageKey) : null

  const setPro = useCallback((owned: boolean) => {
    checkedRef.current = true
    if (owned) proSeqRef.current++
    setIsEntitled(owned)
    void (owned ? AsyncStorage.setItem(PRO_CACHE_KEY, '1') : AsyncStorage.removeItem(PRO_CACHE_KEY)).catch(() => undefined)
  }, [])

  /**
   * 購入の一覧を Pro の状態に反映する（起動時・前面に戻ったとき・購入・復元・リスナーのどこから来ても同じ処理）。
   * 戻り値は Pro か（保留中の支払いは Pro にしない）。Pro にしてから acknowledge する
   * （3日以内に済ませないと Play が払い戻す。失敗しても Pro は止めず、次に読み直したときにもう一度行う）
   */
  const applyPurchases = useCallback(
    async (iap: Iap, purchases: readonly PurchaseLike[]) => {
      const owned = ownsPro(purchases)
      if (owned) setPro(true)
      for (const p of unacknowledgedPro(purchases)) {
        await iap.finishTransaction({ purchase: p as Parameters<Iap['finishTransaction']>[0]['purchase'], isConsumable: false }).catch(() => undefined)
      }
      return owned
    },
    [setPro],
  )

  const ensureListener = useCallback(
    (iap: Iap) => {
      if (listenerRef.current || !mountedRef.current) return
      // 購入の画面の外で届く更新（保留中だった支払いが済んだ など）も Pro に反映する
      listenerRef.current = iap.purchaseUpdatedListener((purchase) => {
        if (!mountedRef.current) return
        void applyPurchases(iap, [purchase]).then((owned) => {
          if (mountedRef.current && owned) setPurchaseMessageKey('purchaseSucceeded')
        })
      })
    },
    [applyPurchases],
  )

  const connect = useCallback(() => {
    if (!connectedRef.current) {
      const p = loadIap().then(async (iap) => {
        await iap.initConnection()
        return iap
      })
      connectedRef.current = p
      // 失敗したら次の操作で接続し直す（その間に作り直した接続は消さない）
      p.catch(() => {
        if (connectedRef.current === p) connectedRef.current = null
      })
    }
    return connectedRef.current.then((iap) => {
      ensureListener(iap)
      return iap
    })
  }, [ensureListener])

  /**
   * 端末の Play ストアにある購入を読み直す。返金されていれば Pro を外す。
   * keepPro: 外さない（Play が「購入済み」と言ったのに、まだ一覧に載っていないとき）
   */
  const refresh = useCallback(
    async (iap: Iap, keepPro = false) => {
      const seq = proSeqRef.current
      const purchases = await iap.getAvailablePurchases()
      const owned = await applyPurchases(iap, purchases)
      if (!owned && !keepPro && proSeqRef.current === seq) setPro(false)
      // 起動時に読めなかった案内は、読めたら消す
      setPurchaseMessageKey((k) => (k === 'purchaseStatusFailed' ? null : k))
      return { owned, pending: hasPendingPro(purchases) }
    },
    [applyPurchases, setPro],
  )

  const loadPrice = useCallback(async (iap: Iap) => {
    const products = await iap.fetchProducts({ skus: [PRO_PRODUCT_ID], type: 'in-app' })
    const product = products?.find((p) => p.id === PRO_PRODUCT_ID)
    // 価格はストアのローカライズ済み文字列をそのまま出す
    if (product) setPriceLabel(product.displayPrice)
    return !!product
  }, [])

  useEffect(() => {
    mountedRef.current = true
    if (blockedReasonKey !== null) return
    let active = true
    // Play を読めるまでは、最後に確かめた状態で始める
    AsyncStorage.getItem(PRO_CACHE_KEY)
      .then((v) => {
        if (active && v === '1' && !checkedRef.current) setIsEntitled(true)
      })
      .catch(() => undefined)
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
    // 前面に戻ったら読み直す（購入・復元の最中は、その処理に任せる）
    const appState = AppState.addEventListener('change', (state) => {
      if (state !== 'active' || purchaseLockRef.current) return
      connect()
        .then((iap) => refresh(iap))
        .catch(() => undefined)
    })
    return () => {
      active = false
      mountedRef.current = false
      appState.remove()
      listenerRef.current?.remove()
      listenerRef.current = null
    }
  }, [blockedReasonKey, connect, refresh, loadPrice])

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
        else {
          // 購入の画面から結果が返ってこなかった（後払いの経路など）。Play を読み直して確かめる
          const { owned, pending } = await refresh(iap, true)
          if (owned) setPurchaseMessageKey('purchaseSucceeded')
          else if (pending) setPurchaseMessageKey('purchasePending')
        }
      } catch (error) {
        const kind = purchaseErrorKind(error)
        if (kind === 'alreadyOwned') {
          // 購入済み（別の端末で買った・入れ直した）。復元と同じように Pro に戻す。
          // 一覧にまだ載っていなければ Pro は外さず、確かめられなかったと伝える
          const { owned } = await refresh(iap, true).catch(() => ({ owned: false }))
          setPurchaseMessageKey(owned ? 'proRestored' : 'purchaseStatusFailed')
        }
        // ユーザーによるキャンセルはエラーではない（「失敗しました」を出さない）
        else if (kind !== 'cancelled') setPurchaseMessageKey('purchaseFailed')
      }
    } finally {
      purchaseLockRef.current = false
      setIsPurchasing(false)
    }
  }, [blockedReasonKey, isReady, connect, loadPrice, applyPurchases, refresh])

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
