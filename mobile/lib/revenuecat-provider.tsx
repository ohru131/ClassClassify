import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { Platform } from 'react-native'
import Purchases, { type CustomerInfo, LOG_LEVEL, type PurchasesPackage } from 'react-native-purchases'

import { resolvePurchaseMessageKey } from './purchase-message'
import { selectOneTimePackageFromOfferings } from './purchase-offering'

// 課金は買い切り（非消費型）1本だけ。サブスクは絶対に売らない（判定は lib/purchase-offering.ts）。
// 設計は既存アプリ UnitCalc の lib/revenuecat-provider.tsx と同じ（不変条件は mobile/README.md）。
export const PRO_ENTITLEMENT_IDENTIFIER = 'pro'

const COPY = {
  purchaseStoreOnly: '購入は iOS / Android のアプリ版でご利用いただけます。',
  revenueCatKeyMissing: 'RevenueCat の公開 SDK キーが設定されていません。',
  customerInfoFetchFailed: '購入情報を取得できませんでした。時間をおいてもう一度お試しください。',
  purchaseSucceeded: 'ご購入ありがとうございます。Pro が有効になりました（買い切りのため、今後の請求はありません）。',
  purchaseNotApplied:
    'お支払いは完了しましたが、Pro をまだ有効にできていません。「購入を復元」を押してください。それでも有効にならない場合はサポートへご連絡ください（二重に請求されることはありません）。',
  purchaseFailed: '購入を完了できませんでした。もう一度お試しください。',
  productLoadFailed: 'ストアから Pro の商品を読み込めませんでした。通信状況を確認して、もう一度お試しください。',
  proRestored: 'Pro の購入を復元しました。',
  noRestorablePurchase: '復元できる Pro の購入が見つかりませんでした。',
  restoreFailed: '購入を復元できませんでした。もう一度お試しください。',
} as const

type PurchaseMessageKey = keyof typeof COPY

type ProContextValue = {
  isPro: boolean
  /** Pro 状態の復元が終わったか（終わるまで広告を出さない） */
  isReady: boolean
  isNativePurchaseAvailable: boolean
  purchaseMessage: string | null
  priceLabel: string | null
  isPurchasing: boolean
  purchasePro: () => Promise<void>
  restorePurchases: () => Promise<void>
}

const ProContext = createContext<ProContextValue | null>(null)

// configure() は多重に呼ぶと SDK 内部の状態がリセットされうるので一度きりにする
let purchasesConfigured = false

function getPlatformKey() {
  if (Platform.OS === 'ios') return process.env.EXPO_PUBLIC_REVENUECAT_IOS_API_KEY
  if (Platform.OS === 'android') return process.env.EXPO_PUBLIC_REVENUECAT_ANDROID_API_KEY
  return undefined
}

const hasProEntitlement = (info: CustomerInfo) => Boolean(info.entitlements.active[PRO_ENTITLEMENT_IDENTIFIER])

/** ユーザーによるキャンセルはエラーではない（「失敗しました」を出さない） */
function isUserCancelledError(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'userCancelled' in error && (error as { userCancelled: unknown }).userCancelled === true
}

export function RevenueCatProvider({ children }: { children: ReactNode }) {
  const [isPro, setIsPro] = useState(false)
  const [isNativeReady, setIsNativeReady] = useState(false)
  const [purchaseMessageKey, setPurchaseMessageKey] = useState<PurchaseMessageKey | null>(null)
  const [oneTimePackage, setOneTimePackage] = useState<PurchasesPackage | null>(null)
  const [isPurchasing, setIsPurchasing] = useState(false)
  // state と Pressable の disabled はコミット後の値なので、同じフレームの二重タップをすり抜ける。
  // 課金 API を叩く経路なので同期フラグで直列化する（購入と復元で共有）。
  const purchaseLockRef = useRef(false)
  const isNativePurchaseAvailable = Platform.OS === 'ios' || Platform.OS === 'android'
  const platformKey = getPlatformKey()

  // Web と SDK キー未設定の環境では初期化する余地が無い。叩けば必ず失敗して本当の原因を隠すので、
  // 購入・復元はこの理由を出して受け付けない。
  const blockedReasonKey: PurchaseMessageKey | null = !isNativePurchaseAvailable ? 'purchaseStoreOnly' : !platformKey ? 'revenueCatKeyMissing' : null
  const isReady = blockedReasonKey !== null ? true : isNativeReady
  const messageKey = resolvePurchaseMessageKey(purchaseMessageKey, blockedReasonKey, isPro)
  const purchaseMessage = messageKey ? COPY[messageKey] : null
  // 価格はストアのローカライズ済み文字列をそのまま出す
  const priceLabel = oneTimePackage ? oneTimePackage.product.priceString : null

  useEffect(() => {
    if (blockedReasonKey !== null || !platformKey) return
    let active = true
    const handleUpdate = (info: CustomerInfo) => {
      if (active) setIsPro(hasProEntitlement(info))
    }
    const configure = async () => {
      try {
        if (__DEV__) Purchases.setLogLevel(LOG_LEVEL.DEBUG)
        if (!purchasesConfigured) {
          Purchases.configure({ apiKey: platformKey })
          purchasesConfigured = true
        }
        const info = await Purchases.getCustomerInfo()
        if (!active) return
        setIsPro(hasProEntitlement(info))
        Purchases.addCustomerInfoUpdateListener(handleUpdate)
        // 商品の取得失敗で Pro 表示まで巻き添えにしない
        try {
          const offerings = await Purchases.getOfferings()
          if (active) setOneTimePackage(selectOneTimePackageFromOfferings(offerings))
        } catch {
          // 購入時にもう一度取りに行く
        }
      } catch {
        if (active) setPurchaseMessageKey('customerInfoFetchFailed')
      } finally {
        if (active) setIsNativeReady(true)
      }
    }
    void configure()
    return () => {
      active = false
      Purchases.removeCustomerInfoUpdateListener(handleUpdate)
    }
  }, [blockedReasonKey, platformKey])

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
      let pkg = oneTimePackage
      if (!pkg) {
        try {
          pkg = selectOneTimePackageFromOfferings(await Purchases.getOfferings())
          if (pkg) setOneTimePackage(pkg)
        } catch {
          // 下で productLoadFailed を出す
        }
      }
      // 買い切り商品が取れないときは購入させない。RevenueCatUI のペイウォールにフォールバックすると
      // offering に残ったサブスク商品を売ってしまう経路になる。
      if (!pkg) {
        setPurchaseMessageKey('productLoadFailed')
        return
      }
      try {
        const { customerInfo } = await Purchases.purchasePackage(pkg)
        // 決済が通っても pro の entitlement が付いてこないことがある（dashboard の紐付け忘れ等）。
        // 確認できたときだけ成功と言う。
        const unlocked = hasProEntitlement(customerInfo)
        setIsPro(unlocked)
        setPurchaseMessageKey(unlocked ? 'purchaseSucceeded' : 'purchaseNotApplied')
      } catch (error) {
        if (!isUserCancelledError(error)) setPurchaseMessageKey('purchaseFailed')
      }
    } finally {
      purchaseLockRef.current = false
      setIsPurchasing(false)
    }
  }, [blockedReasonKey, isReady, oneTimePackage])

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
      const info = await Purchases.restorePurchases()
      const unlocked = hasProEntitlement(info)
      setIsPro(unlocked)
      setPurchaseMessageKey(unlocked ? 'proRestored' : 'noRestorablePurchase')
    } catch {
      setPurchaseMessageKey('restoreFailed')
    } finally {
      purchaseLockRef.current = false
      setIsPurchasing(false)
    }
  }, [blockedReasonKey, isReady])

  const value = useMemo(
    () => ({ isPro, isReady, isNativePurchaseAvailable, purchaseMessage, priceLabel, isPurchasing, purchasePro, restorePurchases }),
    [isPro, isReady, isNativePurchaseAvailable, purchaseMessage, priceLabel, isPurchasing, purchasePro, restorePurchases],
  )
  return <ProContext.Provider value={value}>{children}</ProContext.Provider>
}

export function usePro() {
  const v = useContext(ProContext)
  if (!v) throw new Error('RevenueCatProvider の内部で使用してください。')
  return v
}
