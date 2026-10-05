// Google Play Billing（expo-iap 経由）で Pro を判定する純関数。
// lib/pro-provider.tsx に書くと expo-iap（ネイティブモジュール）を読み込むことになり、
// vitest からこの判定だけを検証できないので切り出してある。
//
// 課金は買い切り（非消費型）1本だけ。サブスクは売らない。購入は consume しない
// （consume すると再び買える＝買い切りではなくなり、復元もできなくなる）。acknowledge だけする。

/** Play Console の一回限りの商品 ID（非消費型として扱う） */
export const PRO_PRODUCT_ID = 'fairclass_pro'

/** expo-iap の Purchase のうち、判定に使う分だけ */
export interface PurchaseLike {
  productId: string
  purchaseState: 'pending' | 'purchased' | 'unknown'
  purchaseToken?: string | null
  isAcknowledgedAndroid?: boolean | null
}

const isPro = (p: PurchaseLike) => p.productId === PRO_PRODUCT_ID

/** 支払いが済んだ Pro の購入があるか（保留中の支払いは Pro にしない） */
export const ownsPro = (purchases: readonly PurchaseLike[]) => purchases.some((p) => isPro(p) && p.purchaseState === 'purchased')

/** 支払いが保留中（コンビニ払いなど）の Pro の購入があるか */
export const hasPendingPro = (purchases: readonly PurchaseLike[]) => purchases.some((p) => isPro(p) && p.purchaseState === 'pending')

/**
 * acknowledge が要る Pro の購入。Play は3日以内に acknowledge されない購入を自動で払い戻すので、
 * 購入の直後だけでなく起動時・復元時にも済ませる（購入の直後にアプリが落ちた場合など）
 */
export const unacknowledgedPro = (purchases: readonly PurchaseLike[]) =>
  purchases.filter((p) => isPro(p) && p.purchaseState === 'purchased' && p.isAcknowledgedAndroid !== true && !!p.purchaseToken)

export type PurchaseErrorKind = 'cancelled' | 'alreadyOwned' | 'other'

/** expo-iap のエラー（code は 'user-cancelled' / 'already-owned' など）の種類 */
export function purchaseErrorKind(error: unknown): PurchaseErrorKind {
  const code = typeof error === 'object' && error !== null && 'code' in error ? String((error as { code: unknown }).code) : ''
  if (code === 'user-cancelled') return 'cancelled'
  if (code === 'already-owned') return 'alreadyOwned'
  return 'other'
}
