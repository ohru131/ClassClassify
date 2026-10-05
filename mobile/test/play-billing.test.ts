import { describe, expect, it } from 'vitest'

import { hasPendingPro, ownsPro, PRO_PRODUCT_ID, purchaseErrorKind, unacknowledgedPro, type PurchaseLike } from '@/lib/play-billing'

const purchase = (over: Partial<PurchaseLike> = {}): PurchaseLike => ({
  productId: PRO_PRODUCT_ID,
  purchaseState: 'purchased',
  purchaseToken: 'token',
  isAcknowledgedAndroid: true,
  ...over,
})

describe('Play Billing の Pro の判定', () => {
  it('商品 ID は Play Console の買い切りの商品', () => {
    expect(PRO_PRODUCT_ID).toBe('fairclass_pro')
  })

  it('支払い済みの Pro の購入があるときだけ Pro', () => {
    expect(ownsPro([purchase()])).toBe(true)
    expect(ownsPro([])).toBe(false)
    // 保留中（コンビニ払いなど）・状態不明はまだ Pro にしない
    expect(ownsPro([purchase({ purchaseState: 'pending' })])).toBe(false)
    expect(ownsPro([purchase({ purchaseState: 'unknown' })])).toBe(false)
    // 別の商品は Pro にしない
    expect(ownsPro([purchase({ productId: 'other' })])).toBe(false)
  })

  it('保留中の Pro の購入を見分ける', () => {
    expect(hasPendingPro([purchase({ purchaseState: 'pending' })])).toBe(true)
    expect(hasPendingPro([purchase()])).toBe(false)
    expect(hasPendingPro([purchase({ productId: 'other', purchaseState: 'pending' })])).toBe(false)
  })

  it('acknowledge が要るのは支払い済みで未 acknowledge の Pro の購入だけ', () => {
    const fresh = purchase({ isAcknowledgedAndroid: false })
    expect(unacknowledgedPro([fresh, purchase()])).toEqual([fresh])
    // 情報が無いときも済ませる（acknowledge し直しても害は無い）
    expect(unacknowledgedPro([purchase({ isAcknowledgedAndroid: null })])).toHaveLength(1)
    expect(unacknowledgedPro([purchase({ isAcknowledgedAndroid: false, purchaseState: 'pending' })])).toEqual([])
    expect(unacknowledgedPro([purchase({ isAcknowledgedAndroid: false, purchaseToken: null })])).toEqual([])
    expect(unacknowledgedPro([purchase({ isAcknowledgedAndroid: false, productId: 'other' })])).toEqual([])
  })

  it('エラーの種類: キャンセル・購入済み・それ以外', () => {
    expect(purchaseErrorKind({ code: 'user-cancelled' })).toBe('cancelled')
    expect(purchaseErrorKind({ code: 'already-owned' })).toBe('alreadyOwned')
    expect(purchaseErrorKind({ code: 'network-error' })).toBe('other')
    expect(purchaseErrorKind(new Error('x'))).toBe('other')
    expect(purchaseErrorKind(null)).toBe('other')
  })
})
