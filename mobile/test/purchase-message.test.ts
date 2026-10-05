import { describe, expect, it } from "vitest";

import { resolvePurchaseMessageKey } from "@/lib/purchase-message";

describe("resolvePurchaseMessageKey", () => {
  it("購入操作で設定されたメッセージを優先する", () => {
    expect(resolvePurchaseMessageKey("purchaseSucceeded", "purchaseStoreOnly", true)).toBe("purchaseSucceeded");
  });

  it("メッセージが無ければ購入できない理由を出す", () => {
    expect(resolvePurchaseMessageKey(null, "purchaseStatusFailed", false)).toBe("purchaseStatusFailed");
  });

  it("メッセージも理由も無ければnullを返す", () => {
    expect(resolvePurchaseMessageKey(null, null, false)).toBeNull();
  });

  // ここがこのモジュールを作った理由。支払いが保留中の案内を出したあと、
  // 購入の更新のリスナー経由でProが有効になったとき、案内が残ると
  // 「Pro利用中」カードと「支払いの完了を待っています」が同時に並んでしまう。
  it("Proが有効になったら purchasePending を出さない", () => {
    expect(resolvePurchaseMessageKey("purchasePending", null, true)).toBeNull();
  });

  it("Proが有効になったら noRestorablePurchase を出さない", () => {
    expect(resolvePurchaseMessageKey("noRestorablePurchase", null, true)).toBeNull();
  });

  it("Proが有効でないうちは purchasePending を出し続ける", () => {
    expect(resolvePurchaseMessageKey("purchasePending", null, false)).toBe("purchasePending");
  });

  it("Proが有効でないうちは noRestorablePurchase を出し続ける", () => {
    expect(resolvePurchaseMessageKey("noRestorablePurchase", null, false)).toBe("noRestorablePurchase");
  });

  it("Proが有効になっても購入の成功・復元の成功メッセージは消さない", () => {
    expect(resolvePurchaseMessageKey("purchaseSucceeded", null, true)).toBe("purchaseSucceeded");
    expect(resolvePurchaseMessageKey("proRestored", null, true)).toBe("proRestored");
  });

  it("Proが有効で古いメッセージを落とすときは購入できない理由へフォールバックする", () => {
    // Web版でProを持っているユーザーには「ストア版で購入できます」が残るが、
    // これはメッセージが未設定のときと同じ既存の振る舞い。
    expect(resolvePurchaseMessageKey("purchasePending", "purchaseStoreOnly", true)).toBe("purchaseStoreOnly");
  });

  it("購入の失敗・商品の読み込み失敗はProの有無で消さない", () => {
    // これらは「Proが無い」と主張していないので、Proが有効でも矛盾しない。
    expect(resolvePurchaseMessageKey("purchaseFailed", null, true)).toBe("purchaseFailed");
    expect(resolvePurchaseMessageKey("productLoadFailed", null, true)).toBe("productLoadFailed");
  });
});
