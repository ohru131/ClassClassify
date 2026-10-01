import { describe, expect, it } from 'vitest'
import { clampPage, offsetForPage, pageFromOffset, pageLayout, stepPage } from '../lib/class-pager'

describe('結果画面のクラスのページ送り', () => {
  it('スクロール位置からいちばん近いページを選ぶ', () => {
    expect(pageFromOffset(0, 360, 4)).toBe(0)
    expect(pageFromOffset(179, 360, 4)).toBe(0)
    expect(pageFromOffset(181, 360, 4)).toBe(1)
    expect(pageFromOffset(720, 360, 4)).toBe(2)
    // 端を越えた引っ張り（バウンス）でも範囲内
    expect(pageFromOffset(-40, 360, 4)).toBe(0)
    expect(pageFromOffset(5000, 360, 4)).toBe(3)
  })

  it('ページ幅が未計測なら 0', () => {
    expect(pageFromOffset(500, 0, 4)).toBe(0)
  })

  it('クラス数が減ったら範囲内に丸める', () => {
    expect(clampPage(5, 4)).toBe(3)
    expect(clampPage(-1, 4)).toBe(0)
    expect(clampPage(2, 0)).toBe(0)
    expect(clampPage(Number.NaN, 3)).toBe(0)
  })

  it('ページの位置と getItemLayout は幅の倍数', () => {
    expect(offsetForPage(3, 358)).toBe(1074)
    expect(pageLayout(358, 2)).toEqual({ length: 358, offset: 716, index: 2 })
  })

  it('画面の回転でページ幅が変わっても、同じページの位置へ戻せる', () => {
    const page = pageFromOffset(offsetForPage(2, 360), 360, 4)
    expect(pageFromOffset(offsetForPage(page, 800), 800, 4)).toBe(2)
  })

  it('前後の矢印は端で止まる', () => {
    expect(stepPage(0, -1, 4)).toBeNull()
    expect(stepPage(0, 1, 4)).toBe(1)
    expect(stepPage(3, 1, 4)).toBeNull()
    expect(stepPage(2, -1, 4)).toBe(1)
  })
})
