// スマホ幅の結果画面で、クラスを1ページずつ横に並べて左右にスワイプで切り替えるための純関数
// （app/(tabs)/results.tsx の ClassPager。React Native に依存しないのでテストから読める）。

/** ページ番号を 0〜count-1 に収める（クラス数が再実行・読み込みで減ったとき） */
export function clampPage(index: number, count: number): number {
  if (count <= 0 || !Number.isFinite(index)) return 0
  return Math.min(count - 1, Math.max(0, Math.round(index)))
}

/** 横スクロールの位置（px）から、いちばん近いページ番号。ページ幅が未計測（0）なら 0 */
export function pageFromOffset(offset: number, pageWidth: number, count: number): number {
  if (!(pageWidth > 0)) return 0
  return clampPage(offset / pageWidth, count)
}

/** ページ番号の左端の位置（px） */
export const offsetForPage = (index: number, pageWidth: number): number => Math.max(0, index) * Math.max(0, pageWidth)

/** FlatList の getItemLayout（全ページが同じ幅） */
export const pageLayout = (pageWidth: number, index: number) => ({ length: pageWidth, offset: offsetForPage(index, pageWidth), index })

/** 前後の矢印で移る先。端では null（ボタンを無効にする） */
export function stepPage(index: number, delta: -1 | 1, count: number): number | null {
  const next = index + delta
  return next < 0 || next >= count ? null : next
}

/**
 * スクロールのイベントごとに「今どのページを選んでいることにするか」を決める。
 * settling はタブ・矢印で動かし始めたときの行き先（アニメーション中の途中のページでタブが点滅しないよう、
 * 着くまでは途中の位置を無視する）。
 * - drag: 利用者が指（マウス）で触った。行き先の予約を捨て、以後は実際の位置に従う
 * - scroll: 途中経過。予約があれば、その行き先に着いたときだけ予約を外す
 * - end: スクロールが止まった（ネイティブは onMomentumScrollEnd、Web は止まってからの一定時間）。
 *   予約の有無に関係なく実際の位置で選び直す（割り込まれて予約が残ったままにならないように）
 */
export function resolvePagerScroll(
  state: { settling: number | null; current: number },
  event: { phase: 'drag' | 'scroll' | 'end'; page: number },
): { settling: number | null; select: number | null } {
  const changed = (page: number) => (page !== state.current ? page : null)
  if (event.phase === 'drag') return { settling: null, select: null }
  if (event.phase === 'end') return { settling: null, select: changed(event.page) }
  if (state.settling !== null) return { settling: event.page === state.settling ? null : state.settling, select: null }
  return { settling: null, select: changed(event.page) }
}
