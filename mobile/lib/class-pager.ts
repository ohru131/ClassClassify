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
