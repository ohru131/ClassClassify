/**
 * 「1クラスの最大人数」を1段階増やす／減らしたときのクラス数。最大人数はクラス数から
 * ceil(n / k) で決まるので、単純に最大人数へ ±1 すると同じ k に戻って変化しないことがある
 * （n=30・k=5 で 6→7 にしても ceil(30/5)=6 のまま）。表示される値が実際に変わる次の k を探す。
 * 戻り値は { k, max }（max は ceil(n / k)＝表示と保存に使う値）。動かせなければ null。
 */
export function stepClassCountByMax(n: number, k: number, dir: 1 | -1): { k: number; max: number } | null {
  const cur = Math.ceil(n / k)
  // 最大人数を増やす ＝ クラスを減らす方向、減らす ＝ 増やす方向
  for (let next = k - dir; next >= 2 && next <= Math.max(2, n); next -= dir) {
    const max = Math.ceil(n / next)
    if (dir === 1 ? max > cur : max < cur) return { k: next, max }
  }
  return null
}
