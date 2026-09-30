import { useWindowDimensions } from 'react-native'

/** タブレット・Chromebook・横向きのレイアウトに切り替える幅（dp） */
export const WIDE_BREAKPOINT = 768

/**
 * 画面幅に応じたレイアウト。useWindowDimensions はウィンドウのリサイズ（Chromebook の
 * フリーフォーム窓・分割画面・回転）に追従するので、起動時の幅で固定しない。
 */
export function useLayout() {
  const { width, height } = useWindowDimensions()
  const isWide = width >= WIDE_BREAKPOINT
  return {
    width,
    height,
    isWide,
    /** 結果のクラスカードを何列で並べるか */
    classColumns: isWide ? Math.max(2, Math.min(4, Math.floor((width - 48) / 280))) : 1,
    /** 本文の最大幅（大画面で横に間延びさせない） */
    contentMaxWidth: isWide ? 1200 : undefined,
  }
}
