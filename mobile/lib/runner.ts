import { createAnnealer, type CompiledProblem } from './solver'

// React Native には Web Worker が無いので、1本の JS スレッドの上で焼きなましを
// 「sliceMs だけ計算 → 画面へ制御を返す」を繰り返して進める。マルチスタートは並列ではなく
// 順番に行い、探索時間 timeMs を各スタートで等分する（各スタートは自分が実際に計算した
// 時間だけで時間を消費するので、UI に返している間は時間を使わない）。

export interface SlicedRunResult {
  classOf: number[]
  cost: number
  iterations: number
  starts: number
}

export interface SlicedRunOptions {
  timeMs: number
  /** スタート回数（省略時は探索時間から決める） */
  starts?: number
  /**
   * 1回に続けて計算する時間の下限。短いほど UI が滑らかになるが、切り替えの負担が増える。
   * UI に返すと戻るまでの時間を測り、計算の割合が COMPUTE_SHARE を下回らないよう自動で伸ばす（上限 maxSliceMs）
   */
  sliceMs?: number
  maxSliceMs?: number
  seed?: number
  /** fraction: 0〜1 の全体の進み具合、best: これまでの最良コスト */
  onProgress?: (fraction: number, best: number) => void
  /** 画面に制御を返す方法（テストで差し替える） */
  yieldToUi?: () => Promise<void>
}

export const defaultStarts = (timeMs: number) => (timeMs >= 20000 ? 3 : timeMs >= 8000 ? 2 : 1)

const setTimeoutYield = () => new Promise<void>((resolve) => setTimeout(resolve, 0))

// 実時間のうち計算に充てる割合の目標。探索時間は計算した時間だけで数えるので、UI に返している
// 時間が長いと「10秒」の設定でも実時間が何倍にも延びる（実機の debug ビルドで約95秒かかった）。
// 0.8 なら実時間は探索時間の 1.25 倍程度に収まる
const COMPUTE_SHARE = 0.8

/** 直前に UI へ返していた時間から、次に続けて計算する時間を決める */
export const nextSliceMs = (yieldMs: number, minMs: number, maxMs: number) =>
  Math.min(maxMs, Math.max(minMs, (yieldMs * COMPUTE_SHARE) / (1 - COMPUTE_SHARE)))

export class CancelledError extends Error {
  constructor() {
    super('cancelled')
  }
}

export function runSliced(problem: CompiledProblem, opt: SlicedRunOptions): { promise: Promise<SlicedRunResult>; cancel: () => void } {
  const starts = Math.max(1, Math.floor(opt.starts ?? defaultStarts(opt.timeMs)))
  const minSliceMs = opt.sliceMs ?? 24
  const maxSliceMs = Math.max(minSliceMs, opt.maxSliceMs ?? 200)
  const yieldToUi = opt.yieldToUi ?? setTimeoutYield
  let sliceMs = minSliceMs
  // UI へ返して戻るまでの実時間を測り、次の計算時間に反映する
  const yieldAndAdapt = async () => {
    const t0 = performance.now()
    await yieldToUi()
    sliceMs = nextSliceMs(performance.now() - t0, minSliceMs, maxSliceMs)
  }
  const baseSeed = (opt.seed ?? Date.now()) >>> 0
  let cancelled = false

  const promise = (async () => {
    let best: SlicedRunResult | null = null
    let iterations = 0
    for (let s = 0; s < starts; s++) {
      const annealer = createAnnealer(problem, {
        timeMs: opt.timeMs / starts,
        seed: (baseSeed + s * 7919) >>> 0,
        onProgress: (cost, frac) => opt.onProgress?.((s + frac) / starts, Math.min(cost, best?.cost ?? Infinity)),
      })
      for (;;) {
        if (cancelled) throw new CancelledError()
        if (annealer.run(sliceMs)) break
        await yieldAndAdapt()
      }
      const r = annealer.result()
      iterations += r.iterations
      if (!best || r.cost < best.cost) best = { classOf: r.classOf, cost: r.cost, iterations: 0, starts }
      opt.onProgress?.((s + 1) / starts, best.cost)
      if (s + 1 < starts) await yieldAndAdapt()
    }
    if (cancelled) throw new CancelledError()
    return { ...best!, iterations }
  })()

  return {
    promise,
    cancel: () => {
      cancelled = true
    },
  }
}
