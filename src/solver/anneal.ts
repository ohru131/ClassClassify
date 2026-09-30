import type { CompiledProblem, SolveResult } from './types'

/** 人数制約・別の組ペア違反・同じ組ペア違反に掛ける重み（属性バランスより十分大きい） */
const HARD = 1000
/** 目標値からのずれ（二乗）に掛ける微小な重み。帯の内側でも中央へ寄せるタイブレーカー */
const TIE = 0.05

export interface AnnealOptions {
  timeMs: number
  seed: number
  onProgress?: (best: number, fraction: number) => void
}

function rng(seed: number) {
  let s = seed >>> 0 || 1
  return () => {
    s ^= s << 13
    s >>>= 0
    s ^= s >>> 17
    s ^= s << 5
    s >>>= 0
    return s / 4294967296
  }
}

class UnionFind {
  parent: number[]
  constructor(n: number) {
    this.parent = Array.from({ length: n }, (_, i) => i)
  }
  find(x: number): number {
    while (this.parent[x] !== x) x = this.parent[x] = this.parent[this.parent[x]]
    return x
  }
  union(a: number, b: number) {
    this.parent[this.find(a)] = this.find(b)
  }
}

/** 途中で止めて再開できる焼きなまし（Web Worker の無い React Native でも UI を止めずに回すため） */
export interface Annealer {
  /**
   * 最大 sliceMs ミリ秒だけ探索を進める。探索時間を使い切って仕上げまで終えたら true。
   * 時間割り当て（opt.timeMs）は、この関数の中で実際に計算していた時間だけで消費する。
   */
  run(sliceMs: number): boolean
  /** run() が true を返した後に呼ぶ */
  result(): SolveResult
}

/**
 * 焼きなまし法によるクラス分け。
 * 「同じ組ペア」は union-find で1ブロックにまとめて常に同じ組に置く（ハード制約）。
 * ブロック単位の「移動」「交換」を近傍とし、人数・別の組ペアはペナルティで扱う。
 */
export function anneal(p: CompiledProblem, opt: AnnealOptions): SolveResult {
  const a = createAnnealer(p, opt)
  while (!a.run(Infinity));
  return a.result()
}

export function createAnnealer(p: CompiledProblem, opt: AnnealOptions): Annealer {
  const t0 = performance.now()
  const rand = rng(opt.seed)
  const { n, k, attr, weights, lo, hi, target } = p
  const A = attr.length

  // --- ブロック化 ---
  const uf = new UnionFind(n)
  for (const [i, j] of p.wantedPairs) uf.union(i, j)
  const rootToBlock = new Map<number, number>()
  const blockOf = new Array<number>(n)
  const members: number[][] = []
  for (let i = 0; i < n; i++) {
    const r = uf.find(i)
    let b = rootToBlock.get(r)
    if (b === undefined) {
      b = members.length
      rootToBlock.set(r, b)
      members.push([])
    }
    blockOf[i] = b
    members[b].push(i)
  }
  const B = members.length
  const bSize = members.map((m) => m.length)
  // ブロックごとの属性合計（dense: bAttr[b*A + a]）
  const bAttr = new Float64Array(B * A)
  for (let b = 0; b < B; b++) for (const i of members[b]) for (let a = 0; a < A; a++) bAttr[b * A + a] += attr[a][i]
  // ブロック間の「別の組」辺（多重度つき）
  const conflictMap: Map<number, number>[] = Array.from({ length: B }, () => new Map())
  for (const [i, j] of p.unwantedPairs) {
    const bi = blockOf[i]
    const bj = blockOf[j]
    if (bi === bj) continue // 同じ組ペアと矛盾 → 避けられないので無視（評価で報告）
    conflictMap[bi].set(bj, (conflictMap[bi].get(bj) ?? 0) + 1)
    conflictMap[bj].set(bi, (conflictMap[bj].get(bi) ?? 0) + 1)
  }
  const conflicts = conflictMap.map((m) => [...m.entries()])

  // --- コスト関数 ---
  const attrCost = (a: number, c: number) => {
    const over = c > hi[a] ? c - hi[a] : c < lo[a] ? lo[a] - c : 0
    const d = c - target[a]
    return weights[a] * (over + TIE * d * d)
  }
  const sizeCost = (s: number) => HARD * (s > p.maxSize ? s - p.maxSize : s < p.minSize ? p.minSize - s : 0)

  // --- 初期解: 大きいブロックから、衝突が少なく人数の少ない組へ ---
  const cls = new Int32Array(B).fill(-1)
  const size = new Int32Array(k)
  const cnt = new Float64Array(A * k) // cnt[a*k + c]
  const order = Array.from({ length: B }, (_, b) => b).sort((x, y) => bSize[y] - bSize[x] || rand() - 0.5)
  for (const b of order) {
    let best = 0
    let bestScore = Infinity
    for (let c = 0; c < k; c++) {
      let conf = 0
      for (const [o, m] of conflicts[b]) if (cls[o] === c) conf += m
      const score = conf * 1e6 + size[c] + bSize[b] * (size[c] + bSize[b] > p.maxSize ? 1e3 : 0) + rand() * 0.5
      if (score < bestScore) {
        bestScore = score
        best = c
      }
    }
    cls[b] = best
    size[best] += bSize[b]
    for (let a = 0; a < A; a++) cnt[a * k + best] += bAttr[b * A + a]
  }

  const confOf = (b: number, c: number, skip = -1) => {
    let s = 0
    for (const [o, m] of conflicts[b]) if (o !== skip && cls[o] === c) s += m
    return s
  }

  let cost = 0
  for (let c = 0; c < k; c++) cost += sizeCost(size[c])
  for (let a = 0; a < A; a++) for (let c = 0; c < k; c++) cost += attrCost(a, cnt[a * k + c])
  for (let b = 0; b < B; b++) cost += (HARD * confOf(b, cls[b])) / 2

  // 移動 b: from → to の差分
  const deltaMove = (b: number, to: number) => {
    const from = cls[b]
    let d = sizeCost(size[from] - bSize[b]) + sizeCost(size[to] + bSize[b]) - sizeCost(size[from]) - sizeCost(size[to])
    const off = b * A
    for (let a = 0; a < A; a++) {
      const v = bAttr[off + a]
      if (v === 0) continue
      const cf = cnt[a * k + from]
      const ct = cnt[a * k + to]
      d += attrCost(a, cf - v) + attrCost(a, ct + v) - attrCost(a, cf) - attrCost(a, ct)
    }
    d += HARD * (confOf(b, to) - confOf(b, from))
    return d
  }
  const applyMove = (b: number, to: number) => {
    const from = cls[b]
    size[from] -= bSize[b]
    size[to] += bSize[b]
    const off = b * A
    for (let a = 0; a < A; a++) {
      const v = bAttr[off + a]
      cnt[a * k + from] -= v
      cnt[a * k + to] += v
    }
    cls[b] = to
  }
  // 交換 b1(c1) ↔ b2(c2) の差分
  const deltaSwap = (b1: number, b2: number) => {
    const c1 = cls[b1]
    const c2 = cls[b2]
    const ds = bSize[b2] - bSize[b1]
    let d = 0
    if (ds !== 0) d += sizeCost(size[c1] + ds) + sizeCost(size[c2] - ds) - sizeCost(size[c1]) - sizeCost(size[c2])
    const o1 = b1 * A
    const o2 = b2 * A
    for (let a = 0; a < A; a++) {
      const v = bAttr[o2 + a] - bAttr[o1 + a]
      if (v === 0) continue
      const x1 = cnt[a * k + c1]
      const x2 = cnt[a * k + c2]
      d += attrCost(a, x1 + v) + attrCost(a, x2 - v) - attrCost(a, x1) - attrCost(a, x2)
    }
    d += HARD * (confOf(b1, c2, b2) - confOf(b1, c1, b2) + confOf(b2, c1, b1) - confOf(b2, c2, b1))
    return d
  }
  const applySwap = (b1: number, b2: number) => {
    const c1 = cls[b1]
    const c2 = cls[b2]
    applyMove(b1, c2)
    applyMove(b2, c1)
  }

  // --- 初期温度: ランダム近傍の改悪幅の中央値程度 ---
  const samples: number[] = []
  for (let s = 0; s < 200 && B > 1; s++) {
    const b1 = (rand() * B) | 0
    const b2 = (rand() * B) | 0
    if (cls[b1] === cls[b2]) continue
    const d = deltaSwap(b1, b2)
    if (d > 0 && d < HARD / 2) samples.push(d)
  }
  samples.sort((x, y) => x - y)
  const T0 = Math.max(samples[samples.length >> 1] ?? 1, 0.05)
  const Tend = 0.002

  let bestCost = cost
  let bestCls = Int32Array.from(cls)
  let iter = 0
  /** これまでの run() で探索に使った時間の合計 */
  let activeMs = 0
  let lastReport = -Infinity
  let finished = !(B > 1 && k > 1)
  let polished = false
  let res: SolveResult | null = null

  // --- 最良解に戻して貪欲に仕上げ ---
  const polish = () => {
    for (let b = 0; b < B; b++) if (cls[b] !== bestCls[b]) applyMove(b, bestCls[b])
    cost = bestCost
    for (let improved = true, pass = 0; improved && pass < 20; pass++) {
      improved = false
      for (let b1 = 0; b1 < B; b1++) {
        for (let to = 0; to < k; to++) {
          if (to === cls[b1]) continue
          const d = deltaMove(b1, to)
          if (d < -1e-9) {
            applyMove(b1, to)
            cost += d
            improved = true
          }
        }
        for (let b2 = b1 + 1; b2 < B; b2++) {
          if (cls[b1] === cls[b2]) continue
          const d = deltaSwap(b1, b2)
          if (d < -1e-9) {
            applySwap(b1, b2)
            cost += d
            improved = true
          }
        }
      }
    }
    const classOf = Array.from({ length: n }, (_, i) => cls[blockOf[i]])
    res = { classOf, cost, elapsedMs: performance.now() - t0, iterations: iter }
  }

  const run = (sliceMsIn: number): boolean => {
    // 0・負・NaN だと1回も進まずに戻り、呼び出し側のループが永久に終わらない。最低 1ms は進める
    const sliceMs = Math.max(1, sliceMsIn || 0)
    if (polished) return true
    if (!finished) {
      const sliceStart = performance.now()
      // ホットループではクロージャ変数ではなくローカル変数を使う
      let c = cost
      let bc = bestCost
      let it = iter
      let T = T0
      let best = bestCls
      const nB = B
      const nK = k
      const r = rand
      for (;;) {
        if ((it & 1023) === 0) {
          const now = performance.now()
          const frac = (activeMs + now - sliceStart) / opt.timeMs
          if (frac >= 1) {
            finished = true
            break
          }
          if (now - sliceStart >= sliceMs) break
          T = T0 * Math.pow(Tend / T0, frac)
          if (opt.onProgress && now - lastReport > 150) {
            lastReport = now
            opt.onProgress(bc, frac)
          }
        }
        it++
        const b1 = (r() * nB) | 0
        if (r() < 0.3) {
          const to = (r() * nK) | 0
          if (to === cls[b1]) continue
          const d = deltaMove(b1, to)
          if (d <= 0 || r() < Math.exp(-d / T)) {
            applyMove(b1, to)
            c += d
          }
        } else {
          const b2 = (r() * nB) | 0
          if (cls[b1] === cls[b2]) continue
          const d = deltaSwap(b1, b2)
          if (d <= 0 || r() < Math.exp(-d / T)) {
            applySwap(b1, b2)
            c += d
          }
        }
        if (c < bc - 1e-9) {
          bc = c
          best = Int32Array.from(cls)
        }
      }
      bestCls = best
      activeMs += performance.now() - sliceStart
      cost = c
      bestCost = bc
      iter = it
      // 途中で止めた: 次の run() で続きから（it & 1023 === 0 の地点なので時間判定から再開する）
      if (!finished) return false
    }
    polish()
    polished = true
    return true
  }

  return {
    run,
    result: () => {
      if (!res) throw new Error('annealer has not finished')
      return res
    },
  }
}
