import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { CancelledError, defaultStarts, nextSliceMs, runSliced } from '../lib/runner'
import { compile, evaluate, parseWorkbook } from '../lib/solver'

const load = (f: string) => {
  const b = readFileSync(new URL(`../../public/${f}`, import.meta.url))
  return parseWorkbook(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength))
}

describe.each(['sample1.xlsx', 'sample2.xlsx', 'sample-group.xlsx'])('%s', (file) => {
  it('時間を区切って進めても、条件違反0・人数差1以内に到達する', async () => {
    const p = load(file)
    const { compiled } = compile(p)
    let yields = 0
    const fractions: number[] = []
    const job = runSliced(compiled, {
      timeMs: 1500,
      starts: 2,
      sliceMs: 10,
      seed: 42,
      onProgress: (f) => fractions.push(f),
      yieldToUi: async () => {
        yields++
      },
    })
    const res = await job.promise
    const report = evaluate(p, res.classOf, p.numClasses)
    expect(report.violations).toEqual([])
    expect(Math.max(...report.sizes) - Math.min(...report.sizes)).toBeLessThanOrEqual(1)
    expect(report.totalExcess).toBe(0)
    // UI へ何度も制御を返している（1回で計算し切っていない）
    expect(yields).toBeGreaterThan(20)
    for (let i = 1; i < fractions.length; i++) expect(fractions[i]).toBeGreaterThanOrEqual(fractions[i - 1] - 1e-9)
    expect(fractions.at(-1)).toBe(1)
    expect(res.starts).toBe(2)
  })
})

describe('runSliced', () => {
  it('中止すると CancelledError で終わる', async () => {
    const p = load('sample1.xlsx')
    const { compiled } = compile(p)
    const job = runSliced(compiled, { timeMs: 10_000, sliceMs: 5, yieldToUi: () => new Promise((r) => setTimeout(r, 0)) })
    setTimeout(job.cancel, 50)
    const t0 = Date.now()
    await expect(job.promise).rejects.toBeInstanceOf(CancelledError)
    expect(Date.now() - t0).toBeLessThan(2000)
  })

  it('UI に返している時間は探索時間に数えない', async () => {
    const p = load('sample-group.xlsx')
    const { compiled } = compile(p)
    let yields = 0
    const t0 = Date.now()
    const res = await runSliced(compiled, {
      timeMs: 300,
      starts: 1,
      sliceMs: 5,
      yieldToUi: async () => {
        yields++
        await new Promise((r) => setTimeout(r, 5))
      },
    }).promise
    // 計算 300ms ＋ 待ち（5ms × 回数）ぶんの実時間がかかる
    expect(Date.now() - t0).toBeGreaterThanOrEqual(300 + yields * 4)
    expect(res.iterations).toBeGreaterThan(0)
  })

  it('UI に返す時間が長くても、計算時間を伸ばして実時間を探索時間の近くに収める', async () => {
    const p = load('sample-group.xlsx')
    const { compiled } = compile(p)
    let yields = 0
    const t0 = Date.now()
    await runSliced(compiled, {
      timeMs: 600,
      starts: 1,
      sliceMs: 5,
      yieldToUi: async () => {
        yields++
        // 実機の debug ビルドの再描画に相当する重さ
        await new Promise((r) => setTimeout(r, 30))
      },
    }).promise
    // 計算を伸ばさないと 600ms ÷ 5ms = 120回 × 30ms ≈ 3.6秒の待ちが加わる
    expect(yields).toBeLessThan(30)
    expect(Date.now() - t0).toBeLessThan(600 * 2)
  })

  it('計算時間は UI に返した時間の4倍（計算8割）を目安に、下限と上限の間に収める', () => {
    expect(nextSliceMs(0, 24, 200)).toBe(24)
    expect(nextSliceMs(10, 24, 200)).toBeCloseTo(40)
    expect(nextSliceMs(1000, 24, 200)).toBe(200)
  })

  it('探索時間からスタート回数を決める', () => {
    expect(defaultStarts(3000)).toBe(1)
    expect(defaultStarts(10000)).toBe(2)
    expect(defaultStarts(30000)).toBe(3)
  })
})
