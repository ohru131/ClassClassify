import { describe, expect, it } from 'vitest'
import { stepClassCountByMax } from '../lib/class-size'

describe('最大人数のステッパー', () => {
  it('n=30・k=5（最大6）で + を押すと、表示が実際に変わる k=4（最大8）へ移る', () => {
    expect(stepClassCountByMax(30, 5, 1)).toEqual({ k: 4, max: 8 })
  })
  it('− は最大人数が減る次の k へ', () => {
    expect(stepClassCountByMax(30, 5, -1)).toEqual({ k: 6, max: 5 })
    expect(stepClassCountByMax(30, 4, -1)).toEqual({ k: 5, max: 6 })
  })
  it('端では動かない（k は2以上・生徒数以下）', () => {
    expect(stepClassCountByMax(30, 2, 1)).toBeNull()
    expect(stepClassCountByMax(30, 30, -1)).toBeNull()
  })
  it('どの n・k でも、+ で最大は増え、− で減る', () => {
    for (let n = 2; n <= 60; n++)
      for (let k = 2; k <= n; k++) {
        const cur = Math.ceil(n / k)
        const up = stepClassCountByMax(n, k, 1)
        const down = stepClassCountByMax(n, k, -1)
        if (up) expect(up.max).toBeGreaterThan(cur), expect(up.max).toBe(Math.ceil(n / up.k))
        if (down) expect(down.max).toBeLessThan(cur), expect(down.max).toBe(Math.ceil(n / down.k))
      }
  })
})
