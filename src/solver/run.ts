import type { CompiledProblem } from './types'
import type { WorkerResponse } from './worker'

export interface RunResult {
  classOf: number[]
  cost: number
  iterations: number
  workers: number
}

/** 複数の Web Worker で並列にマルチスタートし、最良解を返す */
export function runParallel(
  problem: CompiledProblem,
  timeMs: number,
  onProgress: (fraction: number, best: number) => void,
): { promise: Promise<RunResult>; cancel: () => void } {
  const count = Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1))
  const workers: Worker[] = []
  const bests = new Array(count).fill(Infinity)
  let cancel = () => {}
  const promise = new Promise<RunResult>((resolve, reject) => {
    const done: { classOf: number[]; cost: number; iterations: number }[] = []
    cancel = () => {
      workers.forEach((w) => w.terminate())
      reject(new Error('cancelled'))
    }
    for (let w = 0; w < count; w++) {
      const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
      workers.push(worker)
      worker.onerror = (e) => {
        workers.forEach((x) => x.terminate())
        reject(new Error(e.message))
      }
      worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
        const m = e.data
        if (m.type === 'progress') {
          bests[w] = m.best
          onProgress(m.fraction, Math.min(...bests))
          return
        }
        done.push(m)
        worker.terminate()
        if (done.length === count) {
          const best = done.reduce((a, b) => (b.cost < a.cost ? b : a))
          resolve({ ...best, iterations: done.reduce((s, d) => s + d.iterations, 0), workers: count })
        }
      }
      worker.postMessage({ problem, timeMs, seed: (Date.now() + w * 7919) >>> 0 })
    }
  })
  return { promise, cancel }
}
