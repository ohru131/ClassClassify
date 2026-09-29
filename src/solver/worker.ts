import { anneal } from './anneal'
import type { CompiledProblem } from './types'

export type WorkerRequest = { problem: CompiledProblem; timeMs: number; seed: number }
export type WorkerResponse =
  | { type: 'progress'; best: number; fraction: number }
  | { type: 'done'; classOf: number[]; cost: number; iterations: number }

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const { problem, timeMs, seed } = e.data
  const res = anneal(problem, {
    timeMs,
    seed,
    onProgress: (best, fraction) => self.postMessage({ type: 'progress', best, fraction } satisfies WorkerResponse),
  })
  self.postMessage({ type: 'done', classOf: res.classOf, cost: res.cost, iterations: res.iterations } satisfies WorkerResponse)
}
