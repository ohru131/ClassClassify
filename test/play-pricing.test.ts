import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

// @ts-expect-error JavaScript のスクリプト（型定義なし）
import { CSV_PATH, microsToMoney, parsePricingCsv } from '../scripts/push-play-pricing.mjs'

// docs/play-console/pricing.csv（Play の国別価格の唯一の情報源）が、送る前の検証を通ることを確かめる
describe('Pro の国別価格（pricing.csv）', () => {
  const rows = parsePricingCsv(readFileSync(CSV_PATH, 'utf8')) as { region: string; currency: string; display: string; priceMicros: string; status: string }[]
  const by = (r: string) => rows.find((x) => x.region === r)

  it('検証を通り、基準の国が入っている', () => {
    expect(by('JP')).toMatchObject({ currency: 'JPY', display: '980', status: 'set' })
    expect(by('US')).toMatchObject({ currency: 'USD', display: '5.99', status: 'set' })
    for (const r of ['KR', 'GB', 'DE', 'ES', 'PT', 'BR', 'MX', 'CL', 'CO', 'PE', 'AU', 'CA']) expect(by(r)?.status).toBe('set')
  })

  it('ユーロ圏は全加盟国を同じ価格にする', () => {
    const euro = rows.filter((r) => r.currency === 'EUR' && r.status === 'set')
    expect(euro.length).toBe(20)
    expect(new Set(euro.map((r) => r.display))).toEqual(new Set(['5.99']))
  })

  it('壊れた行は送る前に止める', () => {
    const head = 'region,currency,price_display,price_micros,status,basis\n'
    expect(() => parsePricingCsv(head + 'JP,JPY,980,98000000,set,x')).toThrow(/一致しない/)
    expect(() => parsePricingCsv(head + 'DE,USD,5.99,5990000,set,x')).toThrow(/EUR/)
    expect(() => parsePricingCsv(head + 'US,USD,5.9,5900000,set,x')).toThrow(/小数2桁/)
    expect(() => parsePricingCsv(head + 'US,USD,5.99,5990000,maybe,x')).toThrow(/status/)
    expect(() => parsePricingCsv(head + 'US,USD,5.99,5990000,set,x\nUS,USD,5.99,5990000,set,x')).toThrow(/重複/)
    expect(parsePricingCsv(head + 'BR,BRL,19.90,19900000,set,"R$14,90〜19,90"')[0].basis).toBe('R$14,90〜19,90')
  })

  it('API の Money 型（units + nanos）に直す', () => {
    expect(microsToMoney('USD', '5990000')).toEqual({ currencyCode: 'USD', units: '5', nanos: 990000000 })
    expect(microsToMoney('JPY', '980000000')).toEqual({ currencyCode: 'JPY', units: '980', nanos: 0 })
  })
})
