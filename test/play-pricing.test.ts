import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

// @ts-expect-error JavaScript のスクリプト（型定義なし）
import { CSV_PATH, mergeRegionalConfigs, microsToMoney, parseArgs, parsePricingCsv } from '../scripts/push-play-pricing.mjs'

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

  it('値を取る引数に値が無いときは既定値に落とさず止める', () => {
    expect(() => parseArgs(['--key'])).toThrow(/--key に値がない/)
    expect(() => parseArgs(['--key', '--commit'])).toThrow(/--key に値がない/)
    expect(() => parseArgs(['--package', ''])).toThrow(/--package に値がない/)
    expect(() => parseArgs(['--sku', '-x'])).toThrow(/--sku に値がない/)
    // 指定しなければ既定値（指定なしと値なしを区別する）
    expect(parseArgs([])).toMatchObject({ mode: 'dry-run', package: 'com.ohru131.mosaic', sku: 'mosaic_pro', key: null, enableNewRegions: false })
    expect(parseArgs(['--commit', '--key', 'sa.json', '--sku', 'pro2'])).toMatchObject({ mode: 'commit', key: 'sa.json', sku: 'pro2' })
  })

  it('既存の国は価格だけ差し替えて販売の可否を保ち、設定の無い国は明示したときだけ足す', () => {
    const existing = [
      { regionCode: 'JP', price: microsToMoney('JPY', '800000000'), availability: 'AVAILABLE' },
      { regionCode: 'BR', price: microsToMoney('BRL', '9900000'), availability: 'NO_LONGER_AVAILABLE' },
      { regionCode: 'IN', price: microsToMoney('INR', '99000000'), availability: 'AVAILABLE' },
    ]
    const csv = [
      { region: 'JP', currency: 'JPY', priceMicros: '980000000' },
      { region: 'BR', currency: 'BRL', priceMicros: '19900000' },
      { region: 'CL', currency: 'CLP', priceMicros: '3990000000' },
    ]
    const r = mergeRegionalConfigs(existing, csv)
    const by = new Map(r.configs.map((c: { regionCode: string }) => [c.regionCode, c]))
    expect(by.get('JP')).toEqual({ regionCode: 'JP', price: microsToMoney('JPY', '980000000'), availability: 'AVAILABLE' })
    expect(by.get('BR')).toMatchObject({ availability: 'NO_LONGER_AVAILABLE', price: { units: '19' } })
    expect(by.get('IN')).toEqual(existing[2])
    expect(by.has('CL')).toBe(false)
    expect(r).toMatchObject({ updated: ['JP', 'BR'], added: [], newRegions: ['CL'] })

    const withNew = mergeRegionalConfigs(existing, csv, { enableNewRegions: true })
    expect(withNew).toMatchObject({ added: ['CL'], newRegions: [] })
    expect(withNew.configs.find((c: { regionCode: string }) => c.regionCode === 'CL')).toMatchObject({ availability: 'AVAILABLE' })
  })
})
