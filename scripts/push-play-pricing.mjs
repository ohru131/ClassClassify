#!/usr/bin/env node
// docs/play-console/pricing.csv の国別価格を、Google Play の買い切り商品（Mosaic Pro）へ反映する。
// UnitCalc（si-unit-calculator）の scripts/push-play-pricing.mjs を Mosaic 用に移植したもの。
//
//   node scripts/push-play-pricing.mjs                    # ドライラン（既定。通信しない・鍵も要らない）
//   node scripts/push-play-pricing.mjs --list --key sa.json        # 商品の一覧を読むだけ
//   node scripts/push-play-pricing.mjs --commit --key sa.json      # 反映する
//   node scripts/push-play-pricing.mjs --commit --include-unconfirmed   # status=confirm の行も送る
//
// 前提:
// - Play Console のアプリ内アイテム（一回限りの商品）を先に作っておく（既定の商品 ID は mosaic_pro）。
//   このスクリプトは商品を作らない。**既存の購入オプションの国別価格だけ**を書き換え、CSV に無い国は
//   今の設定（Play の自動換算）のまま残す。
// - サービスアカウントの鍵（JSON）は **コミットしない**（.gitignore の play-service-account*.json）。
//   --key か環境変数 GOOGLE_PLAY_SERVICE_ACCOUNT_JSON で渡す。既定はリポジトリ直下の
//   play-service-account.json。Play Console の「ユーザーと権限」でこのアカウントに
//   「財務データの表示・注文と定期購入の管理」の権限を与えること。
// - 2025 年に Play Console の価格 CSV インポートと価格テンプレートは廃止された。まとめて入れる手段は
//   Play Developer API（monetization.onetimeproducts）だけで、このスクリプトはそれを使う。
//
// CSV の status 列: set = 反映する / confirm = Play 側の通貨・為替を確かめてから（既定では送らない）。
import { createSign } from 'node:crypto'
import { existsSync, readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)))
export const CSV_PATH = join(ROOT, 'docs', 'play-console', 'pricing.csv')
const API = 'https://androidpublisher.googleapis.com/androidpublisher/v3'
// mobile/app.config.ts の APP_ID と同じ値（**初回アップロード前に確定させること**。公開後は変えられない）
const DEFAULT_PACKAGE = 'com.ohru131.mosaic'
const DEFAULT_SKU = 'mosaic_pro'
const DEFAULT_KEY = join(ROOT, 'play-service-account.json')
const HEADER = 'region,currency,price_display,price_micros,status,basis'

// 通貨の打ち間違い（ユーロ圏に USD を書く等）を送る前に止めるための一覧。CSV に書く国だけ持つ
const EXPECTED_CURRENCY = {
  US: 'USD', JP: 'JPY', KR: 'KRW', GB: 'GBP', CA: 'CAD', AU: 'AUD', NZ: 'NZD', CH: 'CHF',
  BR: 'BRL', MX: 'MXN', CL: 'CLP', CO: 'COP', PE: 'PEN', EC: 'USD', AR: 'ARS', BG: 'EUR',
  ...Object.fromEntries('AT BE CY DE EE ES FI FR GR HR IE IT LT LU LV MT NL PT SI SK'.split(' ').map((r) => [r, 'EUR'])),
}
// 小数を持たない通貨（Play も整数で受け取る）
const ZERO_DECIMAL = new Set(['JPY', 'KRW', 'CLP', 'COP'])

/** 1行を項目に分ける（basis 列はダブルクォートで囲まれ、中にカンマを含みうる） */
function splitCsvLine(line) {
  const out = []
  let cur = ''
  let quoted = false
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i]
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') {
        cur += '"'
        i += 1
      } else if (ch === '"') quoted = false
      else cur += ch
    } else if (ch === '"') quoted = true
    else if (ch === ',') {
      out.push(cur)
      cur = ''
    } else cur += ch
  }
  out.push(cur)
  return out
}

/** CSV を読んで検証する。不正なら行番号つきで投げる */
export function parsePricingCsv(text) {
  const lines = text.replace(/\r\n/g, '\n').trim().split('\n')
  if (lines.shift() !== HEADER) throw new Error(`pricing.csv の見出しが「${HEADER}」ではない`)
  const rows = []
  const seen = new Set()
  for (const [index, line] of lines.entries()) {
    const at = `pricing.csv ${index + 2}行目`
    const [region, currency, display, micros, status, basis] = splitCsvLine(line)
    if (!/^[A-Z]{2}$/.test(region) || !/^[A-Z]{3}$/.test(currency)) throw new Error(`${at}: 地域・通貨の形式が不正`)
    if (seen.has(region)) throw new Error(`${at}: ${region} が重複している`)
    seen.add(region)
    if (EXPECTED_CURRENCY[region] && EXPECTED_CURRENCY[region] !== currency) throw new Error(`${at}: ${region} の通貨は ${EXPECTED_CURRENCY[region]} のはず（${currency}）`)
    const decimals = ZERO_DECIMAL.has(currency) ? /^\d+$/ : /^\d+\.\d{2}$/
    if (!decimals.test(display)) throw new Error(`${at}: ${currency} の表示価格は ${ZERO_DECIMAL.has(currency) ? '整数' : '小数2桁'}で書く（${display}）`)
    if (!/^\d+$/.test(micros)) throw new Error(`${at}: price_micros が整数ではない`)
    const [units, frac = ''] = display.split('.')
    const expected = BigInt(units) * 1000000n + BigInt((frac + '000000').slice(0, 6))
    if (BigInt(micros) !== expected) throw new Error(`${at}: price_micros（${micros}）が表示価格 ${display} と一致しない（${expected}）`)
    if (status !== 'set' && status !== 'confirm') throw new Error(`${at}: status は set か confirm`)
    if (!basis) throw new Error(`${at}: basis（根拠）が空`)
    rows.push({ region, currency, display, priceMicros: micros, status, basis })
  }
  return rows
}

export function microsToMoney(currencyCode, priceMicros) {
  const micros = BigInt(priceMicros)
  return { currencyCode, units: (micros / 1000000n).toString(), nanos: Number((micros % 1000000n) * 1000n) }
}

function parseArgs(argv) {
  const args = { mode: 'dry-run', package: DEFAULT_PACKAGE, sku: DEFAULT_SKU, key: null, list: false, includeUnconfirmed: false }
  for (let i = 0; i < argv.length; i += 1) {
    const value = argv[i]
    if (value === '--commit') args.mode = 'commit'
    else if (value === '--dry-run') args.mode = 'dry-run'
    else if (value === '--package') args.package = argv[++i]
    else if (value === '--sku') args.sku = argv[++i]
    else if (value === '--key') args.key = argv[++i]
    else if (value === '--list') args.list = true
    else if (value === '--include-unconfirmed') args.includeUnconfirmed = true
    else if (value === '--help' || value === '-h') args.help = true
    else throw new Error(`不明な引数: ${value}`)
  }
  return args
}

async function getAccessToken(keyPath) {
  const key = JSON.parse(readFileSync(keyPath, 'utf8'))
  if (!key.client_email || !key.private_key) throw new Error(`サービスアカウントの鍵に見えない: ${keyPath}`)
  const now = Math.floor(Date.now() / 1000)
  const b64 = (value) => Buffer.from(JSON.stringify(value)).toString('base64url')
  const claim = { iss: key.client_email, scope: 'https://www.googleapis.com/auth/androidpublisher', aud: key.token_uri ?? 'https://oauth2.googleapis.com/token', iat: now, exp: now + 3600 }
  const unsigned = `${b64({ alg: 'RS256', typ: 'JWT' })}.${b64(claim)}`
  const signer = createSign('RSA-SHA256')
  signer.update(unsigned)
  const jwt = `${unsigned}.${signer.sign(key.private_key, 'base64url')}`
  const response = await fetch(claim.aud, {
    method: 'POST',
    headers: { 'content-type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ grant_type: 'urn:ietf:params:oauth:grant-type:jwt-bearer', assertion: jwt }),
  })
  const body = await response.json()
  if (!response.ok) throw new Error(`トークン取得に失敗 (${response.status}): ${JSON.stringify(body)}`)
  return body.access_token
}

async function api(token, method, path, json) {
  const response = await fetch(`${API}${path}`, {
    method,
    headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
    body: json ? JSON.stringify(json) : undefined,
  })
  const text = await response.text()
  let body = null
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    // ゲートウェイのエラーは HTML のことがある
  }
  if (!response.ok) throw new Error(`${method} ${path} が ${response.status}: ${body?.error?.message ?? text.slice(0, 800)}`)
  return body
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  if (args.help) {
    console.log('node scripts/push-play-pricing.mjs [--dry-run|--commit] [--list] [--include-unconfirmed] [--package com.ohru131.mosaic] [--sku mosaic_pro] [--key sa.json]')
    return
  }

  const all = parsePricingCsv(readFileSync(CSV_PATH, 'utf8'))
  const rows = all.filter((r) => r.status === 'set' || args.includeUnconfirmed)
  const skipped = all.filter((r) => !rows.includes(r))
  console.log(`商品: ${args.package} / ${args.sku}`)
  console.log(`送る価格: ${rows.length}地域（CSV に無い国は今の設定＝Play の自動換算のまま）`)
  for (const r of rows) console.log(`  ${r.region}: ${r.currency} ${r.display}`)
  if (skipped.length) console.log(`送らない（status=confirm。確認後に --include-unconfirmed か status=set へ）: ${skipped.map((r) => `${r.region} ${r.currency} ${r.display}`).join(', ')}`)
  if (args.mode === 'dry-run' && !args.list) {
    console.log('ドライラン。反映するには --commit を付ける。')
    return
  }

  const keyPath = args.key ?? process.env.GOOGLE_PLAY_SERVICE_ACCOUNT_JSON ?? DEFAULT_KEY
  if (!existsSync(keyPath)) throw new Error(`サービスアカウントの鍵が無い: ${keyPath}（--key <path> または GOOGLE_PLAY_SERVICE_ACCOUNT_JSON）`)
  const token = await getAccessToken(keyPath)
  if (args.list) {
    const result = await api(token, 'GET', `/applications/${args.package}/oneTimeProducts`)
    const products = result?.oneTimeProducts ?? []
    if (!products.length) console.log('一回限りの商品が見つからない（販売アカウントの設定か商品の作成が必要）')
    for (const p of products) console.log(`${p.productId}: ${p.listings?.[0]?.title ?? 'タイトルなし'} / 購入オプション ${p.purchaseOptions?.length ?? 0}`)
    return
  }

  const current = await api(token, 'GET', `/applications/${args.package}/oneTimeProducts/${args.sku}`)
  if (!current.purchaseOptions?.length) throw new Error('購入オプションが見つからない（Play Console で「購入」の購入オプションを作る）')
  if (!current.regionsVersion?.version) throw new Error('商品の regionsVersion が見つからない')
  console.log(`既存の商品: ${current.productId} / 購入オプション ${current.purchaseOptions.length}`)

  // 先頭の購入オプション（買い切りの「購入」）だけを書き換える。他の国の設定はそのまま残す
  const purchaseOptions = current.purchaseOptions.map((option, index) => {
    if (index !== 0) return option
    const byRegion = new Map((option.regionalPricingAndAvailabilityConfigs ?? []).map((c) => [c.regionCode, c]))
    for (const r of rows) byRegion.set(r.region, { regionCode: r.region, price: microsToMoney(r.currency, r.priceMicros), availability: 'AVAILABLE' })
    return { ...option, regionalPricingAndAvailabilityConfigs: [...byRegion.values()] }
  })
  const query = new URLSearchParams({ updateMask: 'purchaseOptions', 'regionsVersion.version': current.regionsVersion.version })
  const updated = await api(token, 'PATCH', `/applications/${args.package}/oneTimeProducts/${args.sku}?${query}`, { ...current, purchaseOptions })

  // 送った全地域が期待どおりになったかを読み戻して確かめる
  const configs = updated?.purchaseOptions?.[0]?.regionalPricingAndAvailabilityConfigs ?? []
  const actual = new Map(configs.map((c) => [c.regionCode, c.price]))
  const wrong = rows.filter((r) => {
    const p = actual.get(r.region)
    if (!p || p.currencyCode !== r.currency) return true
    return BigInt(p.units ?? 0) * 1000000n + BigInt(p.nanos ?? 0) / 1000n !== BigInt(r.priceMicros)
  })
  if (wrong.length) throw new Error(`反映を確認できない地域: ${wrong.map((r) => r.region).join(', ')}`)
  console.log(`反映した（${rows.length}地域を読み戻して一致を確認）。`)
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(`\n失敗: ${error.message}`)
    process.exitCode = 1
  })
}
