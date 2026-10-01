#!/usr/bin/env node
// docs/play-console/pricing.csv の国別価格を、Google Play の買い切り商品（Mosaic Pro）へ反映する。
// UnitCalc（si-unit-calculator）の scripts/push-play-pricing.mjs を Mosaic 用に移植したもの。
//
//   node scripts/push-play-pricing.mjs                    # ドライラン（既定。通信しない・鍵も要らない）
//   node scripts/push-play-pricing.mjs --list --key sa.json        # 商品の一覧を読むだけ
//   node scripts/push-play-pricing.mjs --plan --key sa.json        # 商品の今の設定と突き合わせるだけ（書き込まない）
//   node scripts/push-play-pricing.mjs --commit --key sa.json      # 反映する（既に設定がある国の価格だけ）
//   node scripts/push-play-pricing.mjs --commit --enable-new-regions --key sa.json  # 設定の無い国も販売開始する
//   node scripts/push-play-pricing.mjs --commit --include-unconfirmed   # status=confirm の行も送る
//
// 前提:
// - Play Console のアプリ内アイテム（一回限りの商品）を先に作っておく（既定の商品 ID は mosaic_pro）。
//   このスクリプトは商品を作らない。**既存の購入オプションの国別価格だけ**を書き換え、CSV に無い国は
//   今の設定（Play の自動換算）のまま残す。販売の可否（availability）は国ごとに今の値を保ち、
//   まだ設定の無い国は --enable-new-regions を付けたときだけ足す。
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

// 値を取る引数。値が無い（末尾）・次が別の引数（- で始まる）ときは、既定値に落とさずに止める
// （`--key --commit` の打ち間違いで既定の鍵・既定の商品へ黙って送らないため）
const VALUE_FLAGS = { '--package': 'package', '--sku': 'sku', '--key': 'key' }

export function parseArgs(argv) {
  const args = { mode: 'dry-run', package: DEFAULT_PACKAGE, sku: DEFAULT_SKU, key: null, list: false, plan: false, includeUnconfirmed: false, enableNewRegions: false }
  for (let i = 0; i < argv.length; i += 1) {
    const value = argv[i]
    if (value in VALUE_FLAGS) {
      const next = argv[i + 1]
      if (next === undefined || next === '' || next.startsWith('-')) throw new Error(`${value} に値がない（例: ${value} <値>）`)
      args[VALUE_FLAGS[value]] = next
      i += 1
    } else if (value === '--commit') args.mode = 'commit'
    else if (value === '--dry-run') args.mode = 'dry-run'
    else if (value === '--list') args.list = true
    else if (value === '--plan') args.plan = true
    else if (value === '--include-unconfirmed') args.includeUnconfirmed = true
    else if (value === '--enable-new-regions') args.enableNewRegions = true
    else if (value === '--help' || value === '-h') args.help = true
    else throw new Error(`不明な引数: ${value}`)
  }
  return args
}

/**
 * 今の購入オプションの国別設定に CSV の価格を重ねる。
 * - 既に設定がある国: 価格だけを差し替え、**availability は今の値のまま**（販売を止めた国を勝手に再開しない）
 * - 設定が無い国: enableNewRegions のときだけ AVAILABLE で足す。既定では足さずに newRegions として返す
 * - CSV に無い国: 触らない
 */
export function mergeRegionalConfigs(existing, rows, { enableNewRegions = false } = {}) {
  const byRegion = new Map((existing ?? []).map((c) => [c.regionCode, c]))
  const updated = []
  const added = []
  const newRegions = []
  for (const r of rows) {
    const price = microsToMoney(r.currency, r.priceMicros)
    const cur = byRegion.get(r.region)
    if (cur) {
      byRegion.set(r.region, { ...cur, price })
      updated.push(r.region)
    } else if (enableNewRegions) {
      byRegion.set(r.region, { regionCode: r.region, price, availability: 'AVAILABLE' })
      added.push(r.region)
    } else newRegions.push(r.region)
  }
  return { configs: [...byRegion.values()], updated, added, newRegions }
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
    console.log('node scripts/push-play-pricing.mjs [--dry-run|--plan|--commit] [--list] [--include-unconfirmed] [--enable-new-regions] [--package com.ohru131.mosaic] [--sku mosaic_pro] [--key sa.json]')
    return
  }

  const all = parsePricingCsv(readFileSync(CSV_PATH, 'utf8'))
  const rows = all.filter((r) => r.status === 'set' || args.includeUnconfirmed)
  const skipped = all.filter((r) => !rows.includes(r))
  console.log(`商品: ${args.package} / ${args.sku}`)
  console.log(`送る価格: ${rows.length}地域（CSV に無い国は今の設定＝Play の自動換算のまま）`)
  for (const r of rows) console.log(`  ${r.region}: ${r.currency} ${r.display}`)
  if (skipped.length) console.log(`送らない（status=confirm。確認後に --include-unconfirmed か status=set へ）: ${skipped.map((r) => `${r.region} ${r.currency} ${r.display}`).join(', ')}`)
  if (args.mode === 'dry-run' && !args.list && !args.plan) {
    console.log('ドライラン（通信しない）。商品の今の設定と突き合わせるには --plan、反映するには --commit を付ける。')
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
  const merged = mergeRegionalConfigs(current.purchaseOptions[0].regionalPricingAndAvailabilityConfigs, rows, { enableNewRegions: args.enableNewRegions })
  console.log(`価格を差し替える国（販売の可否は今のまま）: ${merged.updated.join(', ') || 'なし'}`)
  if (merged.added.length) console.log(`新しく販売を始める国（--enable-new-regions）: ${merged.added.join(', ')}`)
  if (merged.newRegions.length) console.log(`商品にまだ設定が無いので送らない国（足すなら --enable-new-regions）: ${merged.newRegions.join(', ')}`)
  if (args.mode !== 'commit') {
    console.log('--plan は読むだけ。反映するには --commit を付ける。')
    return
  }
  const purchaseOptions = current.purchaseOptions.map((option, index) => (index === 0 ? { ...option, regionalPricingAndAvailabilityConfigs: merged.configs } : option))
  const query = new URLSearchParams({ updateMask: 'purchaseOptions', 'regionsVersion.version': current.regionsVersion.version })
  // 公式の REST パスは get / list が `oneTimeProducts`、patch だけが `onetimeproducts`（小文字）。
  // androidpublisher v3 の discovery 文書（monetization.onetimeproducts）で確認した（2026-10-01）
  const updated = await api(token, 'PATCH', `/applications/${args.package}/onetimeproducts/${args.sku}?${query}`, { ...current, purchaseOptions })

  // 送った地域が期待どおりになったかを読み戻して確かめる
  const sent = new Set([...merged.updated, ...merged.added])
  const configs = updated?.purchaseOptions?.[0]?.regionalPricingAndAvailabilityConfigs ?? []
  const actual = new Map(configs.map((c) => [c.regionCode, c.price]))
  const wrong = rows.filter((r) => sent.has(r.region)).filter((r) => {
    const p = actual.get(r.region)
    if (!p || p.currencyCode !== r.currency) return true
    return BigInt(p.units ?? 0) * 1000000n + BigInt(p.nanos ?? 0) / 1000n !== BigInt(r.priceMicros)
  })
  if (wrong.length) throw new Error(`反映を確認できない地域: ${wrong.map((r) => r.region).join(', ')}`)
  console.log(`反映した（${sent.size}地域を読み戻して一致を確認）。`)
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(`\n失敗: ${error.message}`)
    process.exitCode = 1
  })
}
