#!/usr/bin/env node
// docs/store-listing.md の掲載文と submission-assets/ の画像を、Google Play の掲載情報へ反映する。
// UnitCalc（si-unit-calculator）の scripts/push-play-listing.mjs を FairClass 用に移植したもの。
// ビルド（AAB のアップロード）とは別系統で、いつでも実行できる。
//
//   node scripts/push-play-listing.mjs                                   # ドライラン（既定。通信しない・鍵も要らない）
//   node scripts/push-play-listing.mjs --validate --key sa.json          # edit を作って検証だけ。edit は破棄する
//   node scripts/push-play-listing.mjs --commit   --key sa.json          # 反映する（審査に入る）
//   node scripts/push-play-listing.mjs --lang ja-JP,ko-KR --commit --key sa.json
//   オプション: --skip-text / --skip-images / --package <id>
//
// 文言・画像の順・ファイル名は**このスクリプトに書かず、資料をパースして読む**（資料を直したときに黙って食い違わせないため。読めなければ落とす）:
//   docs/store-listing.md            タイトル・短い説明・詳しい説明（掲載ロケールごと）
//   submission-assets/README.md      言語ごとに Play へ上げる8枚の順
//   submission-assets/screenshots/   phone / tablet7 / tablet10 のスクリーンショット
//   submission-assets/store/         アイコン・フィーチャーグラフィック
//
// 前提:
// - サービスアカウントの鍵（JSON）は**コミットしない**（.gitignore 済み）。--key か環境変数 GOOGLE_PLAY_SERVICE_ACCOUNT_JSON、
//   既定はリポジトリ直下の play-service-account.json。Play Console の「ユーザーと権限」でこのアプリへの
//   「ストアの掲載情報の管理」権限を与えておく。
// - 掲載ロケールは API（PUT listings/{locale}）が作るので、Console で先に追加しなくてよい。
// - API が扱う画像の種別は phone / 7インチ / 10インチ / フィーチャー / アイコンだけ。
//   **Chromebook・デスクトップのスクリーンショットは API に無い**ので、Console で手動で上げる。
import { createSign } from 'node:crypto'
import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)))
const COPY_PATH = join(ROOT, 'docs', 'store-listing.md')
const ASSETS_README = join(ROOT, 'submission-assets', 'README.md')
const SHOTS_DIR = join(ROOT, 'submission-assets', 'screenshots')
const STORE_DIR = join(ROOT, 'submission-assets', 'store')
const API = 'https://androidpublisher.googleapis.com/androidpublisher/v3'
const UPLOAD_API = 'https://androidpublisher.googleapis.com/upload/androidpublisher/v3'
// mobile/app.config.ts の APP_ID と同じ値
const DEFAULT_PACKAGE = 'com.ohru131.mosaic'
const DEFAULT_KEY = join(ROOT, 'play-service-account.json')

// Play の掲載ロケール → 画像の言語（submission-assets/README.md の「掲載のロケールと、使う画像の言語」）
const IMAGE_LANG = {
  'en-US': 'en', 'en-AU': 'en', 'en-GB': 'en', 'ja-JP': 'ja', 'ko-KR': 'ko',
  'es-419': 'es', 'es-ES': 'es', 'de-DE': 'de', 'pt-BR': 'pt-BR',
}
const LIMITS = { title: 30, shortDescription: 80, fullDescription: 4000 }

// folder: screenshots 内のフォルダ / type: API の画像種別 / rule: Play の寸法の規格
const SHOT_KINDS = [
  { folder: 'phone', type: 'phoneScreenshots', rule: { min: 320, max: 3840, maxAspect: 2 } },
  { folder: 'tablet7', type: 'sevenInchScreenshots', rule: { min: 320, max: 3840, maxAspect: 2 } },
  { folder: 'tablet10', type: 'tenInchScreenshots', rule: { min: 1080, max: 7680, maxAspect: 2 } },
]

// 文字数は Python の len()（コードポイント数）。資料の字数表記がその数え方
const charLen = (s) => [...s].length
const readText = (path) => readFileSync(path, 'utf8').replace(/\r\n/g, '\n')

// ---------------------------------------------------------------- 引数

const VALUE_FLAGS = { '--package': 'package', '--key': 'key' }

export function parseArgs(argv) {
  const args = { mode: 'dry-run', package: DEFAULT_PACKAGE, key: null, locales: null, skipImages: false, skipText: false }
  for (let i = 0; i < argv.length; i += 1) {
    const value = argv[i]
    if (value in VALUE_FLAGS || value === '--lang') {
      const next = argv[i + 1]
      // 値が無いまま既定へ落とすと、打ち間違いで意図しない鍵・パッケージへ送ってしまう
      if (next === undefined || next === '' || next.startsWith('-')) throw new Error(`${value} に値がない`)
      if (value === '--lang') args.locales = next.split(',').map((s) => s.trim()).filter(Boolean)
      else args[VALUE_FLAGS[value]] = next
      i += 1
    } else if (value === '--validate') args.mode = 'validate'
    else if (value === '--commit') args.mode = 'commit'
    else if (value === '--dry-run') args.mode = 'dry-run'
    else if (value === '--skip-images') args.skipImages = true
    else if (value === '--skip-text') args.skipText = true
    else throw new Error(`不明な引数: ${value}`)
  }
  return args
}

// ---------------------------------------------------------------- 資料のパース

/** docs/store-listing.md を `## ロケール` ごとに読む */
export function parseListings(text) {
  const listings = {}
  const sections = text.split(/^## /m).slice(1)
  for (const section of sections) {
    const locale = /^([A-Za-z]{2,3}(?:-[A-Za-z0-9]+)?)（/.exec(section)?.[1]
    if (!locale || !IMAGE_LANG[locale]) continue
    const block = (label) => {
      const m = new RegExp(`### ${label}（([\\d,]+)字 / \\d+）\\n\\n\`\`\`\\n([\\s\\S]*?)\\n\`\`\``).exec(section)
      if (!m) throw new Error(`${locale}: 「${label}」のブロックが読めない`)
      return { stated: Number(m[1].replace(/,/g, '')), text: m[2] }
    }
    const title = block('アプリ名')
    const short = block('短い説明')
    const full = block('詳しい説明')
    listings[locale] = {
      title: title.text, shortDescription: short.text, fullDescription: full.text,
      stated: { title: title.stated, shortDescription: short.stated, fullDescription: full.stated },
    }
  }
  return listings
}

/** README の「Play へ上げる順（言語ごと）」の表を読み、画像の言語 → ['04-results' ではなく '04' …] にする */
export function parseShotOrder(text) {
  const start = text.indexOf('### Play へ上げる順（言語ごと）')
  if (start < 0) throw new Error('「Play へ上げる順」の節が見つからない（submission-assets/README.md）')
  const end = text.indexOf('\n\n##', start + 1)
  const section = text.slice(start, end < 0 ? undefined : end)
  const order = {}
  for (const line of section.split('\n')) {
    const cells = line.split('|').map((s) => s.trim())
    // | 言語 | 順番 | 理由 | → cells = ['', 言語, 順番, 理由, '']
    if (cells.length < 4 || !/\d{2}\s*→/.test(cells[2])) continue
    const nums = cells[2].split('→').map((s) => s.trim())
    if (nums.length !== 8 || nums.some((n) => !/^\d{2}$/.test(n))) throw new Error(`枚数・形式が不正な行: ${line}`)
    for (const lang of cells[1].split('・').map((s) => s.trim())) order[lang] = nums
  }
  return order
}

// ---------------------------------------------------------------- 画像

function pngInfo(path) {
  const buf = readFileSync(path)
  if (buf.length < 26 || buf.readUInt32BE(0) !== 0x89504e47) throw new Error(`PNG ではない: ${path}`)
  // colorType 4・6 はアルファ付き
  return { buf, width: buf.readUInt32BE(16), height: buf.readUInt32BE(20), alpha: buf[25] === 4 || buf[25] === 6 }
}

function checkImage(path, rule) {
  const img = pngInfo(path)
  const problems = []
  if (rule.exact && (img.width !== rule.exact[0] || img.height !== rule.exact[1])) problems.push(`寸法が ${rule.exact.join('×')} でない（${img.width}×${img.height}）`)
  if (rule.min && Math.min(img.width, img.height) < rule.min) problems.push(`短辺が ${rule.min}px 未満`)
  if (rule.max && Math.max(img.width, img.height) > rule.max) problems.push(`長辺が ${rule.max}px 超`)
  if (rule.maxAspect && Math.max(img.width, img.height) / Math.min(img.width, img.height) > rule.maxAspect) problems.push(`アスペクト比が ${rule.maxAspect}:1 超`)
  if (rule.noAlpha && img.alpha) problems.push('アルファチャンネルがある（Play は不可）')
  return { ...img, problems }
}

// ---------------------------------------------------------------- 計画と検証

export function buildPlan({ listings, order, locales, skipImages, skipText }) {
  const plan = []
  const problems = []
  for (const locale of locales) {
    const entry = { locale, text: null, shots: [], featureGraphic: null, icon: null }
    if (!skipText) {
      const t = listings[locale]
      if (!t) { problems.push(`${locale}: docs/store-listing.md に掲載文が無い`); continue }
      entry.text = { title: t.title, shortDescription: t.shortDescription, fullDescription: t.fullDescription }
      for (const [field, limit] of Object.entries(LIMITS)) {
        const actual = charLen(t[field])
        if (actual > limit) problems.push(`${locale} ${field}: ${actual}字で上限${limit}字超`)
        // 資料の表記と実測の食い違い＝資料が古い
        if (t.stated[field] !== actual) problems.push(`${locale} ${field}: 資料の表記 ${t.stated[field]}字 と実測 ${actual}字 が違う`)
      }
    }
    if (!skipImages) {
      const lang = IMAGE_LANG[locale]
      const nums = order[lang]
      if (!nums) { problems.push(`${locale}: README に画像の順が無い（${lang}）`); continue }
      for (const kind of SHOT_KINDS) {
        const dir = join(SHOTS_DIR, kind.folder)
        const files = existsSync(dir) ? readdirSync(dir) : []
        for (const n of nums) {
          const name = files.find((f) => f.startsWith(`${lang}-${n}-`) && f.endsWith('.png'))
          if (!name) { problems.push(`${locale} ${kind.folder}: ${lang}-${n}-*.png が無い`); continue }
          const img = checkImage(join(dir, name), kind.rule)
          img.problems.forEach((p) => problems.push(`${locale} ${kind.folder}/${name}: ${p}`))
          entry.shots.push({ ...kind, name, ...img })
        }
      }
      const fg = join(STORE_DIR, `play-feature-graphic-${lang}-1024x500.png`)
      if (!existsSync(fg)) problems.push(`${locale}: フィーチャーグラフィックが無い`)
      else {
        const img = checkImage(fg, { exact: [1024, 500], noAlpha: true })
        img.problems.forEach((p) => problems.push(`${locale} featureGraphic: ${p}`))
        entry.featureGraphic = img
      }
      const icon = join(STORE_DIR, 'play-icon-512.png')
      if (!existsSync(icon)) problems.push('アイコン（store/play-icon-512.png）が無い')
      else {
        const img = checkImage(icon, { exact: [512, 512] })
        img.problems.forEach((p) => problems.push(`icon: ${p}`))
        entry.icon = img
      }
    }
    plan.push(entry)
  }
  return { plan, problems }
}

function printPlan(plan, { mode, pkg }) {
  console.log(`パッケージ: ${pkg}`)
  console.log(`モード: ${mode}${mode === 'dry-run' ? '（通信しない）' : mode === 'validate' ? '（検証のみ・反映しない）' : '（★反映する）'}`)
  for (const e of plan) {
    console.log(`\n[${e.locale}]`)
    if (e.text) {
      console.log(`  アプリ名     ${charLen(e.text.title)}/30  ${e.text.title}`)
      console.log(`  短い説明     ${charLen(e.text.shortDescription)}/80  ${e.text.shortDescription}`)
      console.log(`  詳しい説明   ${charLen(e.text.fullDescription)}/4000`)
    }
    for (const kind of SHOT_KINDS) {
      const shots = e.shots.filter((s) => s.type === kind.type)
      if (shots.length) console.log(`  ${kind.folder} ${shots.length}枚: ${shots.map((s) => s.name.replace(/^[^-]+(?:-BR)?-/, '').replace('.png', '')).join(' → ')}`)
    }
    if (e.featureGraphic) console.log('  フィーチャーグラフィック 1024×500')
    if (e.icon) console.log('  アイコン 512×512')
  }
}

// ---------------------------------------------------------------- 認証・API

// サービスアカウントの JSON 鍵から RS256 の JWT を作ってアクセストークンへ交換する（外部ライブラリなし）
async function getAccessToken(keyPath) {
  const key = JSON.parse(readFileSync(keyPath, 'utf8'))
  if (!key.client_email || !key.private_key) throw new Error(`サービスアカウントの鍵に見えない: ${keyPath}`)
  const now = Math.floor(Date.now() / 1000)
  const b64 = (o) => Buffer.from(JSON.stringify(o)).toString('base64url')
  const aud = key.token_uri ?? 'https://oauth2.googleapis.com/token'
  const unsigned = `${b64({ alg: 'RS256', typ: 'JWT' })}.${b64({ iss: key.client_email, scope: 'https://www.googleapis.com/auth/androidpublisher', aud, iat: now, exp: now + 3600 })}`
  const signer = createSign('RSA-SHA256')
  signer.update(unsigned)
  const res = await fetch(aud, {
    method: 'POST',
    headers: { 'content-type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ grant_type: 'urn:ietf:params:oauth:grant-type:jwt-bearer', assertion: `${unsigned}.${signer.sign(key.private_key, 'base64url')}` }),
  })
  const body = await res.json()
  if (!res.ok) throw new Error(`トークン取得に失敗 (${res.status}): ${JSON.stringify(body)}`)
  return body.access_token
}

async function api(token, method, path, { json, body, contentType } = {}) {
  const res = await fetch(path.startsWith('http') ? path : `${API}${path}`, {
    method,
    headers: { authorization: `Bearer ${token}`, ...(json ? { 'content-type': 'application/json' } : {}), ...(contentType ? { 'content-type': contentType } : {}) },
    body: json ? JSON.stringify(json) : body,
  })
  const text = await res.text()
  let parsed = null
  try { parsed = text ? JSON.parse(text) : null } catch { /* 画像アップロードは空応答のことがある */ }
  if (!res.ok) throw new Error(`${method} ${path} が ${res.status}: ${parsed?.error?.message ?? text.slice(0, 400)}`)
  return parsed
}

async function pushListing({ token, pkg, plan, mode }) {
  const { id } = await api(token, 'POST', `/applications/${pkg}/edits`)
  console.log(`\nedit を作成: ${id}`)
  let committed = false
  try {
    for (const e of plan) {
      const base = `/applications/${pkg}/edits/${id}/listings/${e.locale}`
      if (e.text) {
        await api(token, 'PUT', base, { json: e.text })
        console.log(`  [${e.locale}] 文言を更新`)
      }
      // 並びを変える手段が「消して入れ直す」だけ。アップロード順がそのまま掲載順なので直列で上げる
      const replace = async (type, items) => {
        await api(token, 'DELETE', `${base}/${type}`)
        for (const item of items) await api(token, 'POST', `${UPLOAD_API}${base}/${type}?uploadType=media`, { body: item.buf, contentType: 'image/png' })
        console.log(`  [${e.locale}] ${type} ${items.length}枚を差し替え`)
      }
      for (const kind of SHOT_KINDS) {
        const shots = e.shots.filter((s) => s.type === kind.type)
        if (shots.length) await replace(kind.type, shots)
      }
      if (e.featureGraphic) await replace('featureGraphic', [e.featureGraphic])
      if (e.icon) await replace('icon', [e.icon])
    }
    if (mode === 'commit') {
      await api(token, 'POST', `/applications/${pkg}/edits/${id}:commit`)
      committed = true
      console.log('\n★ commit した。Play Console に反映された（審査に入る）。')
    } else {
      await api(token, 'POST', `/applications/${pkg}/edits/${id}:validate`)
      console.log('\n検証を通過。反映はしていない。')
    }
  } finally {
    // commit 済みの edit は消せない。検証だけのときは必ず片付ける
    if (!committed) {
      await api(token, 'DELETE', `/applications/${pkg}/edits/${id}`)
        .then(() => console.log(`edit ${id} を破棄した（ストアには何も反映されていない）。`))
        .catch((err) => console.error(`edit ${id} の破棄に失敗: ${err.message}`))
    }
  }
}

// ---------------------------------------------------------------- 本体

async function main() {
  const args = parseArgs(process.argv.slice(2))
  const listings = parseListings(readText(COPY_PATH))
  const order = parseShotOrder(readText(ASSETS_README))
  const locales = args.locales ?? Object.keys(IMAGE_LANG)
  for (const l of locales) if (!IMAGE_LANG[l]) throw new Error(`未知のロケール: ${l}（${Object.keys(IMAGE_LANG).join(', ')}）`)

  const { plan, problems } = buildPlan({ listings, order, locales, skipImages: args.skipImages, skipText: args.skipText })
  printPlan(plan, { mode: args.mode, pkg: args.package })
  if (problems.length) {
    console.error(`\n検証で ${problems.length} 件の問題:`)
    problems.forEach((p) => console.error(`  - ${p}`))
    process.exitCode = 1
    return
  }
  console.log('\n検証OK（文字数・画像の寸法・枚数・資料との突き合わせ）。')
  if (args.mode === 'dry-run') {
    console.log('ドライランなので通信しない。送るなら --validate（検証のみ）か --commit（反映）を付ける。')
    return
  }

  const keyPath = args.key ?? process.env.GOOGLE_PLAY_SERVICE_ACCOUNT_JSON ?? DEFAULT_KEY
  if (!existsSync(keyPath)) throw new Error(`鍵が見つからない: ${keyPath}（--key か GOOGLE_PLAY_SERVICE_ACCOUNT_JSON で渡す）`)
  await pushListing({ token: await getAccessToken(keyPath), pkg: args.package, plan, mode: args.mode })
}

// import されたとき（テスト）は実行しない
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((e) => {
    console.error(`\n失敗: ${e.message}`)
    process.exitCode = 1
  })
}
