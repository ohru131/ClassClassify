#!/usr/bin/env node
// submission-assets/ の画像が Google Play の規格に合っているかを確かめる（依存なし。PNG のヘッダーを直接読む）。
//
//   node scripts/check-submission-assets.mjs
//
// 規格（2026-10 時点。Play Console のヘルプ）:
//   アプリのアイコン        512×512、32 ビット PNG（アルファあり可）
//   フィーチャーグラフィック 1024×500、JPEG または 24 ビット PNG（アルファなし）
//   スクリーンショット       JPEG または 24 ビット PNG（アルファなし）、各辺 320〜3840px、長辺:短辺 ≤ 2:1
//                            スマホは 2〜8 枚、7・10 インチタブレット・Chromebook は各 8 枚まで
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)))
const DIR = join(ROOT, 'submission-assets')
const LANGS = ['ja', 'en', 'ko', 'es', 'de', 'pt-BR']
const FORMS = ['phone', 'tablet7', 'tablet10', 'chromebook']

/** PNG の幅・高さ・色の種類（2 = RGB、6 = RGBA）と1色あたりのビット数 */
function pngInfo(file) {
  const b = readFileSync(file)
  if (b.readUInt32BE(0) !== 0x89504e47 || b.toString('ascii', 12, 16) !== 'IHDR') throw new Error(`PNG ではない: ${file}`)
  return { width: b.readUInt32BE(16), height: b.readUInt32BE(20), bitDepth: b[24], colorType: b[25], bytes: b.length }
}

const problems = []
const check = (cond, msg) => cond || problems.push(msg)
let total = 0
let count = 0
const add = (info) => {
  total += info.bytes
  count += 1
}

const icon = pngInfo(join(DIR, 'store', 'play-icon-512.png'))
add(icon)
check(icon.width === 512 && icon.height === 512, `アイコンが 512×512 ではない（${icon.width}×${icon.height}）`)
check(icon.colorType === 6 && icon.bitDepth === 8, 'アイコンが 32 ビット PNG（RGBA・8 ビット）ではない')

for (const lang of LANGS) {
  const fg = pngInfo(join(DIR, 'store', `play-feature-graphic-${lang}-1024x500.png`))
  add(fg)
  check(fg.width === 1024 && fg.height === 500, `フィーチャーグラフィック ${lang} が 1024×500 ではない`)
  check(fg.colorType === 2 && fg.bitDepth === 8, `フィーチャーグラフィック ${lang} が 24 ビット PNG（アルファなし）ではない`)
}

const table = {}
for (const form of FORMS) {
  const files = readdirSync(join(DIR, 'screenshots', form)).filter((f) => f.endsWith('.png'))
  for (const lang of LANGS) {
    const mine = files.filter((f) => f.startsWith(`${lang}-`))
    table[`${form}/${lang}`] = mine.length
    check(mine.length >= (form === 'phone' ? 2 : 4) && mine.length <= 8, `${form}/${lang} の枚数が範囲外（${mine.length}）`)
    for (const f of mine) {
      const info = pngInfo(join(DIR, 'screenshots', form, f))
      add(info)
      const long = Math.max(info.width, info.height)
      const short = Math.min(info.width, info.height)
      check(short >= 320 && long <= 3840, `${form}/${f}: 辺の長さが 320〜3840 の外（${info.width}×${info.height}）`)
      check(long / short <= 2, `${form}/${f}: 縦横比が 2:1 を超える`)
      check(info.colorType === 2, `${form}/${f}: アルファ付き（24 ビット PNG にする）`)
      if (form === 'phone') check(short >= 1080, `${form}/${f}: 短辺が 1080 未満（おすすめ枠の条件）`)
    }
  }
}

const sizes = {}
for (const form of FORMS) {
  const f = readdirSync(join(DIR, 'screenshots', form)).find((x) => x.endsWith('.png'))
  const i = pngInfo(join(DIR, 'screenshots', form, f))
  sizes[form] = `${i.width}×${i.height}`
}
console.log('画面サイズ:', sizes)
console.log('枚数:', table)
console.log(`画像 ${count} 枚・合計 ${(total / 1024 / 1024).toFixed(1)} MB（submission-assets 全体は ${(dirSize(DIR) / 1024 / 1024).toFixed(1)} MB）`)
if (problems.length) {
  console.error(`\n規格に合わないもの ${problems.length} 件:\n- ${problems.join('\n- ')}`)
  process.exitCode = 1
} else console.log('すべて Play の規格に合っている。')

function dirSize(dir) {
  let n = 0
  for (const e of readdirSync(dir, { withFileTypes: true })) n += e.isDirectory() ? dirSize(join(dir, e.name)) : statSync(join(dir, e.name)).size
  return n
}
