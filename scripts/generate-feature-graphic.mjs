#!/usr/bin/env node
// Google Play のフィーチャーグラフィック（1024×500）を言語ごとに、ストアアイコン（512×512）を1枚作る。
//
//   node scripts/generate-feature-graphic.mjs              # 6言語ぶん＋アイコン
//   node scripts/generate-feature-graphic.mjs --lang ja,ko
//
// 出力: submission-assets/store/play-feature-graphic-<lang>-1024x500.png
//       submission-assets/store/play-icon-512.png
// Play の規格: フィーチャーグラフィックは JPEG または 24 ビット PNG（アルファなし）、
// アイコンは 512×512 の 32 ビット PNG（角丸・影は Play が付けるので、四角いまま全面を塗る）。
//
// 見出しは訳文ではなく、docs/store-listing.md の各言語の短い説明と同じ主張（均等に・数秒で・端末内）に
// 揃えてある。**価格・評価・ランキング・「No.1」のような文言は入れない**（Play のメタデータの
// ポリシーで、フィーチャーグラフィックへの載せ方が制限されている）。
//
// フォント: 日本語・韓国語は Noto Sans JP / Noto Sans KR を優先し、無ければ OS の CJK フォント
// （Yu Gothic・Malgun Gothic・Hiragino・Apple SD Gothic Neo）へ落ちる。Linux のサンドボックスで
// 韓国語が豆腐（□）や文泉驛の字形にならないよう、fc-list で Noto Sans KR があるか確かめること
// （submission-assets/README.md の「フォント」）。
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { launchChromium, loadSharp } from './lib/deps.mjs'

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)))
const OUT_DIR = join(ROOT, 'submission-assets', 'store')
const ICON = join(ROOT, 'mobile', 'assets', 'icon.png')
const WIDTH = 1024
const HEIGHT = 500
// Play は端末によって左右を切ることがあるので、文字とアイコンは中央の安全域に収める
const SAFE_WIDTH = 820

// headline: 何をするアプリか / sub: 誰のための・何が違うか（端末内・広告なし）
export const LOCALES = {
  ja: { font: 'ja', headline: 'クラス分けを、<br>偏りなく数秒で。', sub: '男女・学力・支援の必要な子を各クラスに均等に。<br>名簿は端末の中だけ。広告なし。' },
  en: { headline: 'Balanced class lists<br>in seconds.', sub: 'Gender, academics, support needs — spread evenly.<br>Student data stays on your device. No ads.' },
  ko: { font: 'ko', headline: '반 편성을<br>몇 초 만에 고르게.', sub: '성별·학업·지원이 필요한 학생을 반마다 고르게.<br>명단은 기기 안에만. 광고 없음.' },
  es: { headline: 'Grupos equilibrados<br>en segundos.', sub: 'Género, desempeño y apoyos, repartidos por igual.<br>Los datos no salen del dispositivo. Sin anuncios.' },
  de: { headline: 'Ausgewogene Klassen<br>in Sekunden.', sub: 'Geschlecht, Leistung, Förderbedarf – gleichmäßig verteilt.<br>Alle Daten bleiben auf dem Gerät. Ohne Werbung.' },
  'pt-BR': { headline: 'Turmas equilibradas<br>em segundos.', sub: 'Gênero, desempenho e apoio distribuídos por igual.<br>Os dados não saem do aparelho. Sem anúncios.' },
}

const FONT_STACK = {
  ja: '"Noto Sans JP", "Yu Gothic UI", "Yu Gothic", "Hiragino Sans", "Noto Sans CJK JP", sans-serif',
  ko: '"Noto Sans KR", "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans CJK KR", sans-serif',
  latin: '"Noto Sans", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
}

// アプリ内の組の色（mobile/components/theme.ts の classColor の並び）で、4組に均等に散った名簿を描く
const CLASS_DOTS = ['#6366F1', '#D946EF', '#10B981', '#F59E0B']
const KINDS = ['#FFFFFF', 'rgba(255,255,255,.55)', '#FDE68A']

function boardHtml() {
  return CLASS_DOTS.map(
    (color, c) =>
      `<div class="cls"><div class="bar" style="background:${color}"></div><div class="dots">${Array.from({ length: 9 }, (_, i) => `<i style="background:${KINDS[(i + c) % 3]}"></i>`).join('')}</div></div>`,
  ).join('')
}

function buildHtml(lang, iconDataUri) {
  const l = LOCALES[lang]
  const font = FONT_STACK[l.font ?? 'latin']
  return `<!doctype html><html lang="${lang}"><head><meta charset="utf-8"><style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { width: ${WIDTH}px; height: ${HEIGHT}px; }
  body { background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 55%, #C026D3 100%); display: flex; align-items: center; justify-content: center;
    font-family: ${font}; color: #fff; -webkit-font-smoothing: antialiased; }
  .safe { width: ${SAFE_WIDTH}px; display: flex; align-items: center; gap: 44px; }
  .copy { flex: 1; min-width: 0; }
  .brand { display: flex; align-items: center; gap: 14px; font-family: ${FONT_STACK.latin}; font-weight: 900; font-size: 30px; letter-spacing: .3px; }
  .brand img { width: 56px; height: 56px; border-radius: 14px; box-shadow: 0 6px 18px rgba(0,0,0,.25); }
  .headline { font-size: ${l.font ? 46 : 44}px; font-weight: 900; line-height: 1.18; margin-top: 22px; letter-spacing: ${l.font ? '0.5px' : '-0.5px'}; }
  .sub { font-size: 18px; font-weight: 600; line-height: 1.5; margin-top: 18px; color: #EDE9FE; }
  .board { flex: none; display: grid; grid-template-columns: repeat(2, 92px); gap: 12px; padding: 16px; background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.25); border-radius: 22px; }
  .cls { background: rgba(15,23,42,.22); border-radius: 12px; overflow: hidden; }
  .bar { height: 6px; }
  .dots { display: grid; grid-template-columns: repeat(3, 18px); gap: 8px; padding: 12px; justify-content: center; }
  .dots i { width: 18px; height: 18px; border-radius: 50%; display: block; }
</style></head><body><div class="safe">
  <div class="copy">
    <div class="brand"><img src="${iconDataUri}" alt="">Mosaic</div>
    <div class="headline">${l.headline}</div>
    <div class="sub">${l.sub}</div>
  </div>
  <div class="board" aria-hidden="true">${boardHtml()}</div>
</div></body></html>`
}

async function main() {
  const argv = process.argv.slice(2)
  const langs = argv.includes('--lang') ? argv[argv.indexOf('--lang') + 1].split(',') : Object.keys(LOCALES)
  mkdirSync(OUT_DIR, { recursive: true })
  const sharp = await loadSharp()
  const iconPng = readFileSync(ICON)
  const iconDataUri = `data:image/png;base64,${iconPng.toString('base64')}`
  const browser = await launchChromium()
  try {
    const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 1 })
    for (const lang of langs) {
      if (!LOCALES[lang]) throw new Error(`未知の言語: ${lang}`)
      await page.setContent(buildHtml(lang, iconDataUri), { waitUntil: 'load' })
      await page.evaluate(() => document.fonts.ready)
      const buf = await page.screenshot({ clip: { x: 0, y: 0, width: WIDTH, height: HEIGHT } })
      const out = sharp ? await sharp(buf).removeAlpha().png({ compressionLevel: 9 }).toBuffer() : buf
      writeFileSync(join(OUT_DIR, `play-feature-graphic-${lang}-1024x500.png`), out)
      console.log(`  ✓ play-feature-graphic-${lang}-1024x500.png`)
    }

    // ストアアイコン: mobile/assets/icon.png（1024×1024・全面塗り）を 512×512 に縮める。
    // 元画像はアルファの無い RGB なので、Play の「32 ビット PNG」に合わせて RGBA（不透明）で出す
    await page.setViewportSize({ width: 512, height: 512 })
    await page.setContent(`<html><body style="margin:0"><img src="${iconDataUri}" style="width:512px;height:512px;display:block"></body></html>`, { waitUntil: 'load' })
    const iconBuf = await page.screenshot({ clip: { x: 0, y: 0, width: 512, height: 512 } })
    const icon = sharp ? await sharp(iconPng).resize(512, 512, { kernel: 'lanczos3' }).ensureAlpha(1).png({ compressionLevel: 9 }).toBuffer() : iconBuf
    writeFileSync(join(OUT_DIR, 'play-icon-512.png'), icon)
    console.log('  ✓ play-icon-512.png')
  } finally {
    await browser.close()
  }
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
