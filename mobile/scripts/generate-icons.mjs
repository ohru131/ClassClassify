// アプリアイコン（Web 版の favicon.svg と同じ意匠）を PNG に書き出す。Playwright の Chromium が要る:
//   playwright-core を入れた環境で CHROMIUM_PATH=<chromium の実行ファイル> node scripts/generate-icons.mjs
import { chromium } from 'playwright-core'
const out = new URL('../assets/', import.meta.url).pathname
const tiles = (s, pad) => {
  const u = (s - 2 * pad) / 18 // grid of 2 tiles
  const t = 8 * u, g = 2 * u, r = 2 * u
  const x0 = pad, x1 = x0 + t + g
  return `<g fill="#fff"><rect x="${x0}" y="${x0}" width="${t}" height="${t}" rx="${r}"/><rect x="${x1}" y="${x0}" width="${t}" height="${t}" rx="${r}" opacity=".6"/><rect x="${x0}" y="${x1}" width="${t}" height="${t}" rx="${r}" opacity=".6"/><rect x="${x1}" y="${x1}" width="${t}" height="${t}" rx="${r}" opacity=".85"/></g>`
}
const grad = `<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6366f1"/><stop offset="1" stop-color="#d946ef"/></linearGradient></defs>`
const icon = (s) => `<svg xmlns="http://www.w3.org/2000/svg" width="${s}" height="${s}">${grad}<rect width="${s}" height="${s}" fill="url(#g)"/>${tiles(s, s * 0.2)}</svg>`
// アダプティブアイコンの前景は中央 66% が安全領域
const fg = (s) => `<svg xmlns="http://www.w3.org/2000/svg" width="${s}" height="${s}">${tiles(s, s * 0.3)}</svg>`
const b = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH })
const p = await b.newPage()
for (const [name, svg, s, transparent] of [['icon.png', icon(1024), 1024, false], ['adaptive-icon.png', fg(1024), 1024, true], ['favicon.png', icon(48), 48, false]]) {
  await p.setViewportSize({ width: s, height: s })
  await p.setContent(`<html><body style="margin:0;background:transparent">${svg}</body></html>`)
  await p.screenshot({ path: out + name, omitBackground: transparent, clip: { x: 0, y: 0, width: s, height: s } })
}
await b.close()
