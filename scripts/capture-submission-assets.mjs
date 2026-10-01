#!/usr/bin/env node
// Google Play 用のスクリーンショットを、スマホ版の Web 書き出し（mobile/dist）から自動で撮る。
//
// なぜ Web 書き出しなのか: この環境には Android 実機もエミュレータも無い。画面の構成・文言・
// サンプル名簿はネイティブと同じコード（Expo Router + react-native-web）なので、ストアの絵として
// 使える。ただし**フォント・ステータスバー・ナビゲーションバーは実機と違う**ので、
// 提出前に一度は Android 実機・タブレット・Chromebook で見比べること（submission-assets/README.md）。
//
// 使い方（リポジトリ直下で。文言を mobile/lib/copy/*.ts から読むので tsx で動かす）:
//   (cd mobile && npx expo export --platform web)            # mobile/dist を作る
//   npx tsx scripts/capture-submission-assets.mjs              # 6言語 × 4画面サイズ
//   npx tsx scripts/capture-submission-assets.mjs --lang ja,ko --form phone
//   npx tsx scripts/capture-submission-assets.mjs --only 04-results
// Playwright（playwright-core）と sharp はアプリの依存ではない。scripts/lib/deps.mjs を参照。
//
// 撮るもの（言語ごとに、その言語のサンプル名簿で）:
//   01 名簿（サンプルを読み込んだ直後）   02 ペア指定（別の組）   03 編成の設定と実行中の進み具合
//   04 結果（クラスごとの名簿と集計）     05 バランス表           06 手直し（生徒を別の組へ）
//   07 印刷・PDF のプレビュー（?pro=preview）                    08 Pro の画面
// 端末の枠・評価の星・「No.1」のような装飾は付けない（Play のポリシー上も、実際の画面以外を
// 載せないため）。
import { createServer } from 'node:http'
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { extname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { COPY } from '../mobile/lib/copy/index.ts'
import { SAMPLE_FILES } from '../mobile/lib/samples.generated.ts'
import { APP_LANGUAGES, LANGUAGE_META } from '../src/i18n/languages.ts'
import { launchChromium, loadSharp } from './lib/deps.mjs'

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)))
const DIST = join(ROOT, 'mobile', 'dist')
const OUT = join(ROOT, 'submission-assets', 'screenshots')

// Play の規格: PNG / JPEG、各辺 320〜3840px、長辺:短辺 ≤ 2:1。**タブレット（7・10インチ）と Chromebook は
// 16:9 または 9:16** で、10インチと Chromebook は各辺 1080px 以上（7680px まで）。
// css の大きさ × deviceScaleFactor が出力の画素数になる。幅 768dp 以上でタブレット用の
// 横並びレイアウト（mobile/lib/layout.ts の WIDE_BREAKPOINT）に切り替わる。
export const FORMS = {
  // スマホ（9:16、1080×1920）
  phone: { viewport: { width: 360, height: 640 }, scale: 3 },
  // 7インチタブレット（縦 9:16、1296×2304）。幅が 768dp 未満なのでスマホと同じ1列
  tablet7: { viewport: { width: 648, height: 1152 }, scale: 2 },
  // 10インチタブレット（横 16:9、1920×1080）。クラスを横に並べるレイアウト
  tablet10: { viewport: { width: 1280, height: 720 }, scale: 1.5 },
  // Chromebook（横 16:9、1920×1080）。タッチ無しのマウス操作として撮る
  chromebook: { viewport: { width: 1280, height: 720 }, scale: 1.5 },
}

// 撮る画面。Play に上げるのはスマホが最大8枚、タブレット・Chromebook も最大8枚
export const SHOTS = ['01-roster', '02-pairs', '03-run', '04-results', '05-balance', '06-move', '07-print', '08-pro']

const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css', '.json': 'application/json', '.png': 'image/png', '.ttf': 'font/ttf', '.ico': 'image/x-icon', '.svg': 'image/svg+xml' }

// mobile/app.config.ts は web.output = 'single'（SPA）なので、無いパスは index.html を返す
function startServer() {
  if (!existsSync(join(DIST, 'index.html'))) throw new Error(`${DIST} がありません。先に (cd mobile && npx expo export --platform web) を実行してください。`)
  const server = createServer((req, res) => {
    const pathname = decodeURIComponent(new URL(req.url ?? '/', 'http://x').pathname)
    let file = join(DIST, pathname)
    if (!file.startsWith(DIST) || !existsSync(file) || statSync(file).isDirectory()) file = join(DIST, 'index.html')
    res.writeHead(200, { 'content-type': MIME[extname(file)] ?? 'application/octet-stream' })
    res.end(readFileSync(file))
  })
  return new Promise((done) => server.listen(0, '127.0.0.1', () => done({ origin: `http://127.0.0.1:${server.address().port}`, close: () => server.close() })))
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
// 「Together {n}」のような埋め込み付きの文言を、数字を問わない正規表現にする
const pattern = (s) => new RegExp('^' + esc(s).replace(/\\\{\w+\\\}/g, '.*') + '$')

function parseArgs(argv) {
  const get = (name) => (argv.includes(name) ? argv[argv.indexOf(name) + 1].split(',') : null)
  return { langs: get('--lang') ?? [...APP_LANGUAGES], forms: get('--form') ?? Object.keys(FORMS), only: get('--only'), headed: argv.includes('--headed') }
}

export async function toPlayPng(sharp, buf) {
  const quantized = await sharp(buf).flatten({ background: '#ffffff' }).png({ palette: true, quality: 95, effort: 10, dither: 0.5 }).toBuffer()
  return sharp(quantized).removeAlpha().toColourspace('srgb').png({ compressionLevel: 9, adaptiveFiltering: true, palette: false }).toBuffer()
}

async function save(page, file, sharp) {
  const buf = await page.screenshot({ type: 'png' })
  // Play は「24 ビット PNG（アルファなし）」。UI の絵は色数が少ないので、いったん 256 色に減色してから
  // （文字のにじみは目視で分からない程度）24 ビット RGB の PNG に戻す。減色しないと1枚あたり2〜3倍の大きさになる。
  // **パレット（8 ビット）PNG のまま保存しないこと**（Play の規格は 24 ビット）。
  const out = sharp ? await toPlayPng(sharp, buf) : buf
  writeFileSync(file, out)
}

async function captureOne(browser, origin, lang, formName, only, sharp) {
  const form = FORMS[formName]
  const t = COPY[lang]
  const sample = SAMPLE_FILES[lang].find((s) => s.id === 'sample1')
  const dir = join(OUT, formName)
  mkdirSync(dir, { recursive: true })
  const want = (name) => !only || only.includes(name)

  const context = await browser.newContext({ viewport: form.viewport, deviceScaleFactor: form.scale, locale: LANGUAGE_META[lang].intl, hasTouch: formName !== 'chromebook' })
  // 表示言語はアプリの設定（mobile/lib/language-provider.tsx の保存キー）で固定し、前回の名簿は消す
  await context.addInitScript((l) => {
    if (location.protocol.startsWith('http')) {
      localStorage.setItem('mosaic.language.v1', l)
      if (!sessionStorage.getItem('captured')) {
        localStorage.removeItem('mosaic.project.v1')
        sessionStorage.setItem('captured', '1')
      }
    }
  }, lang)
  const page = await context.newPage()
  const errors = []
  page.on('pageerror', (e) => errors.push(e.message))
  const shot = async (name) => {
    if (!want(name)) return
    await sleep(350)
    await save(page, join(dir, `${lang}-${name}.png`), sharp)
    console.log(`  ✓ ${formName}/${lang}-${name}.png`)
  }
  // 下のタブバー（expo-router）。アイコンの字形（私用領域の文字）が名前の先頭に付くので末尾一致で探す
  const tab = (label) => page.getByRole('tablist').last().getByRole('tab', { name: new RegExp(esc(label) + '$') })
  const button = (label) => page.getByRole('button', { name: pattern(label) }).first()

  // ?pro=preview: Web 版だけ Pro の画面を出す（購入はできない。mobile/lib/pro-preview.ts）
  await page.goto(`${origin}/?pro=preview`)
  await button(sample.label).click()
  await page.getByText(t.tabStudents.replace('{n}', ''), { exact: false }).first().waitFor()
  await shot('01-roster')

  // 別の組の指定（韓国では学폭の分離、ドイツでは Klasse 5 の希望など。docs/research/overseas-demand.md）
  await page.getByRole('tab', { name: pattern(t.tabUnwanted) }).click()
  await shot('02-pairs')

  // 設定・実行。標準（10秒）で走らせ、途中の進み具合を撮る
  await tab(t.tabRun).click()
  await page.getByRole('tab', { name: t.standard, exact: true }).click()
  await button(t.runBtn).click()
  await page.getByRole('progressbar').waitFor()
  await sleep(2600)
  await shot('03-run')
  // 終わると結果画面へ移る
  await page.getByRole('tab', { name: t.tabBalance, exact: true }).waitFor({ timeout: 60_000 })
  await sleep(500)
  await shot('04-results')

  await page.getByRole('tab', { name: t.tabBalance, exact: true }).click()
  await shot('05-balance')
  await page.getByRole('tab', { name: t.tabClasses, exact: true }).click()

  // 手直し: 名簿の生徒を1人選ぶと、移動先の組を選ぶパネルが出る
  const studentPrefix = t.studentA11y.split('{no}')[0]
  const students = page.getByRole('button', { name: new RegExp('^' + esc(studentPrefix) + '\\d+ ') })
  await students.nth(formName === 'phone' ? 3 : 5).click()
  await page.getByRole('button', { name: pattern(t.moveTo) }).first().waitFor()
  await shot('06-move')
  await page.getByRole('button', { name: t.close, exact: true }).last().click()

  // 印刷・PDF（Web は新しいウィンドウに HTML を書いて印刷する。印刷ダイアログは止める）
  if (want('07-print')) {
    await page.evaluate(() => {
      const open = window.open.bind(window)
      window.open = (...a) => {
        const w = open(...a)
        if (w) w.print = () => undefined
        return w
      }
    })
    const [popup] = await Promise.all([context.waitForEvent('page'), button(t.printPdf).click()])
    await popup.waitForLoadState('load')
    const html = await popup.content()
    await popup.close()
    // 印刷用 HTML は A4 縦（画面では幅 210mm）。スマホ幅のウィンドウに出すと見出しが折り返して
    // 実際の印刷物と違う絵になるので、A4 が収まる幅で組み、出力の画素数だけ他の画面と揃える。
    // 横長の画面（10インチ・Chromebook）はその幅のまま、用紙を中央に置く。
    const portrait = form.viewport.height > form.viewport.width
    const width = portrait ? 900 : form.viewport.width
    const printContext = await browser.newContext({
      viewport: { width, height: Math.round((width * form.viewport.height) / form.viewport.width) },
      deviceScaleFactor: (form.viewport.width * form.scale) / width,
      locale: LANGUAGE_META[lang].intl,
    })
    const printPage = await printContext.newPage()
    await printPage.setContent(html, { waitUntil: 'load' })
    // 組み終わる前に撮ると縮んだ絵になることがあったので、幅が収まりフォントが揃うのを待つ
    await printPage.waitForFunction((w) => innerWidth === w && document.documentElement.scrollWidth <= w && document.fonts.status === 'loaded', width)
    await printPage.addStyleTag({ content: 'html{background:#e2e8f0} body{background:#fff;margin:20px auto!important;box-shadow:0 4px 18px rgba(15,23,42,.18)}' })
    await sleep(400)
    await save(printPage, join(dir, `${lang}-07-print.png`), sharp)
    console.log(`  ✓ ${formName}/${lang}-07-print.png`)
    await printContext.close()
  }

  // Pro の画面は購入前の状態で撮る（?pro=preview を外す）。Web 版だけに出る
  // 「購入はアプリ版で」の注記は消し、Web で押せない購入ボタンの半透明もネイティブと同じ見た目に戻す。
  if (want('08-pro')) {
    await page.goto(`${origin}/pro`)
    await page.getByText(t.proName, { exact: true }).first().waitFor()
    await page.evaluate(({ note, buy }) => {
      for (const el of document.querySelectorAll('div')) {
        if (el.childElementCount !== 0 || el.textContent !== note) continue
        // 注記の枠ごと消す（文字だけ消すと色の付いた帯が残る）
        let box = el
        while (box.parentElement && box.parentElement.textContent === note) box = box.parentElement
        box.remove()
      }
      const b = [...document.querySelectorAll('[role="button"]')].find((x) => x.getAttribute('aria-label') === buy)
      if (b) b.style.opacity = '1'
    }, { note: t.purchaseStoreOnly, buy: t.buy })
    await page.evaluate(() => window.scrollTo(0, 0))
    await shot('08-pro')
  }

  if (errors.length) console.warn(`  ! ${formName}/${lang}: ページのエラー ${errors.join(' / ')}`)
  await context.close()
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  const sharp = await loadSharp()
  if (!sharp) console.warn('sharp が無いので PNG を再圧縮しません（Playwright の出力のまま保存します）')
  const server = await startServer()
  const browser = await launchChromium({ headless: !args.headed })
  try {
    for (const formName of args.forms) {
      if (!FORMS[formName]) throw new Error(`未知の画面サイズ: ${formName}`)
      for (const lang of args.langs) await captureOne(browser, server.origin, lang, formName, args.only, sharp)
    }
  } finally {
    await browser.close()
    server.close()
  }
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
