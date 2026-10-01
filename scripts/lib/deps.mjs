// 撮影・画像生成スクリプトが使う外部パッケージ（Playwright・sharp）を探して読み込む。
// どちらもアプリの依存ではないので package.json には入れていない。次のどこかにあれば使う:
//   1. このリポジトリの node_modules（`npm i --no-save playwright-core sharp` など）
//   2. 環境変数 SUBMISSION_DEPS で指定したディレクトリの node_modules
//   3. グローバル（`npm root -g`）
// Chromium は CHROMIUM_PATH → /opt/pw-browsers/chromium → Windows の Chrome / Edge の順に探す
// （`playwright install` はしない）。
import { execSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { createRequire } from 'node:module'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'

function candidates() {
  const dirs = [process.cwd()]
  if (process.env.SUBMISSION_DEPS) dirs.push(process.env.SUBMISSION_DEPS)
  try {
    dirs.push(execSync('npm root -g', { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim().replace(/node_modules$/, ''))
  } catch {
    // npm が無ければグローバルは見ない
  }
  return dirs
}

async function load(names, { optional = false } = {}) {
  for (const dir of candidates()) {
    const req = createRequire(join(dir, 'noop.js'))
    for (const name of names) {
      try {
        const resolved = req.resolve(name)
        const mod = await import(pathToFileURL(resolved).href)
        return mod.default ?? mod
      } catch {
        // 次の候補へ
      }
    }
  }
  if (optional) return null
  throw new Error(`${names.join(' / ')} が見つかりません。npm i --no-save ${names[0]} するか、SUBMISSION_DEPS にその node_modules の親ディレクトリを指定してください。`)
}

export const loadPlaywright = () => load(['playwright-core', 'playwright'])
export const loadSharp = () => load(['sharp'], { optional: true })

export function chromiumPath() {
  const list = [
    process.env.CHROMIUM_PATH,
    '/opt/pw-browsers/chromium',
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ].filter(Boolean)
  return list.find((p) => existsSync(p))
}

export async function launchChromium(options = {}) {
  const { chromium } = await loadPlaywright()
  const executablePath = chromiumPath()
  return chromium.launch({ ...options, ...(executablePath ? { executablePath } : {}) })
}
