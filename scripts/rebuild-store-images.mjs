#!/usr/bin/env node
// ストア提出用の画像（スクリーンショット 192 枚・フィーチャーグラフィック 6 枚・アイコン）を一括で作り直す。
// アプリ名・画面の文言・サンプルを変えたあと、画像に写っている名前や文言をまとめて差し替えるために使う。
//
//   node scripts/rebuild-store-images.mjs                       # 全部（20分ほど）
//   node scripts/rebuild-store-images.mjs --lang ja,ko          # 言語を絞る（撮影とフィーチャーグラフィックの両方）
//   node scripts/rebuild-store-images.mjs --form phone --only 01-roster,08-pro   # 撮影だけ絞る
//   node scripts/rebuild-store-images.mjs --skip-export         # mobile/dist を作り直さない（文言だけ変えたときは不可）
//
// 中でやること（どれも単独でも動く既存のスクリプト）:
//   1. (cd mobile && npx expo export --platform web)         … 撮影元の Web 書き出し mobile/dist を作る
//   2. npx tsx scripts/capture-submission-assets.mjs          … スクリーンショット（--lang / --form / --only を渡す）
//   3. node scripts/generate-feature-graphic.mjs              … フィーチャーグラフィックとアイコン（--lang を渡す）
//   4. node scripts/check-submission-assets.mjs               … 大きさ・アルファ・縦横比・枚数の検査
//
// **撮影元は作業ツリー（mobile/）そのもの。** 未コミットの画面の変更があると、それも写り込む。
// コミット済みの画面だけを撮りたいときは、git worktree で取り出したフォルダで実行すること。
// Playwright（playwright-core）と sharp の用意は scripts/lib/deps.mjs と submission-assets/README.md を参照。
import { spawnSync } from 'node:child_process'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)))

const argv = process.argv.slice(2)
const value = (name) => (argv.includes(name) ? argv[argv.indexOf(name) + 1] : null)
const pass = (names) => names.flatMap((n) => (value(n) ? [n, value(n)] : []))

function step(title, cmd, args, cwd = ROOT) {
  console.log(`\n=== ${title}\n$ ${[cmd, ...args].join(' ')}`)
  const t0 = Date.now()
  // Windows の npx は .cmd なので shell 経由で起動する
  const r = spawnSync(cmd, args, { cwd, stdio: 'inherit', shell: process.platform === 'win32' })
  if (r.status !== 0) {
    console.error(`\n${title} が失敗しました（終了コード ${r.status ?? r.signal}）。ここで止めます。`)
    process.exit(r.status ?? 1)
  }
  console.log(`--- ${title}: ${((Date.now() - t0) / 1000).toFixed(0)} 秒`)
}

if (!argv.includes('--skip-export')) {
  step('Web 書き出し', 'npx', ['expo', 'export', '--platform', 'web'], join(ROOT, 'mobile'))
}
step('スクリーンショット', 'npx', ['tsx', 'scripts/capture-submission-assets.mjs', ...pass(['--lang', '--form', '--only'])])
step('フィーチャーグラフィックとアイコン', 'node', ['scripts/generate-feature-graphic.mjs', ...pass(['--lang'])])
step('検査', 'node', ['scripts/check-submission-assets.mjs'])
console.log('\n完了。git diff --stat -- submission-assets で差し替わった画像を確かめてからコミットする。')
