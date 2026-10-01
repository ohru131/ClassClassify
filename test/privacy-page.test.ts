import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { build } from 'vite'
import { afterAll, describe, expect, it } from 'vitest'

import { PRIVACY, PRIVACY_SCOPE } from '../src/i18n/privacy'
import { APP_LANGUAGES } from '../src/i18n/languages'
import { linkify, renderPrivacyContent } from '../src/privacy/render'

// Play Console に登録するプライバシーポリシーの URL（https://ohru131.github.io/ClassClassify/privacy/）が
// 実際にデプロイ物（dist/）に入り、アプリ内の画面と同じ本文を6言語ぶん持っていることを確かめる。

describe('公開用プライバシーポリシー', () => {
  it('6言語ぶんの本文をスマホ版の文言そのままで持つ', () => {
    const html = renderPrivacyContent()
    for (const lang of APP_LANGUAGES) {
      expect(html).toContain(`data-lang="${lang}"`)
      expect(html).toContain(`href="#${lang}"`)
      expect(html).toContain(`id="${lang}"`)
      for (const s of PRIVACY[lang]) expect(html).toContain(s.title.replace(/&/g, '&amp;'))
      expect(PRIVACY_SCOPE[lang].length).toBeGreaterThan(20)
    }
    // JavaScript が動かなくても全言語が読める（隠すのは main.ts が動いたときだけ）
    expect(html).not.toMatch(/\bhidden\b/)
  })

  it('URL だけをリンクにし、他はエスケープする', () => {
    expect(linkify('a <b> https://example.com/x 。')).toBe('a &lt;b&gt; <a href="https://example.com/x" rel="noopener">https://example.com/x</a> 。')
    expect(linkify('GitHub（https://github.com/ohru131/ClassClassify）の')).toContain('href="https://github.com/ohru131/ClassClassify"')
  })

  const outDir = mkdtempSync(join(tmpdir(), 'mosaic-dist-'))
  afterAll(() => rmSync(outDir, { recursive: true, force: true }))

  it('vite build の出力に privacy/index.html が入る', async () => {
    await build({
      configFile: fileURLToPath(new URL('../vite.config.ts', import.meta.url)),
      root: fileURLToPath(new URL('..', import.meta.url)),
      logLevel: 'silent',
      build: { outDir, emptyOutDir: true },
    })
    const file = join(outDir, 'privacy', 'index.html')
    expect(existsSync(file)).toBe(true)
    const html = readFileSync(file, 'utf8')
    expect(html).not.toContain('<!--PRIVACY_CONTENT-->')
    expect(html.match(/<article /g)?.length).toBe(APP_LANGUAGES.length)
    expect(html).toContain('RevenueCat')
    // 相対パス（GitHub Pages のサブパス /ClassClassify/ で動く）
    expect(html).toMatch(/src="\.\.\/assets\/privacy-[^"]+\.js"/)
    expect(existsSync(join(outDir, 'index.html'))).toBe(true)
  }, 120_000)
})
