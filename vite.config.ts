/// <reference types="vitest/config" />
import { defineConfig, runnerImport, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath } from 'node:url'

// 公開用プライバシーポリシー（privacy/index.html → dist/privacy/index.html）。
// Play Console に登録する URL は https://ohru131.github.io/ClassClassify/privacy/ 。
// 本文は src/i18n/privacy.ts（スマホ版と共通）から、ビルド時に静的な HTML として埋め込む。
// 設定ファイルから静的に import すると設定のバンドルにスマホ版の文言まで巻き込むので、
// Vite のモジュールランナー（runnerImport）で必要になったときだけ読む。
const privacyPage = (): Plugin => ({
  name: 'mosaic-privacy-page',
  transformIndexHtml: {
    order: 'pre',
    handler: async (html, ctx) => {
      if (!ctx.path.replace(/\\/g, '/').endsWith('/privacy/index.html')) return html
      const { module } = await runnerImport<typeof import('./src/privacy/render')>(fileURLToPath(new URL('./src/privacy/render.ts', import.meta.url)))
      return html.replace('<!--PRIVACY_CONTENT-->', module.renderPrivacyContent())
    },
  },
})

export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss(), privacyPage()],
  worker: { format: 'es' },
  build: {
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        privacy: fileURLToPath(new URL('./privacy/index.html', import.meta.url)),
      },
    },
  },
  // スマホ版（mobile/）のテストは mobile/ 側の設定で実行する
  test: { exclude: ['**/node_modules/**', 'mobile/**'] },
  resolve: {
    alias: [{ find: /^\.\/cpexcel\.js$/, replacement: fileURLToPath(new URL('./src/stubs/cpexcel.cjs', import.meta.url)) }],
  },
})
