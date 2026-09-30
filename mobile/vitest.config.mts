import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vitest/config'

// 共有ソルバー（../src/solver）が import する xlsx-js-style を mobile/node_modules から解決する
export default defineConfig({
  resolve: {
    alias: [
      { find: /^\.\/cpexcel\.js$/, replacement: fileURLToPath(new URL('./stubs/cpexcel.cjs', import.meta.url)) },
      { find: /^xlsx-js-style$/, replacement: fileURLToPath(new URL('./node_modules/xlsx-js-style/dist/xlsx.min.js', import.meta.url)) },
      { find: /^@\//, replacement: fileURLToPath(new URL('./', import.meta.url)) },
    ],
  },
  test: { include: ['test/**/*.test.ts'] },
})
