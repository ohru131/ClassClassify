/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath } from 'node:url'

export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss()],
  worker: { format: 'es' },
  // スマホ版（mobile/）のテストは mobile/ 側の設定で実行する
  test: { exclude: ['**/node_modules/**', 'mobile/**'] },
  resolve: {
    alias: [{ find: /^\.\/cpexcel\.js$/, replacement: fileURLToPath(new URL('./src/stubs/cpexcel.cjs', import.meta.url)) }],
  },
})
