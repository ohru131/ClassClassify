import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath } from 'node:url'

export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss()],
  worker: { format: 'es' },
  resolve: {
    alias: [{ find: /^\.\/cpexcel\.js$/, replacement: fileURLToPath(new URL('./src/stubs/cpexcel.cjs', import.meta.url)) }],
  },
})
