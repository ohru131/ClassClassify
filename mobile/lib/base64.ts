// Hermes の atob/btoa の有無や挙動に依存しないよう、自前で持つ（ファイルの読み書きにだけ使う）
const CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
const LOOKUP = new Uint8Array(256)
for (let i = 0; i < CHARS.length; i++) LOOKUP[CHARS.charCodeAt(i)] = i

export function base64ToArrayBuffer(b64: string): ArrayBuffer {
  const clean = b64.replace(/[^A-Za-z0-9+/]/g, '')
  const len = Math.floor((clean.length * 3) / 4)
  const out = new Uint8Array(len)
  let o = 0
  for (let i = 0; i < clean.length; i += 4) {
    const a = LOOKUP[clean.charCodeAt(i)]
    const b = LOOKUP[clean.charCodeAt(i + 1)]
    const c = LOOKUP[clean.charCodeAt(i + 2)]
    const d = LOOKUP[clean.charCodeAt(i + 3)]
    const n = (a << 18) | (b << 12) | (c << 6) | d
    if (o < len) out[o++] = (n >> 16) & 255
    if (o < len) out[o++] = (n >> 8) & 255
    if (o < len) out[o++] = n & 255
  }
  return out.buffer
}
