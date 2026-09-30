import * as ns from 'xlsx-js-style'

// xlsx-js-style（CommonJS）の読み込み方の違いを吸収する。Vite・Metro・vitest では名前空間に
// 関数が並ぶが、Node の ESM から直接読む（サンプル生成スクリプトを tsx で動かす）と中身が
// default に入る。どちらでも同じオブジェクトを返す。
const XLSX: typeof ns = (ns as unknown as { default?: typeof ns }).default?.utils ? (ns as unknown as { default: typeof ns }).default : ns
export default XLSX
