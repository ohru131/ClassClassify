// リポジトリ直下の public/sample*.xlsx を base64 にしてバンドルへ埋め込む（lib/samples.generated.ts）。
// アセットとして同梱すると、ネイティブでは expo-asset＋ファイル読み出し、Web では fetch と
// 経路が分かれるため、文字列で持つ方が単純で確実。サンプルを差し替えたら `npm run samples:generate`。
// test/samples.test.ts が public/ の中身と一致していることを検証する。
import { readFileSync, writeFileSync } from 'node:fs'

export const SAMPLES = [
  { id: 'sample1', file: 'sample1.xlsx', label: 'クラス分け（80名・4組）' },
  { id: 'sample2', file: 'sample2.xlsx', label: 'クラス分け（80名・シンプル）' },
  { id: 'sample-group', file: 'sample-group.xlsx', label: 'グループ分け（30名・6班）' },
]

const body = SAMPLES.map(({ id, file, label }) => {
  const b64 = readFileSync(new URL(`../../public/${file}`, import.meta.url)).toString('base64')
  return `  { id: ${JSON.stringify(id)}, file: ${JSON.stringify(file)}, label: ${JSON.stringify(label)}, base64: ${JSON.stringify(b64)} },`
}).join('\n')

writeFileSync(
  new URL('../lib/samples.generated.ts', import.meta.url),
  `// 自動生成（scripts/generate-samples.mjs）。手で編集しないこと。\nexport const SAMPLE_FILES: { id: string; file: string; label: string; base64: string }[] = [\n${body}\n]\n`,
)
console.log('lib/samples.generated.ts を生成しました')
