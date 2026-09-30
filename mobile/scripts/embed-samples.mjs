// リポジトリ直下の public/samples/<lang>/*.xlsx（scripts/generate-samples.ts の出力。Web 版と共通）を
// base64 にしてバンドルへ埋め込む（lib/samples.generated.ts）。サンプルを作り直したら
//   （リポジトリ直下で）npm run samples:generate → （mobile で）npm run samples:embed
// test/stored-project.test.ts が public/samples/ の中身と一致していることを確認する。
import { readFileSync, writeFileSync } from 'node:fs'

const LANGS = ['ja', 'en', 'ko', 'es', 'de', 'pt-BR']
const LABELS = {
  ja: { sample1: 'クラス分け（80名・4組）', sample2: 'クラス分け（80名・シンプル）', 'sample-group': 'グループ分け（30名・6班）' },
  en: { sample1: 'Class placement (80 students, 4 classes)', sample2: 'Class placement (80 students, simple)', 'sample-group': 'Group work (30 students, 6 groups)' },
  ko: { sample1: '반 편성(학생 80명·4개 반)', sample2: '반 편성(학생 80명·간단)', 'sample-group': '모둠 편성(30명·6모둠)' },
  es: { sample1: 'Grupos (80 estudiantes, 4 grupos)', sample2: 'Grupos (80 estudiantes, simple)', 'sample-group': 'Equipos de trabajo (30, 6 equipos)' },
  de: { sample1: 'Klassen (80 Schüler, 4 Klassen)', sample2: 'Klassen (80 Schüler, einfach)', 'sample-group': 'Gruppenarbeit (30 Schüler, 6 Gruppen)' },
  'pt-BR': { sample1: 'Turmas (80 alunos, 4 turmas)', sample2: 'Turmas (80 alunos, simples)', 'sample-group': 'Grupos de trabalho (30 alunos, 6 grupos)' },
}
const IDS = ['sample1', 'sample2', 'sample-group']

const body = LANGS.map((lang) => {
  const list = IDS.map((id) => {
    const b64 = readFileSync(new URL(`../../public/samples/${lang}/${id}.xlsx`, import.meta.url)).toString('base64')
    return `    { id: ${JSON.stringify(id)}, label: ${JSON.stringify(LABELS[lang][id])}, base64: ${JSON.stringify(b64)} },`
  })
  return `  ${JSON.stringify(lang)}: [\n${list.join('\n')}\n  ],`
}).join('\n')

writeFileSync(
  new URL('../lib/samples.generated.ts', import.meta.url),
  `// 自動生成（scripts/embed-samples.mjs）。手で編集しないこと。\nimport type { AppLanguage } from './i18n'\n\nexport const SAMPLE_FILES: Record<AppLanguage, { id: string; label: string; base64: string }[]> = {\n${body}\n}\n`,
)
console.log('lib/samples.generated.ts を生成しました')
