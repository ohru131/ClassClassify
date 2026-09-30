import * as XLSX from 'xlsx'
import { createSpreadsheet, type GoogleFile, type SheetSpec } from './google'

const GUIDE: (string | number)[][] = [
  ['Mosaic 名簿ひな形の使い方'],
  [''],
  ['シート', '書き方'],
  ['設定', 'B列に値を入力。「クラス数」は必須、「1クラスの最大人数」は任意（生徒人数は参考）'],
  ['生徒名簿', '1行目: 各項目の重み（大きいほど優先して均等化、0 で無視）'],
  ['', '2行目: 見出し。A列「NO」、B列「名前」、C列以降に項目名（自由に追加・削除できる）'],
  ['', '3行目以降: 生徒1人1行。該当する項目に ○ を入れる、または 1/2/3 などの段階・点数を入れる'],
  ['同じ組ペア', '1行に、同じ組にしたい生徒の NO を横に並べる（3人以上も可）'],
  ['別の組ペア', '1行に、互いに別の組にしたい生徒の NO を横に並べる'],
  [''],
  ['項目の値の扱い', '値が1種類（○ と空欄など）→ 該当者数を均等化'],
  ['', '値が数種類（1/2/3 など）→ 値ごとの人数を均等化'],
  ['', '7種類以上の数値（点数など）→ クラス平均を均等化'],
  [''],
  ['記入例', '「生徒名簿」などには記入例（架空の生徒80名）が入っている。自分の名簿に書き換えて使う'],
  ['読み込み', 'Mosaic の「Google スプレッドシート」からこのファイルを選ぶ'],
  ['このシート', '「使い方」シートは読み込み時に無視されるので、残しても消してもよい'],
]

const BAND = '#EEF2FF'
const WEIGHT = '#FEF9C3'

/** 記入例（sample1.xlsx）からひな形の各シートを作る */
export function buildTemplateSheets(buf: ArrayBuffer): SheetSpec[] {
  const wb = XLSX.read(buf, { type: 'array' })
  const rows = (name: string) =>
    XLSX.utils.sheet_to_json<(string | number | null)[]>(wb.Sheets[name], { header: 1, defval: '', blankrows: true })

  const roster = rows('生徒名簿')
  return [
    { name: '使い方', rows: GUIDE, bands: [{ row: 0, color: BAND, bold: true }, { row: 2, color: '#F1F5F9', bold: true }], boldCols: [0], colWidths: [140, 640] },
    { name: '設定', rows: rows('設定'), boldCols: [0], colWidths: [180, 80] },
    {
      name: '生徒名簿',
      rows: roster,
      frozenRows: 2,
      frozenCols: 2,
      bands: [
        { row: 0, color: WEIGHT },
        { row: 1, color: BAND, bold: true },
      ],
      colWidths: [56, 120, ...Array.from({ length: Math.max(0, (roster[1]?.length ?? 2) - 2) }, () => 84)],
    },
    { name: '同じ組ペア', rows: rows('同じ組ペア') },
    { name: '別の組ペア', rows: rows('別の組ペア') },
  ]
}

/** 記入例つきのひな形スプレッドシートを利用者の Drive に作成する */
export async function createTemplateSpreadsheet(): Promise<GoogleFile> {
  const buf = await (await fetch('./sample1.xlsx')).arrayBuffer()
  return createSpreadsheet('Mosaic 名簿ひな形', buildTemplateSheets(buf))
}
