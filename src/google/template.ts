import XLSX from '../solver/xlsx'
import { createSpreadsheet, type GoogleFile, type SheetSpec } from './google'
import { LANGUAGE_META, type AppLanguage } from '../i18n/languages'
import { FILE_LABELS } from '../solver/labels'
import { sampleUrl } from '../copy/core'

type Row = (string | number)[]

// 「使い方」シート（言語別）。シート名は src/solver/labels.ts の語彙に合わせる
const GUIDE: Record<AppLanguage, Row[]> = {
  ja: [
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
  ],
  en: [
    ['How to use the Mosaic roster template'],
    [''],
    ['Sheet', 'What to enter'],
    ['Settings', 'Enter values in column B. "Number of classes" is required, "Maximum class size" is optional (number of students is for reference)'],
    ['Roster', 'Row 1: weight of each attribute (higher = balanced first, 0 = ignored)'],
    ['', 'Row 2: headings. Column A "No.", column B "Name", attributes from column C (add or remove freely)'],
    ['', 'Row 3 on: one student per row. Put ✓ for yes/no attributes, or levels such as 1/2/3, or scores'],
    ['Keep together', 'One group per row: the No. of students to place in the same class (3 or more is fine)'],
    ['Keep apart', 'One group per row: the No. of students to place in different classes from each other'],
    [''],
    ['How values are treated', 'One value (e.g. ✓ and blank) → the number of students with it is balanced'],
    ['', 'A few values (e.g. 1/2/3) → the number of students with each value is balanced'],
    ['', '7 or more numbers (e.g. scores) → class averages are balanced'],
    [''],
    ['Example', 'The sheets contain an example (80 fictional students). Replace it with your own roster'],
    ['Loading', 'Choose this file with "Google Sheets" in Mosaic'],
    ['This sheet', 'The "How to use" sheet is ignored when loading, so you can keep or delete it'],
  ],
  ko: [
    ['Mosaic 명단 양식 사용법'],
    [''],
    ['시트', '입력 방법'],
    ['설정', 'B열에 값을 입력합니다. "반 수"는 필수, "반별 최대 인원"은 선택(학생 수는 참고용)'],
    ['학생 명단', '1행: 항목별 가중치(클수록 먼저 고르게 나눔, 0이면 반영하지 않음)'],
    ['', '2행: 제목. A열 "번호", B열 "이름", C열부터 항목 이름(자유롭게 추가·삭제)'],
    ['', '3행부터: 학생 한 명당 한 행. 해당 항목에 ✓, 또는 1/2/3 같은 단계·점수를 입력'],
    ['같은 반 배정', '한 행에 같은 반에 배정할 학생의 번호를 나란히 입력(3명 이상 가능)'],
    ['분리 배정', '한 행에 서로 다른 반에 배정할 학생의 번호를 나란히 입력'],
    [''],
    ['값의 처리', '값이 한 종류(✓와 빈칸 등) → 해당 학생 수를 고르게'],
    ['', '값이 몇 종류(1/2/3 등) → 값별 인원을 고르게'],
    ['', '7종류 이상의 숫자(점수 등) → 반 평균을 고르게'],
    [''],
    ['예시', '시트에는 예시(가상의 학생 80명)가 들어 있습니다. 실제 명단으로 바꿔 쓰세요'],
    ['불러오기', 'Mosaic의 "Google 스프레드시트"에서 이 파일을 선택합니다'],
    ['이 시트', '"사용법" 시트는 불러올 때 무시되므로 남겨 두어도, 지워도 됩니다'],
  ],
  es: [
    ['Cómo usar la plantilla de lista de Mosaic'],
    [''],
    ['Hoja', 'Qué escribir'],
    ['Configuración', 'Escribe los valores en la columna B. «Número de grupos» es obligatorio, «Máximo por grupo» es opcional (el número de estudiantes es de referencia)'],
    ['Lista de estudiantes', 'Fila 1: peso de cada criterio (más alto = se equilibra primero, 0 = no se toma en cuenta)'],
    ['', 'Fila 2: encabezados. Columna A «N.º», columna B «Nombre», criterios desde la columna C (puedes agregar o quitar)'],
    ['', 'Desde la fila 3: un estudiante por fila. Pon ✓ en los criterios de sí/no, o niveles como 1/2/3, o notas'],
    ['Mantener juntos', 'Un conjunto por fila: los N.º de los estudiantes que van en el mismo grupo (pueden ser 3 o más)'],
    ['Separar', 'Un conjunto por fila: los N.º de los estudiantes que van en grupos distintos entre sí'],
    [''],
    ['Cómo se tratan los valores', 'Un solo valor (✓ y vacío) → se equilibra cuántos lo tienen'],
    ['', 'Varios valores (1/2/3, etc.) → se equilibra cuántos hay de cada valor'],
    ['', '7 o más números (notas, etc.) → se equilibran los promedios de los grupos'],
    [''],
    ['Ejemplo', 'Las hojas traen un ejemplo (80 estudiantes ficticios). Reemplázalo con tu lista'],
    ['Cargar', 'Elige este archivo con «Google Sheets» en Mosaic'],
    ['Esta hoja', 'La hoja «Cómo usar» se ignora al cargar; puedes dejarla o borrarla'],
  ],
  de: [
    ['So verwenden Sie die Mosaic-Vorlage'],
    [''],
    ['Blatt', 'Was eintragen'],
    ['Einstellungen', 'Werte in Spalte B eintragen. „Anzahl Klassen“ ist Pflicht, „Höchstzahl pro Klasse“ optional (Anzahl Schüler dient zur Orientierung)'],
    ['Schülerliste', 'Zeile 1: Gewicht jedes Merkmals (höher = zuerst ausgleichen, 0 = ignorieren)'],
    ['', 'Zeile 2: Überschriften. Spalte A „Nr.“, Spalte B „Name“, Merkmale ab Spalte C (frei ergänzen oder löschen)'],
    ['', 'Ab Zeile 3: ein Schüler pro Zeile. Bei Ja/Nein-Merkmalen ✓ eintragen, sonst Stufen wie 1/2/3 oder Noten'],
    ['Zusammen', 'Eine Gruppe pro Zeile: die Nr. der Schüler, die in dieselbe Klasse sollen (auch 3 oder mehr)'],
    ['Trennen', 'Eine Gruppe pro Zeile: die Nr. der Schüler, die in verschiedene Klassen sollen'],
    [''],
    ['Umgang mit Werten', 'Ein Wert (✓ und leer) → die Anzahl wird ausgeglichen'],
    ['', 'Einige Werte (1/2/3 usw.) → die Anzahl je Wert wird ausgeglichen'],
    ['', '7 oder mehr Zahlen (Noten usw.) → die Klassendurchschnitte werden ausgeglichen'],
    [''],
    ['Beispiel', 'Die Blätter enthalten ein Beispiel (80 fiktive Schüler). Ersetzen Sie es durch Ihre Liste'],
    ['Laden', 'Wählen Sie diese Datei in Mosaic über „Google Tabellen“'],
    ['Dieses Blatt', 'Das Blatt „Anleitung“ wird beim Laden ignoriert und kann bleiben oder gelöscht werden'],
  ],
  'pt-BR': [
    ['Como usar o modelo de lista do Mosaic'],
    [''],
    ['Aba', 'O que preencher'],
    ['Configurações', 'Preencha os valores na coluna B. «Número de turmas» é obrigatório, «Máximo por turma» é opcional (o número de alunos é só referência)'],
    ['Lista de alunos', 'Linha 1: peso de cada critério (maior = equilibrado primeiro, 0 = ignorado)'],
    ['', 'Linha 2: cabeçalhos. Coluna A «Nº», coluna B «Nome», critérios a partir da coluna C (adicione ou remova à vontade)'],
    ['', 'A partir da linha 3: um aluno por linha. Use ✓ nos critérios de sim/não, ou níveis como 1/2/3, ou notas'],
    ['Manter juntos', 'Um conjunto por linha: os nº dos alunos que ficam na mesma turma (pode ser 3 ou mais)'],
    ['Separar', 'Um conjunto por linha: os nº dos alunos que ficam em turmas diferentes entre si'],
    [''],
    ['Como os valores são tratados', 'Um único valor (✓ e vazio) → equilibra quantos têm o valor'],
    ['', 'Alguns valores (1/2/3 etc.) → equilibra quantos há de cada valor'],
    ['', '7 ou mais números (notas etc.) → equilibra as médias das turmas'],
    [''],
    ['Exemplo', 'As abas trazem um exemplo (80 alunos fictícios). Substitua pela sua lista'],
    ['Carregar', 'Escolha este arquivo em «Planilhas Google» no Mosaic'],
    ['Esta aba', 'A aba «Como usar» é ignorada ao carregar; pode deixar ou apagar'],
  ],
}

/** 「使い方」シートの名前 */
const GUIDE_SHEET: Record<AppLanguage, string> = { ja: '使い方', en: 'How to use', ko: '사용법', es: 'Cómo usar', de: 'Anleitung', 'pt-BR': 'Como usar' }

const BAND = '#EEF2FF'
const WEIGHT = '#FEF9C3'

/** 記入例（その言語の sample1.xlsx）からひな形の各シートを作る。lang の既定は日本語（従来と同じ） */
export function buildTemplateSheets(buf: ArrayBuffer, lang: AppLanguage = 'ja'): SheetSpec[] {
  const S = FILE_LABELS[LANGUAGE_META[lang].file].sheets
  const wb = XLSX.read(buf, { type: 'array' })
  const rows = (name: string) =>
    XLSX.utils.sheet_to_json<(string | number | null)[]>(wb.Sheets[name], { header: 1, defval: '', blankrows: true })

  const roster = rows(S.roster)
  return [
    { name: GUIDE_SHEET[lang], rows: GUIDE[lang], bands: [{ row: 0, color: BAND, bold: true }, { row: 2, color: '#F1F5F9', bold: true }], boldCols: [0], colWidths: [140, 640] },
    { name: S.settings, rows: rows(S.settings), boldCols: [0], colWidths: [180, 80] },
    {
      name: S.roster,
      rows: roster,
      frozenRows: 2,
      frozenCols: 2,
      bands: [
        { row: 0, color: WEIGHT },
        { row: 1, color: BAND, bold: true },
      ],
      colWidths: [56, 120, ...Array.from({ length: Math.max(0, (roster[1]?.length ?? 2) - 2) }, () => 84)],
    },
    { name: S.wanted, rows: rows(S.wanted) },
    { name: S.unwanted, rows: rows(S.unwanted) },
  ]
}

/** 記入例つきのひな形スプレッドシートを利用者の Drive に作成する */
export async function createTemplateSpreadsheet(lang: AppLanguage = 'ja', title = 'Mosaic 名簿ひな形'): Promise<GoogleFile> {
  // その言語のサンプル（日本語も samples/ja/。旧 ./sample1.xlsx は以前のリンク用に残してある）
  const buf = await (await fetch(sampleUrl(lang, 'sample1'))).arrayBuffer()
  return createSpreadsheet(title, buildTemplateSheets(buf, lang))
}
