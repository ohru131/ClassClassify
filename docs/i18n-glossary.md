# 用語集（スマホ版 Mosaic の多言語化）

スマホ版（`mobile/`）の UI・Excel の語彙（`src/solver/labels.ts`）・ストア掲載文（`docs/store-listing.md`）は
**この表の訳語に揃える**。訳語を変えるときは、先にこの表を直してから各所を直す（訳語のブレを防ぐため）。
根拠は `docs/research/overseas-demand.md` の「O. 言語ごとのローカライズ注意点」と「A.7 用語の差」。

対応言語: `ja`（既存）・`en`・`ko`・`es`（中南米の語彙を基本にする）・`de`・`pt-BR`。

## 1. 対訳表

| 概念 | ja | en | ko | es（中南米寄り） | de | pt-BR |
|---|---|---|---|---|---|---|
| クラス編成（作業・アプリの主題） | クラス編成 | class placement | 반 편성 | distribución de estudiantes en grupos | Klasseneinteilung | enturmação / distribuição de alunos nas turmas |
| 組（クラス）の単位 | 組 / クラス | class | 반 | grupo | Klasse | turma |
| 組の名前 | 1組 | Class 1 | 1반 | Grupo 1 | Klasse 1 | Turma 1 |
| クラス（グループ）数 | クラス数 | number of classes | 반 수 | número de grupos | Anzahl Klassen | número de turmas |
| 1クラスの最大人数 | 最大人数 | maximum class size | 반별 최대 인원 | máximo por grupo | Höchstzahl pro Klasse | máximo por turma |
| 生徒 | 生徒 | student | 학생 | estudiante | Schülerin / Schüler（表では「Schüler」） | aluno |
| 名簿 | 名簿 | roster | 명단 | lista de estudiantes | Schülerliste | lista de alunos |
| 項目（生徒の特性） | 項目 | attribute | 항목 | criterio | Merkmal | critério |
| 重み | 重み | weight | 가중치 | peso | Gewicht | peso |
| 同じ組にする | 同じ組にする | keep together | 같은 반 배정 | mantener juntos | zusammen | manter juntos |
| 別の組にする | 別の組にする | keep apart | 분리 배정 | separar | trennen | separar |
| ペア指定（上の2つの総称） | ペア指定 | pairings | 배정 조건 | condiciones | Wünsche | condições |
| ペア指定のラベル | 同1 / 別1 | T1 / A1（Together / Apart） | 같1 / 분1 | J1 / S1（Juntos / Separar） | Z1 / T1（Zusammen / Trennen） | J1 / S1（Juntos / Separar） |
| 理想範囲（各組の目標人数の帯） | 理想の範囲 | target range | 목표 범위 | rango ideal | Sollbereich | faixa ideal |
| 理想（1組あたりの目標値） | 理想 | target | 목표 | meta | Soll | meta |
| バランス | バランス | balance | 균형 | equilibrio | Ausgewogenheit | equilíbrio |
| 人数差 | 人数差 | size difference | 인원 차 | diferencia de tamaño | Größenunterschied | diferença de tamanho |
| 条件違反 | 条件違反 | unmet conditions | 충족하지 못한 조건 | condiciones no cumplidas | nicht erfüllte Bedingungen | condições não atendidas |
| 探索時間（高速/標準/徹底） | 高速 / 標準 / 徹底 | Quick / Standard / Thorough | 빠르게 / 표준 / 꼼꼼하게 | Rápido / Estándar / A fondo | Schnell / Standard / Gründlich | Rápido / Padrão / Completo |
| 編成を実行 | クラス編成を実行 | Build classes | 반 편성 실행 | Armar grupos | Klassen einteilen | Montar turmas |
| 手動で移動 | 別の組へ移動 | move to another class | 다른 반으로 이동 | mover a otro grupo | in andere Klasse verschieben | mover para outra turma |
| 該当 / カテゴリ / 数値（項目の種類） | 該当 / カテゴリ / 数値 | Yes/no / Category / Number | 해당 / 범주 / 숫자 | Sí/no / Categoría / Número | Ja/Nein / Kategorie / Zahl | Sim/não / Categoria / Número |
| 空欄 | 空欄 | blank | 빈칸 | vacío | leer | vazio |
| ひな形 | ひな形 | template | 양식 | plantilla | Vorlage | modelo |
| 書き出し | 書き出し | export | 내보내기 | exportar | exportieren | exportar |
| 印刷・PDF | 印刷・PDF | print / PDF | 인쇄 / PDF | imprimir / PDF | drucken / PDF | imprimir / PDF |
| 買い切り | 買い切り | one-time purchase | 1회 구매 | pago único | Einmalkauf | compra única |
| 購入を復元 | 購入を復元 | Restore purchase | 구매 복원 | Restaurar compra | Kauf wiederherstellen | Restaurar compra |
| 端末内に保存 | この端末の中だけに保存 | stays on this device | 이 기기에만 저장 | se guarda solo en este dispositivo | bleibt auf diesem Gerät | fica só neste aparelho |

## 2. 言語ごとの表記ルール

### en
- 米英豪で共通に通じる語だけを UI に使う（grade / year の区別が要る語は UI に出さない）。綴りは米式（behavior・color）を既定にし、`en-GB`・`en-AU` の掲載文だけ英式（behaviour）にする。
- 「組」は **class**。homeroom（米）・form（英）は使わない。
- 「FERPA 準拠」「GDPR compliant」などの**法的な保証と読める表現は書かない**（「nothing is uploaded」のように事実だけを書く）。

### ko
- 敬体（합니다체）で統一する。ボタンは名詞形・動詞の基本形（「실행」「내보내기」）。
- 「반 편성」は分かち書きする（UI 全体で統一）。組の名前は「1반」。
- **학교폭력（학폭）の語をアプリ内で使わない**。分離の必要な生徒は「분리 배정」「분리가 필요한 학생」と中立に書く。
- **「다문화」を項目名の既定値・サンプルに置かない**（保護者に見られたときの問題）。

### es（中南米の語彙を基本にする）
- 「組」は **grupo** に統一（`curso` はスペインでは学年、チリでは組を指し、階層がずれるので UI の固定文言に使わない。`sección`・`paralelo` も国で割れるので使わない）。
- 生徒は **estudiante(s)**（包括的な語として中南米で好まれる。スペイン固有の `alumnado` は使わない）。
- 呼びかけは **tú**（ustedes ではなく単数。スペインの vosotros は使わない）。`ordenador`・`móvil` などスペイン固有の語は避け、端末は中立の「dispositivo」と書く。
- 小数点は中南米の多くと同じく Intl に任せる（es-419 はピリオド、es-ES はカンマ）。
- 学力について「同質な組を作る」と読める表現は避け、「学力が偏らないように（equilibrar el desempeño）」と書く（スペインでは成績による同質な組分けが禁止されている）。

### de
- **Sie** で統一する。
- 生徒は表の見出し・ボタンでは「Schüler」、説明文では「Schülerinnen und Schüler」を基本にする（「Schüler:innen」「SuS」などの記号・略語は学校・州で表記が割れるので使わない）。
- 語が長いので、ボタンは短い動詞（「Einteilen」「Exportieren」）を優先する。
- 「DSGVO-konform」と断言しない（判断するのは学校・州）。「Alle Daten bleiben auf dem Gerät」と事実を書く。

### pt-BR
- 呼びかけは **você**。
- 「組」は **turma**、学年は **ano**。
- 「LGPD に準拠」と断言しない。「os dados dos alunos não saem do aparelho」と事実を書く。

## 3. 配慮の必要な項目名（サンプル・ひな形）

性別や支援の必要性は、ラベルそのものが生徒を指す言葉になる。サンプルでは次の表現を使う（`mobile/scripts/generate-samples.ts`）。

| 日本語のサンプル | en | ko | es | de | pt-BR |
|---|---|---|---|---|---|
| 性別♀（該当＝女子） | Girl | 여학생 | Niña | Mädchen | Menina |
| 学習支援 | Learning support | 학습 지원 | Apoyo en el aprendizaje | Lernförderung | Apoio à aprendizagem |
| 登校支援 | Attendance support | 등교 지원 | Apoyo a la asistencia | Unterstützung Anwesenheit | Apoio à frequência |
| 視覚配慮 | Vision support | 시각 지원 | Apoyo visual | Unterstützung Sehen | Apoio visual |
| 情緒面の配慮 | Emotional support | 정서 지원 | Apoyo emocional | Emotionale Unterstützung | Apoio emocional |
| 走力 | Running | 달리기 | Velocidad | Laufen | Corrida |
| ピアノ | Piano | 피아노 | Piano | Klavier | Piano |
| 学習 | Academics | 학업 | Desempeño académico | Leistung | Desempenho |
| 体育 | PE | 체육 | Educación física | Sport | Educação física |
| PTA | Parent committee | 학부모회 | Comité de familias | Elternbeirat | Conselho de pais |
| 協調性 | Teamwork | 협동심 | Trabajo en equipo | Teamfähigkeit | Cooperação |
| 前回の組 | Previous group | 이전 모둠 | Grupo anterior | Vorherige Gruppe | Grupo anterior |

- 診断名・制度名（IEP・NEE・Förderbedarf・특수교육대상 など）は**サンプルに入れない**（制度は国ごとに違い、診断名は機微な情報のため）。学校が自分の名簿で使う名前を入れる。
- 「該当」の印はサンプルでは ja `○`、他言語 `✓`。どちらも1種類の値なら「該当」として扱われる。

## 4. ネイティブスピーカーに確認してほしい箇所

- ko: 「분리 배정」「배정 조건」の語感（学폭を想起させすぎないか）、「꼼꼼하게」（探索時間）。
- es: 「Armar grupos」（チリ・アルゼンチンでは自然、メキシコで違和感が無いか）、「Comité de familias」。
- de: 「Unterstützung Anwesenheit」「Wünsche」（ペア指定の総称。保護者の友だち希望と区別がつくか）。
- pt-BR: 「Enturmação」をシート名に使ってよいか（学校の事務用語として通じるか）。
