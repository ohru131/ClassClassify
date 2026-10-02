# 用語集（スマホ版 FairClass の多言語化）

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
| ペア指定のラベル | 同1 / 別1 | T1 / A1（Together / Apart） | 같1 / 분1 | J1 / S1（Juntos / Separar） | Z1 / G1（zusammen / getrennt。T は英語の Together と紛らわしいので使わない） | J1 / S1（Juntos / Separar） |
| 理想範囲（各組の目標人数の帯） | 理想の範囲 | target range | 목표 범위 | rango ideal | Sollbereich | faixa ideal |
| 理想（1組あたりの目標値） | 理想 | target | 목표 | meta | Soll | meta |
| バランス | バランス | balance | 균형 | equilibrio | Ausgewogenheit | equilíbrio |
| 人数差 | 人数差 | size difference | 인원 차 | diferencia de tamaño | Größenunterschied | diferença de tamanho |
| 条件違反 | 守れなかった指定 | unmet conditions | 충족하지 못한 조건 | condiciones no cumplidas | nicht erfüllte Bedingungen | condições não atendidas |
| 探索時間（高速/標準/徹底） | 考える時間: さっと / ふつう / じっくり | Quick / Standard / Thorough | 빠르게 / 표준 / 꼼꼼하게 | Rápido / Estándar / A fondo | Schnell / Standard / Gründlich | Rápido / Padrão / Completo |
| 編成を実行 | クラスを編成する | Build classes | 반 편성 실행 | Armar grupos | Klassen einteilen | Montar turmas |
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

### 全言語共通
- **「別の組（keep apart）」は保証しない。** 同じ組の指定はソルバーが必ず守る（生徒をまとめて1つの塊として動かす）が、別の組の指定は目的関数の重い罰則で、組数・人数によっては満たせない。UI・掲載文とも「できる限り（as far as possible / en la medida de lo posible / so weit wie möglich / 최대한）離し、満たせなかった指定はすぐに表示する」と書き、「確実に」「必ず」「meets」「cumple」「atende」とは書かない。

### ja
- 先生向けのアプリなので、技術用語（探索・最適化・実行・違反・破棄・無視など）を画面に出さず、やわらかい言い回しにする（「探索中」→「考えています」、「条件違反」→「守れなかった指定」、「中止」→「やめる」、「無視します」→「考えに入れません」）。

### en
- 米英豪で共通に通じる語だけを UI に使う（grade / year の区別が要る語は UI に出さない）。綴りは米式（behavior・color）を既定にし、`en-GB`・`en-AU` の掲載文だけ英式（behaviour）にする。
- 「組」は **class**。homeroom（米）・form（英）は使わない。
- 「FERPA 準拠」「GDPR compliant」などの**法的な保証と読める表現は書かない**（「nothing is uploaded」のように事実だけを書く）。

### ko
- 敬体（합니다체）で統一する。ボタンは名詞形・動詞の基本形（「실행」「내보내기」）。依頼・指示は「-십시오」（다시 실행하십시오）、確認は「-시겠습니까?」（삭제하시겠습니까?）。「-세요」「-할까요?」は使わない。
- 名簿の絞り込みは **필터**（필터 지우기・이 필터 지우기・필터 다시 적용）。「조건」は同じ組・別の組の指定（배정 조건）にだけ使う。
- 数を括弧で補うときは単位を付ける（반 수({k}개)）。
- 「반 편성」は分かち書きする（UI 全体で統一）。組の名前は「1반」。
- **학교폭력（학폭）の語をアプリ内で使わない**。分離の必要な生徒は「분리 배정」「분리가 필요한 학생」と中立に書く。
- **「다문화」を項目名の既定値・サンプルに置かない**（保護者に見られたときの問題）。

### es（中南米の語彙を基本にする）
- 「組」は **grupo** に統一（`curso` はスペインでは学年、チリでは組を指し、階層がずれるので UI の固定文言に使わない。`sección`・`paralelo` も国で割れるので使わない）。
- 生徒は **estudiante(s)**（包括的な語として中南米で好まれる。スペイン固有の `alumnado` は使わない）。
- 引用符は **“…”**（«» は使わない。pt-BR も同じ）。
- 設定・実行のタブは **Configurar**（「Ajustes」は Pro のタブ見出し「Pro y ajustes」と紛らわしい。pt-BR も Configurar）。
- 呼びかけは **tú**（ustedes ではなく単数。スペインの vosotros は使わない）。`ordenador`・`móvil` などスペイン固有の語は避け、端末は中立の「dispositivo」と書く。
- 小数点は中南米の多くと同じく Intl に任せる（es-419 はピリオド、es-ES はカンマ）。
- 学力について「同質な組を作る」と読める表現は避け、「学力が偏らないように（equilibrar el desempeño）」と書く（スペインでは成績による同質な組分けが禁止されている）。

### de
- **Sie** で統一する。
- 生徒は表の見出し・ボタンでは「Schüler」、説明文では「Schülerinnen und Schüler」を基本にする（「Schüler:innen」「SuS」などの記号・略語は学校・州で表記が割れるので使わない）。
- 語が長いので、ボタンは短い動詞（「Einteilen」「Exportieren」）を優先する。設定・実行のタブと画面見出しは「Einteilen」で揃える（「Einstellungen」は Pro のタブ見出しと紛らわしい）。
- 結果の「バランス」はタブ・集計とも **Ausgewogenheit**（Auswertung と混ぜない）。
- ペア指定の記号は **Z / G**（zusammen / getrennt）。
- 生徒を指すときに男性形の代名詞で受けない（「auf einen Schüler, um ihn …」ではなく「auf einen Namen, um ihn …」）。
- 「DSGVO-konform」と断言しない（判断するのは学校・州）。「Alle Daten bleiben auf dem Gerät」と事実を書く。

### pt-BR
- 呼びかけは **você**。
- 「組」は **turma**、学年は **ano**。
- 「LGPD に準拠」と断言しない。「os dados dos alunos não saem do aparelho」と事実を書く。

## 3. サンプル名簿の項目（国ごとに作り替え）

サンプル（`public/samples/<lang>/`、生成は `scripts/generate-samples.ts`、Web 版・スマホ版で共通）は、
**翻訳ではなく、その国の学校がクラス分けで実際に配慮する項目**で作ってある。各サンプルに
「該当（✓）」「カテゴリ（数種類）」「数値（7種類以上の点数 → 平均を揃える）」の3種類と、
同じ組・別の組の指定が入る（`test/samples.test.ts` が全言語で違反0・全項目が理想範囲内を確認）。

### 3.1 項目の一覧

| 言語 | sample1（80名・4組） | sample2（80名・シンプル） | sample-group（30名・6班） |
|---|---|---|---|
| ja | 性別♀・学習支援・登校支援・視覚配慮（該当）、情緒面の配慮・走力・ピアノ・学習・体育（1〜3）、PTA（該当）、**テスト平均（点数）** | 性別♀・学習支援・登校支援・学習・協調性（該当）、走力・ピアノ（カテゴリ）、**テスト平均** | 性別♀・走力・ピアノ・学習（該当）、体育・前回の組（カテゴリ）、**50m走（秒）** |
| en | Gender (F/M)、Reading score（点数）、Math level（1〜3）、IEP/504 plan・English learner・Behavior support・Leadership（✓） | Gender、Reading score、IEP/504 plan、Previous class（A〜D） | Gender、Reading score、Leadership、Previous group（1〜6） |
| ko | 성별(여/남)、학업 성취도（点数）、교우 관계 지원・특수교육 대상・한국어 지원・리더십（✓）、출신 초등학교（4校） | 성별、학업 성취도、특수교육 대상、이전 반（1〜4반） | 성별、학업 성취도、리더십、이전 모둠（1〜6） |
| es | Género (F/M)、Promedio de notas（4,0〜7,0）、NEE (PIE)・Liderazgo（✓）、Convivencia escolar（Sin observaciones/Seguimiento）、Grupo de origen（A〜D） | Género、Promedio de notas、NEE (PIE)、Grupo de origen | Género、Promedio de notas、Liderazgo、Equipo anterior（1〜6） |
| de | Geschlecht (w/m)、Notenschnitt（1,0〜4,0）、Förderbedarf・DaZ・Unterstützung Verhalten（✓）、Herkunftsgrundschule（4校） | Geschlecht、Notenschnitt、Förderbedarf、Herkunftsgrundschule | Geschlecht、Notenschnitt、Teamfähigkeit、Vorherige Gruppe（1〜6） |
| pt-BR | Gênero (F/M)、Média（5,0〜10,0）、AEE・Liderança（✓）、Convivência（Tranquila/Acompanhamento）、Turma de origem（A〜D） | Gênero、Média、AEE、Turma de origem | Gênero、Média、Liderança、Grupo anterior（1〜6） |

全言語共通で、同じ組（友だちの希望）と別の組（離す必要のある生徒）の指定を入れてある（sample-group は別の組のみ。日本語の元サンプルに合わせた）。

### 3.2 根拠

| 項目 | 根拠（資料と節） |
|---|---|
| 学力の平均を揃える（点数・成績の平均） | `pain-points-and-target.md` 1.2「学力: 平均点・成績分布を均等化」、`overseas-demand.md` A.1（ブラジルは学力の異質性＝混ぜる）、O 節 es（スペインは成績で同質な組を作るのが禁止 → 平均を揃える） |
| 前の組・出身校を散らす（Previous class / 출신 초등학교 / Herkunftsgrundschule / Grupo・Turma de origen） | `pain-points-and-target.md` 0 節 10「前年度クラスの分散」、`overseas-demand.md` C（韓国の中学は出身小学校の割合を考慮）、D（Klasse 5 の「出身小学校の大集団は避ける」） |
| 支援の必要性（IEP/504・특수교육 대상・Förderbedarf・NEE (PIE)・AEE・学習支援） | `pain-points-and-target.md` 1.2「支援の必要性」、`overseas-demand.md` O 節（米 IEP/504、韓 특수교육대상、独 Förderbedarf、es-419 NEE、チリ PIE、pt-BR AEE）、A.7 |
| 言語の支援（English learner・한국어 지원・DaZ） | `overseas-demand.md` O 節 en（EAL/ESL/ELL）・de（DaZ）。韓国は O 節 ko の「다문화 を属性名の既定値に置かない」に従い、**家庭の属性ではなく必要な支援（한국어 지원）**で表す |
| 行動・関係（Behavior support・교우 관계 지원・Unterstützung Verhalten・Convivencia escolar・Convivência） | `pain-points-and-target.md` 1.2「行動面: 海外では学力・社会性・行動の3軸」、`overseas-demand.md` B.2（米の基準に behavior）、A.1（ブラジルの「行動の異質性」）。値は「支援が要る/要らない」の中立な語にし、子どもを評価する語（良い/悪い）は使わない |
| リーダー性（Leadership・리더십・Liderazgo・Liderança・Teamfähigkeit） | `pain-points-and-target.md` 1.2「リーダー性・積極性: 学級委員候補を分散」 |
| 別の組（離す） | `pain-points-and-target.md` 1.2「人間関係（分離）」、`overseas-demand.md` C（学校暴力予防法の分離義務。ただしアプリ内では 학교폭력 の語を使わず「분리 배정」と書く） |
| 同じ組（友だちの希望） | `pain-points-and-target.md` 1.2「海外では友だち4人を書かせ最低1人と同じ組」、`overseas-demand.md` B.1・D（Freundschaftswunsch） |
| 日本語だけの項目（ピアノ・PTA・走力・登校支援） | `pain-points-and-target.md` 1.2（合唱の伴奏者を各組1人以上、運動会の戦力、欠席がちな子への配慮）。日本の学校行事に固有なので他言語には入れない |

### 3.3 表現のルール

- 性別は各言語の中立な略記（F/M、여/남、w/m）にし、男女を半々にした。選択肢は学校が自由に決められる（アプリは値の種類を固定しない）。
- 診断名・国籍・家庭事情は項目名にしない。**必要な支援**の名前にする（「障害」ではなく IEP/504・특수교육 대상、「外国籍・多文化」ではなく English learner・한국어 지원・DaZ）。
- es の「Grupo de origen」: 調査資料は「curso de origen」も挙げているが、`curso` はスペインで学年・チリで組を指して階層がずれる（A.7）ため、UI と同じ **grupo** に揃えた。
- **子どもを評価する値のカテゴリにしない。** 以前は ko「교우 관계: 원만/보통/지원 필요」、de「Verhalten: unauffällig/Unterstützung」と3段・2段の値を持たせていたが、「원만」「unauffällig」は子どもへの評価そのもので、名簿に残る（レビューで指摘）。**支援が要る子にだけ印を付ける該当項目**（교우 관계 지원・Unterstützung Verhalten）に変えた。カテゴリの項目は出身校で残るので、各サンプルが該当・カテゴリ・数値の3種類を含むのは変わらない。
- 小数の項目（Notenschnitt・Promedio de notas・Média・50m走）は**数値のセルに表示形式 0.0** で書く。文字列のままだと Excel で「数値が文字列として保存されています」と出て、並べ替え・平均が効かない。
- 氏名は、その国で一般的な名と姓を機械的に組み合わせた架空のもの（性別は半々）。組み合わせが実在の有名人と一致したもの（Owen Wilson・Jack White・Henry Adams・Beatriz Souza）は `scripts/generate-samples.ts` の `EXCLUDED_NAMES` で除いている。見つけたら足す。

## 4. ネイティブスピーカーに確認してほしい箇所

- ko: 「분리 배정」「배정 조건」の語感（学폭を想起させすぎないか）、「꼼꼼하게」（探索時間）。
- es: 「Armar grupos」（チリ・アルゼンチンでは自然、メキシコで違和感が無いか）、「Comité de familias」。
- de: 「Unterstützung Anwesenheit」「Wünsche」（ペア指定の総称。保護者の友だち希望と区別がつくか）。
- pt-BR: 「Enturmação」をシート名に使ってよいか（学校の事務用語として通じるか）。
- サンプルの項目名・値: ko「교우 관계 지원」、「한국어 지원」／de「Unterstützung Verhalten」「Notenschnitt」の向き（1,0 が最良）／es「Convivencia escolar: Seguimiento」「NEE (PIE)」（チリ以外で通じるか）／pt-BR「Convivência: Acompanhamento」「AEE」。

### 4.1 レビュー（PR #10）で直した箇所のうち、ネイティブに確かめてほしいもの

語そのもの（분리 배정・Armar grupos・Wünsche）は上のとおり確認待ちのまま変えていない。今回直した言い回しで、自信が持ちきれないものを挙げる。

- ko: 합니다체への統一（「-십시오」「-시겠습니까?」）が画面の短いボタン・トーストで硬すぎないか。「필터」（絞り込み）が教員に通じるか。
- es: タブ名「Configurar」（「Armar」だけのタブにする案もあった）。「No se pueden compartir archivos…」「presiona Enter」。スペイン向け掲載文（es-ES）で UI が中南米の語彙であることを断っている一文の自然さ。
- pt-BR: タブ名「Configurar」。掲載文の「Para a LGPD, vale saber que…」。
- de: タブと画面見出しを「Einteilen」に揃えたこと。「Ausgewogenheit」をタブ名にしたこと（長いのでソフトハイフンで折っている）。ペア指定の記号 Z / G（zusammen / getrennt）が読み取れるか。「Noch keine Gruppen」「auf einen Namen, um ihn …」。サンプルの「Unterstützung Verhalten」。
- 全言語: 「別の組」の説明を「できる限り離し、満たせなかった指定はすぐに表示する」に変えた掲載文の言い回し（en の "keeps apart the ones you separate as far as possible"、de の „trennt die anderen so weit wie möglich“ など）。
- 全言語: 最大人数を無視したときの警告（「1クラス最大 {max} 人 × {k} クラスでは全員が入らないため、最大人数は無視しました」とその訳）。
