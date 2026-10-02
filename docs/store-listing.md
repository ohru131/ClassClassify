# ストア掲載文（Google Play・スマホ版 FairClass）

- 対象: `mobile/` のアプリ（Android。iOS も同じ文面を流用できる）。訳語は `docs/i18n-glossary.md` に揃えた。
- **直訳ではなく各国のターゲット向けに書いてある**（根拠: `docs/research/overseas-demand.md` の O 節・A 節）。
  米豪英 = 無料・登録不要・端末内、韓国 = 2月の 반 편성と分離配置（학교폭력 の語は使わない）、南米 = 学年始まり・LGPD、独 = Datenschutz を最初に。
- 字数は Python の `len()`（Unicode 文字数）で実測した値。Play の上限はアプリ名 30・短い説明 80・詳しい説明 4000。
- **法的な保証と読める表現は書かない**（「FERPA/DSGVO/LGPD に準拠」とは書かず、「データを送信しない」という事実だけを書く）。
- 価格はストア側で設定する（アプリはストアのローカライズ済みの価格をそのまま表示する）。各ロケールの「価格」は `docs/play-console/pricing.csv`（国別価格の唯一の情報源。根拠は `docs/play-console/pricing.md`）の値で、【推定】を含む。
- 掲載文で言ってよいこと（アプリの実態と一致させる）: 広告なし・アカウント／登録なし、読み込み・編集・編成・手直しは無料、Pro は**買い切り（one-time purchase）**で Excel への書き出しと印刷・PDF、名簿は端末の外へ出ない（外へ出るのは購入確認のための匿名 ID とレシートだけ）、スマホ・Android タブレット・Chromebook 対応。
- **Play Console ではタブレット（7インチ・10インチ）と Chromebook 用のスクリーンショットも登録する**（`submission-assets/screenshots/` の phone 1080×1920・tablet7 1296×2304・tablet10 1920×1080・chromebook 1920×1080。タブレット・Chromebook は 16:9 / 9:16。上げる順は `submission-assets/README.md`）。

| Play のロケール | アプリ名 | 短い説明 | 詳しい説明 |
|---|---|---|---|
| en-US | 26 | 80 | 2258 |
| en-AU | 26 | 77 | 2051 |
| en-GB | 25 | 76 | 1976 |
| ko-KR | 14 | 55 | 1066 |
| es-419 | 23 | 79 | 2281 |
| es-ES | 28 | 78 | 2307 |
| de-DE | 27 | 79 | 2590 |
| pt-BR | 24 | 76 | 2289 |
| ja-JP | 22 | 44 | 898 |

## en-US（英語・米国）

- 狙い: 学区が SaaS を契約していない学校・個人の教員。春（4〜6月）のクラス替え。無料・登録不要・端末内
- 繁忙期（掲載文・スクショを差し替える時期）: 3〜5月（米）
- 価格: **US$5.99**（買い切り。`docs/play-console/pricing.md`）
- ASO キーワード（本文の地の文に入れてある）: `class placement`, `class lists`, `no sign-up`, `Chromebook`, `keep apart`

### アプリ名（26字 / 30）

```
FairClass: Class Placement
```

### 短い説明（80字 / 80）

```
Balanced class lists in seconds. Free, no sign-up, student data stays on device.
```

### 詳しい説明（2258字 / 4000）

```
Building next year's class lists by hand takes hours: balancing boys and girls, academic levels, students who need extra support, and the friendships and conflicts parents and teachers tell you about. FairClass does the class placement for you in seconds, and you stay in control of every decision.

HOW IT WORKS
• Load your roster from an Excel file (.xlsx), start from a sample, or type it in.
• Add any attributes you care about — gender, reading level, behavior, IEP or 504 support, English learners — and give each one a weight.
• Mark students to keep together and students to keep apart.
• Choose the number of classes and tap "Build classes". FairClass spreads every attribute as evenly as possible, always keeps linked students together and keeps apart the ones you separate as far as possible. Any condition it cannot meet is shown right away.
• Check the balance tables, then move any student to another class with one tap. Totals update instantly.

WHY TEACHERS USE IT
• Free to build class lists — no ads, no account, no sign-up.
• Student data never leaves your device. No roster data is uploaded; the optimization runs on your phone, tablet or Chromebook.
• Works on phones, Android tablets and Chromebooks, in portrait or landscape, with a mouse and keyboard too.
• Every class gets a balanced mix, not just an even headcount: you can see the target range for each attribute and whether every class is inside it.
• Great for group work as well: make balanced teams or table groups in the same way.

FAIRCLASS PRO (ONE-TIME PURCHASE)
• Export the results to Excel and Google Drive: placement, class lists, one sheet per class, pairings and a summary.
• Print or save as PDF for staff meetings: class lists with color-coded pairings and a legend, plus the balance tables, on print-ready pages.
• Buy once and keep it — Pro is not a subscription.

FairClass uses the same engine as the free FairClass web app, and Excel files work in both. The class placement uses simulated annealing, an optimization method that tries a huge number of combinations.

Privacy: rosters and results are stored only on your device. To verify a Pro purchase, the app store and RevenueCat receive an anonymous ID and the receipt — never student names or roster data.
```

## en-AU（英語・オーストラリア）

- 狙い: Class Placement Policy に沿って Term 4 に翌年のクラスを作る小学校。Class Creator 等を契約していない学校・担任
- 繁忙期（掲載文・スクショを差し替える時期）: 9〜11月（Term 3〜4）
- 価格: **A$8.99**（NZ は NZ$9.99。`pricing.csv`【推定】）
- ASO キーワード（本文の地の文に入れてある）: `class placement`, `Term 4`, `no sign-up`, `Chromebook`, `keep apart`

### アプリ名（26字 / 30）

```
FairClass: Class Placement
```

### 短い説明（77字 / 80）

```
Next year's classes, balanced in seconds. Free, no sign-up, data stays local.
```

### 詳しい説明（2051字 / 4000）

```
Every Term 4, teachers sit down with sticky notes and spreadsheets to build next year's classes: an even spread of girls and boys, learning levels, behaviour, students with a support plan or EAL/D support, and the friendship requests from families. FairClass turns that into a few seconds of work and leaves every final decision with you.

HOW IT WORKS
• Load your roster from Excel (.xlsx), start from a sample, or enter it in the app.
• Add the attributes your class placement policy uses and give each a weight.
• Record students to keep together (for example, one friend each) and students to keep apart.
• Set the number of classes and tap "Build classes". FairClass spreads every attribute evenly, always keeps linked students together and keeps apart the ones you separate as far as possible. Any condition it cannot meet is shown right away.
• Review the balance tables and move any student to another class with a single tap. Totals update straight away.

WHY SCHOOLS USE IT
• Free to build your classes — no ads, no account and no sign-up.
• Student data stays on your device. No roster data is uploaded; the calculation runs on your phone, tablet or Chromebook.
• Works on Android tablets and Chromebooks in portrait or landscape, with mouse and keyboard.
• See the target range for every attribute and whether each class is inside it, so you can explain the lists to your leadership team.
• Also handy for balanced groups and teams within a class.

FAIRCLASS PRO (ONE-OFF PURCHASE)
• Export results to Excel and Google Drive: placement, class lists, a sheet per class, pairings and a summary.
• Print or save as PDF for staff meetings: class lists with colour-coded pairings and a legend, plus balance tables.
• Pay once and keep it — no subscription.

FairClass uses the same engine as the free FairClass web app, and the Excel files work in both.

Privacy: rosters and results are stored only on your device. To verify a Pro purchase, the store and RevenueCat receive an anonymous ID and the receipt — never student names or roster data.
```

## en-GB（英語・英国）

- 狙い: 2学級以上の小学校の mixing classes（6〜7月）。MIS 連携の SaaS を使っていない学校
- 繁忙期（掲載文・スクショを差し替える時期）: 5〜7月
- 価格: **£4.99**（アイルランドはユーロ圏の €5,99。`pricing.csv`【推定】）
- ASO キーワード（本文の地の文に入れてある）: `mixing classes`, `class lists`, `no sign-up`, `Chromebook`, `keep apart`

### アプリ名（25字 / 30）

```
FairClass: Mixing Classes
```

### 短い説明（76字 / 80）

```
Mix classes fairly in seconds. Free, no sign-up, pupil data stays on device.
```

### 詳しい説明（1976字 / 4000）

```
Mixing classes in a two-form or three-form entry school is one of the hardest jobs of the summer term: balancing girls and boys, attainment, behaviour, pupils with a support plan, EAL, and the friendship groups parents ask about. FairClass does the number-crunching in seconds and leaves every decision with you.

HOW IT WORKS
• Load your class list from Excel (.xlsx), start from a sample, or type it in.
• Add the attributes you care about and give each one a weight.
• Mark pupils to keep together and pupils to keep apart.
• Choose the number of classes and tap "Build classes". FairClass spreads every attribute evenly, always keeps linked students together and keeps apart the ones you separate as far as possible. Any condition it cannot meet is shown right away.
• Check the balance tables and move any pupil to another class with one tap. Totals update instantly.

WHY TEACHERS USE IT
• Free to build class lists — no ads, no account, no sign-up.
• Pupil data stays on your device. No roster data is uploaded; the calculation runs on your phone, tablet or Chromebook.
• Works on Android tablets and Chromebooks, in portrait or landscape, with a mouse and keyboard.
• Shows the target range for every attribute and whether each class sits inside it — useful when you explain the new classes to parents and governors.
• Also works for balanced groups and teams within a class.

FAIRCLASS PRO (ONE-OFF PURCHASE)
• Export results to Excel and Google Drive: placement, class lists, a sheet per class, pairings and a summary.
• Print or save as PDF for staff meetings: colour-coded class lists with a legend and the balance tables on A4.
• Pay once and keep it — no subscription.

FairClass uses the same engine as the free FairClass web app, and the Excel files work in both.

Privacy: class lists and results are stored only on your device. To verify a Pro purchase, the store and RevenueCat receive an anonymous ID and the receipt — never pupil names or class list data.
```

## ko-KR（韓国語）

- 狙い: 2月に全学年で반 편성をする学年担任団。분리가 필요한 학생の分離配置（学폭の語は使わない）
- 繁忙期（掲載文・スクショを差し替える時期）: 12〜2月（2月上〜中旬に반편성）
- 価格: **₩7,900**（手取りで US$5.99 と揃える。₩6,900 は価格テストの候補。`pricing.md`【推定】）
- ASO キーワード（本文の地の文に入れてある）: `반 편성`, `분리 배정`, `모둠 편성`, `크롬북`

### アプリ名（14字 / 30）

```
FairClass 반 편성
```

### 短い説明（55字 / 80）

```
성별·학업·지원 필요 학생을 고르게, 분리 배정도 최대한 반영해 몇 초 만에 반 편성. 가입 없음.
```

### 詳しい説明（1066字 / 4000）

```
매년 2월, 다음 학년 반 편성은 담임 선생님들이 며칠씩 매달리는 일입니다. 남녀 비율, 학업 수준, 학습 지원이 필요한 학생, 같은 반에 두면 안 되는 학생까지 모두 고려해야 하니까요. FairClass는 이 반 편성을 몇 초 만에 끝내고, 최종 판단은 선생님께 맡깁니다.

사용 방법
• 엑셀 파일(.xlsx)로 학생 명단을 불러오거나, 예시로 시작하거나, 앱에서 직접 입력합니다.
• 성별, 학업, 학습 지원, 체육 등 고려할 항목을 넣고 항목마다 가중치를 정합니다.
• 같은 반에 배정할 학생과 서로 다른 반에 배정할 학생(분리 배정)을 지정합니다.
• 반 수를 정하고 "반 편성 실행"을 누르면 모든 항목이 각 반에 고르게 나뉩니다. 같은 반 배정은 항상 지키고, 분리 배정은 최대한 반영합니다.
• 균형 표를 확인하고, 필요하면 학생을 눌러 다른 반으로 옮기세요. 집계는 바로 다시 계산됩니다.

선생님들이 쓰는 이유
• 반 편성은 무료입니다. 광고도, 회원 가입도 없습니다.
• 학생 명단은 이 기기에만 저장되고 어디에도 전송되지 않습니다. 계산도 휴대폰·태블릿·크롬북 안에서 합니다.
• 분리가 필요한 학생은 최대한 다른 반으로 나누고, 충족하지 못한 조건이 있으면 바로 표시합니다.
• 항목마다 목표 범위와 각 반의 인원이 표로 보여서, 학년 협의회에서 설명하기 쉽습니다.
• 안드로이드 태블릿과 크롬북에서 가로·세로 모두, 마우스와 키보드로도 쓸 수 있습니다.
• 수업 중 모둠 편성에도 같은 방식으로 쓸 수 있습니다.

FairClass Pro(1회 구매)
• 결과를 엑셀·Google 드라이브로 내보내기: 반 편성, 반별 명단, 반마다의 시트, 배정 조건, 집계.
• 회의용 인쇄·PDF: 배정 조건을 색으로 구분한 반별 명단과 범례, 균형 표를 A4로.
• 한 번 구매로 계속 사용합니다. 구독이 아닙니다.

무료 웹 버전 FairClass와 같은 엔진을 쓰며, 엑셀 파일은 양쪽에서 그대로 열립니다.

개인정보: 명단과 결과는 기기에만 저장됩니다. Pro 구매 확인을 위해 스토어와 RevenueCat이 익명 식별자와 영수증만 받으며, 학생 이름이나 명단 내용은 보내지 않습니다.
```

## es-419（スペイン語・中南米）

- 狙い: 南半球の学年始まり（2〜3月）前に grupos / cursos を作る学校。データは端末内（チリの新個人情報法・各国の個人情報保護）
- 繁忙期（掲載文・スクショを差し替える時期）: 10〜2月（南半球・コロンビア A）、メキシコは 6〜8月
- 価格: **MX$79・CLP 3.990・COP 12.900・PEN 10,90・US$3.99（EC）**、ARS は Play 側の通貨を確認してから（`pricing.csv` の status=confirm）。その他の中南米は自動換算（`pricing.md`【推定】）
- ASO キーワード（本文の地の文に入れてある）: `armar grupos`, `distribución de estudiantes`, `sin registro`, `Chromebook`, `equipos de trabajo`

### アプリ名（23字 / 30）

```
FairClass: armar grupos
```

### 短い説明（79字 / 80）

```
Arma grupos equilibrados para el año escolar en segundos. Gratis, sin registro.
```

### 詳しい説明（2281字 / 4000）

```
Antes de que empiece el año escolar hay que armar los grupos: equilibrar niñas y niños, el desempeño académico, los estudiantes con NEE que necesitan apoyo y las amistades o conflictos que conocen los docentes. Hacerlo a mano toma días. FairClass hace la distribución de estudiantes en grupos en segundos, y la decisión final siempre es tuya.

CÓMO FUNCIONA
• Carga la lista de estudiantes desde Excel (.xlsx), empieza con un ejemplo o escríbela en la app.
• Agrega los criterios que quieras (género, desempeño, apoyo en el aprendizaje, convivencia) y dale un peso a cada uno.
• Indica qué estudiantes mantener juntos y a quiénes separar.
• Elige el número de grupos y toca “Armar grupos”. FairClass reparte cada criterio de forma pareja, siempre deja juntos a los estudiantes que unes y separa a los demás en la medida de lo posible. Si alguna condición no se puede cumplir, lo ves de inmediato.
• Revisa las tablas de equilibrio y mueve a cualquier estudiante a otro grupo con un toque. Los totales se recalculan al instante.

POR QUÉ LO USAN LOS DOCENTES
• Armar los grupos es gratis: sin anuncios, sin cuenta y sin registro.
• Los datos de los estudiantes se quedan en tu dispositivo. No se sube ningún dato de la lista a internet; el cálculo se hace en tu celular, tablet o Chromebook.
• Grupos heterogéneos y parejos: ves el rango ideal de cada criterio y si cada grupo está dentro, algo fácil de explicar en el consejo de profesores.
• Funciona en tablets Android y Chromebook, en vertical u horizontal, con mouse y teclado.
• También sirve para armar equipos de trabajo dentro de la clase.

FAIRCLASS PRO (PAGO ÚNICO)
• Exporta los resultados a Excel y Google Drive: distribución, listas por grupo, una hoja por grupo, condiciones y resumen.
• Imprime o guarda en PDF para el consejo de profesores: listas por grupo con las condiciones en colores y una leyenda, más las tablas de equilibrio.
• Pagas una vez y es tuyo: no es una suscripción.

FairClass usa el mismo motor que la versión web gratuita, y los archivos de Excel funcionan en ambas.

Privacidad: las listas y los resultados se guardan solo en tu dispositivo. Para verificar la compra de Pro, la tienda y RevenueCat reciben un identificador anónimo y el recibo, nunca nombres de estudiantes ni datos de la lista.
```

## es-ES（スペイン語・スペイン）

- 狙い: 学校の計画書で編成基準（agrupamiento del alumnado）を公開する義務がある公立校。成績による同質な組分けは禁止なので「異質性を保つ」訴求
- 繁忙期（掲載文・スクショを差し替える時期）: 5〜7月
- 価格: **5,99 €**（ユーロ圏は全加盟国で同じ。`pricing.csv`）
- ASO キーワード（本文の地の文に入れてある）: `reparto en grupos`, `agrupamiento del alumnado`, `sin registro`, `Chromebook`

### アプリ名（28字 / 30）

```
FairClass: reparto en grupos
```

### 短い説明（78字 / 80）

```
Reparte al alumnado en grupos equilibrados en segundos. Gratis y sin registro.
```

### 詳しい説明（2307字 / 4000）

```
Cada curso hay que repartir al alumnado en grupos siguiendo los criterios del centro: equilibrio entre niñas y niños, rendimiento variado en cada grupo, alumnado con necesidades de apoyo repartido de forma justa y las incompatibilidades que conoce el equipo docente. FairClass hace ese reparto en grupos en segundos y deja la última palabra al profesorado.

CÓMO FUNCIONA
• Carga la lista desde Excel (.xlsx), empieza con un ejemplo o escríbela en la app. La app usa vocabulario latinoamericano (“Armar grupos”, “estudiantes”), y los ejemplos incluidos usan la escala de notas de 1 a 7; con tu propio Excel puedes usar la de 0 a 10.
• Añade los criterios del centro y dale un peso a cada uno.
• Indica qué estudiantes mantener juntos y a quiénes separar.
• Elige el número de grupos y pulsa “Armar grupos”. FairClass reparte cada criterio de forma equilibrada: los grupos quedan heterogéneos y parecidos entre sí, sin agrupar por rendimiento. Los estudiantes que unes quedan siempre juntos y los que separas, separados en la medida de lo posible; si alguna condición no se puede cumplir, lo ves de inmediato.
• Revisa las tablas de equilibrio y mueve a cualquier estudiante a otro grupo con un toque.

POR QUÉ LO USA EL PROFESORADO
• Hacer los grupos es gratis: sin anuncios, sin cuenta y sin registro.
• Los datos del alumnado no salen del dispositivo: no se sube nada a internet.
• Muestra el rango ideal de cada criterio y si cada grupo lo cumple, útil para justificar el agrupamiento del alumnado ante el claustro.
• Funciona en tablets Android y Chromebook, en vertical u horizontal, con ratón y teclado.
• También sirve para equipos de trabajo dentro del aula.

FAIRCLASS PRO (PAGO ÚNICO)
• Exporta los resultados a Excel y Google Drive: reparto, listas por grupo, una hoja por grupo, condiciones y resumen.
• Imprime o guarda en PDF para la sesión de evaluación: listas por grupo con colores y leyenda, más las tablas de equilibrio.
• Pagas una vez: no es una suscripción.

FairClass usa el mismo motor que la versión web gratuita y los archivos de Excel sirven en las dos.

Privacidad: las listas y los resultados se guardan solo en el dispositivo. Para verificar la compra de Pro, la tienda y RevenueCat reciben un identificador anónimo y el recibo, nunca nombres del alumnado ni datos de la lista.
```

## de-DE（ドイツ語）

- 狙い: Einschulung・Klasse 5 のクラス編成（5〜7月）。DSGVO と学校データ規則で「送信しない」が最も効く（「DSGVO-konform」とは書かない）
- 繁忙期（掲載文・スクショを差し替える時期）: 4〜7月（Einschulung・Klasse 5）
- 価格: **5,99 €**（スイスは CHF 5.00。`pricing.csv`【推定】）
- ASO キーワード（本文の地の文に入れてある）: `Klasseneinteilung`, `Klassenbildung`, `Alle Daten bleiben auf dem Gerät`, `Chromebook`, `Gruppeneinteilung`

### アプリ名（27字 / 30）

```
FairClass Klasseneinteilung
```

### 短い説明（79字 / 80）

```
Ausgewogene Klassen in Sekunden. Alle Daten bleiben auf dem Gerät – ohne Konto.
```

### 詳しい説明（2590字 / 4000）

```
Alle Daten bleiben auf dem Gerät. Keine Cloud, kein Konto: FairClass berechnet die Klasseneinteilung direkt auf Ihrem Smartphone, Tablet oder Chromebook.

Ob Einschulung oder Übergang in Klasse 5 – bei der Klassenbildung müssen viele Kriterien gleichzeitig passen: ausgeglichenes Verhältnis von Mädchen und Jungen, gemischte Leistung, Schülerinnen und Schüler mit Förderbedarf gerecht verteilt, Freundschaftswünsche der Eltern und Kinder, die getrennt werden sollten. FairClass erledigt diese Einteilung in Sekunden – die letzte Entscheidung treffen immer Sie.

SO FUNKTIONIERT ES
• Schülerliste aus Excel (.xlsx) laden, mit einem Beispiel beginnen oder direkt in der App eingeben.
• Merkmale festlegen (z. B. Geschlecht, Leistung, Förderbedarf, DaZ) und jedem Merkmal ein Gewicht geben.
• Freundschaftswünsche als „zusammen“ und Kinder, die getrennt werden sollen, als „trennen“ eintragen.
• Anzahl der Klassen wählen und „Klassen einteilen“ antippen. FairClass verteilt jedes Merkmal möglichst gleichmäßig, hält „zusammen“ immer ein und trennt die anderen so weit wie möglich. Was sich nicht erfüllen lässt, wird sofort angezeigt.
• Verteilungstabellen prüfen und einzelne Kinder mit einem Tipp in eine andere Klasse verschieben. Die Auswertung wird sofort neu berechnet.

WARUM LEHRKRÄFTE ES NUTZEN
• Die Klasseneinteilung ist kostenlos – ohne Werbung, ohne Konto, ohne Registrierung.
• Keine Übertragung von Schülerdaten: Nichts wird hochgeladen.
• Für jedes Merkmal sehen Sie den Sollbereich und ob jede Klasse darin liegt – so lässt sich die Einteilung in der Konferenz gut begründen.
• Große Gruppen aus verketteten Wünschen (A mit B, B mit C …) werden erkannt und gemeinsam eingeteilt.
• Läuft auf Android-Tablets und Chromebooks, im Hoch- und Querformat, auch mit Maus und Tastatur.
• Auch für die Gruppeneinteilung im Unterricht geeignet.

FAIRCLASS PRO (EINMALKAUF)
• Ergebnis nach Excel und Google Drive exportieren: Einteilung, Klassenlisten, ein Blatt pro Klasse, Wünsche und Auswertung.
• Drucken oder als PDF für die Konferenz: Klassenlisten mit farbig markierten Wünschen und Legende sowie Verteilungstabellen auf A4.
• Einmal kaufen, dauerhaft nutzen – kein Abo.

FairClass nutzt dieselbe Berechnung wie die kostenlose Webversion; die Excel-Dateien funktionieren in beiden.

Datenschutz: Schülerlisten und Ergebnisse werden nur auf dem Gerät gespeichert. Zur Prüfung eines Pro-Kaufs erhalten der Store und RevenueCat eine anonyme Kennung und den Beleg – niemals Namen oder Inhalte der Schülerliste. Ob der Einsatz an Ihrer Schule zulässig ist, entscheiden Schule und Land.
```

## pt-BR（ポルトガル語・ブラジル）

- 狙い: 2月の始業前に enturmação をする coordenação pedagógica・secretaria。heterogênea の規範、LGPD（「準拠」とは書かない）、Pix で購入可能
- 繁忙期（掲載文・スクショを差し替える時期）: 11〜2月
- 価格: **R$ 19,90**（A.8 の帯の上端。R$ 14,90 は価格テストの候補。`pricing.md`【推定】）
- ASO キーワード（本文の地の文に入れてある）: `montar turmas`, `enturmação`, `distribuição de alunos`, `sem cadastro`, `Chromebook`, `LGPD`

### アプリ名（24字 / 30）

```
FairClass: montar turmas
```

### 短い説明（76字 / 80）

```
Enturmação equilibrada em segundos. Grátis, sem cadastro e sem enviar dados.
```

### 詳しい説明（2289字 / 4000）

```
Antes do início do ano letivo, a coordenação pedagógica precisa montar as turmas: equilibrar meninas e meninos, desempenho, alunos com deficiência ou atendidos pelo AEE, comportamento e os pedidos das famílias. Fazer a enturmação à mão leva dias. O FairClass faz a distribuição de alunos nas turmas em segundos, e a decisão final é sempre sua.

COMO FUNCIONA
• Carregue a lista de alunos do Excel (.xlsx), comece com um exemplo ou digite no app.
• Adicione os critérios que quiser (gênero, desempenho, apoio à aprendizagem, comportamento) e dê um peso para cada um.
• Indique quais alunos manter juntos e quais separar.
• Escolha o número de turmas e toque em “Montar turmas”. O FairClass distribui cada critério de forma equilibrada, sempre mantém juntos os alunos que você une e separa os outros na medida do possível. Se alguma condição não puder ser atendida, você vê na hora.
• Confira as tabelas de equilíbrio e mova qualquer aluno para outra turma com um toque. Os totais são recalculados na hora.

POR QUE AS ESCOLAS USAM
• Montar turmas é grátis: sem anúncios, sem conta e sem cadastro.
• Os dados dos alunos não saem do aparelho. Nenhum dado da lista é enviado para a internet; o cálculo é feito no seu celular, tablet ou Chromebook.
• Turmas heterogêneas e parecidas entre si: você vê a faixa ideal de cada critério e se cada turma está dentro dela, fácil de mostrar no conselho de classe.
• Funciona em tablets Android e Chromebook, na vertical ou horizontal, com mouse e teclado.
• Também serve para montar grupos de trabalho em sala.

FAIRCLASS PRO (COMPRA ÚNICA)
• Exporte os resultados para Excel e Google Drive: enturmação, listas por turma, uma aba por turma, condições e resumo.
• Imprima ou salve em PDF para o conselho de classe: listas com as condições coloridas e legenda, além das tabelas de equilíbrio.
• Compre uma vez e use para sempre: não é assinatura. Dá para pagar com Pix pelo Google Play.

O FairClass usa o mesmo motor da versão web gratuita, e os arquivos do Excel funcionam nas duas.

Privacidade: as listas e os resultados ficam só no aparelho. Para verificar a compra do Pro, a loja e o RevenueCat recebem um identificador anônimo e o recibo, nunca nomes de alunos nem dados da lista. Para a LGPD, vale saber que a lista de alunos não sai do aparelho.
```

## ja-JP（日本語）

- 狙い: 3月のクラス編成（新年度）を担う学年主任・担任。国内の既存の掲載方針（無料で編成・端末内・買い切り）
- 繁忙期（掲載文・スクショを差し替える時期）: 2〜3月
- 価格: **¥980**（買い切り。`docs/research/competitors.md` 第5節の ¥610〜¥980 帯の上端。`pricing.md`）
- ASO キーワード（本文の地の文に入れてある）: `クラス編成`, `クラス分け`, `班分け`, `Chromebook`

### アプリ名（22字 / 30）

```
FairClass（フェアクラス）クラス編成
```

### 短い説明（44字 / 80）

```
男女・学力・支援の必要な子を各クラスに均等に。同じ組・別の組の指定も反映して数秒で編成。
```

### 詳しい説明（898字 / 4000）

```
新年度のクラス編成は、男女比、学力、支援の必要な子、同じクラスにしたい子・離したい子まで、いくつもの条件を同時に満たす作業です。付箋と名簿で何日もかかることも珍しくありません。FairClass はこのクラス分けを数秒で行い、最後の判断は先生に委ねます。

使い方
・Excel（.xlsx）の名簿を読み込むか、サンプル・新規作成から始めます。
・性別・学力・学習支援・体育など、均等にしたい項目と重みを決めます。
・「同じ組にする」「別の組にする」生徒を指定します。
・クラス数を決めて「クラスを編成する」を押すだけ。すべての項目が各クラスに均等に散らばります。同じ組の指定は必ず守り、別の組の指定はできる限り反映します（満たせなかった指定はすぐに表示されます）。
・バランス表を確認し、必要なら生徒をタップして別の組へ移動。集計はすぐに再計算されます。

選ばれる理由
・クラス編成は無料。広告も会員登録もありません。
・名簿はこの端末の中だけに保存し、外部へ送信しません。計算もスマホ・タブレット・Chromebook の中で行います。
・項目ごとの理想の範囲と各クラスの人数を表で確認できるので、学年会で説明しやすくなります。
・Android タブレット・Chromebook の横画面や分割画面、マウスとキーボードでも使えます。
・授業の班分け・グループ分けにも使えます。

FairClass Pro（買い切り）
・結果を Excel・Google ドライブへ書き出し（組分け・クラス別名簿・各組・ペア指定・集計）。
・会議用に印刷・PDF（ペア指定の色分けと凡例つきのクラス別名簿、集計・バランス表を A4 縦に）。
・一度の購入でずっと使えます。サブスクリプションではありません。

無料の Web 版 FairClass と同じエンジンで、Excel ファイルはどちらでもそのまま使えます。

プライバシー: 名簿と結果は端末の中だけに保存されます。Pro の購入確認のため、ストアと RevenueCat が匿名の識別子とレシートを受け取りますが、生徒の名前や名簿の内容は送りません。
```
