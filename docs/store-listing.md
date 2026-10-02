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
| en-US | 26 | 79 | 2520 |
| en-AU | 26 | 77 | 2506 |
| en-GB | 25 | 76 | 2416 |
| ko-KR | 14 | 57 | 1193 |
| es-419 | 23 | 76 | 2589 |
| es-ES | 28 | 78 | 2709 |
| de-DE | 27 | 80 | 2926 |
| pt-BR | 24 | 79 | 2611 |
| ja-JP | 22 | 52 | 1039 |

## en-US（英語・米国）

- 狙い: 学区が SaaS を契約していない学校・個人の教員。春（4〜6月）のクラス替え（spring class lists）。無料・登録不要・端末内
- 繁忙期（掲載文・スクショを差し替える時期）: 3〜5月（米）
- 価格: **US$5.99**（買い切り。`docs/play-console/pricing.md`）
- ASO キーワード（本文の地の文に入れてある）: `class placement`, `class lists`, `no sign-up`, `Chromebook`, `keep apart`
- 語彙: IEP/504・English learners (ELL)・behavior（米綴り）・principal・parent requests。技術用語（optimization・algorithm・engine・CPU）は使わない

### アプリ名（26字 / 30）

```
FairClass: Class Placement
```

### 短い説明（79字 / 80）

```
Balanced class lists in seconds. Free, no sign-up, student data stays with you.
```

### 詳しい説明（2520字 / 4000）

```
Every spring, the same puzzle: next year's class lists. You want a fair mix of boys and girls and of reading and math levels in every room, students with IEPs, 504 plans and English learners spread thoughtfully, and the parent requests and "please don't put these two together" notes all remembered. FairClass gives you a balanced first draft in seconds, so your team can spend the meeting talking about kids instead of shuffling sticky notes. Every final decision stays with you.

HOW IT WORKS
• Bring in your roster from an Excel file (.xlsx), start from a sample, or type it in.
• Add whatever matters at your school — gender, reading level, math level, behavior support, IEP or 504, English learner, leadership — and choose how much each one counts.
• Note which students should stay together (like a friend request) and which should be kept apart.
• Pick the number of classes and tap "Build classes". FairClass spreads every group of students as evenly as it can, always keeps paired students together, and keeps apart the ones you separate as far as possible. If a request can't be met, it tells you right away.
• Look over the balance tables, then move any student to another class with one tap. The totals update instantly, so you can try ideas with your grade-level team.

WHY TEACHERS LOVE IT
• Free to build class lists — no ads, no account, no sign-up.
• Your students' information stays on your device. FairClass never uploads names or notes.
• Works on phones, Android tablets and Chromebooks, in portrait or landscape, with a mouse and keyboard too.
• More than an even headcount: for each thing you balance, you can see the target range and whether every class is inside it — handy when your principal or a parent asks how the lists were made.
• Great for group work too: make balanced table groups, reading groups or teams the same way.

FAIRCLASS PRO (ONE-TIME PURCHASE)
• Export your results to Excel, then share them to Google Drive or email: class lists, one sheet per class, pairings and a summary.
• Print or save as PDF for your placement meeting: class lists with color-coded pairings and a legend, plus the balance tables.
• Buy once and it's yours — Pro is not a subscription.

Prefer a bigger screen? FairClass also comes as a free web app, and your Excel files open in both.

Privacy: rosters and results are saved only on your device. To confirm a Pro purchase, Google Play and our purchase service (RevenueCat) receive an anonymous ID and the receipt — never student names or roster data.
```

## en-AU（英語・オーストラリア）

- 狙い: Class Placement Policy に沿って Term 4 に翌年のクラスを作る小学校（新学年は1月末〜2月始業）。Class Creator 等を契約していない学校・担任
- 繁忙期（掲載文・スクショを差し替える時期）: 9〜11月（Term 3〜4）
- 価格: **A$8.99**（NZ は NZ$9.99。`pricing.csv`【推定】）
- ASO キーワード（本文の地の文に入れてある）: `class placement`, `Term 4`, `no sign-up`, `Chromebook`, `keep apart`
- 語彙: Term 4・NCCD / adjustments・EAL/D・behaviour／colour（英式綴り）・leadership team。技術用語（optimisation・algorithm・engine・CPU）は使わない

### アプリ名（26字 / 30）

```
FairClass: Class Placement
```

### 短い説明（77字 / 80）

```
Next year's classes, balanced in seconds. Free, no sign-up, data stays local.
```

### 詳しい説明（2506字 / 4000）

```
Term 4 means class placements. Before the new school year starts in late January or February, you're juggling an even spread of girls and boys, learning levels and behaviour, students on the NCCD or with adjustments in place, EAL/D learners, and the friendship requests from families. FairClass gives you a balanced first draft of next year's classes in seconds, so the planning meeting can focus on the kids, not the sticky notes. Every final decision stays with you.

HOW IT WORKS
• Bring in your class lists from Excel (.xlsx), start from a sample, or enter them in the app.
• Add the things your class placement policy looks at — for example gender, literacy and numeracy levels, behaviour, learning support, EAL/D, leadership — and choose how much each one counts.
• Note which students should stay together (for example, at least one friend) and which should be kept apart.
• Set the number of classes and tap "Build classes". FairClass spreads every group of students as evenly as it can, always keeps paired students together, and keeps apart the ones you separate as far as possible. If a request can't be met, it tells you straight away.
• Look over the balance tables and move any student to another class with a single tap. The totals update straight away, so you can try ideas with your year-level team.

WHY TEACHERS LOVE IT
• Free to build your classes — no ads, no account and no sign-up.
• Student information stays on your device. FairClass never uploads names or notes.
• Works on phones, Android tablets and Chromebooks, in portrait or landscape, with mouse and keyboard.
• For each thing you balance, see the target range and whether every class sits inside it — handy when you walk your leadership team through the new classes.
• Also great for balanced reading groups, table groups and sports teams within a class.

FAIRCLASS PRO (ONE-OFF PURCHASE)
• Export your results to Excel, then share them to Google Drive or email: class lists, a sheet per class, pairings and a summary.
• Print or save as PDF for your placement meeting: class lists with colour-coded pairings and a legend, plus the balance tables, on A4.
• Pay once and it's yours — no subscription.

Prefer a bigger screen? FairClass also comes as a free web app, and your Excel files open in both.

Privacy: class lists and results are saved only on your device. To confirm a Pro purchase, Google Play and our purchase service (RevenueCat) receive an anonymous ID and the receipt — never student names or class list data.
```

## en-GB（英語・英国）

- 狙い: 2学級以上の小学校の mixing classes（6〜7月、transition 前のクラス替え）。MIS 連携の SaaS を使っていない学校
- 繁忙期（掲載文・スクショを差し替える時期）: 5〜7月
- 価格: **£4.99**（アイルランドはユーロ圏の €5,99。`pricing.csv`【推定】）
- ASO キーワード（本文の地の文に入れてある）: `mixing classes`, `class lists`, `no sign-up`, `Chromebook`, `keep apart`
- 語彙: pupils・SEND / EHCP・EAL・attainment・transition・Year groups・SLT／governors・behaviour／colour（英式綴り）。技術用語（optimisation・algorithm・engine・CPU）は使わない

### アプリ名（25字 / 30）

```
FairClass: Mixing Classes
```

### 短い説明（76字 / 80）

```
Mix classes fairly in seconds. Free, no sign-up, pupil data stays on device.
```

### 詳しい説明（2416字 / 4000）

```
Mixing classes is one of the trickiest jobs of the summer term, especially in a two- or three-form entry school. Before transition you're weighing up girls and boys, attainment, behaviour, pupils with SEND or an EHCP, EAL learners, and the friendship groups parents ask about. FairClass gives you a balanced first draft of the new class lists in seconds, so your Year group team can spend the meeting talking about the children, not shuffling sticky notes. Every final decision stays with you.

HOW IT WORKS
• Bring in your class list from Excel (.xlsx), start from a sample, or type it in.
• Add whatever matters to your school — gender, reading and maths attainment, behaviour, SEND, EAL, previous class — and choose how much each one counts.
• Note which pupils should stay together (such as a friend) and which should be kept apart.
• Choose the number of classes and tap "Build classes". FairClass spreads every group of pupils as evenly as it can, always keeps paired pupils together, and keeps apart the ones you separate as far as possible. If a request can't be met, it tells you straight away.
• Look over the balance tables and move any pupil to another class with one tap. The totals update instantly, so you can try ideas together.

WHY TEACHERS LOVE IT
• Free to mix your classes — no ads, no account, no sign-up.
• Pupil information stays on your device. FairClass never uploads names or notes.
• Works on phones, Android tablets and Chromebooks, in portrait or landscape, with a mouse and keyboard.
• For each thing you balance, see the target range and whether every class sits inside it — useful when you explain the new classes to SLT, governors or parents.
• Also handy for balanced table groups, reading groups and teams within a class.

FAIRCLASS PRO (ONE-OFF PURCHASE)
• Export your results to Excel, then share them to Google Drive or email: class lists, a sheet per class, pairings and a summary.
• Print or save as PDF for staff meetings: class lists with colour-coded pairings and a legend, plus the balance tables, on A4.
• Pay once and it's yours — no subscription.

Prefer a bigger screen? FairClass also comes as a free web app, and your Excel files open in both.

Privacy: class lists and results are saved only on your device. To confirm a Pro purchase, Google Play and our purchase service (RevenueCat) receive an anonymous ID and the receipt — never pupil names or class list data.
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

### 短い説明（57字 / 80）

```
2월 반 편성, 남녀·학업·지원 필요 학생을 고르게 나누고 분리 배정도 최대한 반영합니다. 가입 없음.
```

### 詳しい説明（1193字 / 4000）

```
매년 2월이면 담임 선생님들이 모여 다음 학년 반 편성을 준비합니다. 남녀 비율, 학업 성취도, 도움이 필요한 학생, 출신 학교, 같은 반이 되면 안 되는 학생까지 하나하나 맞추다 보면 며칠이 금방 지나갑니다. FairClass는 고르게 나눈 반 편성안을 몇 초 만에 만들어 드립니다. 최종 결정은 언제나 선생님께서 내리십니다.

이렇게 사용합니다
• 엑셀 파일(.xlsx)로 학생 명단을 불러오거나, 예시로 시작하거나, 앱에서 직접 입력합니다.
• 성별, 학업 성취도, 특수교육 대상, 출신 학교처럼 반마다 고르게 나누고 싶은 항목을 넣고, 항목마다 얼마나 중요한지(가중치) 정합니다.
• 같은 반에 둘 학생(같은 반 배정)과 서로 다른 반으로 나눌 학생(분리 배정)을 지정합니다.
• 반 수를 정하고 "반 편성하기"를 누르면 모든 항목이 각 반에 고르게 나뉩니다. 같은 반 배정은 항상 지키고, 분리 배정은 최대한 반영합니다. 지키지 못한 조건이 있으면 바로 알려 드립니다.
• 균형 표를 살펴보고, 필요하면 학생을 눌러 다른 반으로 옮길 수 있습니다. 집계도 바로 바뀝니다.

선생님들이 FairClass를 쓰는 이유
• 반 편성은 무료입니다. 광고도, 회원 가입도 없습니다.
• 학생 명단은 이 기기에만 저장되고 어디에도 전송되지 않습니다. 반 편성도 휴대폰·태블릿·크롬북 안에서 이루어집니다.
• 분리가 필요한 학생은 최대한 다른 반으로 나누고, 충족하지 못한 조건이 있으면 바로 표시합니다.
• 항목마다 목표 범위와 반별 인원이 표로 보여서, 학년 협의회에서 편성 근거를 설명하기 쉽습니다.
• 안드로이드 태블릿과 크롬북에서 가로·세로 화면 모두, 마우스와 키보드로도 쓸 수 있습니다.
• 수업 시간의 모둠 편성에도 그대로 쓸 수 있습니다.

FairClass Pro(1회 구매)
• 결과를 엑셀·Google 드라이브로 내보내기: 반 편성, 반별 명단, 반마다의 시트, 배정 조건, 집계.
• 협의회용 인쇄·PDF: 배정 조건을 색으로 구분한 반별 명단과 범례, 균형 표를 A4로.
• 한 번 구매로 계속 사용합니다. 구독이 아닙니다.

무료 웹 버전 FairClass와 똑같은 방식으로 반을 나누며, 엑셀 파일은 양쪽에서 그대로 열립니다.

개인정보: 명단과 결과는 기기에만 저장됩니다. Pro 구매 확인을 위해 스토어와 RevenueCat이 익명 식별자와 영수증만 받으며, 학생 이름이나 명단 내용은 보내지 않습니다.
```

## es-419（スペイン語・中南米）

- 狙い: 南半球の学年始まり（2〜3月）・メキシコの ciclo escolar（8〜9月）の前に grupos を作る学校の先生。tú で語りかけ、技術用語（optimización・algoritmo・motor）は使わない。データは端末内（チリの新個人情報法・各国の個人情報保護。「準拠」とは書かない）
- 繁忙期（掲載文・スクショを差し替える時期）: 10〜2月（南半球・コロンビア A）、メキシコは 6〜8月
- 価格: **MX$79・CLP 3.990・COP 12.900・PEN 10,90・US$3.99（EC）**、ARS は Play 側の通貨を確認してから（`pricing.csv` の status=confirm）。その他の中南米は自動換算（`pricing.md`【推定】）
- ASO キーワード（本文の地の文に入れてある）: `armar grupos`, `distribución de estudiantes`, `ciclo escolar`, `sin registro`, `Chromebook`, `equipos de trabajo`

### アプリ名（23字 / 30）

```
FairClass: armar grupos
```

### 短い説明（76字 / 80）

```
Arma grupos equilibrados para el nuevo ciclo escolar. Gratis y sin registro.
```

### 詳しい説明（2589字 / 4000）

```
Sea en febrero, en marzo o en agosto, antes del inicio del ciclo escolar hay que armar los grupos: que queden parejos en niñas y niños y en desempeño, que los estudiantes con NEE estén bien repartidos, que los amigos que se apoyan sigan juntos y que algunos compañeros no coincidan. Hacerlo a mano, con listas y papelitos, puede llevar días. FairClass te propone una distribución de estudiantes en grupos en segundos, y la última palabra siempre es tuya.

ASÍ DE FÁCIL
• Carga tu lista de estudiantes desde Excel (.xlsx), empieza con un ejemplo o escríbela directamente en la app.
• Agrega lo que tu escuela toma en cuenta (género, desempeño, NEE, convivencia, liderazgo…) y decide cuánta importancia le das a cada criterio.
• Marca qué estudiantes deben ir juntos y a quiénes conviene separar.
• Elige cuántos grupos necesitas y toca “Armar grupos”. FairClass reparte cada criterio de forma pareja. Los estudiantes que marcas como juntos quedan siempre en el mismo grupo, y a los que quieres separar los separa en la medida de lo posible. Si algo no se pudo cumplir, te lo muestra de inmediato.
• Revisa cómo quedó cada grupo y, si quieres, mueve a cualquier estudiante a otro grupo con un toque. Los totales se actualizan al instante.

POR QUÉ LES GUSTA A LOS DOCENTES
• Armar los grupos es gratis: sin anuncios, sin cuenta y sin registro.
• La lista de tus estudiantes se queda en tu dispositivo. No se sube nada a internet: todo sucede en tu celular, tablet o Chromebook.
• Grupos heterogéneos y parejos entre sí: para cada criterio ves el rango ideal y si cada grupo está dentro. Así es fácil explicar en el consejo de profesores, o a las familias, por qué los grupos quedaron así.
• Funciona en celulares, tablets Android y Chromebook, en vertical u horizontal, y también con mouse y teclado.
• También te sirve para armar equipos de trabajo dentro del aula.

FAIRCLASS PRO (PAGO ÚNICO)
• Exporta a Excel y Google Drive: la distribución, las listas por grupo, una hoja por grupo, las condiciones y un resumen.
• Imprime o guarda en PDF para el consejo de profesores: listas por grupo con las condiciones marcadas en colores y una leyenda, más las tablas de equilibrio.
• Pagas una sola vez y es tuyo: no es una suscripción.

¿Ya usas la versión web gratuita de FairClass? Arma los grupos de la misma manera, y tus archivos de Excel funcionan en las dos.

Privacidad: tus listas y resultados se guardan solo en tu dispositivo. Para confirmar la compra de Pro, la tienda y RevenueCat reciben solo un identificador anónimo y el recibo, nunca nombres de estudiantes ni datos de la lista.
```

## es-ES（スペイン語・スペイン）

- 狙い: 学校の計画書で編成基準（agrupamiento del alumnado）を公開する義務がある公立校。9月の新学年の前（6〜7月・9月初め）に組む。成績による同質な組分けは禁止なので「異質性を保つ・desempeño を均等に」と書く。NEAE。技術用語は使わない
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

### 詳しい説明（2709字 / 4000）

```
Al terminar el curso, o justo antes de que empiece el siguiente en septiembre, toca hacer los grupos siguiendo los criterios de agrupamiento del alumnado del centro: equilibrio entre niñas y niños, un desempeño equilibrado entre los grupos, el alumnado con NEAE repartido de forma justa y las incompatibilidades que conoce el equipo docente. Con papel, pósits y hojas de cálculo se van tardes enteras. FairClass hace ese reparto en grupos en segundos, y la última palabra la tiene siempre el profesorado.

CÓMO FUNCIONA
• Carga la lista desde Excel (.xlsx), empieza con un ejemplo o escríbela en la app. Un apunte: la app usa vocabulario latinoamericano (“Armar grupos”, “estudiantes”) y los ejemplos usan notas de 1 a 7; con tu propio Excel puedes usar la escala de 0 a 10.
• Añade los criterios de tu centro (niñas y niños, notas, NEAE, convivencia…) y decide cuánto pesa cada uno.
• Marca qué alumnos y alumnas deben ir juntos y a quiénes conviene separar.
• Elige cuántos grupos necesitas y pulsa “Armar grupos”. FairClass reparte cada criterio de forma equilibrada: los grupos salen heterogéneos y parecidos entre sí, sin agrupar por rendimiento. Quienes marcas como juntos van siempre al mismo grupo, y quienes quieres separar quedan separados en la medida de lo posible; si algo no se puede cumplir, lo ves al momento.
• Revisa cómo ha quedado cada grupo y pasa a quien quieras a otro grupo con un toque. Los totales se actualizan al instante.

POR QUÉ LO USA EL PROFESORADO
• Hacer los grupos es gratis: sin anuncios, sin cuenta y sin registro.
• Los datos del alumnado no salen del dispositivo: no se sube nada a internet, todo se hace en tu móvil, tableta o Chromebook.
• Para cada criterio ves el rango ideal y si cada grupo está dentro: te ayuda a justificar el agrupamiento del alumnado ante el claustro, el equipo de ciclo o las familias.
• Funciona en móviles, tabletas Android y Chromebook, en vertical u horizontal, con ratón y teclado.
• También sirve para hacer equipos de trabajo cooperativo en el aula.

FAIRCLASS PRO (PAGO ÚNICO)
• Exporta a Excel y Google Drive: el reparto, las listas por grupo, una hoja por grupo, las condiciones y un resumen.
• Imprime o guarda en PDF para el claustro o la reunión de equipo: listas por grupo con colores y leyenda, más las tablas de equilibrio.
• Pagas una sola vez: no es una suscripción.

¿Ya usas la versión web gratuita de FairClass? Hace los grupos de la misma forma, y los archivos de Excel sirven en las dos.

Privacidad: las listas y los resultados se guardan solo en el dispositivo. Para confirmar la compra de Pro, la tienda y RevenueCat reciben solo un identificador anónimo y el recibo, nunca nombres del alumnado ni datos de la lista.
```

## de-DE（ドイツ語）

- 狙い: Einschulung と、Grundschule から weiterführende Schule（Klasse 5）への移行時のクラス編成（5〜7月）。DSGVO と学校データ規則で「送信しない」が最も効くので Datenschutz を冒頭に置く（「DSGVO-konform」とは書かない）。Inklusion・Förderbedarf への配慮を明記
- 繁忙期（掲載文・スクショを差し替える時期）: 4〜7月（Einschulung・Klasse 5）
- 価格: **5,99 €**（スイスは CHF 5.00。`pricing.csv`【推定】）
- ASO キーワード（本文の地の文に入れてある）: `Klasseneinteilung`, `Klassenbildung`, `Alle Daten bleiben auf dem Gerät`, `Chromebook`, `Gruppeneinteilung`

### アプリ名（27字 / 30）

```
FairClass Klasseneinteilung
```

### 短い説明（80字 / 80）

```
Klassen fair einteilen. Alle Daten bleiben auf dem Gerät – ohne Konto und Cloud.
```

### 詳しい説明（2926字 / 4000）

```
Alle Daten bleiben auf dem Gerät. Keine Cloud, kein Konto, keine Registrierung: Die Schülerliste verlässt Ihr Smartphone, Tablet oder Chromebook nicht – auch die Klasseneinteilung selbst entsteht direkt dort.

Ob Einschulung oder Übergang von der Grundschule an die weiterführende Schule: Die Klassenbildung kostet jedes Jahr viel Zeit und Fingerspitzengefühl. Mädchen und Jungen sollen ausgewogen verteilt sein, die Leistungen gemischt, Freundschaftswünsche berücksichtigt – und manche Kinder kommen besser nicht in dieselbe Klasse. Damit Inklusion gelingt, sollen auch Schülerinnen und Schüler mit Förderbedarf gut auf alle Klassen verteilt sein. FairClass macht Ihnen in Sekunden einen ausgewogenen Vorschlag. Die Entscheidung treffen immer Sie und Ihr Kollegium.

SO GEHT'S
• Schülerliste aus Excel (.xlsx) laden, mit einem Beispiel beginnen oder die Namen direkt in der App eingeben.
• Festlegen, worauf Sie achten möchten – z. B. Geschlecht, Notenschnitt, Förderbedarf, DaZ oder die Herkunftsgrundschule – und wie wichtig Ihnen jedes Merkmal ist.
• Freundschaftswünsche als „zusammen“ eintragen, Kinder, die getrennt werden sollen, als „trennen“.
• Anzahl der Klassen wählen und auf „Klassen einteilen“ tippen. Jedes Merkmal wird möglichst gleichmäßig auf die Klassen verteilt. „Zusammen“ wird immer eingehalten, „trennen“ so weit wie möglich. Was sich nicht erfüllen lässt, sehen Sie sofort.
• Den Vorschlag in Ruhe prüfen und einzelne Kinder mit einem Tipp in eine andere Klasse verschieben. Die Übersicht passt sich sofort an.

WARUM LEHRKRÄFTE FAIRCLASS NUTZEN
• Die Klasseneinteilung ist kostenlos – ohne Werbung, ohne Konto, ohne Registrierung.
• Keine Übertragung von Schülerdaten: Nichts aus der Schülerliste wird hochgeladen.
• Für jedes Merkmal sehen Sie den Sollbereich und ob jede Klasse darin liegt. So können Sie die Einteilung im Kollegium und gegenüber der Schulleitung gut begründen.
• Lange Ketten von Freundschaftswünschen (A mit B, B mit C …) werden erkannt und gemeinsam eingeteilt.
• Läuft auf Android-Tablets und Chromebooks, im Hoch- und Querformat, auch mit Maus und Tastatur.
• Auch für die Gruppeneinteilung im Unterricht geeignet.

FAIRCLASS PRO (EINMALKAUF)
• Ergebnis nach Excel und Google Drive exportieren: Einteilung, Klassenlisten, ein Blatt pro Klasse, Wünsche und Auswertung.
• Drucken oder als PDF für die Konferenz: Klassenlisten mit farbig markierten Wünschen und Legende sowie Tabellen zur Ausgewogenheit auf A4.
• Einmal kaufen, dauerhaft nutzen – kein Abo.

FairClass teilt die Klassen genauso ein wie die kostenlose Webversion; Ihre Excel-Dateien können Sie in beiden verwenden.

Datenschutz: Schülerlisten und Ergebnisse werden nur auf dem Gerät gespeichert. Zur Prüfung eines Pro-Kaufs erhalten der Store und RevenueCat eine anonyme Kennung und den Beleg – niemals Namen oder Inhalte der Schülerliste. Ob der Einsatz an Ihrer Schule zulässig ist, entscheiden Schule und Land.
```

## pt-BR（ポルトガル語・ブラジル）

- 狙い: 2月の始業前に enturmação をする professores・coordenação pedagógica・secretaria。você で語りかけ、技術用語（otimização・algoritmo・motor）は使わない。heterogênea の規範、AEE・inclusão、LGPD は事実だけ（「準拠」「em conformidade」とは書かない）、Pix で購入可能
- 繁忙期（掲載文・スクショを差し替える時期）: 11〜2月
- 価格: **R$ 19,90**（A.8 の帯の上端。R$ 14,90 は価格テストの候補。`pricing.md`【推定】）
- ASO キーワード（本文の地の文に入れてある）: `montar turmas`, `enturmação`, `distribuição de alunos`, `ano letivo`, `sem cadastro`, `Chromebook`, `LGPD`

### アプリ名（24字 / 30）

```
FairClass: montar turmas
```

### 短い説明（79字 / 80）

```
Monte turmas equilibradas para o ano letivo em segundos. Grátis e sem cadastro.
```

### 詳しい説明（2611字 / 4000）

```
Todo ano é a mesma coisa: antes do início do ano letivo, em fevereiro, é preciso montar as turmas. Equilibrar meninas e meninos, desempenho e comportamento, distribuir com cuidado os alunos com deficiência ou atendidos pelo AEE, manter juntos os amigos que se apoiam e separar quem não deve ficar junto. Fazer a enturmação à mão, com listas e papeizinhos, leva dias. O FairClass faz a distribuição de alunos nas turmas em segundos, e a decisão final é sempre sua.

COMO FUNCIONA
• Carregue a lista de alunos do Excel (.xlsx), comece com um exemplo ou digite direto no app.
• Inclua o que a sua escola leva em conta (gênero, desempenho, AEE, comportamento, liderança…) e diga quanto pesa cada critério.
• Marque quais alunos devem ficar juntos e quais é melhor separar.
• Escolha o número de turmas e toque em “Montar turmas”. O FairClass distribui cada critério de forma equilibrada. Quem você marca para ficar junto fica sempre na mesma turma, e quem você quer separar fica separado na medida do possível. Se alguma condição não puder ser atendida, você vê na hora.
• Confira como ficou cada turma e, se quiser, mova qualquer aluno para outra turma com um toque. Os totais se atualizam na hora.

POR QUE PROFESSORES E COORDENADORES GOSTAM
• Montar turmas é grátis: sem anúncios, sem conta e sem cadastro.
• Os dados dos alunos não saem do aparelho. Nada é enviado para a internet: tudo acontece no seu celular, tablet ou Chromebook.
• Turmas heterogêneas e parecidas entre si: você vê a faixa ideal de cada critério e se cada turma está dentro dela. Fica fácil explicar a enturmação no conselho de classe ou para as famílias.
• Funciona em celulares, tablets Android e Chromebook, na vertical ou horizontal, com mouse e teclado.
• Também serve para montar grupos de trabalho em sala.

FAIRCLASS PRO (COMPRA ÚNICA)
• Exporte para Excel e Google Drive: enturmação, listas por turma, uma aba por turma, condições e resumo.
• Imprima ou salve em PDF para o conselho de classe ou a secretaria: listas por turma com as condições em cores e legenda, além das tabelas de equilíbrio.
• Pague uma vez e pronto: não é assinatura. Dá para pagar com Pix pelo Google Play.

Já usa a versão web gratuita do FairClass? Ela monta as turmas do mesmo jeito, e os arquivos do Excel funcionam nas duas.

Privacidade: as listas e os resultados ficam só no aparelho. Para confirmar a compra do Pro, a loja e o RevenueCat recebem apenas um identificador anônimo e o recibo, nunca nomes de alunos nem dados da lista. Sobre a LGPD: a lista de alunos não sai do aparelho; se o uso é adequado à sua escola, quem decide é a própria escola.
```

## ja-JP（日本語）

- 狙い: 3月のクラス編成（新年度）を担う学年主任・担任。国内の既存の掲載方針（無料で編成・端末内・買い切り）。学年会のたたき台・引き継ぎに使える、と先生に寄り添う言葉で書く
- 繁忙期（掲載文・スクショを差し替える時期）: 2〜3月
- 価格: **¥980**（買い切り。`docs/research/competitors.md` 第5節の ¥610〜¥980 帯の上端。`pricing.md`）
- ASO キーワード（本文の地の文に入れてある）: `クラス編成`, `クラス分け`, `班分け`, `Chromebook`

### アプリ名（22字 / 30）

```
FairClass（フェアクラス）クラス編成
```

### 短い説明（52字 / 80）

```
3月のクラス編成に。男女・学力・支援の必要な子を各クラスに均等に。名簿は端末の中だけで、登録も不要です。
```

### 詳しい説明（1039字 / 4000）

```
3月、新年度に向けたクラス編成、本当におつかれさまです。男女比、学力、支援の必要な子、同じクラスにしたい子・離したい子……いくつもの条件を見比べながら、名前カードや付箋を何度も並べ替える。学年会で何日もかかることも珍しくありません。
FairClass は、条件をそろえたクラス分けの案を数秒でつくります。学年会のたたき台にして、最後は先生方の目で決めてください。

使い方
・Excel（.xlsx）の名簿を読み込むか、サンプルや新しい名簿から始めます。
・性別・学力・学習支援・登校支援・体育・ピアノなど、各クラスにそろえたい項目と、それぞれをどのくらい大事にするか（重み）を決めます。
・「同じ組にする」「別の組にする」子を指定します。
・クラス数を決めて「クラスを編成する」を押すだけ。すべての項目が各クラスに均等に散らばります。同じ組の指定は必ず守り、別の組の指定はできる限り反映します（守れなかった指定はすぐに表示されます）。
・バランス表を見ながら、必要なら生徒をタップして別の組へ移動できます。集計もすぐに更新されます。

先生方に選ばれる理由
・クラス編成は無料。広告も会員登録もありません。
・名簿はこの端末の中だけに保存し、外部へ送信しません。クラス分けもスマホ・タブレット・Chromebook の中で行います。
・項目ごとの理想の範囲と各クラスの人数が表でひと目でわかるので、学年会や管理職に「偏りのない編成です」と説明しやすくなります。
・Android タブレット・Chromebook の横画面や分割画面、マウスとキーボードでも使えます。
・授業の班分け・グループ分けにも使えます。

FairClass Pro（買い切り）
・結果を Excel・Google ドライブへ書き出し（組分け・クラス別名簿・各組・ペア指定・集計）。
・学年会や新しい担任への引き継ぎ用に印刷・PDF（ペア指定の色分けと凡例つきのクラス別名簿、集計・バランス表を A4 縦に）。
・一度の購入でずっと使えます。サブスクリプションではありません。

無料の Web 版 FairClass と同じしくみでクラスを分けるので、Excel ファイルはどちらでもそのまま使えます。

プライバシー: 名簿と結果は端末の中だけに保存されます。Pro の購入確認のため、ストアと RevenueCat が匿名の識別子とレシートを受け取りますが、生徒の名前や名簿の内容は送りません。
```
