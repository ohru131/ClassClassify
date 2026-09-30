# 海外需要調査（どの言語版を作るか）

- 調査日: 2026-09-30
- 対象: Mosaic — クラス編成オプティマイザー（属性の均等化＋同じ組／別の組の制約、端末内処理。モバイル版は画面上で全機能無料・Pro 買い切り約¥980で xlsx 書き出しと印刷/PDF、広告なし。Web版は無料）
- 前提資料: `docs/research/pain-points-and-target.md`、`docs/research/competitors.md`

> **調査方法と限界**: 一次情報は WebSearch の検索結果（タイトルと抜粋）。WebFetch は多くのサイトで egress ブロックされる前提で、抜粋で確認できなかった事項は【要確認】、筆者の推測は【推定】と書く。
> 学校数・教員数は各国統計の概数（検索抜粋より）で、年度や定義（公立のみ／私立含む）が揃っていない。桁の比較にだけ使うこと。


---

## A. 南米（中南米）— 独立節

> オーナーの追加関心により、スペイン語・ポルトガル語を一括りにせず国ごとに見る。**南半球の国（ブラジル・アルゼンチン・チリ・ペルー）とコロンビア（カレンダーA）は2〜3月始業で日本（4月）とほぼ同じ時期に編成作業が来る**ので、日本向けと同じ季節に告知・ASO強化ができる、というのがこの地域の最大の運用上の利点。

### A.1 ブラジル（pt-BR）

- **慣行**: 1学年を複数の turma（例: 5º ano A/B/C）に分ける作業は **enturmação** と呼ばれ、研究対象になるほど一般的。基準は学校が決め、ベロオリゾンテ市立校の校長調査では「年齢の均質性」が最多、次いで「学力の異質性（混ぜる）」、学校によっては「行動の異質性」（規律の良い子と悪い子を混ぜ、問題の多い子が1組に集中しないようにする）。一方で**学力や年齢（留年による遅れ）で固める enturmação はクラス間格差を広げる**という批判的研究も多い。出典: https://www.redalyc.org/pdf/551/55150202.pdf , https://repositorio.ufmg.br/bitstream/1843/46295/1/DISSERTAC%cc%a7A%cc%83O%20TATIANA.pdf , https://observatoriodeeducacao.institutounibanco.org.br/api/assets/observatorio/6c56f0fd-555e-4cd9-abb5-b4ecb594c037
  - → **「混ぜる（heterogênea）＝公平」という規範が研究・行政側にあり、Mosaic の「均等化」はその規範に合う。** 逆に「学力別に固める」運用の学校には刺さらない。
- **担い手と時期**: 学校の direção（校長）と **coordenador(a) pedagógico(a)**、実務は secretaria escolar。matrícula（1〜2月）の確定後、**2月の始業直前**に組む【推定：学年暦からの推定。公立の州・市で差あり】。ブラジルの学年は2月始業・12月終業。
- **規模**: 2024年の Censo Escolar で **学校17.93万校・生徒4,710万人・教員236.8万人**、私立は生徒の **20.2%**。出典: https://www.correiobraziliense.com.br/euestudante/educacao-basica/2025/04/7107754-eja-em-queda-especialistas-alertam-para-evasao-e-fechamento-de-escolas-no-df.html , https://pt.org.br/efeito-lula-cresce-numero-de-matriculas-em-tempo-integral-e-na-educacao-profissional/
  - 1学年1クラスの小規模校（農村部）も多く【推定】、対象校はその一部。ただし都市部の公立・私立は1学年複数 turma が普通【推定】。
- **端末**: スマホの Android 比率 **約89%**（StatCounter の抜粋）。出典: https://gs.statcounter.com/android-version-market-share/mobile-os/brazil 。学校の Chromebook 普及は州ごとの施策で差がある【要確認】。
- **支払い能力**: 公立教員の法定最低給（piso）は **2025年に R$4,867.77/月（40時間）**、ただし1/3の自治体は未払いとの報道。出典: https://www.opovo.com.br/noticias/politica/2025/01/31/piso-dos-professores-mec-oficializa-reajuste-de-627-e-valor-passa-a-rs-4-86777.html , https://portaldeprefeitura.com.br/brasil/um-terco-dos-munipios-do-brasil-nao-pagam-piso-professores/604340/ 。教員の自腹購入率の統計は見つからなかった【要確認】。
- **Play の支払い**: **Pix が全 Android ユーザーで使える**（ほかにクレジットカード・ギフトカード・PayPal・PicPay・Mercado Pago）。**カードを持たない教員でも買える**のは大きい。出典: https://blog.google/intl/pt-br/produtos/android-chrome-play/google-play-pix-chega-a-loja-de-apps-e-amplia-opcoes-de-pagamento/ , https://tecnoblog.net/noticias/picpay-agora-e-aceito-na-google-play-para-pagamentos-sem-cartao-de-credito/
- **個人情報**: **LGPD 第14条**で子どものデータには保護者の特定の同意が要る。**ANPD は連邦直轄区の教育局が Google Forms で約3,000人の子どもの健康情報を露出させた件で制裁**しており、「クラウドに生徒の配慮情報を置かない」ことは現実の懸念。出典: https://confidata.com.br/blog/anpd-sancao-seedf-educacao-licoes , https://confidata.com.br/blog/consentimento-pais-lgpd-escolas-guia
  - → **端末内処理は訴求になる**（「dados dos alunos não saem do aparelho」）。
- **競合**: ポルトガル語の専用 enturmação ツールは見つからなかった。校務システム（i-Educar、SIGE 等）は「enturmar＝在籍登録」の意味で、**属性を均等化する自動編成機能は確認できなかった**【要確認】。実務は Excel / Google Sheets の手作業【推定】。
- **評価**: 需要 中〜強（慣行はあるが公立は自治体の基準で固める所もある）、市場規模 最大級、WTP 低い、競合 ほぼ無し、ローカライズ 中（pt-BR）。

### A.2 アルゼンチン（es-419 / es-AR）

- **慣行**: 小学校の同学年の組は **división** または **sección**（「3.º grado, división A」「sección B」）【要確認：州ごとに呼称が揺れる】。進級時に divisiones を **mezclar**（混ぜる）かどうかは学校裁量で、混ぜると保護者から反発が出る、という話は検索では具体例を拾えなかった【要確認】。
- **時期**: 2026年は多くの州で**2月最終週〜3月2日に始業**（ブエノスアイレス州は3月2日）。編成は12月〜2月【推定】。出典: https://www.cronista.com/informacion-gral/confirmado-adelantan-el-comienzo-de-clases-y-los-chicos-deberan-regresar-antes-a-las-aulas-2/
- **規模**: 私立（gestión privada）が生徒の **27.8%**（2022年 Relevamiento Anual、323万人）。ブエノスアイレス市は51%が私立。出典: https://www.lacapitalmdp.com/en-argentina-casi-un-28-por-ciento-de-estudiantes-cursan-en-escuelas-de-educacion-privada/
- **端末**: Android 約86%（抜粋）。出典: https://gs.statcounter.com/android-version-market-share/mobile-os/brazil （同検索の抜粋。国別ページは未確認）【要確認】
- **WTP**: インフレと為替規制のため個人のアプリ購入は難しい【推定】。
- **個人情報**: Ley 25.326（EU から十分性認定あり）【要確認：本調査で出典を取得していない】。
- **評価**: 需要 中、規模 中、WTP 低、競合 無し、ローカライズ 小（es-419 で共通化可）。

### A.3 チリ（es-419 / es-CL）

- **慣行**: 同学年の並行クラスは **curso**（「4.º básico A」）。私立・特許補助校では規程として **「Reglamento de fusión, fisión y mezcla de cursos」** を持つ学校があり、ある学校の規程は「mezcla（組の混合）は学年全体への帰属感・社会性・適応力を育て、学校の共生（convivencia）を促す目的で行う」、1組は18〜28人、目標22〜25人と定める。**「混ぜる」ことが制度化されている**のは Mosaic に好材料。出典: https://mackay.cl/wp-content/uploads/2023/09/Reglamento-de-Fusion-Fision-y-Mezcla-y-Cambio-de-Cursos-2024.pdf
  - 担い手は校長・**jefe(a) de UTP**（技術教育部門長）・orientador/convivencia escolar の担当【推定】。
- **時期**: 2026年度は**3月4日始業**（教員は3月2日から）。編成は12月〜2月【推定】。出典: https://www.elmostrador.cl/datos-utiles/2026/02/26/calendario-escolar-2026-revisa-las-fechas-clave-de-inicio-de-clases-y-vacaciones-de-invierno/
- **規模**: 2025年の生徒 **354万人**、**particular subvencionado（私立の補助校）が53.9%**、自治体立22.3%、SLEP 12.7%。私立補助校は6,000校超（学校全体の約半分）。出典: https://cooperativa.cl/noticias/pais/educacion/colegios/caida-de-la-natalidad-en-chile-se-percibio-en-cantidad-de-escolares/2025-11-04/162732.html
- **個人情報**: **新しい個人情報保護法（Ley 21.719）が2026年12月1日施行予定**で、子ども（NNA）のデータを最高度の保護対象にする（延期の議論あり）。学校は保護者・生徒データの扱いを見直す時期に入っている。出典: https://www.ecija.com/actualidad-insights/colegios-y-proteccion-de-datos-los-desafios-frente-a-la-nueva-ley-de-datos-personales/ , https://www.bcn.cl/leychile/navegar?i=1209272
  - → **施行直後の2027年度編成（2026年12月〜2027年2月）は「端末内処理」を打ち出す好機。**
- **WTP**: 南米では相対的に所得が高く【推定】、私立校の比率も高い。
- **評価**: 需要 中〜強、規模 小、WTP 中、競合 無し、ローカライズ 小（es-419）。**南米でいちばん当たりやすい国。**

### A.4 コロンビア（es-419 / es-CO）

- **慣行**: 同学年の組は **grupo** または **curso**（中等は「601, 602」と番号で呼ぶ学校が多い）【要確認】。生徒の79%が公立（sector oficial）。出典: https://www.javeriana.edu.co/recursosdb/5581483/7093554/INFORME-14-LEE-PUJ-RADIOGRAFIA-EDUCACIO%CC%81N-ESCOLAR.pdf
- **時期**: **カレンダーA（公立・大半の私立）は1月19〜26日始業**（ボゴタは1月26日）。**カレンダーB（主に高所得の私立・国際校）は8〜9月始業**。出典: https://www.eltiempo.com/vida/educacion/calendario-escolar-2026-estas-son-las-fechas-de-regreso-a-clases-y-vacaciones-en-las-diferentes-regiones-del-pais-3509782
  - → 告知時期が2回（11〜1月と6〜8月）ある。カレンダーBの私立は WTP が高い層【推定】。
- **編成基準**: 公開規程を検索では拾えなかった【要確認】。
- **評価**: 需要 中【推定】、規模 中、WTP 低〜中、競合 無し、ローカライズ 小。

### A.5 ペルー（es-419 / es-PE）

- **慣行**: 同学年の組は **sección**（「2.º grado, sección A」）。**全国学力調査（ECE 2016）の校長アンケートで、中学2年に複数セクションを持つ校長の40%は生徒を無作為（くじ・五十音順・入学順）に割り振り、残りは性別・行動・年齢・学力などで振り分ける。** 全国規範は無く、学校ごとにばらばら。出典: https://cies.org.pe/wp-content/uploads/2019/01/3if-cies-nd_jk_gm_0.pdf
  - → **「無作為で割っている40%」は、手間をかけていない層＝ツールがあれば基準を持ち込める余地**とも、**そもそも困っていない層**とも読める。後者の可能性が高い【推定】。
- **時期**: 2026年は公立 **3月16日始業**、私立は3月2〜9日。出典: https://www.infobae.com/peru/2026/02/27/inicio-de-clases-2026-esta-es-la-fecha-oficial-que-fijo-el-minedu-para-alumnos-de-colegios-estatales/
- **規模**: 基礎教育の学校の **23.4%が私立**、私立の生徒は約350万人（34%）、**私立の教員22.8万人（全体の36%）**。公立教員43.1万人（2025年）。出典: https://escale.minedu.gob.pe/documents/inicio/Publicaciones/Boletin%20regional/Magnitudes-PERU.pdf , https://gestion.pe/economia/empresas/peru-atrae-a-universidades-extranjeras-mientras-crece-la-demanda-de-educacion-flexible-noticia/ 【数字の出典年が混在。要確認】
- **評価**: 需要 弱〜中、規模 中、WTP 低、競合 無し、ローカライズ 小。

### A.6 メキシコ（参考、es-419 / es-MX）

- 同学年の組は **grupo**（「3.º A」）。学年暦は **8月下旬〜9月始業**（北半球型）【要確認：本調査で2026年の出典未取得】。Android 比率 約84%（抜粋）。
- 公立初等教育は1学年の組替えを毎年する学校としない学校がある【要確認】。検索では編成基準の公開例を拾えなかった（引っかかったのはスペイン本国の規程ばかり）。
- 規模は中南米最大級（スペイン語圏で最大の人口）だが、北半球型の暦なので**告知時期がスペイン・米国と同じ（5〜8月）**。

### A.7 用語の差（es-ES と es-419、pt-BR と pt-PT）

| 概念 | es-ES（スペイン） | es-419 共通案 | 国別の揺れ | pt-BR | pt-PT |
|---|---|---|---|---|---|
| 学年 | curso（「3.º de Primaria」） | grado（初等）／año | CL は「curso」を組の意味でも使う（4.º básico **A** の全体が curso） | ano / série | ano |
| 同学年の組 | **grupo**／clase／aula | **grupo** または **sección** | AR: división・sección／CL: curso／PE: sección／CO: grupo・curso／MX: grupo／EC: **paralelo** | **turma** | **turma** |
| 組分け作業 | **agrupamiento del alumnado**／reparto／distribución del alumnado（Andalucía の規程名） | distribución de estudiantes | CL: mezcla de cursos／AR: armado de divisiones【要確認】 | **enturmação**／distribuição de alunos nas turmas | **constituição de turmas**（規程名） |
| 生徒 | alumnado／alumnos | estudiantes（包括的表現として好まれる）【推定】 | — | alunos／estudantes | alunos |
| 特別支援 | **NEAE／ACNEAE**（alumnado con necesidad específica de apoyo educativo） | **NEE**（necesidades educativas especiales） | CL: NEE（PIE＝学校統合プログラム）【要確認】 | **alunos com deficiência / público-alvo da educação especial**、AEE | NEE → 現行は「medidas de suporte à aprendizagem」（DL 54/2018）【要確認】 |

- **結論: スペイン語は es-ES と es-419 の2本に分ける価値がある**（「組」の語が grupo/curso/sección/paralelo で割れ、しかも「curso」が es-ES では学年、チリでは組を指す＝**同じ語が別の階層を意味する**）。ただし最初は es-419 1本で「grupo（sección）」と併記し、ES 版は後から足す手もある。UI では「組」を**ユーザーが呼び名を選べる設定**にしておく（A/B/C の組名も含め）と国別の揺れを吸収できる【推定】。
- **ポルトガル語は「turma」で共通**なので、まず pt-BR 1本で足りる。pt-PT は語彙（ecrã/tela、ficheiro/arquivo、telemóvel/celular）の差が UI 文言に出るので、ポルトガル本国を狙うなら別ロケールが要る。

### A.8 南米を狙う場合の進め方（案）

1. **言語コード**: `pt-BR` を先に、次に `es-419`（Play Console のロケールは `es-419`。アプリ内は `es` をベースに `es-419` の語彙）。`es-ES` と `pt-PT` は後回し。
2. **価格**: ¥980（約 US$6.5・約 R$36）は**ブラジルでは高い**。Play の国別価格で **ブラジル R$ 14,90〜19,90、チリ CLP 2.990〜3.990、コロンビア COP 12.900〜16.900、ペルー PEN 9,90〜14,90、アルゼンチン は Play の自動換算の最低帯**を目安にする【推定：現地のユーティリティ系アプリの相場感と購買力からの推定。要確認】。無料で画面上は全機能が使えるので、Pro は「印刷・Excel が要る学校の事務担当が1回買う」位置づけ。
3. **告知時期**: ブラジル・チリ・アルゼンチン・ペルーは **11月〜2月**（日本の2〜3月の山の直前）、コロンビアのカレンダーAは **11月〜1月**。**日本と同じ「年度替わり前の3か月」に南北がそろう**ので、ASO（ストア文言の季節語: "montar turmas 2027", "armar cursos 2027"）を 10〜11月に差し替える。
4. **訴求**: pt-BR は「enturmação equilibrada em minutos」「dados dos alunos não saem do aparelho (LGPD)」。es-419 は「arma grupos/cursos equilibrados」「mezcla de cursos」＋チリの新法施行（2026年12月）に合わせた「datos en tu dispositivo」。
5. **優先度**: 南米単独では日本語の次に来るほど強くない（WTP が低い）。**es-419 はスペイン本国・米国のスペイン語話者にも流用できる**ので、スペイン語を作るなら南米向けの語彙を最初から入れておくのが費用対効果が高い。

---

## B. 英語圏（en）

### B.1 オーストラリア・ニュージーランド
- **慣行が最も制度化されている地域。** 公立小学校が「Class Placement Policy」を公開するのが普通で、**毎年、翌年度のクラスを学力・行動・社会性（相性）・性別のバランスで作り直す**。作業は学年末（Term 4、10〜12月）の数か月前から始まり、「友だちを4人挙げさせ、そのうち少なくとも1人と同じ組にする」学校もあれば、友人要望フォームを使わず教員が判断する学校もある。**担任の指名（保護者の希望）は受け付けない**が定型文。出典: https://www.geeastps.vic.edu.au/wp-content/uploads/files/Class-Placement-Policy.pdf , https://connellspt-p.schools.nsw.gov.au/content/dam/doe/sws/schools/c/connellspt-p/2023-documents/ClassPlacementGuidelines.pdf , https://meltonwestps.vic.edu.au/wp-content/uploads/2025/08/Class-Placement-Policy-2025-.pdf
- NZ も Term 4 に保護者の意見フォームを集め、12月に発表する形。出典: https://northcoteint.school.nz/wp-content/uploads/2023/11/Term-4-Week-8-Principals-Newsletter.pdf , https://snellsbeach.school.nz/?p=33781
- **規模**: 豪州 2025年 9,673校・生徒416万人（ABS）。出典: https://www.abs.gov.au/statistics/people/education/schools/2025
- **競合が最も濃い**: Class Creator（豪発・Sentral/Wonde 連携・US$1.60/生徒/年、最低$400）、Class Maker（豪・$199/年〜）、Sentral / School Bytes の Class Builder（学校管理システムの標準機能）。詳しくは `competitors.md` 第3節。**学校単位で既に買われている市場**なので、Mosaic は「予算の無い学校・小規模校・個人の教員」「まず試す」の入口になる。
- **学年暦**: 1月下旬〜2月始業（南半球）→ 編成は**8〜12月**。

### B.2 米国
- **毎年春（4〜5月）に保護者の Parent Input Form を集め、5月にロスターを作り、夏に在籍数の変動で直す**のが典型。基準は男女比・学力・行動・才能・性格・支援の必要性・人間関係の衝突・教員の強みとの相性。出典: https://www.svsd410.org/client-page-academics/elementary-class-roster-development-process , https://www.cbsd.org/cms/lib/PA01916442/Centricity/Domain/2510/Classroom%20Placement%20Planning%20Procedures%2020232024%20Final.pdf , https://avery.webster.k12.mo.us/class-placement
- **規模**: 公立 99,297校・教員325万人（2023–24、NCES）。出典: https://nces.ed.gov/fastfacts/display.asp?id=372
- **WTP が最も高い**: 教員の約9割が自腹で教室の費用を払い、**2024–25 年の平均自腹額は $895**（AdoptAClassroom.org）。出典: https://www.adoptaclassroom.org/2025/06/09/2025-teacher-survey-spending-stats-classroom-needs/
- **障壁**: FERPA・COPPA・州の生徒データ保護法のため、**学区が承認したアプリ以外は生徒データに使えない**という規程が多い（承認リスト方式）。出典: https://studentprivacycompass.org/resource/the-educators-guide-to-student-privacy , https://www.sparksd.org/departments/technology/third-party-apps
  - → **「端末内処理・アカウント不要・送信しない」は承認審査の説明を簡単にする材料**になるが、承認リスト自体を通らないと公式には使えない。**プライバシーポリシーと「データを送信しない」技術説明を英語で用意し、学区の審査（Student Data Privacy Consortium の DPA 等）に出せる形にする**必要がある【推定】。
- **端末**: K-12 の購入端末は Chromebook が約半数（Futuresource、2010年代後半の値）。出典: https://www.gsmarena.com/nearly_50_of_all_of_us_classroom_devices_are_now_chromebooks-blog-15333.php 【最新値は要確認】。Chromebook で Play アプリを使えるかは学区の管理設定次第【要確認】。Web 版（無料）が Chromebook では現実的な入口。
- **学年暦**: 8〜9月始業 → 編成は **4〜6月**、夏に手直し。

### B.3 英国（イングランド）・アイルランド
- 1学年1学級（one-form entry）の小学校も多く、**2学級以上（two-form entry 以上）の学校で「mixing classes（組の混ぜ替え）」をするかどうかが毎回の論争**になる（「30人×2組を性別・学力・年齢・性格・友人関係で均衡させるのは難しく、発表後は涙と苦情が出る」）。出典: https://www.teachwire.net/news/seating-plans-mixing-classes/ , https://www.mumsnet.com/talk/education/791511-mixing-classes
- **規模**: イングランドの学校 24,499校・生徒890万人（2025/26）、幼児学級の平均25.9人。出典: https://explore-education-statistics.service.gov.uk/find-statistics/school-pupils-and-their-characteristics/2025-26
- Class Creator は英国の MIS 連携基盤 Wonde で提供されている＝英国にも競合が入っている。出典: https://wonde.com/uk/application/class-creator/overview
- 英国 GDPR（DPA 2018）で学校はデータ処理者との契約を求められる【要確認】→ 端末内処理は訴求。
- **学年暦**: 9月始業 → 混ぜ替えは**6〜7月**（学期末に発表）。

### B.4 カナダ
- 春に翌年度の案を作り、**9月の在籍確定後に「September reorganization」でクラスを組み直す**（オンタリオ州。複式学級＝combined grades も一般的）。出典: https://www.hwdsb.on.ca/gatestone/files/2022/09/Gatestone-Reorganization-Sept.-2022.pdf , https://efis.fma.csc.gov.on.ca/faab/Memos/SB2014/SB10E_AODA.pdf
- → **年に2回（6月と9月）作業がある**のは Mosaic の「再実行が数秒」という強みが生きる場面。ケベックは仏語（fr-CA）。

### B.5 英語圏のまとめ
- **需要は世界で最も強く、しかも教員の自腹が最も多い。** 反面、学校単位の SaaS が確立していて、公式利用には学区・学校の承認が要る。
- Mosaic のポジション: 「**学校が SaaS を契約していない**（小規模校・私立・予算の無い学区・英国の2学級校）」「**契約はあるが、担任が事前に案を試したい**」「**班分け・グループ分けを毎週したい**（ClassDojo の Group Maker はランダム＋除外だけ。`competitors.md` 3.2）」の3つ。
- 南半球（豪NZ）と北半球（米英加）で作業時期が半年ずれるので、**英語版だけで年に2回の需要の山**がある。

---

## C. 韓国（ko）

- **慣行: 毎年、全学年で 반편성（クラス替え）をする。** 3月始業に向け、従来2月下旬だった반편성を**2月上〜中旬に前倒しする**動きがあり（担任が早く生徒を把握し、保護者も準備できるように）、時期は日本とほぼ同じ。中学では出身小学校の割合と男女比を考慮する。出典: https://dhnews.co.kr/news/view/179515738490486 , https://hangyo.com/mobile/article.html?no=68289
  - 担い手は学年の担任団（現担任が「가편성」＝仮編成を作り、教務部が確定する）【推定：日本と同型。要確認】。
  - 小1の学級編成で「금쪽이」（手のかかる子）の配置が課題として論じられている。出典: https://hangyo.com/mobile/article.html?no=105990
- **「別の組」制約の法的な必要性がある。** 学校暴力予防法により、学校長は学校暴力を認知したら**加害生徒と被害生徒を遅滞なく分離しなければならない**。加害生徒への措置には **7号「学級交替」** がある。**「加害者と同じ中学に配置された」「分離されずに同じ教室で受験した」ことが報道・訴訟になる**ほど敏感。出典: https://www.heraldk.com/article/2026020413450015864 , https://www.segye.com/newsView/20250205502008 , https://www.jjan.kr/articleAmp/20221121580103
  - → **Mosaic の「別の組ペア（ハード制約）」は韓国では『学폭 분리』として最も強い訴求点になる**【推定】。一方、学폭の記録は極めて機微な情報なので、**端末内処理が前提条件**になる。
- **規模**: 教員50.6万人（2025、うち小学校は減少中）、中学の1学級平均24.9人。**少子化で1年生は2年で15%減り30万人を割った**＝学級数は減っていくが、都市部は1学年複数学級が標準。出典: https://en.fnnews.com/news/202508191552256510 , https://asianews.network/?p=251079 , https://v.daum.net/v/20250913160142029
- **競合**: 韓国語の반편성専用ツールは検索で見つからなかった（見つかったのは「1人1役割り当て」の Web ツール程度）。教員コミュニティ（인디스쿨）で Excel マクロが共有されている可能性が高い【要確認】。出典: https://tools.devcomma.com/tools/one-person-one-role
- **端末・課金**: スマホ普及率は極めて高い。Play ストアでの個人購入に障壁は無い【推定】。教員の自費購入の統計は未取得【要確認】。
- **個人情報**: 個人情報保護法（PIPA）は厳格で、学校の個人情報は外部クラウドへの持ち出しに制限がある【要確認：本調査で具体的な教育庁指針を取得できず】。
- **学年暦**: 3月始業 → **12月〜2月が山**（日本と同時期。**日本版のキャンペーンと同じ時期に回せる**）。
- **評価**: 需要 強（毎年・全学年・分離が法的義務）、規模 中、WTP 中、競合 無し、ローカライズ 中（CJK フォントは日本語版で既に対応済みのはず【要確認】）。

---

## D. ドイツ語圏（de）

- **ドイツ**: 小学校（Grundschule、4年制）は**入学時に組を作り、原則4年間そのまま**（担任制）【要確認：州・学校で途中の混ぜ替えもあるが少数派】。組分けが発生するのは **入学時（Einschulung）と Klasse 5（中等への進学）**。入学時の基準は学級人数・友人関係・学級の社会構成・男女比・学力・特別支援・通学路（ある小学校の公開規程）。**保護者の友人希望は受け付けるが、他の基準が多いので余地は限られる。** 出典: https://grundschule.spardorf.de/wp-content/uploads/2022/10/GS-Spardorf_Klassenbildung_Allgemeine-Regelungen_Kriterien_11.10.2022.pdf
  - Klasse 5 のギムナジウムでは「同じ組にしたい友だちを3人まで」書かせ、**「連鎖する希望（A→B→C…）や出身小学校の大集団は避ける」**という運用が見られる＝まさに「同じ組ペアの連結成分が大きくなりすぎる」問題。出典: https://gymnasium-ohmoor.hamburg.de/wp-content/uploads/sites/730/2020/12/Ohmooer_FAQ.pdf
  - → **Mosaic の union-find（同じ組ブロック）とブロックの大きさの警告は、ドイツの Klasse 5 の運用にそのまま合う**【推定】。
- **スイス**: 学校（Schule）ごとに「Klasseneinteilung」の方針を公開し、**段階（Zyklus）の切り替わりで組を混ぜ直す**学校が多い（入学時と中間段階の移行時）。出典: https://www.kirchberg-schulen.ch/schulbetrieb/klasseneinteilung-.html/69/print/pdf , https://www.schule.stallikon.ch/aktuelles/schul-abc/klasseneinteilungen.html/510/print/pdf , https://www.beobachter.ch/bildung/schule/einschulung-mit-dem-freund-zur-schule
- **オーストリア**: Volksschule 4年・固定【推定】。
- **競合**: ドイツ語の専用ツールは見つからなかった（検索で出るのは英語 SaaS のドイツ語ページ＝Class Solver の softwareadvice.de 掲載）。出典: https://www.softwareadvice.de/software/248420/class-solver
- **個人情報**: GDPR＋各州の学校データ保護規則が厳しく、**生徒データを米国クラウドに載せることへの抵抗が強い**【推定：一般的に知られた傾向。個別出典は要確認】→ **端末内処理は最強の訴求**。
- **学年暦**: 8〜9月始業（州ごと）→ 入学・Klasse 5 の組分けは **5〜7月**。
- **評価**: 需要 中（毎年全学年ではなく、入口の学年だけ）、規模 中〜大（DE+AT+CH）、WTP 高、競合 無し、ローカライズ 中（語が長く UI 崩れ注意）。

---

## E. フランス語圏（fr）

- **フランスの小学校は毎年組み直す**（学年ごとに担任が替わるため）。**「生徒の学級への割り振り（répartition des élèves）は校長（directeur）の権限で、教員会議（conseil des maîtres）の意見を聞いて決める」、基準は明示し議事録に残すのが望ましい**とする視学区の通達がある。保護者からの問い合わせには定型の回答を用意するのが普通。出典: https://belfort1.circo90.ac-besancon.fr/wp-content/uploads/sites/2/2016/06/NOTE-DE-SERVICE-N°-9-REPARTITION-des-ELEVES.pdf , https://directeurs-01.blog.ac-lyon.fr/wordpress/wp-content/uploads/2024/10/2024-avril_Cafe-the-dir_Attribution.et_.constitution.des_.classes_Synthese.pdf
- **中学（collège）の6e の学級編成**は「人数・男女・学力の多様性・行動の特性」で均衡させ、**生徒同士の関係（残す組み合わせ・離す組み合わせ）、自律的な子・発言の多い子と控えめな子の配分**まで考える。出典: https://gphilipe.loire-atlantique.e-lyco.fr/wp-content/uploads/sites/30/2024/06/Note-aux-familles-des-eleves-entrant-au-6eme-au-college-G-Philipe.pdf
- **ただしフランスの小学校は複式（cours double: CE1-CE2 など）が非常に多く**、編成の悩みの中心は「どの学年を何人ずつどの組に入れるか（structure）」。この部分は **Teetsh の無料ツール**（学年ごとの人数とクラス数、最少・最多人数、学年数の上限から構成案を自動生成）が既にある。**Mosaic は「学年の混在」を扱えない**ので、その前段は対象外。出典: https://outilstice.com/outil-gratuit-pour-creer-une-structure-decole-repartir-les-eleves/
- **規模**: 初等教育 47,000校・生徒615.5万人（2025年度）。出典: https://www.education.gouv.fr/les-effectifs-dans-le-premier-degre-6155-millions-d-eleves-scolarises-la-rentree-2025-451624
- 競合: Teetsh（構成）、Keamk（仏発のチーム分け、1〜5のレベル合計を揃える）。出典: https://outilstice.com/en/keamk-creer-des-equipes-par-niveau/
- **学年暦**: 9月始業 → 小学校の割り振りは **6月**（conseil des maîtres）、6e は **6〜7月**。
- ベルギー・スイス仏語圏・ケベック（上記オンタリオと同じく9月の再編成あり）にも流用可。
- **評価**: 需要 中〜強（毎年）、規模 大、WTP 中、競合 中（Teetsh・Keamk が無料）、ローカライズ 中。**「複式の学年を扱えない」ことが仏語圏では弱点**。

---

## F. スペイン（es-ES）

- **アンダルシア州などの公立校は「生徒の編成の基準（criterios para establecer los agrupamientos del alumnado）」を学校の計画書に書いて公開する義務があり**、実際に公開文書が大量にある。基準の典型は「各組の人数を揃える・男女を均等に・生まれ月（成熟度）を均等に・特別な教育的支援が要る子を均等に」、**成績で同質な組を作ることは禁止**（segregación にあたる）。**第3学年・第5学年など段階の区切りで組を組み替える（reorganización de grupos）学校がある。** 出典: https://blogsaverroes.juntadeandalucia.es/ceipencarnacion/files/2024/10/2.N.-Criterios-agrupamientos.pdf , https://blogsaverroes.juntadeandalucia.es/ceipreyescatolicos/files/2022/11/24-Fundamentación-Reorganización-de-Grupos.pdf , https://www.edu.xunta.gal/centros/ceipcouto/print/1225
  - 全国調査では、初等の編成基準で最も多いのは「男女の均衡」、次いで「異質性」「アルファベット順」、最も少ないのが「学力」。出典: https://educacionfpydeportes.gob.es/inee/dam/jcr:fca16bf2-407d-4014-9873-7882443749c9/2009p3.pdf
- **規模**: 初等教育を行う学校 13,778校（2025–26）、非大学教育の教員 79.3万人（2024–25）。出典: https://www.educacionfpydeportes.gob.es/dam/jcr:e7a0bdc6-411e-4a1d-9758-746d87ac5837/nota-avance2025-26.pdf , https://lamoncloa.gob.es/serviciosdeprensa/notasprensa/educacion-fp-deportes/Documents/2024/110924-%20datos-y-cifras-curso-escolar-2024-2025.pdf
- 競合: スペイン語の専用ツールは見つからなかった（Keamk のスペイン語紹介のみ）。
- **学年暦**: 9月始業 → 編成は **6〜7月（または9月初め）**。
- **評価**: 需要 中〜強（規程化されている）、規模 中、WTP 中、競合 無し、ローカライズ 小（es-419 と用語だけ差し替え）。

---

## G. イタリア（it）

- **入口の学年（小1・中1＝classe prima）だけ組を作り、以後5年/3年固定。** その代わり**学校ごとに編成基準を明文化**し、委員会（commissione）が作る。基準は「クラス間は均質、クラス内は異質」、**男女・外国籍の子・障害のある子（diversamente abili）・社会経済的に不利な子・DSA/BES（学習障害・特別な教育的ニーズ）を均等に**、出身の園・小学校（plessi di provenienza）の配分。**公表後は変更しない。** 出典: https://nuvola.madisoft.it/file/api/public-file-preview/TOIC85000C/fced2261-427e-43af-b3ff-739ccfed90d6 , https://nuvola.madisoft.it/file/api/public-file-preview/MOIC83300X/4637b0a0-5132-4543-8ced-859e08ceea2a
- **Mosaic の属性設計（該当者数の均等化）と基準がほぼ1対1で対応する**のが特徴。時期は **6〜7月**（9月始業）。
- **評価**: 需要 中（入口学年のみ、ただし全校が毎年やる）、規模 中、WTP 中、競合 無し【要確認】、ローカライズ 中。

---

## H. オランダ・北欧

- **オランダ**: 学級（groep）の編成は学校の裁量。**複式（combinatiegroep）が多く、在籍数の変動で組を組み替える（groepen husselen）と子ども・保護者が反発する**事例が報道される（「グループ3から一緒だった6・7年生が抗議」）。出典: https://www.nhnieuws.nl/nieuws/286856/kinderen-basisschool-de-uilenburcht-in-protest-wij-willen-niet-uit-elkaar , https://zoek.officielebekendmakingen.nl/kst-31293-351.pdf
  - → 需要はあるが頻度は低い。英語で足りる教員も多い【推定】。
- **スウェーデン等**: 学年の区切り（1・4・7年）で組を作り直す学校がある【要確認：本調査で具体的な規程を取得できず】。人口が小さく、言語ごとのローカライズ費用に見合いにくい。
- **評価**: いずれも需要 弱〜中、規模 小。**当面は英語版で拾う**。

---

## I. 中国語圏（zh-Hant / zh-Hans）

- **台湾は編成方法が法令で決まっている。** 「國民小學及國民中學常態編班及分組學習準則」と各市の補充規定により、**常態編班（能力別にしない）・男女混合・人数均衡・障害のある子への配慮**を原則とし、**小1は電腦亂數（コンピュータの乱数）、小3・小5の再編成は学校の基準で並べてから S 型（蛇行）で配る**。編班委員会（行政・教員・保護者代表）が公開で行い、**名簿を15日以上掲示**する。出典: https://laws.gov.taipei/law/LawSearch/LawExport/FL036911?type=0 , https://www.tmups.tp.edu.tw/wp-content/uploads/doc/tmups9201/教務處_註冊組法規_編班_國民小學及國民中學常態編班及分組學習準則980714.pdf
  - → **編成方法そのものが「乱数」「S 型」と決まっているため、最適化で案を作る余地が小さい**。使える場面は「S 型の前の並べ方の検討」「法令の対象外の場面（班分け・私立）」に限られる【推定】。**公開性が要件なので「なぜこの配置か」を説明できないと使いにくい。**
- **香港**: 小学校の分班は学校裁量で、成績による分班の研究がある。中学は教育局の能力組別（Band 1〜3）で**学校間で振り分け済み**。出典: https://bibliography.lib.eduhk.hk/tc/bibs/0e2bb01c
- **中国本土**: 義務教育で**重点班・快慢班を禁止し、「均衡編班・随機生成・陽光公開」**を求める（安徽省「五坚持三严禁」など）。均衡化の需要自体はあるが、**方法は「ランダム＋公開」が求められ、しかも Google Play が無い**。出典: https://www.163.com/dy/article/L3DQNERB05566S5I.html , https://news.ycwb.com/ikimvkotjl/content_54066242.htm
- **評価**: 台湾 需要 中（ただし方法が規制）、香港 弱、本土 配布不可。**優先度は低い。**

---

## J. 東南アジア（インドネシア・ベトナム・タイ）

- **インドネシア**: 新入学年（SMP の7年・SMA の10年）の **rombel（学級）分け**は手作業が多く、「成績の低い子から高い子まで・男女・名前の頭文字・行動の評価」を均等にする手順や、K-means で分ける研究がある。**需要は明確だが教員の購買力は低い**【推定】。出典: https://openjournal.unpam.ac.id/index.php/PROKASDADIK/article/view/38525/21333
- **ベトナム**: 中学（lớp 6）は居住地で割り当て、学校内での学級分けの規範は検索で拾えなかった。**「lớp chọn（選抜クラス）」文化が根強い**【推定】。出典: https://plo.vn/tuyen-sinh-lop-6-chon-ai-bo-ai-post342807.html
- **タイ**: 中等は **ห้องคิง（成績上位の組）や EP・Gifted などの特別クラス**が一般的で、成績で並べる運用が多い。出典: https://so06.tci-thaijo.org/index.php/jomld/article/view/259498
- **評価**: 需要 弱〜中（能力別が主流）、WTP 低。**優先度は低い**（インドネシアは Web 版で様子見）。

---

## K. ポルトガル（pt-PT、参考）

- 学校群（agrupamento）ごとに **「turmas の構成基準（critérios de constituição de turmas）」** を毎年文書化する（マデイラ・アソーレス・本土とも公開例多数）。考慮事項は出身の幼稚園・前段階のグループ、**男女の均衡、年齢、信条・人種・出身国の多様化**、インクルージョンと異質性の原則。コーディネーターが案を出し、校長が承認する。出典: https://aevn.pt/docs/Criterios_Gerais_Matriculas_Constituicao_de_Turmas_Horarios.pdf , https://ebipv.edu.azores.gov.pt/wp-content/uploads/2023/05/Criterios-de-Formacao-de-Turmas_25_26.pdf , https://www.aepombal.edu.pt/wp-content/uploads/2026/07/A6PE_CCTurmas_signed.pdf
- 市場は小さい（人口約1,000万）。**pt-BR 版をまず作り、pt-PT は語彙差し替えで後から**。9月始業 → 7月に編成。

---

## L. 副次用途: 授業内のグループ分け・チーム分け

- 英語圏・仏語圏では**無料の Web ツールと ClassDojo の Group Maker（ランダム＋除外）**が行き渡っていて、単体では課金につながりにくい（`competitors.md` 3.2）。仏発の Keamk（レベル合計を揃える）、豪 AI Classroom Planner（衝突の定義）など「バランス＋除外」まで持つものもある。出典: https://outilstice.com/en/keamk-creer-des-equipes-par-niveau/ , https://apps.apple.com/us/app/-/id6748561222
- **Mosaic が勝てるのは「複数属性を同時に均等化」＋「同じ組／別の組のハード制約」＋「名簿を一度入れれば班分けとクラス編成の両方に使い回せる」**点。**クラス編成は年1回なので、班分けの頻繁な利用がアプリを端末に残す理由になる**【推定】。
- 各国とも「班分け」の語は別に要る: en `groups / teams`、ko `모둠 편성`（韓国の授業のグループは 모둠）、es `equipos / grupos de trabajo`、pt-BR `grupos de trabalho`、de `Gruppeneinteilung`、fr `groupes / îlots`【推定：一般的な用語。要確認】。

---

## M. スコアリング

評価は5段階（5が最も有利）。**「競合」は5＝競合が少ない**、**「ローカライズ費用」は5＝安い**。重みは 需要×3・規模×2・WTP×2・競合×1・ローカライズ×1（合計45点満点）。点数はこの調査の定性的な判断で【推定】。

| 言語（主な国） | 需要の強さ | 市場規模 | WTP | 競合（少なさ） | ローカライズ費用（安さ） | 加重合計 | コメント |
|---|---|---|---|---|---|---|---|
| **en**（US/AU/NZ/UK/CA/IE） | 5 | 5 | 5 | 2 | 5 | **42** | 毎年・全学年の再編成が制度化、自腹 $895/年。SaaS 競合は学校契約で、個人・小規模校の入口が空いている |
| **ko**（KR） | 5 | 3 | 3 | 5 | 3 | **35** | 毎年全学年の반편성、学폭の分離が法的義務＝「別の組」が刺さる。日本と同時期（2月） |
| **es**（ES + 中南米） | 4 | 4 | 2 | 5 | 4 | **33** | スペインは基準の公開が義務、中南米は mezcla de cursos・私立多い。1言語で約20か国 |
| **de**（DE/AT/CH） | 3 | 4 | 5 | 5 | 3 | **35** | 入学時と Klasse 5 だけだが GDPR で端末内処理が最強。WTP 高 |
| **fr**（FR/BE/CH/CA） | 4 | 4 | 3 | 3 | 3 | **32** | 毎年組み直すが複式が多く、構成の段階は Teetsh（無料）が押さえている |
| **pt-BR**（BR、+PT） | 3 | 5 | 1 | 5 | 4 | **30** | 最大の母数・Pix・LGPD。WTP が最も低い |
| **it**（IT） | 3 | 3 | 3 | 5 | 3 | **29** | 入口学年のみだが基準が Mosaic の属性と1対1 |
| **zh-Hant**（TW/HK） | 2 | 2 | 3 | 4 | 3 | **23** | 台湾は乱数・S 型が法定、香港は学校間で能力別 |
| **nl / 北欧** | 2 | 2 | 4 | 5 | 3 | **26** | 頻度が低く英語で代替可 |
| **id**（ID） | 3 | 4 | 1 | 5 | 4 | **28** | rombel 分けの需要はあるが購買力が低い |
| **vi / th** | 1 | 3 | 1 | 5 | 3 | **19** | 能力別（lớp chọn・ห้องคิง）が主流 |
| **zh-Hans**（CN） | 2 | 5 | 2 | 3 | 3 | — | Play が無いので対象外（Web 版のみ） |

- **ko と de は同点（35）。** 順位で ko を上にしたのは、(1) ko は毎年・全学年、de は入口の学年だけで**利用頻度（＝アプリが端末に残る理由）が違う**、(2) ko は日本版と同じ2月の山に同時展開でき運用費が小さい、の2点。**課金単価を最優先するなら de を先にしてよい。**
- es（33）は de より点が低いが、**中南米をまとめて取れること（オーナーの関心）**と、UI 文言が短く作業量が小さいことで3位に置いた。

---

## N. 推奨（作る順番）

| 順位 | 言語・ロケール | 理由（1行） |
|---|---|---|
| **1** | **`en`**（Play の掲載は `en-US` を基準に `en-AU` `en-GB` も） | 需要・規模・WTP が全部最上位。**南北半球で年2回（豪NZ 8〜12月・米英 4〜7月）山**があり、班分け用途もいちばん広い |
| **2** | **`ko`** | 毎年全学年で2月に반편성、**学校暴力の分離義務で「別の組」制約の価値が法的に裏付けられている**。日本版と同じ季節に回せ、競合も見当たらない |
| **3** | **`es`**（アプリ内は `es` 1本＋語彙を `es-419` 寄りに、Play は `es-419` と `es-ES` の2掲載） | スペインは編成基準の公開が義務、中南米（チリ・アルゼンチン・コロンビア・ペルー・メキシコ）まで1言語で届く |
| **4** | **`de`** | 毎年ではないが、**GDPR とドイツの学校データ規則で「送信しない」がいちばん効く市場**。WTP も高い |
| **5** | **`pt-BR`**（または `fr`） | pt-BR はオーナーの南米関心と母数の大きさ・Pix で購入可能。**課金額を重視するなら `fr` が先**（WTP 中・毎年の再編成） |

- **最初の一手は `en` と `ko` の2つ。** 英語はストア掲載と Web 版だけでも効果が出やすく、韓国は日本と同じ2月の山に間に合わせられる（**2026年12月〜2027年2月に向け、11月までに出す**のが理想）。
- `es` と `pt-BR` を作るなら、**南半球の編成期（11〜2月）に合わせて 10月までに**。`de`・`fr`・`it`・`es-ES` は **5〜7月**に合わせる。

---

## O. 言語ごとのローカライズ注意点

### en
- 用語: クラス編成 = **class placement / class lists / building classes**（豪NZ・米）、**mixing classes**（英）。「同じ組にしたい」= **keep together / pair**、「別の組にしたい」= **separate / keep apart**（Class Creator・Class Maker の語）。クラス = class / homeroom（米）/ form（英の中等）。学年 = grade（米加）/ year（英豪NZ）。**ロケールで grade/year を切り替える**。
- 属性の語: 性別は **gender**（ただし入力値は学校に任せ、選択肢を固定しない）。特別支援は米 **IEP / 504**、英 **SEN / SEND・EHCP**、豪 **NCCD / funded student**。英語学習者は **EAL / ESL / ELL**。**「behaviour」は英豪綴り**、米は behavior。
- **マーケティング時期**: 豪NZ 9〜11月（Term 3〜4）、米 3〜5月、英 5〜7月、加 5〜6月と9月（reorganization）。
- 訴求: 「No sign-up. Student data never leaves your device.」「FERPA-friendly: nothing is uploaded」は**法的な保証と誤解されない書き方**にする（「complies with FERPA」とは書かない）。
- 価格: US$4.99〜6.99（¥980 相当）は、自腹 $895/年の層には十分に安い【推定】。

### ko
- 用語: クラス編成 = **반편성**（学級編成は **학급 편성**）、組 = **반**（1반・2반）、学年 = **학년**。担任 = **담임**。「同じ組」= **같은 반 배정**、「別の組」= **분리 배정 / 다른 반 배정**。班分け = **모둠 편성**。
- **学폭（학교폭력）関係の分離**を訴求するときは、**アプリが「学폭の記録を扱う」と書かない**（機微すぎる）。「분리가 필요한 학생」程度の中立的な表現にする【推定】。
- 属性: 特別支援 = **특수교육대상 / 통합학급**、**다문화**（多文化家庭）の配慮は一般的な語だが、**属性名として既定値に置かない**（保護者に見られたときの問題）【推定】。
- 時期: **12月〜2月**。人気の教員コミュニティ（인디스쿨）での紹介が効く【推定】。

### es（es-ES / es-419）
- 用語は A.7 の表を参照。**「組」は es-ES `grupo`、es-419 `grupo / sección`（UI では組の呼び名を設定で変えられるようにする）**。**`curso` はスペインでは学年、チリでは組**を指すので、UI の固定文言には使わない。
- 「同じ組」= **mantener juntos**、「別の組」= **separar**。クラス編成 = es-ES **agrupamiento del alumnado / reparto en grupos**、es-419 **distribución de estudiantes / armado de cursos**。
- 属性: es-ES **NEAE**、es-419 **NEE**。性別は **sexo / género** のどちらも使われるが、スペインの公的文書は「equilibrio entre niños y niñas」が多い。
- **スペインでは成績で同質な組を作ることが禁止**されているので、「学力の平均を揃える（異質性を保つ）」と書く。
- 時期: es-ES 5〜7月、中南米（南半球・コロンビアA）10〜2月、メキシコ 6〜8月。

### de
- 用語: クラス編成 = **Klasseneinteilung / Klassenbildung**、組 = **Klasse**（5a, 5b）、「同じ組」= **zusammen (Freundschaftswunsch)**、「別の組」= **trennen**。班分け = **Gruppeneinteilung**。
- 属性: 特別支援 = **Förderbedarf / sonderpädagogischer Förderbedarf**、DaZ（第二言語としてのドイツ語）。
- **Datenschutz を最初に書く**（「Alle Daten bleiben auf dem Gerät. Keine Cloud, kein Konto.」）。**DSGVO konform と断言しない**（判断するのは学校・州）。
- 敬称: 教員向けアプリは **Sie** で統一【推定】。語が長いので UI のボタン幅に注意。
- 時期: 4〜7月（Einschulung・Klasse 5 の組分け）。スイスは学期の区切りで。

### pt-BR
- 用語: クラス編成 = **enturmação / montagem de turmas / distribuição de alunos nas turmas**、組 = **turma**、学年 = **ano**、「同じ組」= **manter juntos**、「別の組」= **separar**。コーディネーター = **coordenação pedagógica**。
- 属性: 特別支援 = **alunos com deficiência / público-alvo da educação especial / AEE**。**「comportamento」の均等化はブラジルの enturmação で実際に使われる基準**（A.1）。
- 訴求: **LGPD**（「os dados dos alunos não saem do aparelho」）。
- 価格: R$14,90〜19,90【推定】。**Pix で買える**ことを告知に書く。
- 時期: **11月〜2月**。

### fr
- 用語: **répartition des élèves / constitution des classes**、組 = **classe**、「同じ組」= **garder ensemble**、「別の組」= **séparer**、教員会議 = **conseil des maîtres**。
- **複式（cours double）の学年配分は扱えない**ことを明記し、「学年ごとの人数配分が決まった後の、生徒の割り振り」に絞って訴求する（Teetsh との棲み分け）。
- 属性: 特別支援 = **PAP / PPS / AESH**（支援員が付く子）。
- 時期: 5〜6月。

---

## P. 未確認事項（リリース前に確かめること）

- [ ] Google Play で「class placement」「반편성」「Klasseneinteilung」「distribución de alumnos」「enturmação」を実際に検索し、同種アプリの有無と DL 数を確認する（本調査は Play を直接検索できていない）。
- [ ] 韓国の教育庁（시도교육청）の個人情報指針で、教員の私物端末に生徒名簿を置くことがどこまで許されるか。
- [ ] 米国の学区の承認リスト（Student Data Privacy Consortium の DPA）に、データを送信しないアプリがどう扱われるか。
- [ ] ドイツ各州の「私物端末での生徒データ処理（Dienstvereinbarung / Genehmigung）」の要件。日本と同じく**私物端末の利用自体が制限される**可能性がある。
- [ ] 中南米の Play の国別価格の実勢（R$/CLP/COP/PEN）と、教員の自腹購入の実態。
- [ ] アルゼンチン・コロンビア・メキシコの組の呼称（división/sección/grupo）と、毎年組を混ぜるかどうかの慣行（A 節で【要確認】にしたもの）。
- [ ] Android の国別シェア（アルゼンチン・チリ・コロンビア・ペルー）を StatCounter の国別ページで確認する（本調査は抜粋のみ）。

---

## Q. 要約（5行）

1. **クラス替えを毎年・全学年でやり、しかも手作業が痛みとして語られているのは、英語圏（豪NZ・米・加）・韓国・フランス・スペイン**。ドイツ・イタリアは入口学年だけ、台湾・中国は方法が乱数/S 型で規制、東南アジアは能力別が主流。
2. **英語圏は需要・規模・WTP（教員の自腹 平均$895/年）が最大**だが、学校契約の SaaS（Class Creator 等）が確立している。Mosaic は「無料・登録不要・端末内」で個人と小規模校の入口を取る。
3. **韓国は日本と同じ2月に全学年で반편성し、学校暴力の加害・被害の分離が法的義務**で「別の組」制約がそのまま刺さる。競合が見当たらない。
4. **南米は南半球の2〜3月始業で日本とほぼ同時期に告知できる**。チリ（私立補助校54%・mezcla de cursos の規程・2026年12月の新個人情報法）が最有望、ブラジルは母数最大・Pix で購入可能だが WTP が低い。
5. **推奨順: `en` → `ko` → `es`（es-419 語彙＋es-ES 掲載）→ `de` → `pt-BR`（課金重視なら `fr`）**。まず `en` と `ko` を2026年11月までに出し、2027年2月の山に間に合わせる。
