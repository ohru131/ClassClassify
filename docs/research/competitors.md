# 競合・類似製品調査（クラス編成・クラス分け・班分け・席替え）

- 調査日: 2026-09-30
- 対象: Mosaic — クラス編成オプティマイザー（焼きなまし法で属性を均等化、同じ組／別の組の制約、名簿エディタ、結果の手動調整、Excel / Google スプレッドシート入出力、端末内処理、Web版は無料。Expo版は広告なしの無料＋Pro買い切り＝xlsx 書き出しと印刷・PDF）
  - 2026-10-01 追記: 調査時点の前提「Expo版は広告付き・Pro で広告除去」は、その後の方針変更（広告なし・買い切りの Pro は書き出しと印刷・PDF）に合わせて書き直した。調査結果そのものは変えていない

> **調査方法と限界（必ず読むこと）**
> - 一次情報源は WebSearch の検索結果（タイトルと抜粋）。WebFetch は、この環境の egress ポリシーで
>   apps.apple.com / play.google.com / capterra.com / classsolver.com / bizfrsoft.com / soft-egao.com などが**ブロックされた**。
>   そのため、**ストアページ本文・価格ページ・レビュー本文を直接読めていない製品が多い**。
> - 各項目の出典URLは、その事実が抜粋に出ていた検索結果のもの。抜粋で確認できなかった事項は **「未確認」** と書く。
> - **ダウンロード数（Google Play の「N万回以上」）は、どの製品も取得できていない**（Play のページを読めなかったため）。
>   ストア評価も、抜粋に出ていたものだけを載せている。**リリース前に手作業で確認すること。**

---

## 1. 結論（先に要約）

1. **学校向けの「本格的なクラス編成」は、海外では年額の学校ライセンス型 SaaS が主流**
   （Class Creator ＝ US$1.60/生徒/年・最低 $400、Class Composer ＝ US$699/年〜、Class Maker ＝ 年$199〜、Class Solver ＝ 価格非公開）。
   どの製品も「属性バランス＋ペア／分離＋ドラッグ＆ドロップの手直し」を持っていて、**Mosaic の機能の核はこの業界の標準装備**と言える。
2. **日本語で使えるクラス編成の専用ソフトは少ない。** 見つかったのは PC 向けの「学級編成支援プログラム」（bizfrsoft）と「学級編成」（教育ソフト 心）の2つだけ。
   大手の校務支援システム（EDUCOM マネージャー C4th、スズキ校務）については、**自動クラス編成の機能を持つかどうか公開情報で確認できなかった**。
3. **スマホアプリはほぼ全部「班分け・チーム分け・席替え」の軽いもの**で、ランダムか「レベル1〜9の平均をそろえる」程度。
   **複数の属性を同時に均等化し、同じ組／別の組のハード制約を守るクラス編成をスマホでできるアプリは、見つからなかった**
   （日本語・英語とも。ただしストア本体を検索できていないので、見落としはありうる）。
4. **データの扱いで差が出る。** 海外 SaaS はクラウド保存（Class Creator は豪NZのデータを豪州で保管）。
   **端末内処理をうたうのは Class Maker（ローカル保存）と席替えメーカー（名前をサーバーへ送らない）くらい。**
   日本の学校は個人情報の持ち出しに厳しいので、Mosaic の「端末内処理」は強い差別化になる。
5. 価格の相場: 個人向けアプリの買い切りは **$0.99〜$4.99／¥190〜¥300 程度**、学校向け SaaS は **年 $199〜$700＋**。
   Mosaic の Pro 買い切りは**個人の教員が自腹で払える ¥480〜¥980** あたりが現実的（詳しくは第5節）。

---

## 2. 日本の製品

### 2.1 校務支援システム

| 製品 | 提供形態・規模 | クラス編成機能 | 出典 |
|---|---|---|---|
| EDUCOM マネージャー C4th（EDUCOM） | 校務支援の統合システム。2021年10月時点で全国の小中学校約8,800校に導入。都立高校向けの C4th for High School もある | **未確認**（成績・出欠・保健・学籍などの機能は確認できたが、自動クラス編成は検索範囲に出てこなかった） | https://resemom.jp/release/prtimes/20211110/76429.html , https://www.itreview.jp/compares/educom-manager-c4th_vs_sukorev2 |
| スズキ校務（スズキ教育ソフト） | 名簿・出欠・成績・通知表など、機能ごとに買える校務ソフトのシリーズ。LINEスクール連絡帳と名簿を連携 | **未確認** | https://www.itreview.jp/products/suzukikotsutomu/profile , https://www.lycorp.co.jp/ja/news/release/018406/ |

- 校務支援システムは自治体単位で調達される（豊中市の授業支援ソフト仕様書など）。**個人の教員が選んで入れるものではない**ので、Mosaic の直接の競合というより「すでにある名簿データの出どころ」と見る方が正確。
  出典: https://www.city.toyonaka.osaka.jp/kosodate/kyoikucenter/R7_jugyoushien_propo.files/02.jugyoushiensoft_shiyouyouken.pdf
- **生成AIで学級編成を時短した事例**が、文科省のリーディングDXスクールの資料に載っている（児童の特性を項目にして数値化し、生成AIにプロンプトを入れると大幅に時短でき、ヒューマンエラーも減ったという報告）。
  **これは実質的な競合（汎用AIで代用する）になる。** ただし児童の個人情報を外部のAIに渡すことになるので、Mosaic の端末内処理はここでの対抗軸になる。
  出典: https://leadingdxschool.mext.go.jp/files/achieve_r5/jirei/C103232100035_05.pdf

### 2.2 クラス編成の専用ソフト

| 製品 | 主な機能 | アルゴリズム | 価格・動作環境 | 出典 |
|---|---|---|---|---|
| 学級編成支援プログラム（bizfrsoft） | 児童数・男女比・元のクラスの人数を各クラスへ均等に配分。成績を5段階に分け、段階ごとの人数をクラス間でそろえる。特別支援学級・要支援・落ち着きのない子・リーダー性のある子の指定、同じクラスにしたい／したくない子の指定、自動で組み分けたあと手で移動できる | 自動配分（方式は**未確認**） | **未確認**（ページを取得できなかった） | https://bizfrsoft.com/gakkyuhenseisien/ |
| クラス編成ソフト「学級編成」（教育ソフト 心） | PC上で瞬時にクラス編成。手作業で編成したい人向けに個人データをカードにもでき、PCで編成したあと手作業で直すこともできる | **未確認** | **未確認** | https://soft-egao.com/classorganization/ |

- **どちらも機能の範囲は Mosaic とほぼ重なる**（属性の均等化・同じ／別の指定・手動での調整）。差が出るのは、**Web・スマホで動くか、Google スプレッドシートを使えるか、無料で試せるか**の3点になりそう（相手側の価格・動作環境が未確認なので断定はできない）。
- 学術研究としては、小規模なクラス編成の支援システムや、クラス編成問題への最適化手法の適用がある（競合ではなく参考）。
  出典: https://n-junshin.repo.nii.ac.jp/record/37/files/11_yoshihara.pdf , https://bunkyo.repo.nii.ac.jp/records/6803

### 2.3 フリーソフト・Excelマクロ・Webサービス

| 製品 | 種別 | 内容 | データの扱い | 出典 |
|---|---|---|---|---|
| バス座席表 | Excelマクロ（フリー） | 名簿から教室の席順表・連絡網・バスの座席・家庭訪問や三者面談の日程・賞状などを印刷。**クラス編成の機能は無い** | ローカル | https://forest.watch.impress.co.jp/docs/review/20120619_538532.html |
| この席がえ〜 | Windowsのフリーソフト | 男女の配置や座席の条件を指定してランダムに席替え。ルーレット風の演出 | ローカル | https://forest.watch.impress.co.jp/docs/serial/okiniiri/310877.html |
| 席替えメーカー（Seat Shuffle / luft.co.jp） | Web（無料） | 名前を貼り付けて行・列を決めるだけで席替え。教卓の位置、目の悪い子を前へ、背の高い子を後ろへ、隣にしないペア、の4条件。PNG・CSVで保存 | **入力した名前をサーバーへ送らない**と明記 | https://www.luft.co.jp/cgi/seat-shuffle.php |

- **日本語で「クラス替えを自動でやる Web サービス」は、この調査の範囲では見つからなかった**
  （「クラス替え 自動 Webサービス」で検索しても、出てくるのはクラス替えの決め方を解説する記事ばかり）。
  → **Mosaic の Web 版は、日本語でこの種類の検索をする人にとって空白地帯になっている可能性が高い。**
  出典（決め方の記事の例）: https://news.allabout.co.jp/articles/o/110799/ , https://hugkum.sho.jp/411212

### 2.4 スマホアプリ（日本語）

| アプリ | OS | 機能 | 価格・課金 | 評価（抜粋に出ていたもの） | 出典 |
|---|---|---|---|---|---|
| グループわけ | iOS | グループごとの人数と名前を入れると自動でグループ分け。前回との重複チェックと重複を避ける機能 | 無料 | 3.1 | https://apps.apple.com/jp/app/id1396507973 |
| グループわけPRO | iOS | メンバーにレベル（1〜9）を付け、グループごとのレベル平均を均等に近づける。レベルの均等化と重複の回避の比重を調整できる。最大16グループ | 有料（金額は**未確認**） | 5.0（2件） | https://apps.apple.com/jp/app/%E3%82%B0%E3%83%AB%E3%83%BC%E3%83%97%E3%82%8F%E3%81%91pro/id1466498428 |
| グループ分け。チーム分け。 | iOS | カードを引く形式の抽選。最大26人・26グループ | **未確認** | **未確認** | https://apps.apple.com/jp/app/%E3%82%B0%E3%83%AB%E3%83%BC%E3%83%97%E5%88%86%E3%81%91-%E3%83%81%E3%83%BC%E3%83%A0%E5%88%86%E3%81%91/id6469519497 |
| かんたんチーム分け（FlexibleGrouping） | iOS | 2対3・4対4対2のようにグループごとの人数を自由に決められる。最大10グループ・1グループ30人まで | **未確認** | **未確認** | https://apps.apple.com/jp/app/id1492907581 |
| 車割作成 | iOS | 性別や学年がばらけるようにランダムにグループ分け | **未確認** | 4.6（33件） | https://apps.apple.com/jp/app/%E8%BB%8A%E5%89%B2%E4%BD%9C%E6%88%90-%E7%B0%A1%E5%8D%98%E3%81%AB%E3%82%B0%E3%83%AB%E3%83%BC%E3%83%97%E5%88%86%E3%81%91/id1627910277 |
| TeamMaker | iOS | シンプルなランダムのグループ作成 | **未確認** | 4.6（5件） | https://apps.apple.com/jp/app/teammaker/id6475189247 |
| Let's席替え〜学校の先生もラクラク席替え〜 | iPhone（iOS 17以降） | 元教員が「毎回紙でくじを作っている」のを見て作った席替えアプリ | 無料＋アプリ内課金（¥300・¥190） | **未確認** | https://apps.apple.com/jp/app/id1531555002 |
| 瞬速！席替え抽選 | iPad | 行・列・空席を自由に決め、最大100人まで抽選 | 有料 $4.99（広告なし） | **未確認** | https://apps.apple.com/us/app/%E7%9E%AC%E9%80%9F-%E5%B8%AD%E6%9B%BF%E3%81%88%E6%8A%BD%E9%81%B8/id6738973589 |
| いちごプラス 席替えするよ | iOS | 席替え（詳しい機能は**未確認**） | **未確認** | **未確認** | https://apps.apple.com/us/app/%E3%81%84%E3%81%A1%E3%81%94%E3%83%97%E3%83%A9%E3%82%B9%E5%B8%AD%E6%9B%BF%E3%81%88%E3%81%99%E3%82%8B%E3%82%88/id1603062396 |

- **Google Play の日本語アプリは、検索結果にほとんど出てこなかった**（出てくるのは App Store ばかり）。Android では、この種のアプリがさらに手薄な可能性がある。**ただしこれは検索エンジンの偏りかもしれないので、Play ストアで直接確かめる必要がある（未確認）。**
- 見つかった日本語アプリは、**どれも「班・チーム・席」が対象**で、学年全体をクラスに分けるもの（数十〜数百人、属性が複数、ペアのハード制約）は無かった。
  いちばん近いのは「グループわけPRO」（1属性の平均をそろえる＋重複の回避）。

---

## 3. 海外の製品

### 3.1 学校向けのクラス編成（class placement）

| 製品 | 形態・対象 | 主な機能 | アルゴリズム | 価格 | データの扱い | 評価 | 出典 |
|---|---|---|---|---|---|---|---|
| **Class Solver** | Web / SaaS。K-12（小中高・学区） | 属性（characteristics）を自由に定義、ペア／分離の要望を無制限に扱い年度をまたいで保持、全員に友だちがいるかの確認、ソシオグラム、教員アンケート | 「数千通りの組み合わせを評価する最適化エンジン」とうたう | **非公開**（Pricing model: Other） | クラウド | Capterra 4.9/5（48件）。使いやすさ 4.9・サポート 5.0・コスパ 4.8 | https://www.capterra.com/p/205067/Class-Solver/ , https://www.getapp.com/k-12-software/a/class-solver , https://account.sais.org/vendors/class-solver |
| **Class Creator** | Web / SaaS（豪発、2014年〜）。Sentral・Wonde と連携 | 教員アンケート（行動・学力・特別支援・友人関係・EAL/ESL・タグ）、バランスの取れたクラス作成、D&Dでの編集、生徒どうしのペア／分離、配置アラート、履歴の保存、複式学級 | 自動の均衡化（方式は**未確認**） | **US$1.60/生徒/年**（初年度25%引きで$1.20）、AUD$2.50/生徒/年（税抜）、**最低 $400**、任意のサポート US$295/年、日割りなし、自動更新なし | クラウド（豪NZの学校のデータは豪州で保管） | 公式の推薦文は好評（中立の不満レビューは見つからず） | https://www.classcreator.io/pricing , https://www.classcreator.io/faq/ , https://www.classcreator.io/features , https://www.classcreator.io/testimonials |
| **Class Composer** | Web。小学校向け | 紙のカードと付箋の代わりにデジタルのカードで属性を扱う。作業時間を最低50%減らすとうたう | **未確認** | **US$699/年〜の学校定額**、1学年ぶんの無料トライアル | クラウド | **未確認** | https://www.capterra.com/p/227680/ClassComposer/ , https://wp.schooldataleadership.org/systems/student-registration/class-composer |
| **Class Maker** | PCソフト（豪）。小学校向け | 友人の要望から翌年度のクラスを自動作成、一緒にする／離す、男女比、クラスの平均学力の監視、担任の指定、**手で直すと即座にフィードバック**。Excelのテンプレートで入力し、Excel互換で出力 | 自動（方式は**未確認**） | **学校の年間ライセンス $199〜** | **ローカル保存（Webに置かない）**とうたう | **未確認** | https://www.top4.com.au/business/class-maker-156086 , https://www.g2.com/products/class-maker/discuss |
| **Sorting Wizard** | Web（元は Excel） | 能力のバランス、男女の均等、友人関係、問題を起こしやすい子の分離。Excel の取り込み、結果のまとめを表示 | **未確認** | **未確認** | **未確認** | **未確認** | https://teachmag.com/a-new-way-to-create-class-lists-introducing-the-sorting-wizard/ |
| **Sentral Class Builder** | 学校管理システム Sentral（豪）のモジュール | 性別・学力・EAL/D・特別支援の記録、友人／分離、生徒と教員のペア／分離、自動生成、D&D、引き継ぎメモ | 「logic algorithms」による配置の最適化 | Sentral の契約に依存（**未確認**） | クラウド | **未確認** | https://www.sentral.com.au/class-builder , https://help.sentral.com.au/modules/classbuilder/faq/1878400 |
| **School Bytes Class Builder** | 学校管理システム School Bytes（豪）のモジュール | クラスの人数・男女・学力・友人／分離・生徒と教員のペア／分離で自動生成、D&D、確定したクラスを SIS へ同期 | **未確認** | **未確認** | クラウド | **未確認** | https://www.schoolbytes.education/features/class-builder/ |
| PowerSchool / Arbor（学校管理システム） | SIS / MIS | **属性をバランスさせる自動編成機能は検索範囲で確認できなかった**（PowerSchool はクラス発表の配信先、Arbor は年度更新と手動登録の手順） | — | — | — | — | https://edms.blackgold.ca/news/welcome-back-2023-2024-–-école-dansereau-meadows-school-1733951530327 , https://schoolsweb.buckinghamshire.gov.uk/media/108623/eoy-arbor-primary-2025.pdf |

- **「Class Balancer」「Schoolbox」という名前のクラス編成製品は、この調査では確認できなかった**（依頼文にあった名前だが、実在の確認が取れていない）。
- 米国の学校では、クラス分けの優先順位は「クラスの人数・性別・学力・行動・支援の必要性・人間関係」で、保護者の意見フォームも使う。Mosaic の属性の設計はこれとそのまま合う。
  出典: https://wasatch.provo.edu/es/wp-json/wp/v2/posts/26886 , https://pubsonline.informs.org/doi/10.1287/ited.2013.0111
- **gruepr**（大学のプロジェクトチーム分け）は、オープンソースの C++ 製で Windows / macOS 向け。**遺伝的アルゴリズムで最大200人**を最適なチームに分ける。無料で、Google フォームのアンケートを取り込む。
  → 「最適化するグループ分け」を無料で配っている、Mosaic にいちばん考え方が近い先行例。
  出典: https://peer.asee.org/gruepr-an-open-source-program-for-creating-student-project-teams , https://strategy.asee.org/gruepr-an-open-source-program-for-creating-student-project-teams.pdf

### 3.2 グループ分け・チーム分け

| 製品 | OS | 機能 | アルゴリズム | 価格 | 出典 |
|---|---|---|---|---|---|
| Team Shake | iOS | 名前を入れて端末を振るとチームを作る。スキル5段階・性別でバランス、ファイルから取り込み | ランダム＋バランス | 買い切り $1.99。評価 2.0/5 と書く出典が1件ある（件数は少ない） | https://apps.apple.com/app/id390812953 , https://ctl.mesacc.edu/crossroads/team-shake/ |
| InstaGroups | iPad | 性別の配分・能力（1〜5）・出欠を考えて均衡したチーム、2〜10チーム、総当たり・トーナメント表 | 「smart team balancing algorithm」 | 無料＋**サブスク**（月額／年額。金額は**未確認**）。Pro はクラス・生徒が無制限、同僚と共有 | https://apps.apple.com/us/app/-/id6757898148 |
| Teacher Group Maker | iOS | 協同学習のための小グループ分け | **未確認** | **未確認** | https://apps.apple.com/us/app/id1506675959 |
| GroupMaker | iOS | 性別・成績・民族でグループを組む。クラス写真の顔検出で名簿を取り込む | **未確認** | **未確認** | https://ctl.mesacc.edu/crossroads/groupmaker/ |
| Keamk | Web（仏発、スマホでも使える） | 1〜5のレベルの合計をチーム間でそろえる、性別を混ぜる、**Excel の取り込みと書き出し** | 合計値の均等化 | **未確認** | https://outilstice.com/en/keamk-creer-des-equipes-par-niveau/ |
| Flippity（Random Name Picker） | Web＋Google スプレッドシート | 名簿を貼るとペア・4人組などをランダムに作る | ランダム | 完全無料 | https://www.techlearning.com/how-to/best-flippity-tips-and-tricks-for-teachers , https://www.cristinacabal.com/?p=15004 |
| ClassDojo Toolkit の Group Maker | Web / iOS / Android（ClassDojo アプリ内） | 任意の人数でランダムに組む、「一緒にしない」組を設定できる | ランダム＋除外 | 無料 | https://classdojo.com/toolkit/groupmaker , https://help.classdojo.com/hc/en-us/articles/115003740283-What-is-Toolkit |
| Randomizer / Strategic Group Maker（Google Workspace のアドオン） | Google スプレッドシート | 名前のランダム化、ペア・グループ作成、特定の組み合わせを防ぐ | ランダム＋除外 | **未確認** | https://workspace.google.com/marketplace/app/randomizer/734030437098?hl=ja , https://workspace.google.com/marketplace/app/strategic_group_maker/997854293694?hl=ja |
| Google Classroom の生徒グループ | Web / アプリ | グループへの課題配布・メール・グループごとの採点（2025年6月に機能追加、2025年8月に API） | 手動 | Education Plus / Teaching and Learning アドオンのみ | https://workspaceupdates.googleblog.com/2025/04/new-student-group-capabilities-google-classroom.html , https://workspaceupdates.googleblog.com/2025/08/create-manage-student-groups-classroom-api.html |
| SORT / Team Picker Wheel / YouWare Student Group Generator ほか | Web | ランダム、性別のミックス、登録不要 | ランダム | 無料 | https://www.youware.com/features/student-group-generator , https://outilstice.com/en/?p=9463 |

### 3.3 席替え（seating chart）

| 製品 | OS | 機能 | 価格 | 出典 |
|---|---|---|---|---|
| SeatCharter | iPad | ワンタップでシャッフル（教室の配置は変えない） | $0.99 | https://apps.apple.com/pg/app/seatcharter/id428800413 |
| Seats: Smart Seating Charts | iPad | おしゃべりな子を離す、前列に置く子を指定、能力が多様／同質なグループ | **未確認** | https://apps.apple.com/us/app/-/id1139266448 |
| Seating Chart - sekigae sensei | iOS | 教室の配置をD&Dで設計、制約を守って公平に並べるアルゴリズム、ルーレット演出 | **未確認** | https://apps.apple.com/app/id6760364497 |
| SC Class Assistant / Pro | iPad | 座席・出欠・宿題・行動の記録、ランダムな指名とグループ | **未確認** | https://apps.apple.com/us/app/student-centered-class-asst/id6745229100 |
| Smart Seat | iPad / iPhone | 座席表、協同学習のグループ、出欠 | **未確認** | https://www.commonsense.org/education/reviews/smart-seat |
| Mega Seating Plan（seatingplan.com） | Web | 個人: Bronze 無料（座席表1つ）/ Silver（無制限・写真・色分け・名前を覚えるツール）。学校: Platinum（SIS と同期）£7.03/教員/年〜、Gold（CSV・API）£5.63/教員/年〜 | 上記 | https://help.seatingplan.com/en/article/how-much-does-mega-seating-plan-cost-4j5mrb/ |

---

## 4. 比較表（主要な競合と Mosaic）

凡例: ◎ 強い／○ あり／△ 限定的／× なし／? 未確認

| 製品 | Web | iOS | Android | Windows | Chromebook | 複数の属性を均等化 | 同じ組／別の組 | 手動調整 | Excel | Google スプレッドシート | 最適化 | 端末内処理 | 価格 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Mosaic** | ◎ | ○（開発中） | ○（開発中） | ◎（ブラウザ） | ◎（ブラウザ・アプリ） | ◎ | ◎（ハード制約） | ◎（D&D＋即時再集計） | ◎ 入出力 | ◎ 入出力 | ◎ 焼きなまし法 | ◎ | Web無料／アプリは広告＋Pro買い切り |
| Class Solver | ◎ | ? | ? | ブラウザ | ブラウザ | ◎ | ◎ | ○ | ? | ? | ◎（とうたう） | × | 非公開（学校向け） |
| Class Creator | ◎ | ? | ? | ブラウザ | ブラウザ | ◎ | ◎ | ◎ | ○ 取り込み | ? | ○ | × | $1.60/生徒/年・最低$400 |
| Class Composer | ◎ | ? | ? | ブラウザ | ブラウザ | ○ | ? | ○（カード） | ? | ? | ? | × | $699/年〜 |
| Class Maker | × | × | × | ○ | ? | ○ | ○ | ◎ | ◎ | ? | ? | ◎ | $199/年〜 |
| 学級編成支援プログラム | ? | ? | ? | ?（PCとみられる） | ? | ○ | ○ | ○ | ? | ? | ? | ?（ローカルとみられる） | 未確認 |
| 学級編成（心） | ? | ? | ? | ?（PC） | ? | ○ | ? | ○（カード） | ? | ? | ? | ? | 未確認 |
| グループわけPRO | × | ○ | ? | × | × | △（1属性の平均） | △（重複の回避） | ? | × | × | ○ | ○ | 有料（金額は未確認） |
| Keamk | ◎ | ブラウザ | ブラウザ | ブラウザ | ブラウザ | △（1属性＋性別） | ? | ? | ◎ 入出力 | ? | △（合計値） | ? | 未確認 |
| InstaGroups | × | ○（iPad） | × | × | × | ○（性別・能力・出欠） | ? | ? | ? | ? | ○ | ? | サブスク |
| ClassDojo Group Maker | ○ | ○ | ○ | ブラウザ | ブラウザ | × | △（除外のみ） | ? | × | × | × ランダム | × | 無料 |
| Flippity | ○ | ブラウザ | ブラウザ | ブラウザ | ブラウザ | × | × | × | × | ◎ | × ランダム | × | 無料 |
| gruepr | × | × | × | ○ | × | ◎ | ○ | ? | CSV | Google フォーム経由 | ◎ 遺伝的アルゴリズム | ◎ | 無料（OSS） |
| 生成AI（ChatGPT / Gemini） | ◎ | ◎ | ◎ | ◎ | ◎ | △（制約を守る保証が無い） | △ | ○ | ○ | ○ | × | × | 無料〜 |

---

## 5. Mosaic への示唆

### 5.1 差別化できるところ

1. **「スマホ・タブレット・Chromebook で動く、本格的なクラス編成」は空いている。**
   本格的なものは海外の学校向け SaaS（年額・学校単位の契約・英語のみ）か、日本の PC ソフトしか無く、ストアにあるのは班分け・席替えの軽いアプリだけ。
2. **ハード制約を守る最適化。** 班分けアプリはランダムか「1属性の平均」まで。生成AIは制約を守る保証が無い。
   Mosaic は「同じ組」を union-find で束ねて**常に**守り、属性ごとの人数を理想の整数範囲 `[⌊合計/K⌋, ⌈合計/K⌉]` に収めようとする。**これは数字で示せる強み**（結果画面の集計表そのものが証拠になる）。
3. **端末内処理。** 個人情報（特別支援・家庭の事情・人間関係）を外に出さない。海外 SaaS はクラウド、生成AIは外部送信になる。
   ストアの説明とデータセーフティの申告で**いちばん前に出すべき点**。
4. **Google スプレッドシートとの入出力。** 日本の GIGA スクールでは Google Workspace の比率が高いので、Chromebook と組み合わせると強い。
   競合で Google スプレッドシートを正面から扱うのは Flippity（ランダムのみ）と Workspace アドオン程度。
5. **日本語と、日本の学校の慣習に合った属性**（リーダー性・要支援・ピアノ伴奏など）をそのまま扱える。海外製品は日本語非対応（**未確認だが、公開情報に日本語対応の記載は無かった**）。

### 5.2 弱いところ・リスク

- **教員アンケートによるデータ収集が無い。** Class Creator・Class Solver の核は「担任ごとにアンケートで属性を集める」流れで、Mosaic は Excel を1人が作る前提。複数の担任で名簿を分担する運用には弱い。
- **年度をまたいだ履歴が無い**（Class Solver は分離の要望を年度をまたいで保持、Class Creator は配置の履歴を保存）。
- **ソシオグラム・人間関係の可視化が無い**（Class Solver にはある）。
- **学校として導入する窓口が無い**（請求書払い・学校ライセンス）。学校のお金で買うには、個人の買い切りだと「立て替えて経費精算」になる。
- スマホの小さい画面で、数百人の名簿を編集したりクラス間でD&Dしたりするのは操作が難しい。**タブレット・Chromebook を主な対象にし、スマホは「確認と小さな手直し」に絞る**方が現実的かもしれない。
- 日本のPCソフト2本の価格と動作環境が未確認なので、**価格で勝っているかはまだ言えない**。

### 5.3 価格設計

- **個人向けアプリの相場（確認できたもの）**: SeatCharter $0.99、Team Shake $1.99、瞬速！席替え抽選 $4.99、Let's席替え のアプリ内課金 ¥190 / ¥300。
- **学校向けの相場**: Class Maker 年$199〜、Class Creator 最低 年$400（≒$1.60 × 250人）、Class Composer 年$699〜、Mega Seating Plan £5.63〜7.03/教員/年。
- **提案**:
  - Pro の買い切りは **¥610〜¥980 帯**（App Store / Play の価格帯で）。席替えアプリ（¥190〜¥300）より上にしてよい根拠は「年に1度の、何時間もかかる仕事が数分になる」ことで、それでも海外の学校向け（年数万円）より2桁安い。
  - **無料版で xlsx の書き出しを塞ぐのは、使ったあとに初めて課金の壁に当たる形**になる。クラス編成は「結果を持ち出せない＝作業が完了しない」ので、Proへの動機としては強いが、不満のレビューにもなりやすい。
    **Google スプレッドシートへの書き出し、または画面で結果を見る・印刷することは無料でもできる**ようにしておくと、「騙された」という感想を避けられる（Web版は無料で書き出せるので、アプリだけが塞いでいると比べられる点にも注意）。
  - **学校ライセンスは、今すぐは要らない**。ストアの買い切りは個人のアカウントに紐づき、学校の端末に配るには Apple School Manager（数量購入）や Managed Google Play が要る。
    需要（「学校で買いたい」という問い合わせ）が出てから、Web版の Pro（請求書払い）として別に用意する方が軽い。**これは推測で、国内の学校の調達慣行については未確認。**
  - サブスクは避ける。クラス編成は年に1〜2回しか使わないので、月額は割に合わない（InstaGroups はサブスクだが、毎週チーム分けをする体育の先生が対象）。

### 5.4 ASO キーワード候補

**日本語**（アプリ名・サブタイトル・説明文の地の文へ入れる候補）
- 第一候補: `クラス分け` / `クラス替え` / `クラス編成` / `学級編成`
- 周辺: `班分け` / `グループ分け` / `チーム分け` / `名簿` / `男女 均等` / `学力 バランス` / `先生` / `教員` / `小学校` / `中学校` / `校務` / `Excel` / `スプレッドシート` / `Chromebook`
- 席替えは**あえて主軸にしない**（競合が多く、Mosaic の強みが出ない）。ただし将来、席替えモードを足すなら `席替え` は検索量が大きいとみられる（**未確認**）。
- 例（アプリ名の案）: 「Mosaic - クラス分け・班分け自動編成」

**英語**
- 第一候補: `class placement` / `class list` / `class lists` / `class builder` / `student grouping`
- 周辺: `group maker` / `team generator` / `balanced groups` / `random groups` / `teacher tools` / `class roster` / `seating chart`（参考）
- `class creator` / `class solver` は他社の製品名なので**使わない**（商標のトラブルを避けるため）。

---

## 6. 未確認事項（リリース前に手で確かめること）

- [ ] Google Play で「クラス分け」「班分け」「グループ分け」「席替え」を検索し、上位10件のダウンロード数・評価・課金モデル・低評価レビューの内容を記録する（この調査では Play のページを1件も読めていない）。
- [ ] App Store の同じ検索で、上位アプリの価格と低評価レビュー（広告の多さ・人数の上限・データが消えた、などの傾向）を記録する。
- [ ] 学級編成支援プログラム（bizfrsoft）と「学級編成」（教育ソフト 心）の価格・動作環境（Windows / Excel のバージョン）・体験版の有無。
- [ ] EDUCOM マネージャー C4th・スズキ校務・その他の校務支援システム（例: 富士通・NEC・Sky系などの製品。この調査では名前も確認していない）に、自動の学級編成機能があるかどうか。
- [ ] Class Solver の価格（見積もり制とみられる）、Keamk・Sorting Wizard・InstaGroups の価格。
- [ ] 「Class Balancer」「Schoolbox」がクラス編成製品として実在するか。
