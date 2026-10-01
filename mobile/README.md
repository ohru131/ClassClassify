# Mosaic クラス編成（スマホ・タブレット版）

Web 版（リポジトリ直下）と同じソルバー（`../src/solver`）を使う Expo / React Native アプリ。
Web 版は従来どおり無料。このアプリは**広告なしの無料版＋ Pro（買い切り）**で配布し、収益は買い切り一本にする（広告・サブスクは無い）。

| | 無料 | Pro（買い切り） |
| --- | --- | --- |
| 名簿の読み込み（.xlsx・サンプル・新規作成） | ✓ | ✓ |
| 名簿の編集（生徒・項目・重み・同じ組／別の組） | ✓ | ✓ |
| 編成の実行・結果の確認・タップで手動移動 | ✓ | ✓ |
| 広告 | なし | なし |
| 結果・名簿を Excel（.xlsx）で書き出し → 共有シート（Excel・Google ドライブ・メール等） | ボタンは表示（押すと Pro 画面へ） | ✓ |
| 結果の印刷・PDF（A4 縦: クラス別名簿＋ペア指定の色分けと凡例＋集計・バランス表＋作成日） | ボタンは表示（押すと Pro 画面へ） | ✓ |

**価格は ¥980 / US$5.99 を基準に国別に決めてある**（`../docs/play-console/pricing.md`・`pricing.csv`。根拠は `docs/research/competitors.md` 第5節・`overseas-demand.md`）。価格はストア側で設定し、アプリはストアのローカライズ済みの価格文字列をそのまま表示する（コードでは決めない）。

**Google Play への提出**は `../docs/play-console/submission-checklist.md` の順に進める（データセーフティ・レーティング・審査員向けの手順・国別価格も同じフォルダ）。ストア用の画像は `../submission-assets/`。

Google 連携（OAuth）は入れていない。書き出した .xlsx は OS の共有シートから Excel・Google ドライブ・メール等へ送る。

## 構成

```
mobile/
  app/                  expo-router の画面（名簿 / 設定・実行 / 結果 / Pro・設定、プライバシーポリシー）
  components/           画面の部品（名簿の表・生徒の編集・項目・ペア指定）
  lib/
    solver.ts           共有ソルバーへの唯一の入口（../../src/solver を re-export）
    runner.ts           時間を区切って焼きなましを進めるランナー（Web Worker の代わり）
    project-store.tsx   名簿・設定・結果の状態と AsyncStorage への保存
    revenuecat-provider.tsx / purchase-offering.ts / purchase-message.ts   課金（買い切りのみ）
    print-html.ts       印刷・PDF 用の結果の HTML（純関数・テストあり。利用者の入力は必ずエスケープ）
    print(.web).ts      ネイティブ: expo-print（printAsync / printToFileAsync → 共有）、Web: 新しいウィンドウで window.print()
    pro-preview.ts      Web 限定の Pro 表示プレビュー（?pro=preview。スクリーンショット用、ネイティブでは無効）
    xlsx-files(.web).ts ファイルの選択と書き出し（ネイティブ: ファイル＋共有シート、Web: ダウンロード）
    samples.generated.ts  public/sample*.xlsx を base64 で埋め込んだもの（npm run samples:generate）
  plugins/              config plugin（大画面対応・release から不要な権限を外す）
  test/                 vitest（ランナー・保存データ・課金の判定・Excel の往復・印刷用 HTML）
```

### ソルバーの共有

- `metro.config.js` が `../src/solver` を `watchFolders` に入れ、共有コードが import する `xlsx-js-style` を
  **mobile/node_modules から**解決する（EAS ビルドではリポジトリ直下の node_modules が無いため）。
  旧 .xls 用の文字コード表（`cpexcel.js`）は Web 版と同じくスタブに差し替える。
- `tsconfig.json` も `xlsx-js-style` の型を mobile/node_modules に向けている。
- React Native には Web Worker が無いので、`createAnnealer(...).run(sliceMs)`（`src/solver/anneal.ts`）で
  数十ミリ秒ずつ計算しては UI に制御を返す（`lib/runner.ts`）。探索時間はこの計算時間だけで消費し、
  マルチスタートは順番に行う（10秒なら2回、30秒なら3回）。進捗バーと中止ボタンが効く。
- React Native の `Blob` は ArrayBuffer から作れないので、Excel は `writeXlsx(wb, 'base64')` で
  キャッシュに書いてから `expo-sharing` で共有する。

## 多言語対応（ja / en / ko / es / de / pt-BR）

- **`APP_LANGUAGES`（`src/i18n/languages.ts`、`lib/i18n.ts` から re-export）が唯一の情報源**（型・端末ロケールの判定・言語名＝endonym・Intl のロケール）。既定は端末の言語（`expo-localization`）で、未対応の言語は英語。ポルトガル語はブラジル以外も pt-BR。「Pro・設定」で言語を選ぶと端末に保存する。
- UI 文言は `lib/copy/en.ts`（`EN_COPY` のキー集合が正）と各言語のファイル。`Record<CopyKey, string>` なので**キーが欠けると型エラー**になり、`test/i18n.test.ts` が空文字と埋め込み（`{n}` など）の食い違い、日本語の混入を検出する。プライバシーポリシーは `lib/copy/privacy.ts`。
- **訳語は `docs/i18n-glossary.md` に揃える**（先に用語集を直してから各所を直す）。スペイン語は中南米の語彙（grupo・estudiantes・tú）を基本にしている。
- Excel の語彙（シート名・見出し・設定の項目名・結果の見出し）は共有ソルバーの `src/solver/labels.ts`。**読み込みは全言語のシート名・見出しを受け付け**（日本語を最優先に探すので Web 版の挙動は変わらない）、書き出しは選択中の言語で出す。ひな形も選択中の言語で生成して共有できる（無料）。
- 日付・数値は `Intl`（`lib/locale-format.ts`）。印刷用 HTML の `lang` 属性もロケールに合わせる。
- ホーム画面のアプリ名は `locales/*.json`（`app.config.ts` の `locales`）で言語ごとに出し分ける。
- サンプル名簿は Web 版と共通（リポジトリ直下で `npm run samples:generate` → `public/samples/<lang>/`）。国ごとにその国の学校で配慮される項目で作ってあり、スマホ版は `npm run samples:embed`（`scripts/embed-samples.mjs`）でそれを埋め込む。各言語のサンプルが違反0・全項目が理想範囲内に届くことをテストで確認している。
- 言語の定義・判定（`APP_LANGUAGES` など）はリポジトリ直下の `src/i18n/languages.ts` を Web 版と共有している（`lib/i18n.ts` は re-export。Metro の `watchFolders` に `src/i18n` を追加済み）。
- ストア掲載文は `docs/store-listing.md`（Play の en-US / en-AU / en-GB / ko-KR / es-419 / es-ES / de-DE / pt-BR / ja-JP）。

## セットアップ

```bash
cd mobile
npm install
npm test            # vitest
npm run check       # tsc --noEmit
npx expo export --platform web   # Web で動作確認用に書き出す（課金は無効。?pro=preview で Pro の画面を確認できる）
```

課金（react-native-purchases）と印刷（expo-print）はネイティブモジュールなので
**Expo Go では動かない**。実機では開発ビルドを使う:

```bash
npx eas build --profile development --platform android   # または expo run:android（要 Android SDK）
npx expo start --dev-client
```

`npx expo prebuild` で生成される `android/` `ios/` はコミットしない（.gitignore 済み。EAS がビルド時に生成する）。

## 環境変数

`.env`（コミットしない）か EAS の環境変数（`eas env:create`）に設定する。

| 変数 | 内容 | 未設定のとき |
| --- | --- | --- |
| `EXPO_PUBLIC_REVENUECAT_ANDROID_API_KEY` | RevenueCat の Android 公開 SDK キー（`goog_…`） | 購入・復元ボタンは「キーが設定されていません」と出す |
| `EXPO_PUBLIC_REVENUECAT_IOS_API_KEY` | RevenueCat の iOS 公開 SDK キー（`appl_…`） | 同上 |

## EAS ビルド

1. `npx eas login` → `npx eas init`（`app.config.ts` に `extra.eas.projectId` が追加される）
2. 上の環境変数を EAS に登録
3. `npx eas build --profile production --platform android`（`autoIncrement` で versionCode を EAS が採番）
4. `npx eas submit --platform android`、または AAB を Play Console に手動でアップロード

`eas.json` の `appVersionSource: "remote"` は、EAS 側のカウンタが未初期化だと versionCode 1 から始まる。
既存の Play アプリへ上書きする場合は、先に `eas build:version:set` で合わせる。
**アプリ ID（`com.ohru131.mosaic`）は公開後に変えないこと。**

## ローカルビルド（gradle）

EAS を使わず手元でビルドする手順（既存アプリ UnitCalc と同じ構成）。要 JDK 17・Android SDK。

```bash
npx expo prebuild -p android --no-install          # android/ を作り直す（コミットしない）
cd android && ./gradlew.bat installDebug --console=plain    # 初回は約15分
```

- **debug は `com.ohru131.mosaic.debug`（ホーム画面の名前は「Mosaic dev」）の別アプリとして入る**（`plugins/withDebugPackageSuffix.js`）。Play 版と署名が違っても上書きにならないので、Play 版の名簿を消さずに試せる。引き換えに debug では課金を試せない（RevenueCat の商品はパッケージ名に紐づく）。
- 実機の ABI だけビルドすれば速い: `./gradlew.bat installDebug -PreactNativeArchitectures=arm64-v8a`。
- JS だけの変更なら debug APK の作り直しは要らない（dev-client が Metro から読む）。作り直すのは依存・`app.config.ts`・`plugins/` が変わったときだけ。
- Metro は `npx expo start --dev-client --port 8082`（この開発機は 8081 を別プロセスが占有している）→ `adb reverse tcp:8081 tcp:8082` と `adb reverse tcp:8082 tcp:8082` → `adb shell am start -a android.intent.action.VIEW -d "'mosaic://expo-development-client/?url=http%3A%2F%2Flocalhost%3A8082'" com.ohru131.mosaic.debug`。
- `expo start` が `tsconfig.json` を整形し直す（`expo-env.d.ts` が include から消える）ことがある。コミットに混ぜない。
- **release の署名**: `credentials.json`（`android.keystore` に `keystorePath` / `keystorePassword` / `keyAlias` / `keyPassword`）を置くと、`plugins/withLocalReleaseSigning.js` が prebuild のたびに `android/keystore.properties` を作り、`bundleRelease` を upload key で署名する。無ければ debug 鍵で署名される（Play に弾かれる）。`credentials.json`・`credentials/`・`keystore.properties` は .gitignore 済み。**絶対にコミットしない。**
- バージョンは `app.config.ts` の `version` と `android.versionCode`（Play で未使用の値）を上げてから prebuild する。
- release AAB は `cd android && ./gradlew.bat bundleRelease --console=plain`（約15分）→ `android/app/build/outputs/bundle/release/app-release.aab`。`keytool -printcert -jarfile <aab>` の SHA1 が upload key（EAS の Build Credentials・Default）と一致することを確かめる。
- **`react-native-gesture-handler` / `react-native-reanimated` / `react-native-worklets` は Expo SDK の推奨版を直接の依存に置く**（`npx expo install` で入れる）。直接の依存に無いと expo-router 経由で新しすぎる版が入り、gesture-handler 3.x では release の C++ 中間ファイルのパスが Windows の 260 文字上限を超えて `Filename longer than 260 characters` で落ちる（debug は中間フォルダ名が短いので通ってしまう）。worklets も expo-modules-core の対応範囲外になる。
- **ネイティブの依存の版を変えたら、`node_modules/*/android/.cxx` と `node_modules/*/android/build` を消してからビルドする。** prebuild は `android/` しか作り直さないので、ライブラリ側に古い CMake のキャッシュが残り、`libworklets.so ... missing and no known rule to make it` のように消えた出力先を参照して落ちる。

## RevenueCat の設定（買い切りのみ）

1. Play Console で **一回限りの商品（定期購入ではない）** を作る（例: `mosaic_pro`）。iOS も出すなら App Store Connect で「非消費型」。
2. RevenueCat でプロジェクト・アプリを作り、その商品を登録する。
3. **Entitlement `pro`** を作り、商品を紐付ける（紐付け忘れると、支払っても Pro にならない。アプリはその場合「復元」とサポートへ案内する）。
4. **RevenueCat のダッシュボードでその商品を Non-consumable（非消費型）に設定する（必須）。** Google Play の一回限りの商品を
   consume するかどうかは RevenueCat の商品設定で決まり（SDK は RevenueCat が返す `should_consume` に従う）、既定は消費型。
   設定しないと Pro が consume され、復元できず再購入できてしまう。**Lifetime パッケージに入れるだけでは非消費型にならない。**
5. Offering（current）に **Lifetime パッケージ**としてその商品を入れる。
6. 内部テストで購入 → 入れ直して「購入を復元」で Pro が戻ることを確かめる（`../docs/play-console/submission-checklist.md` 5節）。

アプリ側の不変条件（既存アプリ UnitCalc と同じ）:

- **サブスクリプションは絶対に売らない。** `lib/purchase-offering.ts` が `SUBSCRIPTION` と `subscriptionPeriod` を持つ商品を、lifetime 枠に入っていても弾く（`test/purchase-offering.test.ts`）。
- **RevenueCatUI のペイウォールは使わない**（offering に残ったサブスクをそのまま売ってしまう）。買い切り商品が取れなければ購入させず理由を出す。
- ユーザーのキャンセル（`userCancelled`）はエラーとして扱わない。
- 購入・復元は同期フラグ（`purchaseLockRef`）で直列化する（二重タップで二重に課金 API を叩かない）。
- `purchasePackage()` の後に `pro` entitlement を確認できたときだけ「ありがとうございます」を出す。
- SDK キー未設定・Web では購入・復元を受け付けず、理由を出す。
- 審査員向けには Play のプロモコード（ライセンステスター）を使う（買い切りに無料トライアルは無い）。

## 印刷・PDF（Pro）

- HTML は `lib/print-html.ts` の純関数 `buildResultPrintHtml` で組む（`@page { size: A4 portrait }`、クラスのカードは `break-inside: avoid` で途中で切らず、集計・バランスは改ページして次のページから）。
- ネイティブ: 「印刷」は `Print.printAsync`、「PDF で共有」は `Print.printToFileAsync` → `クラス編成結果_日付.pdf` に名前を変えて `expo-sharing`。
- Web: `expo-print` の Web 実装は渡した HTML ではなく今の画面を印刷してしまうので使わず、新しいウィンドウに HTML を書いて `window.print()`（PDF はブラウザの印刷画面から保存）。

## 大画面・Chromebook 対応

学校のタブレットと Chromebook（Play ストアの Android アプリとして動作）で使う前提。

- **向きを固定しない**（`orientation: "default"`）、`ios.supportsTablet: true`。
- **リサイズ可能なウィンドウを妨げない**: `resizeableActivity` を false にしない（`plugins/withLargeScreenSupport.js` が万一 false なら外す）。Chromebook のフリーフォーム窓・分割画面で使える。
- **タッチパネル必須にしない**: `android.hardware.touchscreen` を `required="false"` で宣言する（宣言しないと Play がタッチの無い Chromebook に配信しない）。
- **レイアウトは幅で出し分ける**（`lib/layout.ts`、幅 768dp 以上で大画面）。`useWindowDimensions` を使うのでウィンドウのリサイズ・回転に追従する。
  - 名簿: 大画面は**表形式（横スクロール可）＋右側に編集欄**、スマホはカードの一覧＋編集はモーダル。
  - 結果: 大画面は**クラスを 2〜4 列のグリッド**で全部並べる、スマホは**1組ずつ横に並べて左右のスワイプ**（上のタブ・前後の矢印でも移れる。横向き `FlatList` の `pagingEnabled`、ページ位置の計算は `lib/class-pager.ts`）。
  - 設定: 大画面はクラス数・最大人数・探索時間を横に並べる。
- **マウス・キーボードで完結する**: 長押し・ドラッグ・スワイプ前提の操作は無い（生徒の移動も「タップ → 移動先の組を押す」）。ボタンはホバー・フォーカスで見た目が変わり、テキスト入力は Enter（物理キーボード）で確定する。
- 追加の権限は宣言しない（`android.permissions: []`）。expo-file-system が足す旧来のストレージ権限は `blockedPermissions` で外している（取り込みは OS のファイル選択、書き出しはキャッシュ＋共有シートで権限が要らない）。広告 SDK を入れていないので `AD_ID`・`ACCESS_ADSERVICES_*` などの広告系の権限も載らない。

**Play Console では「タブレット」（7インチ・10インチ）と「Chromebook」向けのスクリーンショットを別に登録する**と、大画面向けの掲載として表示される（タブレット・Chromebook の画像は 16:9 か 9:16。撮影は `../submission-assets/README.md`）。

## データとプライバシー

- 名簿・設定・結果は AsyncStorage（端末内）にだけ保存し、端末の外へは送らない。「Pro・設定」から消去できる（書き出した Excel・PDF とファイル選択時のコピーもキャッシュから消す）。
- **Android はアプリのバックアップを無効にしている**（`android.allowBackup: false`）。有効のままだと AsyncStorage の名簿が Google ドライブへの自動バックアップに含まれる。iOS は、利用者が管理する端末自体のバックアップ（iCloud・パソコン）に含まれうる（プライバシーポリシーにもそう書いてある）。
- 外部へ送るのは **RevenueCat（購入検証のための端末生成の匿名 ID とレシート）だけ**。広告・解析 SDK は入れていない（`app/privacy.tsx`）。
- ストアの掲載に使うプライバシーポリシーの URL は **https://ohru131.github.io/ClassClassify/privacy/**。この画面と同じ `lib/copy/privacy.ts` から Web 版のビルドで作る（`../privacy/index.html`・`../src/privacy/render.ts`）ので、**本文を変えたらアプリと公開ページの両方が変わる**（公開ページは main へのマージでデプロイされる）。

### Play Console のデータセーフティ（申告の目安）

回答の全文と根拠のコードは `../docs/play-console/data-safety.md`。要点:

- 「データの収集」: **購入履歴**（アプリの機能＝購入の確認のため、RevenueCat 経由）と、**アプリ情報と動作 > その他の ID**（RevenueCat の匿名のアプリユーザー ID）。どちらも「任意ではない（購入時）」「暗号化して送信」。
- 名簿（生徒の名前・特性）は端末外へ送らないので「収集」に当たらない。「共有」（第三者への提供）は無し。Android の自動バックアップも無効（`allowBackup: false`）なので、名簿が Google のバックアップ経由で端末外へ出ることもない。
- 広告 ID は使わない（「広告 ID を使用しますか」→ いいえ）。広告は表示しない。
- データ削除: アプリ内の「この端末の名簿と結果を消去」またはアンインストール。
- 申告前に RevenueCat のドキュメント（データセーフティ／App Privacy の案内）で、SDK の収集項目が変わっていないか確認すること。

## 検証状況

確認済み（この環境）:

- `npm test`（vitest 92件）: ランナーが3つのサンプルで条件違反0・人数差1以内・全項目が理想範囲内に到達すること、中止、UI へ返した時間を探索時間に数えないこと、保存データの検証、課金の判定、Excel（base64 経路）の往復、印刷用 HTML（エスケープ・A4 縦・各組・凡例・改ページ・各言語）、Web 限定の Pro プレビュー、多言語（端末の言語の判定・全言語の文言・各言語のサンプルが違反0・言語別の Excel の往復・読み込みエラーの言語）。
- `npx tsc --noEmit`、`npx expo export --platform web`。
- Web 書き出しを Playwright で 390×844・1280×800・800×1280 で操作: サンプル読み込み → 生徒編集 → 実行（進捗バーが進む）→ 結果（人数差0・バランス完全・違反0）→ 手動移動で即時再計算 → 無料時に Excel・印刷ボタンが Pro 画面へ案内する。`?pro=preview` で印刷用ウィンドウ（A4 の HTML）が開く。名前の Enter 確定・項目の追加・同じ組の作成・再読み込み後の復元・ウィンドウ幅を狭めたときのレイアウト切り替え。

- スマホ幅の結果画面のページ送り（390×844）: 組のタブのタップ・前後の矢印（端で無効）・タッチの左右ドラッグ・横ホイールで隣の組へ移り、タブとドット・「2 / 4」が追従する。縦のタッチスクロールでは組が変わらない。生徒を別の組へ移しても今の組に留まり、幅を 600 → 390 に変えても同じ組の位置に戻る。
- 6言語 × 390×844・1280×800 で サンプル → 実行 → 結果 → バランス → Pro 画面、印刷用ウィンドウを Playwright で撮影し、ボタン・タブの文字のはみ出しが無いことを確認（独・西の長い語はセグメントを2行まで折り返す）。

実機（moto g52j 5G・Android。debug ビルド＋ Metro と、upload key で署名した release APK）で確認済み:

- 起動 → サンプル（80名・4組）の読み込み → 実行（標準10秒）→ 結果で人数差0・バランス完全・条件違反0。Pro 画面の表示（SDK キー未設定の案内）。
- **探索時間が実時間に対して大幅に延びていた**のを直した。24ms 計算するたびに UI へ返していたが、UI 側（進捗の再描画）の時間は探索時間に数えないため、「10秒」で実時間が約95秒かかっていた。`lib/runner.ts` が UI に返した時間を測り、計算が実時間の8割を占めるよう1回の計算時間を 24〜200ms で伸ばす。進捗は1%刻みに丸めて、値が変わらない再描画を省く。debug ビルドで約26秒、**release ビルドで約13秒**（10秒で76%・12秒で90%・14秒で結果画面。結果は人数差0・バランス完全・違反0）。
- 英数字がセリフ体に見えるのは端末のシステムフォント設定（この端末は Roboto Slab）に従っているため。アプリはフォントを指定していない。

**実機では未検証**:

- 240名規模で違反0に届くか（release の実時間は80名で確認済み）。
- 結果画面の左右スワイプ（Android の横向き `FlatList` の `pagingEnabled` と、画面全体の縦スクロールとの振り分け）。
- `expo-document-picker` で .xlsx が選べるか（端末によって MIME が違う。`application/octet-stream` も受け付けている）、共有シートから Excel・Google ドライブ・Gmail へ .xlsx が渡るか。
- 購入・復元・Pro の反映（本番の RevenueCat 設定が必要）。
- `expo-print` の印刷画面・PDF の見た目（Android の WebView / iOS の WKWebView で組版がブラウザと違うことがある。日本語フォント、改ページ、背景色の印刷）。
- Chromebook（タッチなし機を含む）・Android タブレットでの表示、フリーフォーム窓のリサイズ、物理キーボードでの Tab 移動・Enter 確定、戻るキー。
- iOS 版（`supportsTablet` 含む）は一度もビルドしていない。
- Hermes の `Intl`（日付・数値の書式）と、ホーム画面のアプリ名の言語別表示（`locales`）。
- 翻訳はネイティブスピーカーの確認を受けていない（`docs/i18n-glossary.md` 第4節に確認してほしい箇所をまとめた）。
