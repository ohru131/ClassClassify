# Mosaic クラス編成（スマホ・タブレット版）

Web 版（リポジトリ直下）と同じソルバー（`../src/solver`）を使う Expo / React Native アプリ。
Web 版は従来どおり無料で、このアプリは無料版（広告あり）＋ Pro（買い切り）で配布する。

| | 無料 | Pro（買い切り） |
| --- | --- | --- |
| 名簿の読み込み（.xlsx・サンプル・新規作成） | ✓ | ✓ |
| 名簿の編集（生徒・項目・重み・同じ組／別の組） | ✓ | ✓ |
| 編成の実行・結果の確認・タップで手動移動 | ✓ | ✓ |
| バナー広告 | 表示 | なし |
| 結果・名簿を Excel（.xlsx）で書き出し → 共有シート | ボタンは表示（押すと Pro 画面へ） | ✓ |

Google 連携（OAuth）は入れていない。書き出した .xlsx は OS の共有シートから Excel・Google ドライブ・メール等へ送る。

## 構成

```
mobile/
  app/                  expo-router の画面（名簿 / 設定・実行 / 結果 / Pro・設定、プライバシーポリシー）
  components/           画面の部品（名簿の表・生徒の編集・項目・ペア指定・広告バナー）
  lib/
    solver.ts           共有ソルバーへの唯一の入口（../../src/solver を re-export）
    runner.ts           時間を区切って焼きなましを進めるランナー（Web Worker の代わり）
    project-store.tsx   名簿・設定・結果の状態と AsyncStorage への保存
    revenuecat-provider.tsx / purchase-offering.ts / purchase-message.ts   課金（買い切りのみ）
    ads-provider.tsx / ads-native-init(.web).ts                          広告
    xlsx-files(.web).ts ファイルの選択と書き出し（ネイティブ: ファイル＋共有シート、Web: ダウンロード）
    samples.generated.ts  public/sample*.xlsx を base64 で埋め込んだもの（npm run samples:generate）
  plugins/              config plugin（AdMob の依存固定・大画面対応・release から不要な権限を外す）
  test/                 vitest（ランナー・保存データ・課金の判定・Excel の往復）
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

## セットアップ

```bash
cd mobile
npm install
npm test            # vitest
npm run check       # tsc --noEmit
npx expo export --platform web   # Web で動作確認用に書き出す（広告・課金は無効）
```

課金（react-native-purchases）と広告（react-native-google-mobile-ads）はネイティブモジュールなので
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
| `EXPO_PUBLIC_ADMOB_ANDROID_APP_ID` | AdMob の Android App ID（`ca-app-pub-…~…`） | Google のテスト App ID。**EAS production ビルドは失敗させる** |
| `EXPO_PUBLIC_ADMOB_IOS_APP_ID` | AdMob の iOS App ID | 同上（iOS ビルド時） |
| `EXPO_PUBLIC_ADMOB_BANNER_UNIT_ID` | バナー広告ユニット ID（`ca-app-pub-…/…`） | Google のテスト広告ユニット |

## EAS ビルド

1. `npx eas login` → `npx eas init`（`app.config.ts` に `extra.eas.projectId` が追加される）
2. 上の環境変数を EAS に登録
3. `npx eas build --profile production --platform android`（`autoIncrement` で versionCode を EAS が採番）
4. `npx eas submit --platform android`、または AAB を Play Console に手動でアップロード

`eas.json` の `appVersionSource: "remote"` は、EAS 側のカウンタが未初期化だと versionCode 1 から始まる。
既存の Play アプリへ上書きする場合は、先に `eas build:version:set` で合わせる。
**アプリ ID（`com.ohru131.mosaic`）は公開後に変えないこと。**

## RevenueCat の設定（買い切りのみ）

1. Play Console で **アプリ内アイテム（非消費型・定期購入ではない）** を作る（例: `mosaic_pro`）。iOS も出すなら App Store Connect で「非消費型」。
2. RevenueCat でプロジェクト・アプリを作り、その商品を登録する。
3. **Entitlement `pro`** を作り、商品を紐付ける（紐付け忘れると、支払っても Pro にならない。アプリはその場合「復元」とサポートへ案内する）。
4. Offering（current）に **Lifetime パッケージ**としてその商品を入れる。

アプリ側の不変条件（既存アプリ UnitCalc と同じ）:

- **サブスクリプションは絶対に売らない。** `lib/purchase-offering.ts` が `SUBSCRIPTION` と `subscriptionPeriod` を持つ商品を、lifetime 枠に入っていても弾く（`test/purchase-offering.test.ts`）。
- **RevenueCatUI のペイウォールは使わない**（offering に残ったサブスクをそのまま売ってしまう）。買い切り商品が取れなければ購入させず理由を出す。
- ユーザーのキャンセル（`userCancelled`）はエラーとして扱わない。
- 購入・復元は同期フラグ（`purchaseLockRef`）で直列化する（二重タップで二重に課金 API を叩かない）。
- `purchasePackage()` の後に `pro` entitlement を確認できたときだけ「ありがとうございます」を出す。
- SDK キー未設定・Web では購入・復元を受け付けず、理由を出す。
- 審査員向けには Play のプロモコード（ライセンステスター）を使う（買い切りに無料トライアルは無い）。

## 広告（AdMob）

- 無料版だけ、画面の最上部にバナーを出す（操作ボタンから離すため）。Web では出さない。
- Pro の人には同意フォームの表示も SDK の初期化も行わない。
- EEA などでは UMP の同意フォームを出してから広告をリクエストする（AdMob 側で「プライバシーとメッセージ」の GDPR メッセージを作成・公開しておく）。
- `app.config.ts` の `userTrackingUsageDescription` は iOS の ATT ダイアログの文言。

## 大画面・Chromebook 対応

学校のタブレットと Chromebook（Play ストアの Android アプリとして動作）で使う前提。

- **向きを固定しない**（`orientation: "default"`）、`ios.supportsTablet: true`。
- **リサイズ可能なウィンドウを妨げない**: `resizeableActivity` を false にしない（`plugins/withLargeScreenSupport.js` が万一 false なら外す）。Chromebook のフリーフォーム窓・分割画面で使える。
- **タッチパネル必須にしない**: `android.hardware.touchscreen` を `required="false"` で宣言する（宣言しないと Play がタッチの無い Chromebook に配信しない）。
- **レイアウトは幅で出し分ける**（`lib/layout.ts`、幅 768dp 以上で大画面）。`useWindowDimensions` を使うのでウィンドウのリサイズ・回転に追従する。
  - 名簿: 大画面は**表形式（横スクロール可）＋右側に編集欄**、スマホはカードの一覧＋編集はモーダル。
  - 結果: 大画面は**クラスを 2〜4 列のグリッド**で全部並べる、スマホは組を切り替えて1組ずつ。
  - 設定: 大画面はクラス数・最大人数・探索時間を横に並べる。
- **マウス・キーボードで完結する**: 長押し・ドラッグ・スワイプ前提の操作は無い（生徒の移動も「タップ → 移動先の組を押す」）。ボタンはホバー・フォーカスで見た目が変わり、テキスト入力は Enter（物理キーボード）で確定する。
- 追加の権限は宣言しない（`android.permissions: []`）。expo-file-system が足す旧来のストレージ権限は `blockedPermissions` で外している（取り込みは OS のファイル選択、書き出しはキャッシュ＋共有シートで権限が要らない）。

**Play Console では「タブレット」（7インチ・10インチ）と「Chromebook」向けのスクリーンショットを別に登録する**と、大画面向けの掲載として表示される（1280×800 や 800×1280 の画面で撮る）。

## データとプライバシー

- 名簿・設定・結果は AsyncStorage（端末内）にだけ保存する。外部へは送らない。「Pro・設定」から消去できる。
- RevenueCat は購入確認のため端末生成の匿名 ID と購入・レシート情報を、AdMob は広告配信・計測のため端末 ID・広告 ID を受け取る（`app/privacy.tsx`）。**「個人情報を一切集めていない」とは書かないこと。** Play Console のデータセーフティもこれに合わせて申告する。
- ストアの掲載には、この画面と同じ内容のプライバシーポリシーを Web で公開した URL が要る。

## 検証状況

確認済み（この環境）:

- `npm test`（vitest 36件）: ランナーが3つのサンプルで条件違反0・人数差1以内・全項目が理想範囲内に到達すること、中止、UI へ返した時間を探索時間に数えないこと、保存データの検証、課金の判定、Excel（base64 経路）の往復。
- `npx tsc --noEmit`、`npx expo export --platform web`。
- Web 書き出しを Playwright で 390×844・1280×800・800×1280 で操作: サンプル読み込み → 生徒編集 → 実行（進捗バーが進む）→ 結果（人数差0・バランス完全・違反0）→ 手動移動で即時再計算 → Pro 限定ボタンで Pro 画面へ。名前の Enter 確定・項目の追加・同じ組の作成・再読み込み後の復元・ウィンドウ幅を狭めたときのレイアウト切り替え。

**実機では未検証**（ネイティブのビルドはこの環境で行っていない）:

- Hermes 上のソルバーの速度。V8（Web）より大幅に遅い見込みで、同じ探索時間でも反復回数は少ない。サンプル規模（80名）で違反0に届くか、240名規模で十分かを実機で確認すること。足りなければ `lib/runner.ts` のスタート回数を減らす（1回あたりを長くする）のが最初の調整点。
- 計算中の UI の滑らかさ（`sliceMs` 24ms ごとに制御を返す）。
- `expo-document-picker` で .xlsx が選べるか（端末によって MIME が違う。`application/octet-stream` も受け付けている）、共有シートから Excel・Google ドライブ・Gmail へ .xlsx が渡るか。
- 購入・復元・Pro の反映、広告の表示と同意フォーム（本番の RevenueCat・AdMob 設定が必要）。
- Chromebook（タッチなし機を含む）・Android タブレットでの表示、フリーフォーム窓のリサイズ、物理キーボードでの Tab 移動・Enter 確定、戻るキー。
- iOS 版（`supportsTablet` 含む）は一度もビルドしていない。
