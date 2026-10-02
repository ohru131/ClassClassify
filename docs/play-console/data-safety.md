# データセーフティ（Data safety）の回答

Play Console →「アプリのコンテンツ」→「データセーフティ」に入れる値と、その根拠になるコード。
**「実際に端末の外へ送っているもの」だけを申告する**（Google の定義では、端末の中だけで処理するデータは「収集」に当たらない）。
Play の定義上、サードパーティ SDK（ここでは RevenueCat）が送るものも**開発者の収集として申告する**。

> 最終確認: 2026-10-01、`main`（ff598b5 以降）のコードで確認。
> 提出前に (1) RevenueCat の公式ドキュメント（Google Play のデータセーフティの記入例）で SDK の収集項目が変わっていないか、
> (2) リリース AAB のマージ済みマニフェストに広告 ID などの権限が増えていないか、を確かめること（下の「提出前の確認」）。

## 根拠（コードで確かめたこと）

| 事実 | 根拠 |
|---|---|
| 名簿（生徒の名前・特性・ペア指定）と結果は端末内の AsyncStorage にだけ保存する | `mobile/lib/project-store.tsx`（`STORAGE_KEY = 'mosaic.project.v1'`、`AsyncStorage.setItem`） |
| 編成の計算は端末内（JS スレッド上の焼きなまし） | `mobile/lib/runner.ts`、`src/solver/` |
| アプリのコードから外部へ HTTP を送る箇所は無い（`fetch` は Web 版のファイル選択で `blob:` を読むだけ） | `mobile/lib/xlsx-files.web.ts`。`grep -rn "fetch(\|https\?://" mobile/app mobile/components mobile/lib` |
| 外部へ送るのは RevenueCat SDK だけ（`Purchases.configure` → `getCustomerInfo` / `getOfferings` / `purchasePackage` / `restorePurchases`） | `mobile/lib/revenuecat-provider.tsx` |
| 広告・解析・クラッシュ報告の SDK は入っていない | `mobile/package.json`（`react-native-purchases` 以外に通信する SDK が無い） |
| Android の自動バックアップ（Google ドライブ）を無効にしている | `mobile/app.config.ts` の `android.allowBackup: false` |
| 追加の権限は宣言しない。旧来のストレージ権限は外している | `mobile/app.config.ts` の `permissions: []`・`blockedPermissions` |
| 書き出した Excel・PDF は OS の共有シートで利用者が送り先を選ぶ（アプリが送るのではない） | `mobile/lib/xlsx-files.ts`、`mobile/lib/print.ts`、`mobile/lib/app-files.ts` |
| 「この端末の名簿と結果を消去」で保存データと書き出しのキャッシュを消せる | `mobile/lib/project-store.tsx` の `clearProject`、`mobile/app/(tabs)/pro.tsx` |
| アカウント・ログインは無い | アプリ全体（認証の画面・SDK が無い） |
| プライバシーポリシーの本文 | `src/i18n/privacy.ts`（公開ページ `https://ohru131.github.io/ClassClassify/privacy/` も同じ文言から作る） |

## フォームの回答

### 1. データの収集とセキュリティ

| 質問 | 回答 |
|---|---|
| 必要なユーザーデータの種類のいずれかを、アプリで収集または共有していますか？ | **はい**（RevenueCat の購入確認のため。下の2種類だけ） |
| ユーザーデータはすべて転送時に暗号化されていますか？ | **はい**（RevenueCat SDK は HTTPS で通信する） |
| ユーザーがデータの削除をリクエストできる方法を提供していますか？ | **はい**。端末内のデータはアプリ内の「この端末の名簿と結果を消去」またはアンインストールで消える。購入の記録（RevenueCat の匿名 ID と購入履歴）は、問い合わせ先へ依頼すれば開発者が RevenueCat のダッシュボードで顧客を削除できる（ストアの購入記録そのものは Google Play が保持する） |
| アカウントの作成 | **アカウントを作成できない**（ログインも無い。したがって「アカウント削除の URL」も不要） |
| 独立したセキュリティ審査（MASA） | 受けていない（任意） |
| ファミリー ポリシーへの準拠 | 対象外（ターゲットは 13 歳以上。13 歳未満を含めない。`content-rating.md` のターゲット層を参照） |

### 2. 収集するデータの種類（この2つだけにチェック）

| Play の分類 | 何か | 収集 / 共有 | 処理は一時的か | 必須 / 任意 | 目的 |
|---|---|---|---|---|---|
| **金融情報 > 購入履歴** | Pro を買ったかどうか・レシート（RevenueCat が Google Play の購入を検証して保持する） | **収集**のみ（共有はしない） | いいえ | 必須（購入・復元を使う人のみ実際に発生） | **アプリの機能**（購入の確認と復元）、**分析**（RevenueCat が購入データを集計してダッシュボードの売上・チャートに使う） |
| **デバイスまたはその他の ID** | RevenueCat の匿名アプリユーザー ID（`$RCAnonymousID:…`。端末で生成され、名前やメールとは結び付かない） | **収集**のみ（共有はしない） | いいえ | **必須**（SDK の設定キーが入ったビルドでは起動時に `getCustomerInfo` が走るため） | **アプリの機能**（購入状態の照会） |

**購入履歴の目的に「分析」を入れる理由**: RevenueCat の Google Play データセーフティの記入例は、購入履歴の目的を「アプリの機能」と「分析」にしている（RevenueCat は受け取った購入データを開発者向けの売上・チャートの集計に使う）。目的は実態より狭く申告しない方が安全。**この環境からは RevenueCat のドキュメントを開けなかった**ので、提出前に記入例で確かめること。デバイスまたはその他の ID の目的は「アプリの機能」のままでよい（同じ記入例で確かめる）。

**「共有」にしない理由**: Play の定義では、開発者に代わってデータを処理するサービスプロバイダ（RevenueCat）への送信は「共有」に当たらない。

### 3. チェックしないもの（よく迷う項目）

| 項目 | チェックしない理由 |
|---|---|
| 個人情報（名前・メールアドレス・ユーザー ID など） | 生徒の名前は**端末の外へ出さない**（Google の定義で「収集」に当たらない）。教員の名前・メールも尋ねない |
| 健康・フィットネス／その他の機密情報 | 名簿に「支援の必要」などの列を入れられるが、**端末の外へ出さない**ので申告対象ではない |
| ファイルとドキュメント | Excel の読み込みは OS のファイル選択でアプリ内に取り込むだけ（外部へ送らない）。書き出しは利用者が共有シートで送り先を選ぶ（アプリが送信するのではない） |
| アプリのアクティビティ・アプリ情報とパフォーマンス（クラッシュログ・診断） | 解析・クラッシュ報告の SDK を入れていない |
| 位置情報・連絡先・写真・音声・カレンダー・メッセージ | 権限を宣言していない（`android.permissions: []`） |
| 広告 ID | 使わない（広告 SDK なし）。「広告 ID の申告」でも **いいえ** |

## 関連する別の申告

| 申告 | 回答 |
|---|---|
| 広告 | **広告は含まれていない**（`src/i18n/privacy.ts`「広告は表示しません」） |
| 広告 ID（Advertising ID）の使用 | **いいえ**。リリース AAB のマニフェストに `com.google.android.gms.permission.AD_ID` が入っていないことを確認する |
| プライバシーポリシー | `https://ohru131.github.io/ClassClassify/privacy/`（6言語・`?lang=ja` などで切り替え） |

## 提出前の確認（実物で）

1. リリース AAB の権限を実測する: `bundletool dump manifest --bundle app-release.aab | grep uses-permission`
   （または `npx expo prebuild -p android` 後に `cd android && ./gradlew :app:processReleaseManifest` して
   `android/app/build/intermediates/merged_manifest/release/.../AndroidManifest.xml` を見る）。
   期待値は **`INTERNET`・`ACCESS_NETWORK_STATE`・`com.android.vending.BILLING`** 程度（RevenueCat / Play Billing 由来）。
   `AD_ID`・`ACCESS_ADSERVICES_*`・`READ/WRITE_EXTERNAL_STORAGE`・`SYSTEM_ALERT_WINDOW` があれば原因を調べる
   （`android/` は prebuild の生成物なのでコミットしない）。**この環境ではネイティブのビルドをしていないので未確認。**
2. RevenueCat の Google Play データセーフティの案内を開き、SDK のバージョン（`react-native-purchases` ^10.7）で
   収集項目（購入履歴・ID）が増えていないか確かめる。
3. プライバシーポリシー（`src/i18n/privacy.ts`）とこの表の食い違いが無いか読み比べる。どちらかを変えたら両方直す。
