# データセーフティ（Data safety）の回答

Play Console →「アプリのコンテンツ」→「データセーフティ」に入れる値と、その根拠になるコード。
**「実際に端末の外へ送っているもの」だけを申告する**（Google の定義では、端末の中だけで処理するデータは「収集」に当たらない）。
Play の定義上、サードパーティ SDK が送るものも開発者の収集として申告するが、**FairClass は課金に Google Play Billing を直接使い（`expo-iap`）、
外部の課金サービスや解析の SDK を入れていない**ので、アプリから端末の外へ送るデータは無い。

> 最終確認: 2026-10-05、Google Play Billing を直接使う版（v1.4.0 の次）のコードで確認。
> 提出前に、リリース AAB のマージ済みマニフェストに広告 ID などの権限が増えていないかを確かめること（下の「提出前の確認」）。

## 根拠（コードで確かめたこと）

| 事実 | 根拠 |
|---|---|
| 名簿（生徒の名前・特性・ペア指定）と結果は端末内の AsyncStorage にだけ保存する | `mobile/lib/project-store.tsx`（`STORAGE_KEY = 'mosaic.project.v1'`、`AsyncStorage.setItem`） |
| 編成の計算は端末内（JS スレッド上の焼きなまし） | `mobile/lib/runner.ts`、`src/solver/` |
| アプリのコードから外部へ HTTP を送る箇所は無い（`fetch` は Web 版のファイル選択で `blob:` を読むだけ） | `mobile/lib/xlsx-files.web.ts`。`grep -rn "fetch(\|https\?://" mobile/app mobile/components mobile/lib` |
| 課金は Google Play Billing を直接使う（`expo-iap`）。購入の状態は端末の Play ストアに問い合わせるだけ（`initConnection` → `getAvailablePurchases` / `fetchProducts` / `requestPurchase` / `finishTransaction`）。購入の情報・識別子をアプリから開発者のサーバーや第三者へ送らない | `mobile/lib/pro-provider.tsx`、`mobile/lib/play-billing.ts` |
| 広告・解析・クラッシュ報告・外部の課金サービスの SDK は入っていない | `mobile/package.json`（通信するネイティブ依存は Google Play の一部である Play Billing＝`expo-iap` だけ） |
| Android の自動バックアップ（Google ドライブ）を無効にしている | `mobile/app.config.ts` の `android.allowBackup: false` |
| 追加の権限は宣言しない。旧来のストレージ権限は外している | `mobile/app.config.ts` の `permissions: []`・`blockedPermissions` |
| 書き出した Excel・PDF は OS の共有シートで利用者が送り先を選ぶ（アプリが送るのではない） | `mobile/lib/xlsx-files.ts`、`mobile/lib/print.ts`、`mobile/lib/app-files.ts` |
| 「この端末の名簿と結果を消去」で保存データと書き出しのキャッシュを消せる | `mobile/lib/project-store.tsx` の `clearProject`、`mobile/app/(tabs)/pro.tsx` |
| アカウント・ログインは無い | アプリ全体（認証の画面・SDK が無い） |
| プライバシーポリシーの本文 | `src/i18n/privacy.ts`（公開ページ `https://ohru131.github.io/ClassClassify/privacy/` も同じ文言から作る） |

**決済のデータの扱い**: 支払いは Google Play が処理し、購入の記録も Google Play が持つ（Google のプライバシーポリシーに従う）。
アプリは端末の Play ストアから「`fairclass_pro` を買ったか」を読むだけで、その結果を端末の外へ送らない。
Play Billing は Google Play の一部で、そこでの決済の処理は開発者の「収集」として申告するものではない【要確認：Play Console のヘルプの最新の記述】。

## フォームの回答

### 1. データの収集とセキュリティ

| 質問 | 回答 |
|---|---|
| 必要なユーザーデータの種類のいずれかを、アプリで収集または共有していますか？ | **いいえ**（端末の外へ送るデータが無い） |
| ユーザーデータはすべて転送時に暗号化されていますか？ | 「いいえ」を選ぶと聞かれない（送るデータが無い） |
| ユーザーがデータの削除をリクエストできる方法を提供していますか？ | 「いいえ」を選ぶと聞かれない。端末内のデータはアプリ内の「この端末の名簿と結果を消去」またはアンインストールで消える |
| アカウントの作成 | **アカウントを作成できない**（ログインも無い。したがって「アカウント削除の URL」も不要） |
| 独立したセキュリティ審査（MASA） | 受けていない（任意） |
| ファミリー ポリシーへの準拠 | 対象外（ターゲットは 13 歳以上。13 歳未満を含めない。`content-rating.md` のターゲット層を参照） |

「いいえ」にすると、ストアのページには「データは収集されません」「第三者と共有されるデータはありません」と出る。

### 2. 収集するデータの種類

**チェックしない**（上で「いいえ」を選ぶので、この画面は出ない）。

### 3. チェックしないもの（よく迷う項目）

| 項目 | チェックしない理由 |
|---|---|
| 金融情報 > 購入履歴 | 購入の記録は Google Play が持ち、アプリは端末の Play ストアから読むだけで外へ送らない |
| デバイスまたはその他の ID | アプリは利用者の識別子を生成も送信もしない |
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
   2026-10-05 の release ビルド（expo-iap 5.8.2・openiap-google 3.6.2・Play Billing Library 8.3.0）の実測:
   | 権限 | 由来 |
   | --- | --- |
   | `com.android.vending.BILLING` | expo-iap / openiap-google（Play Billing） |
   | `android.permission.ACCESS_NETWORK_STATE` | Play Billing Library が依存する `com.google.android.datatransport`（Google の部品） |
   | `android.permission.INTERNET` | React Native のテンプレート・expo-file-system |
   | `android.permission.VIBRATE` | Expo のテンプレート |
   | `com.ohru131.mosaic.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION` | AndroidX（アプリ内だけの受信の保護。利用者に見える権限ではない） |

   `AD_ID`・`ACCESS_ADSERVICES_*`・`READ/WRITE_EXTERNAL_STORAGE`・`SYSTEM_ALERT_WINDOW` があれば原因を調べる
   （`android/` は prebuild の生成物なのでコミットしない）。
2. `mobile/package.json` に通信する SDK（解析・クラッシュ報告・外部の課金サービスなど）が増えていないか確かめる。増えたらこの表を「収集あり」で作り直す。
3. プライバシーポリシー（`src/i18n/privacy.ts`）とこの表の食い違いが無いか読み比べる。どちらかを変えたら両方直す。
4. 以前に「収集あり（購入履歴・デバイスまたはその他の ID）」で申告済みなら、この版を出すときにフォームを「いいえ」に更新する
   （申告と実態が違うと審査で指摘される。実態より広い申告は違反ではないが、ストアのページに不要な「収集」が出る）。
