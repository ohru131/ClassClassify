# 審査員向けの手順（App access・テストの案内）

Play Console →「アプリのコンテンツ」→「アプリへのアクセス」と、審査・クローズドテストの依頼文に使う。

## アプリへのアクセス（App access）の回答

- **「すべての機能を、特別なアクセス権なしで利用できる」**を選ぶ。ログイン・アカウント・招待コードは無い。
- Pro（Excel 書き出し・印刷 / PDF）は買い切りの課金で開く機能で、ログインで隠しているわけではないので、
  App access の「ログイン情報」は要らない。審査で Pro まで見てもらいたいときは、下の「Pro を試す」を依頼文に添える。

## 審査員に渡す説明（英語。Play Console の「その他の手順」欄・審査への返信にそのまま貼れる）

```
FairClass is a class placement tool for teachers. No account or login is required, and all data stays on the device.

How to test (about 2 minutes):
1. Open the app. On the "Roster" tab, under "Try a sample", tap "Class placement (80 students, 4 classes)".
   (Samples are bundled in the app for every language; no file is needed.)
2. Optional: open the "Apart" tab to see students that must be placed in different classes.
3. Go to "Setup & run" and tap "Build classes". The progress bar runs for about 10 seconds.
4. The "Results" tab shows the classes, the balance tables and any unmet condition.
   Tap a student, then a class button, to move them; the totals update immediately.
5. Under "Export as a file", "Excel", "PDF" and "Print" are Pro features (one-time in-app purchase, not a subscription).
   Without Pro they open the Pro screen. License testers can buy Pro without being charged.

Importing an Excel file: use "Choose an Excel file (.xlsx)". Sample files are available at
https://ohru131.github.io/ClassClassify/samples/en/sample1.xlsx (also ja, ko, es, de, pt-BR).

The app shows no ads and has no account. Privacy policy: https://ohru131.github.io/ClassClassify/privacy/
```

> ボタン名は英語 UI（`mobile/lib/copy/en.ts`）のもの。審査員の端末が日本語なら「名簿 > サンプルで試す > クラス分け（80名・4組）」
> 「設定・実行 > クラス編成を実行」「結果 > ファイルで書き出す > Excel・PDF・印刷」になる（`mobile/lib/copy/ja.ts`）。**文言を変えたらここも直す。**

## Pro を試す（課金せずに）

| 方法 | 使いどころ | 手順 |
|---|---|---|
| **ライセンステスター** | 審査員・クローズドテストの参加者・自分の実機確認 | Play Console →「設定」→「ライセンス テスト」に Google アカウント（Gmail）を追加。その端末で購入すると「テスト用のカード」が選べ、課金されない。RevenueCat には sandbox の購入として届き、`pro` entitlement が付く |
| **プロモーション コード** | 先行して使ってもらう先生・レビュー依頼 | Play Console →「収益化」→「プロモーション」で `fairclass_pro` のコードを発行（四半期ごとの上限あり）。Play ストアのアプリで「コードを利用」 |

- **買い切りには無料トライアルを設定できない**（Play の無料試用期間は定期購入だけの機能）。審査員にはライセンステスターかプロモコードを使ってもらう。
- 審査用・テスト用のビルドにも RevenueCat の Android 公開 SDK キー（`EXPO_PUBLIC_REVENUECAT_ANDROID_API_KEY`）を入れておく。
  入っていないと購入・復元ボタンが「キーが設定されていません」と出て Pro を確かめられない（`mobile/lib/revenuecat-provider.tsx`）。

## Excel の読み込みを試す

- アプリにはサンプル名簿が言語ごとに3つ入っている（`mobile/lib/samples.generated.ts`。ファイルは不要）。
- ファイルから読み込む経路を試すときは、Web 版で公開しているサンプルを端末にダウンロードして「Excel ファイルを選ぶ」から選ぶ:
  - `https://ohru131.github.io/ClassClassify/samples/<lang>/sample1.xlsx`（80名・4組、ペア指定あり）
  - `…/sample2.xlsx`（80名・シンプル）、`…/sample-group.xlsx`（30名・6班）
  - `<lang>` は `ja` / `en` / `ko` / `es` / `de` / `pt-BR`。**どの言語のシート名・見出しのファイルも読み込める。**
- Excel のひな形は「Excel のひな形を入手」で共有シートから保存できる（無料）。

## 事前リリースレポート（Pre-launch report）で起きうること

- Robo テストは OS のファイル選択画面（Excel の読み込み）から先へ進めない。サンプルのボタンで名簿を作れるので、
  そこから先（実行・結果・手直し）は Robo でも回る。
- 書き出し・印刷を押すと Pro の画面へ移る（無料版の仕様）。購入ダイアログはテスト端末では完了しないことがある。
- 指摘されやすいのはアクセシビリティ（コントラスト・タップ領域・ラベル）。ブロッカーではないが、出たら次の版で直す。
