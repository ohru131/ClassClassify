# Google スプレッドシート連携の設定手順

Google Cloud で「OAuth クライアント ID」「API キー」を作り、GitHub のリポジトリ変数に登録する。無料・約10分。

アクセス範囲は `drive.file` スコープのみ。アプリが触れられるのは、ユーザーが選択画面（Google Picker）で選んだファイルと、アプリが新規作成したファイルだけ。Drive 全体は見えない。

## 1. プロジェクトを作る

1. https://console.cloud.google.com/ を開き、上部のプロジェクト選択 →「新しいプロジェクト」→ 名前 `Mosaic` などで作成
2. 「ダッシュボード」（または「プロジェクトの設定」）に表示される **プロジェクト番号**（数字のみ）を控える → `GOOGLE_APP_ID`

## 2. API を有効にする

「API とサービス」→「ライブラリ」で次の3つを検索し、それぞれ「有効にする」。

- Google Drive API
- Google Sheets API
- Google Picker API

## 3. OAuth 同意画面（Google Auth Platform）

1. 「API とサービス」→「OAuth 同意画面」→「開始」
2. アプリ名 `Mosaic`、サポートメールを入力。対象は **外部**（Google Workspace の組織内だけで使うなら **内部**）
3. 「データアクセス」→「スコープを追加」で `.../auth/drive.file` を追加して保存
4. 「対象」→ 公開ステータスを **本番環境に公開**
   - `drive.file` は審査不要のスコープのため、公開してもすぐに誰でも使える
   - 「テスト」のままだと、テストユーザーに登録した Google アカウントしかログインできない

## 4. OAuth クライアント ID

1. 「API とサービス」→「認証情報」→「認証情報を作成」→「OAuth クライアント ID」
2. アプリケーションの種類: **ウェブ アプリケーション**
3. 「承認済みの JavaScript 生成元」に追加:
   - `https://ohru131.github.io`
   - `http://localhost:5173`（ローカル開発する場合）
4. 作成後に表示される **クライアント ID**（`xxxx.apps.googleusercontent.com`）を控える → `GOOGLE_CLIENT_ID`
   - クライアントシークレットは不要

## 5. API キー

1. 「認証情報を作成」→「API キー」
2. 作成したキーの「キーを編集」で制限をかける（推奨）
   - アプリケーションの制限: **ウェブサイト** → `https://ohru131.github.io/*`（必要なら `http://localhost:5173/*`）
   - API の制限: **Google Picker API** のみ
3. キーを控える → `GOOGLE_API_KEY`

## 6. GitHub に登録してデプロイ

1. https://github.com/ohru131/ClassClassify/settings/variables/actions を開く
2. 「Variables」タブ →「New repository variable」で3つ登録（Secrets ではなく Variables）

   | Name | Value |
   | --- | --- |
   | `GOOGLE_CLIENT_ID` | 手順4のクライアント ID |
   | `GOOGLE_API_KEY` | 手順5の API キー |
   | `GOOGLE_APP_ID` | 手順1のプロジェクト番号 |

3. https://github.com/ohru131/ClassClassify/actions/workflows/deploy.yml →「Run workflow」で再デプロイ

クライアント ID・API キーはブラウザに配信される前提の値で、秘密情報ではない（上記の生成元・リファラ制限で他サイトからの悪用を防ぐ）。

## 使い方

- ひな形: 「スプレッドシートでひな形を作成」で、記入例つきの `Mosaic 名簿ひな形` を自分の Drive に作成（`使い方` シート付き）。書き換えて使う
- 読み込み: 手順1のカード右側「Google スプレッドシート」→ Google でログイン → 名簿のシートを選択。ひな形と同じシート構成（`設定` / `生徒名簿` / `同じ組ペア` / `別の組ペア`）にしておく。Drive 上の .xlsx も選べる
- 書き出し: 結果画面の「元のシートに書き出す」で、読み込んだスプレッドシートに `組分け_MMDD-HHmm` などのタブを追加。Excel から読み込んだ場合は「スプレッドシートに保存」で新しいスプレッドシートを作る

## うまくいかないとき

| 症状 | 対処 |
| --- | --- |
| `redirect_uri_mismatch` / `origin_mismatch` | 手順4の JavaScript 生成元に `https://ohru131.github.io` があるか確認（末尾スラッシュなし、パスなし） |
| 「このアプリは Google で確認されていません」 | 同意画面が「テスト」のまま。手順3-4で本番環境に公開するか、テストユーザーに追加 |
| Picker が「API キーが無効」 | API キーのリファラ制限と、Google Picker API の有効化を確認 |
| シートを選んでも 404 / 権限エラー | `GOOGLE_APP_ID` がプロジェクト番号（数字）になっているか確認。プロジェクト ID（英字）ではない |
| Google ボタンが表示されない | 3つの Variables が登録されているか確認し、ワークフローを再実行 |
