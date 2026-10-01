# Mosaic — クラス編成オプティマイザー

生徒の特性（性別・学力・支援の必要性など）が各クラスに均等に散らばるよう、クラス分け・グループ分けを自動で行う Web アプリ。

**▶ https://ohru131.github.io/ClassClassify/**

- 登録・トークン不要、完全無料
- 名簿データはブラウザ内だけで処理（サーバーへ送信しない）
- 「同じ組にしたい」「別の組にしたい」組み合わせに対応
- 名簿エディタ: 生徒・特性の一覧表示、項目ごとの絞り込み・並べ替え、セル編集、生徒・項目の追加削除、チェックした生徒から「同じ組」「別の組」を一括作成、矛盾する指定の警告
- 結果をドラッグ＆ドロップで手直し → 集計を即時再計算
- 結果を Excel で保存
- Google スプレッドシートから読み込み・結果の書き出しに対応（[設定手順](docs/google-setup.md)）
- 日本語・English・한국어・Español・Deutsch・Português (Brasil) に対応（ヘッダーで切り替え）

### 言語

- 既定の言語は、URL の `?lang=`（`ja` / `en` / `ko` / `es` / `de` / `pt-BR`。`pt` だけでも可）→ 前回選んだ言語（ブラウザに保存）→ ブラウザの言語 → 英語 の順で決まる。例: `https://ohru131.github.io/ClassClassify/?lang=ko`
- Excel のシート名・見出し・結果の出力、ひな形、Google スプレッドシートの出力は選択中の言語で出す。**読み込みはどの言語のシート名・見出しでも受け付ける**（日本語のファイルは従来どおり）。
- サンプル名簿は言語ごとに、その国の学校でクラス分けに配慮される項目で作ってある（`public/samples/<lang>/`、生成は `npm run samples:generate`。項目と根拠は [docs/i18n-glossary.md](docs/i18n-glossary.md) 第3節）。
- 訳語は [docs/i18n-glossary.md](docs/i18n-glossary.md) に揃える。言語の定義はスマホ版と共通（`src/i18n/languages.ts`）、UI 文言は `src/copy/`（英語のキー集合が正で、欠けると型エラー）。

### スマホ・タブレット版（`mobile/`）

同じソルバーを使う Expo / React Native アプリ（Android・iOS、学校の Chromebook・タブレット対応）。
名簿の読み込み・編集・編成・手動移動は無料（広告なし）、Excel での書き出しと印刷・PDF は Pro（買い切り・サブスクなし）。
名簿データは端末内だけに保存し、外部へ送るのは購入確認（RevenueCat）の匿名 ID とレシートだけ。セットアップ・ビルド・課金の設定は [mobile/README.md](mobile/README.md)。
日本語・英語・韓国語・スペイン語・ドイツ語・ポルトガル語（ブラジル）に対応（用語集は [docs/i18n-glossary.md](docs/i18n-glossary.md)、ストア掲載文は [docs/store-listing.md](docs/store-listing.md)）。Web 版はこれまでどおり無料のまま。

Google Play への提出一式:

- [docs/play-console/submission-checklist.md](docs/play-console/submission-checklist.md) — Play Console の全項目を触る順に（データセーフティ・レーティング・審査員向けの手順・国別価格は同じフォルダ）
- プライバシーポリシー: **https://ohru131.github.io/ClassClassify/privacy/**（`privacy/index.html`。本文はスマホ版の `mobile/lib/copy/privacy.ts` からビルド時に6言語ぶん埋め込む。`?lang=ko` などで切り替え）
- ストア用の画像: [submission-assets/README.md](submission-assets/README.md)（アイコン・フィーチャーグラフィック・スマホ／タブレット／Chromebook のスクリーンショット）

## 使い方

1. 画面の「Excel ひな形」から取得、または「スプレッドシートでひな形を作成」で Google ドライブに作成し、名簿を記入
2. Excel をアップロード、または Google スプレッドシートを選択（サンプルで試すことも可）
3. 項目ごとの重み・クラス数・探索時間を調整して「クラス編成を実行」

### Excel の形式

| シート | 内容 |
| --- | --- |
| `設定` | A列に項目名、B列に値。`クラス数`（必須）、`1クラスの最大人数`（任意） |
| `生徒名簿` | 1行目: 重み（C列以降）、2行目: 見出し（`NO`, `名前`, 項目…）、3行目以降: 生徒 |
| `同じ組ペア` | 1行に同じ組にしたい生徒の NO を並べる（3人以上も可） |
| `別の組ペア` | 1行に互いに別の組にしたい生徒の NO を並べる |

項目の値は自動判定する。

- 値が1種類（○ と空欄など）→「該当」者数を均等化
- 値が数種類（1/2/3 など）→ 値ごとの人数を均等化
- 7種類以上の数値（点数など）→ クラス平均を均等化

## 最適化アルゴリズム

旧版は QUBO（二次制約なし二値最適化）に定式化し、Fixstars Amplify / OpenJij のアニーリングで解いていた。
QUBO では「1人1クラス」「人数」「ペア条件」をすべてペナルティ項として混ぜるため、重みの調整次第で制約違反の解が出る。

新版は問題構造を直接扱う専用の焼きなまし法（Simulated Annealing）をブラウザ上で実行する。

- 解表現: 各生徒のクラス番号（1人1クラスは常に成立）
- 「同じ組ペア」: union-find で1ブロックに束ね、常に同じ組へ配置（ハード制約）
- 近傍: ブロックの移動・交換。人数・別の組ペアは大きなペナルティ
- 目的関数: 各項目のクラス別人数が理想の整数範囲 `[⌊合計/K⌋, ⌈合計/K⌉]` から外れた量 × 重み ＋ 中央へ寄せる微小な二乗項
- 差分計算で毎秒数百万回の近傍評価、Web Worker で CPU コア数ぶん並列マルチスタート、最後に貪欲法で局所最適化

同一条件（240名・8組・30項目、10秒）での比較では、ずれの合計が本手法 21、OR-Tools CP-SAT 43、HiGHS（MILP）128 だった。
付属サンプルはいずれも数秒で「全項目が理想範囲内・条件違反 0」の解に到達する。

## 開発

```bash
npm install
npm run dev     # 開発サーバー
npm test        # ソルバー・多言語・サンプルのテスト
npm run samples:generate   # 言語別のサンプル名簿を作り直す（public/samples/）
npm run build   # dist/ に静的ファイルを出力
```

`main` への push で GitHub Actions がビルドし、GitHub Pages へデプロイする（Settings → Pages → Source: GitHub Actions）。
Web 版（`index.html`）とプライバシーポリシー（`privacy/index.html` → `/privacy/`）の2ページを出す。

ストア提出用のスクリプト（Playwright・sharp はアプリの依存に入れていない。`scripts/lib/deps.mjs` を参照）:

```bash
(cd mobile && npx expo export --platform web)       # スクリーンショットの元になる Web 書き出し
npx tsx scripts/capture-submission-assets.mjs        # 6言語 × スマホ／7・10インチ／Chromebook のスクリーンショット
node scripts/generate-feature-graphic.mjs            # フィーチャーグラフィック（6言語）とストアアイコン
node scripts/check-submission-assets.mjs             # Play の規格（大きさ・アルファ・枚数）を検査
node scripts/push-play-pricing.mjs                   # Pro の国別価格（既定はドライラン。--commit で反映）
```

ローカルで Google 連携を試すには `.env.local` に `VITE_GOOGLE_CLIENT_ID` / `VITE_GOOGLE_API_KEY` / `VITE_GOOGLE_APP_ID` を書く。

技術スタック: React 19 / TypeScript / Vite / Tailwind CSS v4 / SheetJS
