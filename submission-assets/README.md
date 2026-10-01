# ストア提出用の画像（Google Play）

Mosaic スマホ・タブレット版（`mobile/`）を Google Play に掲載するための画像一式。
**すべてスクリプトで作り直せる**（手で加工した画像は無い）。作り方は下の「作り直す」。

```
submission-assets/
├── store/
│   ├── play-icon-512.png                         アプリのアイコン（全言語共通）
│   └── play-feature-graphic-<lang>-1024x500.png  フィーチャーグラフィック（6言語）
└── screenshots/
    ├── phone/<lang>-NN-<name>.png       スマートフォン      1080×1920（9:16）
    ├── tablet7/<lang>-NN-<name>.png     7 インチ タブレット 1200×1920（縦）
    ├── tablet10/<lang>-NN-<name>.png    10 インチ タブレット 1920×1200（横）
    └── chromebook/<lang>-NN-<name>.png  Chromebook          1920×1080（横 16:9）
```

`<lang>` は `ja` / `en` / `ko` / `es` / `de` / `pt-BR`。各言語・各画面サイズに8枚ずつ（6言語 × 4サイズ × 8枚 = 192枚）。

## Play Console のどの枠へ上げるか

「拡大」→「メインのストアの掲載情報」（言語ごとに「翻訳を管理」で切り替える）。

| Play の枠 | 規格 | 上げるファイル |
|---|---|---|
| アプリのアイコン | 512×512・32 ビット PNG | `store/play-icon-512.png`（全言語同じ。角丸と影は Play が付けるので四角いまま） |
| フィーチャー グラフィック | 1024×500・JPEG / 24 ビット PNG（アルファなし） | `store/play-feature-graphic-<lang>-1024x500.png` |
| スマートフォンのスクリーンショット | 2〜8枚・各辺 320〜3840px・長辺:短辺 ≤ 2:1 | `screenshots/phone/<lang>-*.png` |
| 7 インチ タブレットのスクリーンショット | 最大8枚・同上 | `screenshots/tablet7/<lang>-*.png` |
| 10 インチ タブレットのスクリーンショット | 最大8枚・同上 | `screenshots/tablet10/<lang>-*.png` |
| Chromebook のスクリーンショット | 最大8枚・同上 | `screenshots/chromebook/<lang>-*.png` |

### 掲載のロケールと、使う画像の言語

| Play のロケール（`docs/store-listing.md`） | 画像の `<lang>` |
|---|---|
| en-US（既定）・en-AU・en-GB | `en` |
| ja-JP | `ja` |
| ko-KR | `ko` |
| es-419・es-ES | `es`（アプリの UI は中南米寄りのスペイン語 1 種類） |
| de-DE | `de` |
| pt-BR | `pt-BR` |

### 各スクリーンショットの中身（どの言語もその言語のサンプル名簿「80名・4組」で撮影）

| NN-name | 写っているもの |
|---|---|
| `01-roster` | サンプルを読み込んだ直後の名簿（生徒80名・項目・同じ組／別の組の件数）。大画面は表＋右の編集欄 |
| `02-pairs` | 「別の組」の指定の一覧（互いに別の組にする生徒の組み合わせ） |
| `03-run` | 設定・実行: クラス数・最大人数・探索時間・均等にする項目と、計算中の進み具合 |
| `04-results` | 結果: クラス数・人数差 0・バランス「完全」・条件違反 0 と、ペア指定の色分け |
| `05-balance` | バランス分析: 項目ごとのクラス別人数と理想の範囲 |
| `06-move` | 手直し: 生徒をタップして移動先の組を選ぶパネル（スマホは1組ずつ、左右スワイプで切り替え） |
| `07-print` | 印刷・PDF のプレビュー（A4 縦。クラス別名簿・ペア指定の凡例・集計）。Pro の機能 |
| `08-pro` | Pro の画面（買い切りで Excel 書き出し・印刷 / PDF、言語の選択、データの扱い） |

### Play へ上げる順（言語ごと）

一覧の最初の2〜3枚が検索結果・掲載ページで目に入る。**最初は「結果が均等になっている」絵**にする。

| 言語 | 順番（NN） | 理由 |
|---|---|---|
| ja・en・es・de・pt-BR | 04 → 05 → 06 → 02 → 01 → 03 → 07 → 08 | 結果（均等）→ 根拠の表 → 手直しできる → 条件を指定できる → 入力 → 実行 → 印刷 → Pro |
| ko | 04 → 02 → 05 → 06 → 01 → 03 → 07 → 08 | 韓国は「分離配置」（別の組）がいちばんの訴求点（`docs/research/overseas-demand.md` C 節）なので2枚目に |

タブレット・Chromebook も同じ順で8枚すべて上げる（大画面向けの掲載・おすすめの対象になるには各4枚以上が要る【要確認】）。

## 撮り方と注意

- **スマホ版の Web 書き出し（`mobile/dist`）を Playwright（Chromium）で操作して撮っている。** 画面の構成・文言・サンプルは
  ネイティブと同じコードだが、**フォント（Noto Sans）・ステータスバー・ナビゲーションバー・スクロールバーは実機と違う**。
  端末の枠・評価・「No.1」のような装飾は付けていない（実際の画面以外を載せない）。
- 画面サイズ（dp）× 倍率: phone 360×640 ×3、tablet7 600×960 ×2、tablet10 1280×800 ×1.5、chromebook 1280×720 ×1.5。
  幅 768dp 以上で大画面のレイアウト（クラスの横並び・名簿の表）に切り替わる（`mobile/lib/layout.ts`）ので、7 インチ（縦 600dp）はスマホと同じ1列。
- `07-print` は Pro の印刷用 HTML（`mobile/lib/print-html.ts`）を `?pro=preview`（Web 版だけで有効な確認用のモード。購入はできない）で開き、
  A4 の幅で組んだものを他の画像と同じ画素数で撮っている。
- `08-pro` は購入前の状態。Web 版にだけ出る「購入はアプリ版で」の注記を消し、Web では押せない購入ボタンの半透明をネイティブと同じ見た目に戻している。
  ネイティブではボタンに Play の価格（例: 「Pro を購入 – ¥980」）が入る。
- 名簿の名前はすべてサンプル用の架空の名前（`public/samples/`、`scripts/generate-samples.ts`）。
- PNG はアルファなしの **24 ビット RGB**。容量を抑えるため、保存の前に 256 色へ減色してから RGB に戻している（sharp があれば。UI の絵は色数が少なく、文字のにじみは目視で分からない）。画像 199 枚で約 32 MB。**8 ビット（パレット）PNG のまま上げないこと**（Play の規格は 24 ビット）。`check-submission-assets.mjs` が検査する。

### フォント（サンドボックスで撮るとき）

この環境（Linux）の標準フォントは日本語が IPA ゴシック（太字なし）、韓国語は文泉驛（中国語の字形）しかなく、
韓国語が不自然になる。撮影の前に Noto Sans・Noto Sans JP・Noto Sans KR を `~/.local/share/fonts/` に入れ、
`~/.config/fontconfig/fonts.conf` で sans-serif の優先順を Noto Sans → Noto Sans JP → Noto Sans KR にした
（フォントは npm の `@expo-google-fonts/noto-sans{,-jp,-kr}` の TTF を使った。リポジトリには入れていない）。
`fc-match 'sans-serif:lang=ko'` が `Noto Sans KR` を返すことを確かめてから撮る。Windows・macOS では OS の CJK フォントで足りる。

## 作り直す

```bash
# 依存（アプリの依存ではないので package.json には入れていない）
#   playwright-core と sharp が node_modules にあるか、SUBMISSION_DEPS=<それを入れたディレクトリ> を指定する
#   Chromium は CHROMIUM_PATH → /opt/pw-browsers/chromium → Windows の Chrome / Edge の順に探す（playwright install はしない）
npm i --no-save playwright-core sharp

(cd mobile && npx expo export --platform web)     # mobile/dist を作る
npx tsx scripts/capture-submission-assets.mjs      # 全部（6言語 × 4サイズ × 8枚、10分ほど）
npx tsx scripts/capture-submission-assets.mjs --lang ko --form phone,tablet10 --only 04-results
node scripts/generate-feature-graphic.mjs          # フィーチャーグラフィック6枚とアイコン
node scripts/check-submission-assets.mjs           # 大きさ・アルファ・縦横比・枚数を検査
```

- 撮影スクリプトは画面の文言を `mobile/lib/copy/*.ts`・サンプル名を `mobile/lib/samples.generated.ts` から読むので、
  文言を変えてもスクリプトの直しは要らない（ボタンが見つからなければタイムアウトで止まる）。
- フィーチャーグラフィックの見出しは `scripts/generate-feature-graphic.mjs` の `LOCALES`。`docs/store-listing.md` の短い説明と同じ主張に揃える。
- 画面を変えたら撮り直し、`check-submission-assets.mjs` を通してからコミットする。
