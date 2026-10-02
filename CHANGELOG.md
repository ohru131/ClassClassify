# Changelog

このファイルは [Keep a Changelog](https://keepachangelog.com/ja/1.1.0/) の形式に従う。
各 PR が `## [Unreleased]` に1行足し、リリース時にその塊を `## [X.Y.Z] - 日付` へ改名して同じコミットにタグを打つ。
Play の「このバージョンの新機能」はここから写す（`node scripts/release.mjs notes`）。

## [Unreleased]

## [1.0.0] - 2026-10-02

- 初回リリース（Google Play クローズドテスト）。教員向けのクラス編成ツール（Android）。サンプル名簿・Excel の読み込み・クラス編成・結果の手直しに対応する。
- Google ドライブ（OS のファイル選択）から名簿 Excel を読み込み、結果を保存できる。
- 結果の Excel 書き出し・印刷・PDF 共有は Pro（買い切り）で提供する。
- 日本語・英語・ドイツ語・スペイン語・韓国語・ポルトガル語に対応する。

[Unreleased]: https://github.com/ohru131/ClassClassify/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ohru131/ClassClassify/releases/tag/v1.0.0
