// 本文は Web 版の公開ページと共通（リポジトリ直下の src/i18n/privacy.ts）。
// mobile/ の外に置くのは、Web 版のビルドが mobile/tsconfig.json（expo/tsconfig.base を継承）を
// 読まずに済むようにするため。CI は mobile/ の依存を入れないので、ここに置くと変換に失敗する。
export * from '../../../src/i18n/privacy'
