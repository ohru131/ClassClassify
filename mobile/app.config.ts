import type { ExpoConfig } from 'expo/config'

// ストアに出すアプリの識別子。**公開後は変えないこと**（別アプリ扱いになり、購入の復元もできなくなる）。
const APP_ID = 'com.ohru131.mosaic'

const config: ExpoConfig = {
  // 端末の言語に合わせてホーム画面の名前を変える（locales/*.json）。既定は短い製品名
  name: 'Mosaic',
  slug: 'mosaic-class-classify',
  version: '1.0.0',
  // 学校のタブレット・Chromebook では横向き・分割画面で使うので、向きを固定しない。
  orientation: 'default',
  icon: './assets/icon.png',
  scheme: 'mosaic',
  userInterfaceStyle: 'light',
  ios: {
    supportsTablet: true,
    bundleIdentifier: APP_ID,
    infoPlist: { ITSAppUsesNonExemptEncryption: false },
  },
  android: {
    package: APP_ID,
    adaptiveIcon: { foregroundImage: './assets/adaptive-icon.png', backgroundColor: '#6366F1' },
    // 追加の権限は宣言しない（通知・カメラ・マイク等は使わない）。
    // ネイティブ依存（課金 = Play Billing）が持つ権限はマニフェストのマージで自動的に入る。
    permissions: [],
    // 名簿は端末の外へ出さない。Android の自動バックアップ（Google ドライブへのアプリデータの
    // バックアップ）に AsyncStorage の名簿が含まれないよう、アプリのバックアップを無効にする。
    allowBackup: false,
    // expo-file-system のプラグインが足す旧来のストレージ権限は使わない（取り込みは OS の
    // ファイル選択、書き出しはアプリのキャッシュ＋共有シートで、どちらも権限が要らない）。
    blockedPermissions: ['android.permission.READ_EXTERNAL_STORAGE', 'android.permission.WRITE_EXTERNAL_STORAGE'],
  },
  locales: {
    ja: './locales/ja.json',
    en: './locales/en.json',
    ko: './locales/ko.json',
    es: './locales/es.json',
    de: './locales/de.json',
    'pt-BR': './locales/pt-BR.json',
  },
  web: {
    bundler: 'metro',
    output: 'single',
    favicon: './assets/favicon.png',
  },
  plugins: [
    // タッチパネル必須にしない・リサイズ可能なウィンドウを妨げない（Chromebook・タブレット）
    './plugins/withLargeScreenSupport',
    // release ビルドから開発メニュー用の SYSTEM_ALERT_WINDOW を外す
    './plugins/withoutReleaseOverlayPermission',
    'expo-router',
    'expo-localization',
    'expo-sharing',
    'expo-document-picker',
    ['expo-build-properties', { android: { minSdkVersion: 24 } }],
  ],
  experiments: {
    typedRoutes: false,
  },
}

export default config
